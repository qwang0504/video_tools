import numpy as np
from numpy.typing import NDArray
from scipy import stats
from typing import Protocol, Tuple, Optional
from collections import deque
from image.imconvert import im2gray, im2single
from multiprocessing import Process, Event, Pool, cpu_count
from multiprocessing.sharedctypes import RawArray, Value
import ctypes
from tqdm import tqdm
import cv2
from abc import ABC, abstractmethod
from functools import partial


def my_mode(x: NDArray) -> NDArray:
    return stats.mode(x, axis=2, keepdims=False).mode

def mode(arr: NDArray, num_processes: int = cpu_count()):
    '''
    multiprocess computation of mode along 3rd axis
    '''
    
    # create iterable
    chunk_size = int(arr.shape[0] / num_processes) 
    chunks = [arr[i:i + chunk_size] for i in range(0, arr.shape[0], chunk_size)] 

    # distribute work
    with Pool(processes=num_processes) as pool:
        res = pool.map(my_mode, chunks)

    # reshape result
    out = np.vstack(res)
    out.reshape(arr.shape[:-1])

    return out

class BackgroundSubtractor(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def subtract_background(self, image: NDArray) -> NDArray:
        pass

    @abstractmethod
    def get_background_image(self) -> Optional[NDArray]:
        pass

    def is_initialized(self):
        return self.initialized
    
class VideoReader(Protocol):
    def next_frame(self) -> Tuple[bool,NDArray]:
        """return the next frame in the movie, 
        and a boolean if the operation succeeded"""

    def get_number_of_frame(self) -> int:
        """return number of frames in the movie"""

    def seek_to(self, index) -> None:
        """go to a specific frame retrieveable with a call to next_frame"""

    def get_width(self) -> int:
        """return width"""
    
    def get_height(self) -> int:
        """return height"""

    def get_num_channels(self) -> int:
        """return number of channels"""

    def get_type(self) -> np.dtype:
        """return data type"""

    def reset_reader(self) -> None:
        """reset reader to video beginning"""
        
class NoBackgroundSub(BackgroundSubtractor):
    def initialize(self):
        self.initialized = True

    def get_background_image(self) -> Optional[NDArray]:
        return None

    def subtract_background(self, image: NDArray) -> NDArray:
        return image 

class BackroundImage(BackgroundSubtractor):
    def __init__(self, image_file_name) -> None:
        super().__init__()
        self.image_file_name = image_file_name
        self.background = None

    def initialize(self) -> None:
        image = cv2.imread(self.image_file_name)
        self.background = im2single(im2gray(image)) 
        self.initialized = True
    
    def get_background_image(self) -> Optional[NDArray]:
        if self.initialized:
            return self.background
        else:
            return None

    def subtract_background(self, image: NDArray) -> NDArray:
        return image - self.background

class StaticBackground(BackgroundSubtractor):
    '''
    Use this if you want to track if you already have the full video
    and the background doesn't change with time
    '''
    def __init__(
            self,
            video_reader: VideoReader, 
            num_sample_frames: int = 500
        ) -> None:
        super().__init__()
        self.video_reader = video_reader
        self.num_sample_frames = num_sample_frames
        self.background = None

    def sample_frames_evenly(self) -> NDArray:
        '''
        Sample frames evenly from the whole video and add to collection
        '''
        height = self.video_reader.get_height()
        width = self.video_reader.get_width()
        numframes = self.video_reader.get_number_of_frame()
        sample_indices = np.linspace(0, numframes-1, self.num_sample_frames, dtype = np.int64)
        sample_frames = np.empty((height, width, self.num_sample_frames), dtype=np.float32)
        for i,index in enumerate(tqdm(sample_indices)):
            self.video_reader.seek_to(index)
            rval, frame = self.video_reader.next_frame()
            if rval:
                sample_frames[:,:,i] = im2single(im2gray(frame))
            else:
                RuntimeError('StaticBackground::sample_frames_evenly frame not valid')
        return sample_frames

    def compute_background(self, frame_collection: NDArray) -> None:
        """
        Take sample images from the video and return the mode for each pixel
        Input:
            sample_frames: m x n x k numpy.float32 array where k is the number of 
            frames
        Output:
            background: m x n numpy.float32 array
        """
        self.background = mode(frame_collection)

    def initialize(self):
        print('Static background')
        print('Getting sample frames from video...')
        frame_collection = self.sample_frames_evenly()
        print('Compute background...')
        self.compute_background(frame_collection)
        self.video_reader.reset_reader()
        print('...done')
        self.initialized = True

    def get_background_image(self) -> Optional[NDArray]:
        if self.initialized:
            return self.background
        else:
            return None

    def subtract_background(self, image: NDArray) -> NDArray:
        return image - self.background 

class DynamicBackground(BackgroundSubtractor):
    '''
    Use this if you want to extract background from a streaming source 
    (images arriving continuously) or if the background is changing 
    with time. Recomputing the background takes time, do not use 
    for time sensitive applications 
    '''
    def __init__(
        self, 
        num_sample_frames: int, 
        sample_every_n_frames: int
        ) -> None:

        super().__init__()
        self.num_sample_frames = num_sample_frames
        self.sample_every_n_frames = sample_every_n_frames
        self.frame_collection = deque(maxlen=num_sample_frames)
        self.curr_image = 0
        self.background = None

    def compute_background(self):
        frames = np.asarray(self.frame_collection).transpose((1,2,0))
        self.background = mode(frames)

    def subtract_background(self, image: NDArray) -> NDArray: 
        if self.curr_image % self.sample_every_n_frames == 0:
            self.frame_collection.append(image)
            self.compute_background()
        self.curr_image = self.curr_image + 1
        return image - self.background
    
    def initialize(self) -> None:
        self.initialized = True

    def get_background_image(self) -> Optional[NDArray]:
        if self.initialized:
            return self.background
        else:
            return None

class BoundedQueue:
    def __init__(self, size, maxlen):
        self.size = size
        self.maxlen = maxlen
        self.itemsize = np.prod(size)

        self.numel = Value('i',0)
        self.insert_ind = Value('i',0)
        self.data = RawArray(ctypes.c_float, int(self.itemsize*maxlen))
    
    def append(self, item) -> None:
        data_np = np.frombuffer(self.data, dtype=np.float32).reshape((self.maxlen, *self.size))
        np.copyto(data_np[self.insert_ind.value,:,:], item)
        self.numel.value = min(self.numel.value+1, self.maxlen) 
        self.insert_ind.value = (self.insert_ind.value + 1) % self.maxlen

    def get_data(self):
        if self.numel.value == 0:
            return None
        else:
            data_np = np.frombuffer(self.data, dtype=np.float32).reshape((self.maxlen, *self.size))
            return data_np[0:self.numel.value,:,:]

class DynamicBackgroundMP(BackgroundSubtractor):
    '''
    Use this if you want to extract background from a streaming source 
    (images arriving continuously) or if the background is changing 
    with time. Recomputes the background in a different process
    for time sensitive applications.
    '''
    def __init__(
        self, 
        width,
        height,
        num_images = 500, 
        every_n_image = 100,
    ) -> None:

        super().__init__()
        
        self.width = width
        self.height = height
        self.num_images = num_images
        self.every_n_image = every_n_image
        self.counter = 0
        
        self.stop_flag = Event()
        self.background = RawArray(ctypes.c_float, width*height)
        self.image_store = BoundedQueue((width,height),maxlen=num_images)

    def start(self):
        self.proc_compute = Process(
            target=self.compute_background, 
            args=(self.stop_flag, self.image_store, self.background)
        )
        self.proc_compute.start()
        
    def stop(self):
        self.stop_flag.set()
        self.proc_compute.join()

    @staticmethod
    def compute_background(
        stop_flag: Event, 
        image_store: BoundedQueue, 
        background: RawArray
    ):

        while not stop_flag.is_set():
            data = image_store.get_data().transpose((1,2,0))
            if data is not None:
                bckg_img = mode(data)
                background[:] = bckg_img.flatten()

    def get_background(self) -> NDArray:
        return np.frombuffer(self.background, dtype=np.float32).reshape((self.width,self.height))
    
    def subtract_background(self, image : NDArray) -> NDArray:
        """
        Input an image and update the background model
        """
        if self.counter % self.every_n_image == 0:
            self.image_store.append(image)
            if self.counter == 0:
                self.background[:] = image.flatten()
        self.counter = self.counter + 1
        bckg = self.get_background()
        return image - bckg

    def initialize(self) -> None:
        self.start()
        self.initialized = True
    
    def get_background_image(self) -> Optional[NDArray]:
        if self.initialized:
            bckg = self.get_background()
            return bckg
        else:
            return None
