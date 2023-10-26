import cv2
from numpy.typing import NDArray
import numpy as np
from typing import Tuple, Optional
import queue
from multiprocessing import Queue, Process, Event 

# TODO: add resizing as an option 
# TODO: Check index error (+-1). Make sure that number of frames is correct (end index valid)
class OpenCV_VideoReader:
    
    def __init__(self):
        self.open = False

    def open_file(
            self, 
            filename: str, 
            safe: bool = False, 
            crop: Optional[Tuple[int,int,int,int]] = None,
            resize: Optional[float] = None
        ) -> None:
            
        self._filename = filename
        self._capture = cv2.VideoCapture(filename)
        self._current_frame = 0
        self._number_of_frames = 0
        self._width = 0
        self._height = 0
        self._num_channels = 0
        self._safe = safe
        self._crop = crop # [left,bottom,width,height]
        self._resize = resize
            
        # count number of frames
        if safe:
            # loop over the whole video and count each frame
            counter = 0
            while True:
                rval, frame = self._capture.read()
                if not rval:
                    break
                counter = counter + 1
            self._number_of_frames = counter

            # reinitialize reader
            self._capture.release()
            self._capture = cv2.VideoCapture(filename)
        else:
            # Trust opencv to return video properties. This is fast but there are known issues 
            # with this. Use at your own risk.
            self._number_of_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # get video properties from first frame
        rval, frame = self._capture.read()
        if not rval:
            raise(RuntimeError(f"Error while reading the first frame")) 
        
        if self._crop is not None:
            self._width = self._crop[2]
            self._height = self._crop[3]
        else:
            self._width = frame.shape[1]
            self._height = frame.shape[0]
        if len(frame.shape) > 2:
            self._num_channels = frame.shape[2]
        else:
            self._num_channels = 1
        self._type = frame.dtype  

        self.reset_reader()
        self.open = True

    def is_open(self) -> bool:
        return self.open
    
    def reset_reader(self) -> None:
        """
        TODO
        """
        self._capture.release()
        self._capture = cv2.VideoCapture(self._filename)
        self._current_frame = 0

    def next_frame(self) -> Tuple[bool,NDArray]:
        """
        TODO
        """
        rval, frame = self._capture.read()
        if rval:
            self._current_frame = self._current_frame + 1
            if self._crop is not None:
                frame = frame[
                    self._crop[1]:self._crop[1]+self._crop[3],
                    self._crop[0]:self._crop[0]+self._crop[2]
                ]
            if self._resize is not None:
                frame = cv2.resize(
                    frame,
                    None,
                    None,
                    self._resize,
                    self._resize,
                    cv2.INTER_NEAREST
                )
        return (rval, frame)
    
    def previous_frame(self) -> Tuple[bool,NDArray]:
        self.seek_to(self._current_frame - 1) # -1 or -2 ?
        return self.next_frame()
    
    def seek_to(self, index):

        # check arguments value
        if not 0 <= index < self._number_of_frames:
            raise(ValueError(f"index should be between 0 and {self._number_of_frames-1}, got {index}"))
        
        if self._safe:
            # if you need to rewind, start from the beginning, otherwise keep going
            if index < self._current_frame:
                # reinitialize video reader
                self.reset_reader()

            # go through the video until index
            counter = self._current_frame
            while counter < index-1:
                rval, frame = self.next_frame()
                if not rval:
                    raise(RuntimeError(f"movie ended while seekeing to frame {index}"))
                counter = counter + 1
 
        else:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, index-1)
            self._current_frame = index-1
    
    def read_frames(self, start: int, stop: int) -> NDArray:
        """
        Read frames between indices start and stop and store them in a numpy array in RAM
        """

        # check arguments value
        if not 0 <= start < self._number_of_frames:
            raise(ValueError(f"start index should be between 0 and {self._number_of_frames-1}"))
        if not start <= stop < self._number_of_frames:
            raise(ValueError(f"stop index should be between {start} and {self._number_of_frames-1}"))
        
        self.seek_to(start)

        # preinitialize arrray. Be aware that this could take a lot of RAM depending on 
        # resolution and number of frames
        if self._num_channels > 1:
            frames = np.empty((self._height, self._width, self._num_channels, stop-start), dtype=self._type)
        else:
            frames = np.empty((self._height, self._width, stop-start), dtype=self._type)

        # read frames
        counter = self._current_frame
        while counter < stop:
            rval, frame = self.next_frame()
            if not rval:
                raise(RuntimeError(f"movie ended while seeking to frame {stop}"))
            frames[:,:,:,counter-start] = frame
            counter = counter + 1

        return frames

    def play(self) -> None:
        """
        TODO
        """

        print("press q. to stop")
        cv2.namedWindow(self._filename)
        
        while True:
            rval, frame = self.next_frame()
            if not rval:
                break
            cv2.imshow(self._filename,frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        cv2.destroyAllWindows()

    def get_width_max(self) -> int:
        return int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height_max(self) -> int:
        return int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def get_width(self) -> int:
        return self._width
    
    def get_height(self) -> int:
        return self._height
    
    def get_num_channels(self) -> int:
        return self._num_channels
    
    def get_type(self) -> np.dtype:
        return self._type
    
    def get_current_frame_index(self) -> int:
        return self._current_frame

    def get_filename(self) -> str:
        return self._filename
    
    def get_number_of_frame(self) -> int:
        return self._number_of_frames

class Buffered_OpenCV_VideoReader(Process):
    '''
    Buffer video file reading. Cannot seek to a specific frame. Faster 
    when you need to process each video frame sequentially
    '''

    def open_file(
            self, 
            filename: str, 
            safe: bool = False, 
            crop: Optional[Tuple[int,int,int,int]] = None
        ):
            
        self._filename = filename
        self._capture = cv2.VideoCapture(filename)
        self._current_frame = 0
        self._number_of_frames = 0
        self._width = 0
        self._height = 0
        self._num_channels = 0
        self._safe = safe
        self._crop = crop # [left,bottom,width,height]
        self._queue = Queue(maxsize=100)
        self._stop = Event()
            
        # count number of frames
        if safe:
            # loop over the whole video and count each frame
            counter = 0
            while True:
                rval, frame = self._capture.read()
                if not rval:
                    break
                counter = counter + 1
            self._number_of_frames = counter

            # reinitialize reader
            self._capture.release()
            self._capture = cv2.VideoCapture(filename)
        else:
            # Trust opencv to return video properties. This is fast but there are known issues 
            # with this. Use at your own risk.
            self._number_of_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # get video properties from first frame
        rval, frame = self._capture.read()
        if not rval:
            raise(RuntimeError(f"Error while reading the first frame")) 
        
        if self._crop is not None:
            self._width = self._crop[2]
            self._height = self._crop[3]
        else:
            self._width = frame.shape[1]
            self._height = frame.shape[0]
        if len(frame.shape) > 2:
            self._num_channels = frame.shape[2]
        else:
            self._num_channels = 1
        self._type = frame.dtype  

        self._capture.release()

    def run(self) -> None:
        self._capture = cv2.VideoCapture(self._filename)
        self._current_frame = 0
        while not self._stop.is_set():
            rval, frame = self.read_frame()
            if not rval:
                break
            self._queue.put((rval, frame))
            
    def next_frame(self):
        return self._queue.get()
    
    def exit(self):
        self._stop.set()

    def read_frame(self) -> Tuple[bool,NDArray]:
        rval, frame = self._capture.read()
        if rval:
            self._current_frame = self._current_frame + 1
            if self._crop is not None:
                frame = frame[
                    self._crop[1]:self._crop[1]+self._crop[3],
                    self._crop[0]:self._crop[0]+self._crop[2]
                ]
        return (rval, frame)
    
    def reset_reader(self) -> None:
        self._capture.release()
        self._capture = cv2.VideoCapture(self._filename)
        self._current_frame = 0

    def get_width(self) -> int:
        return self._width
    
    def get_height(self) -> int:
        return self._height
    
    def get_num_channels(self) -> int:
        return self._num_channels
    
    def get_type(self) -> np.dtype:
        return self._type
    
    def get_current_frame_index(self) -> int:
        return self._current_frame

    def get_filename(self) -> str:
        return self._filename
    
    def get_number_of_frame(self) -> int:
        return self._number_of_frames