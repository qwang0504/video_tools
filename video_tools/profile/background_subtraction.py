from video_tools import (
    InMemory_OpenCV_VideoReader, Polarity,
    NoBackgroundSub, BackroundImage, InpaintBackground, 
    StaticBackground, DynamicBackground, DynamicBackgroundMP
)
from image_tools import im2single, im2gray
from tqdm import tqdm
import cProfile
import pstats

# video reader
video_reader = InMemory_OpenCV_VideoReader()
video_reader.open_file(
    filename='toy_data/freely_swimming_param.avi',
    safe=False,
    memsize_bytes=4e9,
    single_precision=True,
    grayscale=True
)
num_frames = video_reader.get_number_of_frame()

# background subtraction
background_sub = BackroundImage(
    polarity = Polarity.DARK_ON_BRIGHT,
    image_file_name = 'toy_data/freely_swimming_param.png',
    use_gpu=True
)
background_sub.initialize()

with cProfile.Profile() as pr:

    print('Background subtraction ...')
    for i in tqdm(range(num_frames)):
        (rval, frame) = video_reader.next_frame()
        if not rval:
            raise RuntimeError('VideoReader was unable to read the whole video')
        
        # background expects single precision grayscale
        frame_gray = im2single(im2gray(frame))
        
        # background sub
        background_sub.subtract_background(frame_gray)

video_reader.close()
ps = pstats.Stats(pr)
ps.dump_stats('raw_speed.prof')