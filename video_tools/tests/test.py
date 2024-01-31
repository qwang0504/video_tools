import unittest
import numpy as np
from video_tools import (
    OpenCV_VideoWriter, FFMPEG_VideoWriter_GPU, FFMPEG_VideoWriter_CPU
)

class test_video_writer(unittest.TestCase):

    def test_OpenCV_VideoWriter(self):
        writer = OpenCV_VideoWriter(
            filename = 'test_00.avi',
            height = 256,
            width = 256,
            fps = 25, 
        )
        writer.write_frame(np.zeros((256,256,3), dtype=np.uint8))
        writer.write_frame(255*np.ones((256,256,3), dtype=np.uint8))
        writer.write_frame(np.zeros((256,256,3), dtype=np.uint8))
        writer.close()

    def test_FFMPEG_VideoWriter_GPU(self):
        writer = FFMPEG_VideoWriter_GPU(
            filename = 'test_01.avi',
            height = 256,
            width = 256,
            fps = 25, 
        )
        writer.write_frame(np.zeros((256,256,3), dtype=np.uint8))
        writer.write_frame(255*np.ones((256,256,3), dtype=np.uint8))
        writer.write_frame(np.zeros((256,256,3), dtype=np.uint8))
        writer.close()

    def test_FFMPEG_VideoWriter_CPU(self):
        writer = FFMPEG_VideoWriter_CPU(
            filename = 'test_02.avi',
            height = 256,
            width = 256,
            fps = 25, 
        )
        writer.write_frame(np.zeros((256,256,3), dtype=np.uint8))
        writer.write_frame(255*np.ones((256,256,3), dtype=np.uint8))
        writer.write_frame(np.zeros((256,256,3), dtype=np.uint8))
        writer.close()

class test_video_reader(unittest.TestCase):

    def test_r(self):
        pass

class test_background_subtraction(unittest.TestCase):
    
    def test_(self):
        pass   

class test_video_pocessor(unittest.TestCase):
    
    def test_(self):
        pass   

if __name__ == '__main__':
    unittest.main()
