import unittest
import numpy as np
from video_tools import (
    OpenCV_VideoWriter, FFMPEG_VideoWriter_GPU, FFMPEG_VideoWriter_CPU,
    OpenCV_VideoReader, Buffered_OpenCV_VideoReader, InMemory_OpenCV_VideoReader
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

    def test_OpenCV_VideoReader(self):
        reader = OpenCV_VideoReader()
        reader.open_file(filename = 'test_00.avi')
        
        # check metadata are correctly retrieved
        self.assertEqual(reader.get_height(), 256)
        self.assertEqual(reader.get_width(), 256)
        self.assertEqual(reader.get_fps(), 25)
        self.assertEqual(reader.get_num_channels(), 3)
        self.assertEqual(reader.get_number_of_frame(), 3)
        
        # check frame content
        rval, frame_0 = reader.next_frame()
        rval, frame_1 = reader.next_frame()

        self.assertTrue(
            np.allclose(
                frame_0, 
                np.zeros((256,256,3), dtype=np.uint8),
                atol=5
            )
        )

        self.assertTrue(
            np.allclose(
                frame_1, 
                255*np.ones((256,256,3), dtype=np.uint8),
                atol=5
            )
        )

        reader.close()

    def test_Buffered_OpenCV_VideoReader(self):
        reader = Buffered_OpenCV_VideoReader()
        reader.open_file(filename = 'test_00.avi')
        
        # check metadata are correctly retrieved
        self.assertEqual(reader.get_height(), 256)
        self.assertEqual(reader.get_width(), 256)
        self.assertEqual(reader.get_fps(), 25)
        self.assertEqual(reader.get_num_channels(), 3)
        self.assertEqual(reader.get_number_of_frame(), 3)
        
        # check frame content
        rval, frame_0 = reader.next_frame()
        rval, frame_1 = reader.next_frame()
        
        self.assertTrue(
            np.allclose(
                frame_0, 
                np.zeros((256,256,3), dtype=np.uint8),
                atol=5
            )
        )

        self.assertTrue(
            np.allclose(
                frame_1, 
                255*np.ones((256,256,3), dtype=np.uint8),
                atol=5
            )
        )

        reader.close()

    def test_InMemory_OpenCV_VideoReader(self):
        reader = InMemory_OpenCV_VideoReader()
        reader.open_file(filename = 'test_00.avi')
        
        # check metadata are correctly retrieved
        self.assertEqual(reader.get_height(), 256)
        self.assertEqual(reader.get_width(), 256)
        self.assertEqual(reader.get_fps(), 25)
        self.assertEqual(reader.get_num_channels(), 3)
        self.assertEqual(reader.get_number_of_frame(), 3)
        
        # check frame content
        rval, frame_0 = reader.next_frame()
        rval, frame_1 = reader.next_frame()
        
        self.assertTrue(
            np.allclose(
                frame_0, 
                np.zeros((256,256,3), dtype=np.uint8),
                atol=5
            )
        )

        self.assertTrue(
            np.allclose(
                frame_1, 
                255*np.ones((256,256,3), dtype=np.uint8),
                atol=5
            )
        )

        reader.close()

class test_background_subtraction(unittest.TestCase):
    
    def test_(self):
        pass   

class test_video_pocessor(unittest.TestCase):
    
    def test_(self):
        pass   

if __name__ == '__main__':
    unittest.main()
