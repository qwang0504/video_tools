import cv2
from numpy.typing import NDArray
import subprocess
import numpy as np
from abc import ABC

# TODO maybe add a multiprocessing queue

class VideoWriter(ABC):
    def write_frame(self, image: NDArray) -> None:
        pass

    def close(self) -> None:
        pass

# video writer opencv
class OpenCV_VideoWriter(VideoWriter):

    def __init__(
            self, 
            height: int, 
            width: int, 
            fps: int = 25, 
            filename: str = 'output.avi',
            fourcc: str = 'XVID'
        ) -> None:
        
        self.height = height
        self.width = width
        self.fps = fps
        self.filename = filename
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        color = True
        self.writer = cv2.VideoWriter(filename, self.fourcc, fps, (width, height), color)

    def write_frame(self, image: NDArray) -> None:
        if len(image.shape) == 2:
            image = np.dstack((image,image,image))
        self.writer.write(image)

    def close(self) -> None:
        self.writer.release()

# video writer ffmpeg
class FFMPEG_VideoWriter(VideoWriter):

    def write_frame(self, image: NDArray) -> None:
        # requires RGB images
        if len(image.shape) == 2:
            image = np.dstack((image,image,image))
        self.ffmpeg_process.stdin.write(image.astype(np.uint8).tobytes())

    def close(self) -> None:
        self.ffmpeg_process.stdin.flush()
        self.ffmpeg_process.stdin.close()
        self.ffmpeg_process.wait()

class FFMPEG_VideoWriter_GPU(FFMPEG_VideoWriter):
    # To check which encoders are available, use:
    # ffmpeg -encoders
    #
    # To check which profiles and presets are available for a given encoder use:
    # ffmpeg -h encoder=h264_nvenc

    SUPPORTED_VIDEO_CODECS = ['h264_nvenc', 'hevc_nvenc']
    SUPPORTED_PRESETS = ['p1','p2','p3','p4','p5','p6','p7']
    SUPPORTED_PROFILES = ['main']

    def __init__(
            self, 
            height: int, 
            width: int, 
            fps: int = 25, 
            q: int = 23,
            filename: str = 'output.avi',
            codec: str = 'h264_nvenc',
            profile: str = 'main',
            preset: str = 'p2'
        ) -> None:

        if not codec in self.SUPPORTED_VIDEO_CODECS:
            raise ValueError(f'wrong video_codec type, supported codecs are: {self.SUPPORTED_VIDEO_CODECS}') 

        ffmpeg_cmd_prefix = [
            "ffmpeg",
            "-hide_banner", 
            "-loglevel", "error",
            "-y",  # Overwrite output file if it exists
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-r", str(fps),  # Frames per second
            "-s", f"{width}x{height}",  # Specify image size
            "-i", "-",  # Input from pipe
            "-c:v", codec 
        ]

        ffmpeg_cmd_suffix = [
            "-pix_fmt", "yuv420p",  # Pixel format (required for compatibility)
            filename,
        ]
        
        ffmpeg_cmd_options = []
        if (codec == 'h264_nvenc') or (codec == 'hevc_nvenc'):

            if not profile in self.SUPPORTED_PROFILES:
                raise ValueError(f'wrong profile, supported profile are: {self.SUPPORTED_PROFILES}') 

            if not preset in self.SUPPORTED_PRESETS:
                raise ValueError(f'wrong preset, supported preset are: {self.SUPPORTED_PRESETS}') 

            ffmpeg_cmd_options = [
                "-profile:v", profile,
                "-preset", preset, 
                "-cq:v", str(q),
            ]
        else:
            pass

        ffmpeg_cmd = ffmpeg_cmd_prefix + ffmpeg_cmd_options + ffmpeg_cmd_suffix
        self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        
# video writer ffmpeg
class FFMPEG_VideoWriter_CPU(FFMPEG_VideoWriter):
    # To check which encoders are available, use:
    # ffmpeg -encoders
    #
    # To check which profiles and presets are available for a given encoder use:
    # ffmpeg -h encoder=h264

    SUPPORTED_VIDEO_CODECS = ['h264', 'hevc', 'mjpeg']
    SUPPORTED_PRESETS = ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']
    SUPPORTED_PROFILES = ['main']

    def __init__(
            self, 
            height: int, 
            width: int, 
            fps: int = 25, 
            q: int = 23,
            filename: str = 'output.avi',
            codec: str = 'h264',
            profile: str = 'main',
            preset: str = 'veryfast'
        ) -> None:

        if not codec in self.SUPPORTED_VIDEO_CODECS:
            raise ValueError(f'wrong codec, supported codecs are: {self.SUPPORTED_VIDEO_CODECS}') 
        
        ffmpeg_cmd_prefix = [
            "ffmpeg",
            "-hide_banner", 
            "-loglevel", "error",
            "-y",  # Overwrite output file if it exists
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-r", str(fps),  # Frames per second
            "-s", f"{width}x{height}",  # Specify image size
            "-i", "-",  # Input from pipe
            "-c:v", codec 
        ]

        ffmpeg_cmd_suffix = [
            "-pix_fmt", "yuv420p",  # Pixel format (required for compatibility)
            filename,
        ]

        ffmpeg_cmd_options = []
        if (codec == 'h264') or (codec == 'hevc'):

            if not profile in self.SUPPORTED_PROFILES:
                raise ValueError(f'wrong profile, supported profile are: {self.SUPPORTED_PROFILES}') 

            if not preset in self.SUPPORTED_PRESETS:
                raise ValueError(f'wrong preset, supported preset are: {self.SUPPORTED_PRESETS}') 

            ffmpeg_cmd_options = [
                "-profile:v", profile,
                "-preset", preset, 
                "-crf", str(q),
            ]
        else:
            pass

        ffmpeg_cmd = ffmpeg_cmd_prefix + ffmpeg_cmd_options + ffmpeg_cmd_suffix
        self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

        