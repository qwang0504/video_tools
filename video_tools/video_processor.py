import subprocess
import os
from typing import Optional, List
import numpy as np
import datetime
import tempfile
from abc import abstractmethod, ABC
# Strongly inspired by DeepLabCut video functions :
# https://github.com/DeepLabCut/DeepLabCut/blob/main/deeplabcut/utils/auxfun_videos.py

# GENERAL NOTE: Since compressed video formats use temporal information to achieve 
# good compression (keyframes), splitting or merging videos is not entirely straightforward.
# Re-encoding the videos ensure consistent output and a re-definition of keyframes

class VideoProcessor(ABC):
    '''
    split, shorten, crop, resize videos using ffmpeg
    '''

    def __init__(self, input_video_path: str) -> None:
        
        self.input_video_path = input_video_path
        name, format = os.path.splitext(os.path.basename(input_video_path))
        self.input_path = os.path.dirname(input_video_path)
        self.input_name = name
        self.input_format = format

    def make_output_path(
            self, 
            suffix: Optional[str], 
            dest_folder: Optional[str]
        ) -> None:

        if dest_folder is None:
            dest_folder = os.getcwd()

        if suffix is None:
            res = os.path.join(dest_folder, f"{self.input_name}{self.input_format}")
        else:
            res = os.path.join(dest_folder, f"{self.input_name}_{suffix}{self.input_format}")

        return res
    
    def get_input_video_metadata(self) -> None:
        # note that ffprobe doesn't show entries in the order specified in the command
        # but has it's own internal order (see https://ffmpeg.org/ffprobe.html#Main-options)
        command = [
            'ffprobe',
            '-v', 'error',
            '-show_entries',  'stream=width,height,r_frame_rate,duration,nb_frames',
            '-of', 'csv=print_section=0:nokey=1',
            f'{self.input_video_path}'
        ]
        
        metadata = subprocess.check_output(command, text=True)
        width, height, fps_frac, duration, num_frames = metadata.split(',')
        fps_numerator, fps_denominator = fps_frac.split('/')
        fps = float(fps_numerator)/float(fps_denominator)
        return int(width), int(height), fps, int(num_frames), float(duration)

    @abstractmethod
    def shorten(
            self, 
            start: str, 
            stop: str,
            suffix: Optional[str] = 'short', 
            dest_folder: Optional[str] = None
        ) -> None:
        pass

    @abstractmethod
    def shorten_frame(
            self, 
            frame_start_inclusive: int, 
            frame_stop_inclusive: int,
            suffix: Optional[str] = 'short', 
            dest_folder: Optional[str] = None
        ) -> None:
        pass

    @abstractmethod
    def merge(
            self,
            file_list: List[str],
            suffix: Optional[str] = 'merged', 
            dest_folder: Optional[str] = None
        ):
        pass

    @abstractmethod
    def crop(
        self, 
        left: int, 
        bottom: int, 
        width: int, 
        height: int,
        suffix: Optional[str] = 'cropped', 
        dest_folder: Optional[str] = None
        ) -> None:
        pass

    @abstractmethod
    def rescale(
            self, 
            scale: float,
            suffix: Optional[str] = 'scaled', 
            dest_folder: Optional[str] = None
        ) -> None:
        pass

    @abstractmethod
    def rotate(
            self, 
            angle_degrees: float,
            suffix: Optional[str] = 'rotated', 
            dest_folder: Optional[str] = None
        ) -> None:
        pass

    def reindex(
            self,
            suffix: Optional[str] = 'reindexed', 
            dest_folder: Optional[str] = None
        ):
    
        output_path = self.make_output_path(suffix, dest_folder)
        command = [
            'ffmpeg', 
            '-i', f'{self.input_video_path}',
            '-c', 'copy', 
            '-fflags', '+genpts',
            f'{output_path}'
        ]
        subprocess.call(command)

    def split(
            self, 
            n: int,
            suffix: Optional[str] = 'split', 
            dest_folder: Optional[str] = None
        ) -> None:
        '''split based on time'''

        width, height, fps, num_frames, duration = self.get_input_video_metadata()
        chunk_dur = duration / n
        splits = np.arange(n + 1) * chunk_dur
        hh_mm_ss = lambda val: str(datetime.timedelta(seconds=val))
        for n, (start, end) in enumerate(zip(splits, splits[1:]), start=1):
            self.shorten(
                    hh_mm_ss(start),
                    hh_mm_ss(end),
                    f"{suffix}_{n:03}",
                    dest_folder
                )

    def split_frame(
            self, 
            n: int,
            suffix: Optional[str] = 'split', 
            dest_folder: Optional[str] = None
        ) -> None:
        '''split based on frames using filter'''

        width, height, fps, num_frames, duration = self.get_input_video_metadata()
        chunk_starts = np.linspace(0, num_frames, n+1, dtype=int)
        chunks =  [(chunk_starts[i], chunk_starts[i+1] - 1) for i in range(n)]
        for n, (start, end) in enumerate(chunks, start=1):
            self.shorten_frame(
                    start,
                    end,
                    f"{suffix}_{n:03}",
                    dest_folder
                )
            
class GPU_VideoProcessor(VideoProcessor):
    '''
    Use the h264 or h265 codec with NVIDIA GPU harware acceleration   
    '''

    def __init__(
            self, 
            input_video_path: str,
            codec: str = 'h264_nvenc',
            profile: str = 'main',
            preset: str = 'p2',
            quality: int = 26
        ) -> None:

        super().__init__(input_video_path)
        self.codec = codec
        self.profile = profile
        self.preset = preset
        self.quality = quality

    def crop(
            self, 
            left: int, 
            bottom: int, 
            width: int, 
            height: int,
            suffix: Optional[str] = 'cropped', 
            dest_folder: Optional[str] = None
        ) -> None:

        output_path = self.make_output_path(suffix, dest_folder)
        command = [
            'ffmpeg', '-n', 
            '-i', f'{self.input_video_path}',
            '-filter:v' , f'crop={width}:{height}:{left}:{bottom}',
            '-c:v', self.codec, 
            '-cq', f'{self.quality}',
            '-profile:v', self.profile, 
            '-preset', self.preset,
            f'{output_path}'
        ]
        subprocess.call(command)

    def rescale(
            self, 
            scale: float,
            suffix: Optional[str] = 'scaled', 
            dest_folder: Optional[str] = None
        ) -> None:

        output_path = self.make_output_path(suffix, dest_folder)
        width, height, fps, num_frames, duration = self.get_input_video_metadata()
        new_width = int(width*scale)
        command = [
            'ffmpeg', '-n', 
            '-i', f'{self.input_video_path}', 
            '-filter:v', f'scale={new_width}:-1', 
            '-c:v', self.codec, 
            '-cq', f'{self.quality}',
            '-profile:v', self.profile, 
            '-preset', self.preset,
            f'{output_path}'
        ]
        subprocess.call(command)

    def rotate(
            self, 
            angle_degrees: float,
            suffix: Optional[str] = 'rotated', 
            dest_folder: Optional[str] = None
        ) -> None:

        angle_radians = np.deg2rad(angle_degrees)
        output_path = self.make_output_path(suffix, dest_folder)
        command = [
            'ffmpeg', '-n', 
            '-i', f'{self.input_video_path}',
            '-filter:v' , f"rotate={angle_radians}:'ow=rotw({angle_radians}):oh=roth({angle_radians})'",
            '-c:v', self.codec, 
            '-cq', f'{self.quality}',
            '-profile:v', self.profile, 
            '-preset', self.preset,
            f'{output_path}'
        ]
        subprocess.call(command)

    def shorten(
            self, 
            start: str, 
            stop: str,
            suffix: Optional[str] = 'short', 
            dest_folder: Optional[str] = None
        ) -> None:

        output_path = self.make_output_path(suffix, dest_folder)
        command = [
            'ffmpeg', '-n', 
            '-ss', f'{start}',
            '-i', f'{self.input_video_path}',
            '-to', f'{stop}',
            '-c:v', self.codec, 
            '-cq', f'{self.quality}',
            '-profile:v', self.profile, 
            '-preset', self.preset,
            f'{output_path}'
        ]
        subprocess.call(command)

    def shorten_frame(
            self, 
            frame_start_inclusive: int, 
            frame_stop_inclusive: int,
            suffix: Optional[str] = 'short', 
            dest_folder: Optional[str] = None
        ) -> None:

        output_path = self.make_output_path(suffix, dest_folder)
        command = [
            'ffmpeg', '-n', 
            '-i', f'{self.input_video_path}',
            '-filter:v', f"select='between(n, {frame_start_inclusive}, {frame_stop_inclusive})', setpts=PTS-STARTPTS",
            '-c:v', self.codec, 
            '-cq', f'{self.quality}',
            '-profile:v', self.profile, 
            '-preset', self.preset,
            f'{output_path}'
        ]
        subprocess.call(command)

    def merge(
            self,
            file_list: List[str],
            suffix: Optional[str] = 'merged', 
            dest_folder: Optional[str] = None
        ):
        
        with tempfile.NamedTemporaryFile(mode='wt', delete=False) as fd:

            fd.write('ffconcat version 1.0\n')
            for file in file_list:
                fd.write(f"file {file}\n")
            fd.close()

            os.system(f"cat {fd.name}")
            
            # call ffmpeg
            output_path = self.make_output_path(suffix, dest_folder)
            command = [
                'ffmpeg', '-n',
                '-f', 'concat',
                '-safe', '0',
                '-i', fd.name,
                '-c:v', self.codec, 
                '-cq', f'{self.quality}',
                '-profile:v', self.profile, 
                '-preset', self.preset,
                f'{output_path}'
            ]
            subprocess.call(command)

class CPU_VideoProcessor(VideoProcessor):
    '''
    Use the libx264 or libx265 codec on the CPU
    '''

    def __init__(
            self, 
            input_video_path: str,
            codec: str = 'libx264',
            profile: str = 'main',
            preset: str = 'superfast',
            quality: int = 26
        ) -> None:

        super().__init__(input_video_path)

        self.codec = codec
        self.profile = profile
        self.preset = preset
        self.quality = quality

    def crop(
            self, 
            left: int, 
            bottom: int, 
            width: int, 
            height: int,
            suffix: Optional[str] = 'cropped', 
            dest_folder: Optional[str] = None
        ) -> None:

        output_path = self.make_output_path(suffix, dest_folder)
        command = [
            'ffmpeg', '-n', 
            '-i', f'{self.input_video_path}',
            '-filter:v' , f'crop={width}:{height}:{left}:{bottom}',
            '-c:v', self.codec, 
            '-crf', f'{self.quality}',
            '-profile:v', self.profile, 
            '-preset',  self.preset,
            f'{output_path}'
        ]
        subprocess.call(command)

    def rescale(
            self, 
            scale: float,
            suffix: Optional[str] = 'scaled', 
            dest_folder: Optional[str] = None
        ) -> None:

        output_path = self.make_output_path(suffix, dest_folder)
        width, height, fps, num_frames, duration = self.get_input_video_metadata()
        new_width = int(width*scale)
        command = [
            'ffmpeg', '-n', 
            '-i', f'{self.input_video_path}', 
            '-filter:v', f'scale={new_width}:-1', 
            '-c:v', self.codec, 
            '-crf', f'{self.quality}',
            '-profile:v', self.profile, 
            '-preset', self.preset,
            f'{output_path}'
        ]
        subprocess.call(command)

    def rotate(
            self, 
            angle_degrees: float,
            suffix: Optional[str] = 'rotated', 
            dest_folder: Optional[str] = None
        ) -> None:

        angle_radians = np.deg2rad(angle_degrees)
        output_path = self.make_output_path(suffix, dest_folder)
        command = [
            'ffmpeg', '-n', 
            '-i', f'{self.input_video_path}',
            '-filter:v' , f"rotate={angle_radians}:'ow=ceil(rotw({angle_radians})/2)*2:oh=ceil(roth({angle_radians})/2)*2'",
            '-c:v', self.codec, 
            '-crf', f'{self.quality}',
            '-profile:v', self.profile, 
            '-preset', self.preset,
            f'{output_path}'
        ]
        subprocess.call(command)

    def shorten(
            self, 
            start: str, 
            stop: str,
            suffix: Optional[str] = 'short', 
            dest_folder: Optional[str] = None
        ) -> None:
        output_path = self.make_output_path(suffix, dest_folder)
        command = [
            'ffmpeg', '-n', 
            '-ss', f'{start}',
            '-i', f'{self.input_video_path}',
            '-to', f'{stop}',
            '-c:v', self.codec, 
            '-crf', f'{self.quality}',
            '-profile:v', self.profile, 
            '-preset',  self.preset,
            f'{output_path}'
        ]
        subprocess.call(command)

    def shorten_frame(
            self, 
            frame_start_inclusive: int, 
            frame_stop_inclusive: int,
            suffix: Optional[str] = 'short', 
            dest_folder: Optional[str] = None
        ) -> None:
        output_path = self.make_output_path(suffix, dest_folder)
        command = [
            'ffmpeg', '-n', 
            '-i', f'{self.input_video_path}',
            '-filter:v', f"select='between(n, {frame_start_inclusive}, {frame_stop_inclusive})', setpts=PTS-STARTPTS",
            '-c:v', self.codec, 
            '-crf', f'{self.quality}',
            '-profile:v', self.profile, 
            '-preset',  self.preset,
            f'{output_path}'
        ]
        subprocess.call(command)

    def merge(
            self,
            file_list: List[str],
            suffix: Optional[str] = 'merged', 
            dest_folder: Optional[str] = None
        ):
        
        with tempfile.NamedTemporaryFile(mode='wt', delete=False) as fd:

            fd.write('ffconcat version 1.0\n')
            for file in file_list:
                fd.write(f"file {file}\n")
            fd.close()

            os.system(f"cat {fd.name}")
            
            # call ffmpeg
            output_path = self.make_output_path(suffix, dest_folder)
            command = [
                'ffmpeg', '-n',
                '-f', 'concat',
                '-safe', '0',
                '-i', fd.name,
                '-c:v', self.codec, 
                '-crf', f'{self.quality}',
                '-profile:v', self.profile, 
                '-preset',  self.preset,
                f'{output_path}'
            ]
            subprocess.call(command)