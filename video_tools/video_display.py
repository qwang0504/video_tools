from multiprocessing import Process, Queue, Event
import cv2
import time
from numpy.typing import NDArray
from queue import Empty

# TODO I seem to have a problem as soon as a cv2 window 
# is opened before that, VideoDisplay objects crash
# (no window display)
# Workaround for now: create and start the object at
# the top of the script before anything else
# This seems to be a fork specific issue: looks like
# forking brings unwanted opencv context that fucks things up
# I have no issue when forcing to spawn instead

class VideoDisplay(Process):
    def __init__(
            self, 
            queue: Queue,
            fps: int = 30,
            winname: str = 'display',
            *args,
            **kwargs
        ) -> None:
        super().__init__(*args, **kwargs)
        self.fps = fps
        self.queue = queue
        self.winname = winname
        self.last_image_time = 0
        self.stop_event = Event()

    def queue_image(self, image: NDArray) -> None:
        '''check fps, discard frames if time not elapsed'''
        t = time.time_ns()
        if ((t - self.last_image_time)*1e-9) >= (1/self.fps):
            self.queue.put(image)
            self.last_image_time = t
            print('queued')

    def exit(self) -> None:
        self.stop_event.set()

    def run(self) -> None:

        # try to reinitialize opencv after fork
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        print('test0')
        cv2.namedWindow(self.winname)
        print('test1')
        last_disp_time = time.time_ns()
        while not self.stop_event.is_set():
            print('test2')
            try:
                frame = self.queue.get(timeout=2/self.fps)
            except Empty:
                pass
            timestamp = time.time_ns()
            fps_hat = 1/((timestamp - last_disp_time)*1e-9)
            cv2.imshow(self.winname, frame)
            cv2.displayStatusBar(self.winname, f'Display FPS: {fps_hat:.2f}', 1)
            cv2.waitKey(1)
            print('disped')
            last_disp_time = time.time_ns()

        cv2.destroyWindow(self.winname)

