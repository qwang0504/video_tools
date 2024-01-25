from multiprocessing import Process, Queue
import cv2
import time
from numpy.typing import NDArray

# TODO test that it works on windows

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

    def queue_image(self, image: NDArray) -> None:
        '''check fps, discard frames if time not elapsed'''
        t = time.time_ns()
        if ((t - self.last_image_time)*1e-9) >= (1/self.fps):
            self.queue.put(image)
            self.last_image_time = t

    def exit(self) -> None:
        self.queue.put(None)

    def run(self) -> None:
        cv2.namedWindow(self.winname)
        last_disp_time = time.time_ns()
        while True:
            frame = self.queue.get()
            if frame is None:
                break
            timestamp = time.time_ns()
            fps_hat = 1/((timestamp - last_disp_time)*1e-9)
            cv2.imshow(self.winname, frame)
            cv2.displayStatusBar(self.winname,f'Display FPS: {fps_hat:.2f}',1)
            cv2.waitKey(1)
            last_disp_time = time.time_ns()
        cv2.destroyWindow(self.winname)

