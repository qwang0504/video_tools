from video_tools import VideoDisplay
import cv2
import numpy as np
from multiprocessing import Queue, set_start_method
import time 


FAILS = False
ARRAY = np.random.randint(0,255,(255,255,3), dtype= np.uint8)


def test():    

    cv2.imshow('array',ARRAY)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    q = Queue()
    vdis = VideoDisplay(
        queue=q,
        fps=10,
        winname='disp'
    )
    vdis.start()
    for i in range(100): 
        vdis.queue_image(ARRAY)
        time.sleep(0.1)

    vdis.exit()
    vdis.join()


if __name__ == "__main__":
    
    if FAILS:
        set_start_method('fork')
        test()
    else:
        set_start_method('spawn')
        test()