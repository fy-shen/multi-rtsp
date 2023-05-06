import time
import cv2
import math
import numpy as np
from threading import Thread

from utils.common import StatTime, LOGGER


class LoadStreams:
    def __init__(self, sources, vid_stride=1):
        self.sources = [sources] if isinstance(sources, str) else sources
        self.vid_stride = vid_stride

        n = len(self.sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
        for i, s in enumerate(sources):
            st = f'{i + 1}/{n}: {s}... '
            s = eval(s) if s.isnumeric() else s
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30

            _, self.imgs[i] = cap.read()  # guarantee first frame
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f'{st} Success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)')
            self.threads[i].start()
        LOGGER.info('')

    def update(self, i, cap, stream):
        n, f = 0, self.frames[i]
        while cap.isOpened() and n < f:
            n += 1
            cap.grab()
            if n % self.vid_stride == 0:
                success, img = cap.retrieve()
                if success:
                    self.imgs[i] = img
                else:
                    LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                    self.imgs[i] = np.zeros_like(self.imgs[i])
                    cap.open(stream)
            time.sleep(0.0)

    def __iter__(self):
        self.count = -1
        return self

    def __len__(self):
        return len(self.sources)

    def __next__(self):
        self.count += 1
        if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration
        return self.imgs

