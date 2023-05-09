import time
import cv2
import math
import numpy as np
from threading import Thread
import queue

from utils.common import StatTime, LOGGER


class LoadRTSPs:
    def __init__(self, sources, vid_stride=1, skip_frame=125, show_log=True):
        self.sources = [sources] if isinstance(sources, str) else sources
        self.vid_stride = vid_stride
        self.skip_frame = skip_frame
        self.show_log = show_log
        self.reset = True

        n = len(self.sources)
        self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [], [None] * n
        self.dts = [StatTime()] * n
        # 初始化
        for i, s in enumerate(sources):
            st = f'{i + 1}/{n}: {s}... '
            cap = cv2.VideoCapture(s)
            assert cap.isOpened(), f'{st}Failed to open {s}'

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps[i] = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30

            _, self.imgs[i] = cap.read()
            self.threads[i] = Thread(target=self.update, args=([i, cap, s]), daemon=True)
            LOGGER.info(f'{st} Success ({w}x{h} at {self.fps[i]:.2f} FPS)')
            self.frames.append(queue.Queue())

        for i in range(n):
            self.threads[i].start()

    def update(self, i, cap, stream):
        n, fq, dt = 0, self.frames[i], self.dts[i]
        while cap.isOpened() and n < self.skip_frame and self.reset:
            n += 1
            cap.grab()
            time.sleep(0.0)

        self.reset = False
        n = 0
        while cap.isOpened():
            with dt:
                n += 1
                cap.grab()
                if n % self.vid_stride == 0:
                    success, img = cap.retrieve()
                    msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if success:
                        self.imgs[i] = img
                    else:
                        LOGGER.warning('WARNING ⚠️ Video stream unresponsive, please check your IP camera connection.')
                        self.imgs[i] = np.zeros_like(self.imgs[i])
                        cap.open(stream)
                time.sleep(0.0)
            fq.put([{'n': n, 'dt': dt.dt, 'cv_msec': msec}, self.imgs[i]])
            if self.show_log:
                LOGGER.info(f"update {i}: {n:>4}, {dt.dt:>6.2f}, {msec:>8.2f}")

    def __iter__(self):
        self.count = -1
        return self

    def __len__(self):
        return len(self.sources)

    def __next__(self):
        while True:
            frames_qsize = []
            for fq in self.frames:
                frames_qsize.append(fq.qsize())
            if (np.array(frames_qsize) > 0).all():
                break

        frames = []
        for i, fq in enumerate(self.frames):
            frame = fq.get()
            frame[0]['qsize'] = frames_qsize[i]
            frames.append(frame)
        return frames
