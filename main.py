import cv2
from ultralytics import YOLO

from utils.dataloaders import LoadRTSPs
from utils.common import LOGGER

rtsp_urls = [
    'rtsp://admin:akd123456@192.168.1.168/Streaming/Channels/101',
    'rtsp://admin:akd123456@192.168.1.168/Streaming/Channels/201',
    'rtsp://admin:akd123456@192.168.1.168/Streaming/Channels/301',
    'rtsp://admin:akd123456@192.168.1.168/Streaming/Channels/401',
]

model = YOLO('weights/yolov8n.pt')
dataloader = LoadRTSPs(rtsp_urls, skip_frame=250, show_log=False)
model(dataloader.imgs)
# for i in range(len(rtsp_urls)):
#     cv2.namedWindow(str(i), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

for frames in dataloader:
    imgs = []
    for i, frame in enumerate(frames):
        log, img = frame
        imgs.append(img)
        # cv2.imshow(str(i), img)
        LOGGER.info(f"{i}:{log['qsize']:>4},{log['n']:>5},{log['dt']:>6.2f},{log['cv_msec']:>9.2f}")
    results = model(imgs)
    # for i, result in enumerate(results):
    #     annotated_frame = result.plot()
    #     cv2.imshow(str(i), annotated_frame)
    #
    # if cv2.waitKey(1) == ord('q'):
    #     cv2.destroyAllWindows()
    #     exit()
