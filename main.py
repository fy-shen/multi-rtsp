import cv2

from utils.dataloaders import LoadStreams


rtsp_urls = [
    'rtsp://admin:nike@2021@117.85.174.233:1554/video0',
    'rtsp://admin:nike@2021@117.85.174.233:1557/video0'
]

dataloader = LoadStreams(rtsp_urls)
for imgs in dataloader:
    for i, img in enumerate(imgs):
        cv2.imshow(str(i), img)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            exit()

