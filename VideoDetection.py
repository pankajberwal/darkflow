import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold':0.3,
    
}

tfnet = TFNet(option)

capture = cv2.VideoCapture('Testvideo.mp4')
frame_width = int(capture.get(3))
frame_height = int(capture.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]
obj=['laptop','cell phone','knife','pendrive']
while (capture.isOpened()):
    stime = time.time()
    ret, frame = capture.read()
    flag=False
    count=0
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            count=count+1
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            label = result['label']
            if label in obj:
                flag=True;
            frame = cv2.rectangle(frame, tl, br, color, 3)
            if label in obj:
                frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
            else:
                frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
        text1 = 'Count-->{0}'.format(count)
        if flag==True:
            text2='Detected'
        else:
            text2='Undetected'
        frame = cv2.putText(

                frame, text1,(50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        frame = cv2.putText(

                frame, text2,(250,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
        cv2.imshow('frame', frame)
        out.write(frame)
        if flag==True:
            cv2.imwrite('img.png',frame)
        print('FPS {:.1f}'.format(1 / (time.time() - stime)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        capture.release()
        out.release()
        cv2.destroyAllWindows()
        break
    
