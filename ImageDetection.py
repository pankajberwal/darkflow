import cv2

from darkflow.net.build import TFNet

import numpy as np

import time

options = {

    'model': 'cfg/yolo.cfg',

    'load': 'bin/yolov2.weights',

    'threshold': 0.4,

    

}

tfnet = TFNet(options)

colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]



capture = cv2.VideoCapture(0)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

obj=['laptop','cell phone','earphone','knife','pendrive']

while True:

    stime = time.time()

    ret, frame = capture.read()

    results = tfnet.return_predict(frame)
    flag=False;
    count=0;
    if ret:

        for color, result in zip(colors, results):

            tl = (result['topleft']['x'], result['topleft']['y'])

            br = (result['bottomright']['x'], result['bottomright']['y'])

            label = result['label']

            if label in obj:
                flag=True;

            count=count+1
            confidence = result['confidence']

            text = '{}: {:.0f}%'.format(label, confidence * 100)

            frame = cv2.rectangle(frame, tl, br, color, 5)

            frame = cv2.putText(

                frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        
        text1 = 'Count-->{0}'.format(count)
        if flag==True:
            text2='Detected'
        else:
            text2='Undetected'
        frame = cv2.putText(

                frame, text1,(300,100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        frame = cv2.putText(

                frame, text2,(800,100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('frame', frame)
        if flag==True:
            cv2.imwrite('img.png',frame)
        

        print('FPS {:.1f}'.format(1 / (time.time() - stime)))

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break



capture.release()

cv2.destroyAllWindows()
