from keras.models import load_model
import cv2
import numpy as np
import mediapipe as mp

model = load_model('mask_inception.h5')
label = ['no-mask', 'mask']
fd = mp.solutions.face_detection
detect = fd.FaceDetection(min_detection_confidence=0.25)
# draw = mp.solutions.drawing_utils


def mask_detection(frame):
    # frame = cv2.resize(frame, (440, 330))
    try:
        # **Processing rgb images and integrating mediapipe with inception model**
        ht, wd, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = detect.process(rgb)
        if output.detections:
            for id, det in enumerate(output.detections):
                bbox = det.location_data.relative_bounding_box
                x, y, w, h = int(bbox.xmin*wd), int(bbox.ymin*ht), int(bbox.width*wd), int(bbox.height*ht)
                x1, y1, x2, y2 = (x-w//10), (y-h//5), (x+w), (y+h)
                cropped = rgb[y1:y2, x1:x2]
                img = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_NEAREST)
                img = img/255.0
                pred = model.predict(img.reshape(1, 224, 224, 3))[0]
                num = label[np.argmax(pred)]
                del img, cropped, rgb
                if num == 'mask':
                    col = (68, 148, 31)
                if num == 'no-mask':
                    col = (83, 38, 204)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.rectangle(frame, (x1-5, y1), (x2+5, y1-15), col, cv2.FILLED)
                cv2.putText(frame, f'{num} | id:{(id+1)}', (x1-3, y1-2),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    except:
        return frame
