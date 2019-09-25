import sys

import cv2
from vision.ssd.predictor import Predictor
from vision.ssd.config import mobilenetv1_ssd_config as config
from WebcamCapture import WebcamVideoStream
import deepviewrt as rt

from vision.utils.misc import Timer

if len(sys.argv) < 3:
    print('Usage: python run_ssd_example.py <model path>')
    sys.exit(0)

net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]

if len(sys.argv) >= 5:
    cap = cv2.VideoCapture(sys.argv[4])  # capture from file
else:
    # cap = cv2.VideoCapture('http://192.168.1.65:8080/videofeed')  # capture from camera
    cap = cv2.VideoCapture(0)
    cap.set(3, 480)
    cap.set(4, 480)

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

print(rt.version())
client = rt.Context()
in_file = open(model_path, 'rb')
client.load(in_file.read())

predictor = Predictor(client,
                      nms_method=None,
                      iou_threshold=config.iou_threshold,
                      candidate_size=200,
                      sigma=0.5,
                      config=config)

cap = WebcamVideoStream(cap).start()

timer = Timer()
while True:
    orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    interval = timer.end()
    print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.shape[0]))
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (box[0] + 20, box[1] + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.stop()
cv2.destroyAllWindows()
