import tensorflow as tf
import numpy as np
import cv2
from model import build_model
# from process import resize_with_pad



model = build_model((128,128,3))
checkpoints_dir = "./checkpoints/"
best = tf.train.latest_checkpoint(checkpoints_dir)
model.load_weights(best)
model.summary()
class_names = ['0','1','2','3']
vid = cv2.VideoCapture(0)

while(True):
    ret, frame = vid.read()
    image = frame.copy()
    image = cv2.imread("/home/ahmed/hand_gestures/IMG_1159.JPG")

    image = cv2.resize(image,(128,128))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
    image = np.expand_dims(image,0)
    # print(image.shape)
    predictions = model.predict(image)
    # print(predictions[0])
    cls = np.argmax(predictions)
    score = tf.nn.softmax(predictions[0])
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
