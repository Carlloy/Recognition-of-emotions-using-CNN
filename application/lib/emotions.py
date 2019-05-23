from keras.preprocessing.image import img_to_array
import cv2
from keras.models import load_model
import numpy as np


class Emotions(object):

    def __init__(self):
        # model paths
        self.emotion_model_path = '/Users/karolina/PycharmProjects/Recognition-of-emotions-using-CNN/training/models/model_CNN.30-0.66.hdf5'

        # without compile=False code won't compile - Tensorflow
        self.emotion_model = load_model(self.emotion_model_path, compile=False)

        self.emotions_labels = ['Złość', 'Zniesmaczenie', 'Strach', 'Radość', 'Smutek', 'Zaskoczenie', 'Obojętność']
        self.last_predictions = np.array([0] * 7)
        self.predictions_array = []
        self.predictions_mean = np.array([0] * 7)
        self.emotions_measured = False

    def predict(self, frame, faces):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # When face is detected
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                # resize the image to be passed to the neural network
                resized_face = cv2.resize(face, (48, 48))
                resized_face = resized_face.astype("float32") / 255.0
                resized_face = img_to_array(resized_face)
                resized_face = np.expand_dims(resized_face, axis=0)

                predictions = self.emotion_model.predict(resized_face)[0]
                self.last_predictions = predictions

            # create history of predictions to minimize prediction errors
            if len(self.predictions_array) < 15:
                self.predictions_array.append(self.last_predictions)
            else:
                self.predictions_array.pop(0)
                self.predictions_array.append(self.last_predictions)

            self.predictions_mean = np.array(self.predictions_array).mean(axis=0)
            self.emotions_measured = True
            return self.predictions_mean

        return None

    def get_last_prediction(self):
        return self.predictions_mean
