import cv2
from threading import Thread, Event
from queue import Queue
from tensorflow.keras.models import Model, load_model
from anomaly_detector.network.loss_function import total_loss


class Visualizer(Thread):
    def __init__(self, model_output: Queue, predict_event: Event):
        Thread.__init__(self)
        # self.model = model
        self.model_output = model_output
        self.predict_event = predict_event

    def run(self):
        try:
            cv2.namedWindow('Prediction', cv2.WINDOW_NORMAL)

            while not (self.predict_event.is_set() and self.model_output.empty()):
                if not self.model_output.empty():
                    print('Visualizer: got item')
                    raw_result = self.model_output.get()
                    for i in range(len(raw_result)):
                        cv2.imshow('Prediction', raw_result['prediction'][i].squeeze())
                        cv2.waitKey(500)
        except Exception as e:
            print(e)
        finally:
            cv2.destroyAllWindows()
