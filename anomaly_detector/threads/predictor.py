import cv2
from threading import Thread, Event
from queue import Queue
from tensorflow.keras.models import Model, load_model
from anomaly_detector.network.loss_function import total_loss


class Predictor(Thread):
    def __init__(self, model_path: str, model_input: Queue, model_output: Queue, read_event: Event,
                 predict_event: Event):
        Thread.__init__(self)
        self.model_input = model_input
        self.model_output = model_output
        self.read_event = read_event
        self.predict_event = predict_event

        model = load_model(model_path, custom_objects={'total_loss': total_loss})
        self.model = Model(inputs=model.input, outputs=model.output[0])

    def run(self):
        try:
            while not (self.read_event.is_set() and self.model_input.empty()):
                if not self.model_input.empty():
                    print('Predictor: got item')
                    x = self.model_input.get()
                    y = self.model.predict(x['seq'])
                    self.model_output.put({'prediction': y, 'g_truth': x['target']})
        except Exception as e:
            print(e)
        finally:
            self.predict_event.set()
