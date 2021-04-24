import cv2
import time
import logging
from threading import Thread, Event
from queue import Queue
from tensorflow.keras.models import Model, load_model
from anomaly_detector.network.loss_function import total_loss

root_logger = logging.getLogger('Stream')
logger = root_logger.getChild('Predictor')


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
        logger.info('Initialization complete')

    def run(self):
        elapsed = -time.time()

        try:
            while not (self.read_event.is_set() and self.model_input.empty()):
                # while not self.read_event.is_set():
                if not self.model_input.empty():
                    x = self.model_input.get()
                    y = self.model.predict(x['seq'])
                    elapsed += time.time()
                    logger.info(f'Predicting frames took {elapsed} s')

                    self.model_output.put({'prediction': y, 'g_truth': x['target']})
                else:
                    logger.debug('Still empty')

                elapsed = -time.time()

        except Exception as e:
            logger.error(e)
        finally:
            self.predict_event.set()
            logger.info('Completed')
