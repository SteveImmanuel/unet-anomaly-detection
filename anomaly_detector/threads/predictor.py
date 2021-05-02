import cv2
import time
import logging
from threading import Thread, Event
from queue import Queue
from typing import List, Tuple
from tensorflow.keras.models import Model
from anomaly_detector.utils.evaluation import locate_anomaly, calculate_loss
from anomaly_detector.config import DIM, POOL_SIZE, SMOOTH_STRENGTH

root_logger = logging.getLogger('Stream')
logger = root_logger.getChild('Predictor')


class Predictor(Thread):
    def __init__(self, model: Model, error_patcher: Model, model_input: Queue, model_output: Queue,
                 read_event: Event, predict_event: Event):
        Thread.__init__(self)
        self.model_input = model_input
        self.model_output = model_output
        self.read_event = read_event
        self.predict_event = predict_event
        self.error_patcher = error_patcher
        self.model = model
        logger.info('Initialization complete')

    def set_normalization_param(self, param: Tuple[float, float]):
        self.normalization_param = param

    def run(self):
        elapsed = -time.time()

        try:
            while not (self.read_event.is_set() and self.model_input.empty()):
                if not self.model_input.empty():
                    x = self.model_input.get()
                    y = self.model.predict(x['seq'])
                    elapsed += time.time()
                    logger.info(f'Predicting frames took {elapsed} s')

                    raw_losses, losses = calculate_loss(self.error_patcher,
                                                        y,
                                                        x['target'],
                                                        min_loss=self.normalization_param[0],
                                                        max_loss=self.normalization_param[1],
                                                        smooth_strength=SMOOTH_STRENGTH)
                    error_map = locate_anomaly(y, x['target'], raw_losses, losses, POOL_SIZE, 0.55)
                    logger.debug(error_map.shape)

                    self.model_output.put({
                        'prediction': y,
                        'g_truth': x['target'],
                        'error_map': error_map
                    })
                else:
                    logger.debug('Still empty')

                elapsed = -time.time()

        except Exception as e:
            logger.error(e)
        finally:
            self.predict_event.set()
            logger.info('Completed')
