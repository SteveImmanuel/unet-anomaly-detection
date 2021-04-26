import cv2
import logging
import numpy as np
from threading import Thread, Event
from queue import Queue
from tensorflow.keras.models import Model, load_model
from anomaly_detector.network.loss_function import total_loss

root_logger = logging.getLogger('Stream')
logger = root_logger.getChild('Visualizer')


class Visualizer(Thread):
    def __init__(self, model_output: Queue, predict_event: Event, fps: int):
        Thread.__init__(self)
        self.model_output = model_output
        self.predict_event = predict_event
        self.fps = fps
        logger.info('Initialization complete')

    def run(self):
        try:
            cv2.namedWindow('Prediction', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Prediction', 160 * 3 * 3, 120 * 3)

            while not (self.predict_event.is_set() and self.model_output.empty()):
                if not self.model_output.empty():
                    logger.debug('Receive item')
                    raw_result = self.model_output.get()

                    g_truth = np.repeat(raw_result['g_truth'], 3, axis=-1)
                    prediction = np.repeat(raw_result['prediction'], 3, axis=-1)

                    for i in range(len(raw_result['prediction'])):
                        combined_image = np.concatenate(
                            (g_truth[i], prediction[i], raw_result['error_map'][i]), axis=1)
                        cv2.imshow('Prediction', combined_image)
                        cv2.waitKey(1000 // self.fps)
        except Exception as e:
            logger.error(e)
        finally:
            cv2.destroyAllWindows()
            logger.info('Completed')
