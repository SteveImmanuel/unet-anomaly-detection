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
            cv2.resizeWindow('Prediction', 160 * 7, 120 * 7)

            while not (self.predict_event.is_set() and self.model_output.empty()):
                if not self.model_output.empty():
                    logger.debug('Receive item')
                    raw_result = self.model_output.get()

                    g_truth = np.repeat(raw_result['g_truth'], 3, axis=-1)
                    prediction = np.repeat(raw_result['prediction'], 3, axis=-1)

                    for i in range(len(raw_result['prediction'])):
                        combined_image = np.zeros(160 * 120 * 4 * 3).reshape(240, 320, 3)
                        combined_image += 0.25
                        combined_image[:120, :160] = g_truth[i]
                        combined_image[:120, 160:] = prediction[i]
                        combined_image[120:, 80:240] = raw_result['error_map'][i]

                        cv2.imshow('Prediction', combined_image)
                        cv2.waitKey(1000 // self.fps)
        except Exception as e:
            logger.error(e)
        finally:
            input('Press enter to continue...')
            cv2.destroyAllWindows()
            logger.info('Completed')
