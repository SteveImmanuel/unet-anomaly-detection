# import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import time
import logging
from anomaly_detector.config import MODEL_NAME, DIM, BATCH_SIZE, SEQ_LEN, TRAIN_PATH, VALIDATION_PATH, TEST_PATH, LEARNING_RATE, EPSILON, EPOCHS
from anomaly_detector.threads import Predictor, Reader, Visualizer
from queue import Queue
from threading import Event

logging.basicConfig()
logger = logging.getLogger('Stream')
logger.setLevel(logging.INFO)

read_event = Event()
predict_event = Event()
model_input = Queue()
model_output = Queue()

reader = Reader('01.mp4', model_input, BATCH_SIZE, SEQ_LEN, DIM, read_event)
predictor = Predictor(f'pretrained/{MODEL_NAME}/{MODEL_NAME}_end_train', model_input, model_output,
                      read_event, predict_event)
visualizer = Visualizer(model_output, predict_event, 30)

reader.start()
predictor.start()
visualizer.start()

try:
    while True:
        time.sleep(0.01)
except KeyboardInterrupt:
    logger.info('Stopping thread')
    read_event.set()
    predict_event.set()
    reader.join()
    predictor.join()
    visualizer.join()