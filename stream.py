# import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# from anomaly_detector.network.model import get_model
from anomaly_detector.config import MODEL_NAME, DIM, BATCH_SIZE, SEQ_LEN, TRAIN_PATH, VALIDATION_PATH, TEST_PATH, LEARNING_RATE, EPSILON, EPOCHS
from anomaly_detector.threads import Predictor, Reader, Evaluator
from queue import Queue
from threading import Event
import time

read_event = Event()
predict_event = Event()
model_input = Queue()
model_output = Queue()

reader = Reader('01.mp4', model_input, BATCH_SIZE, SEQ_LEN, DIM, read_event)
predictor = Predictor(f'pretrained/{MODEL_NAME}/{MODEL_NAME}_end_train', model_input, model_output,
                      read_event, predict_event)
evaluator = Evaluator(model_output, predict_event)

print('Initializing reader')
reader.start()
print('Initializing predictor')
predictor.start()
print('Initializing evaluator')
evaluator.start()

reader.join()
predictor.join()
evaluator.join()
# try:
#     while True:
#         time.sleep(0.01)
# except KeyboardInterrupt:
#     print('stopping thread')
#     finish_event.set()