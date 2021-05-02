# import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import time
import logging
from tensorflow.keras.models import load_model, Model
from queue import Queue
from threading import Event

from anomaly_detector.config import DIM, BATCH_SIZE, SEQ_LEN, TRAIN_PATH, VALIDATION_PATH, TEST_PATH, LEARNING_RATE, EPSILON, EPOCHS, POOL_SIZE
from anomaly_detector.threads import Predictor, Reader, Visualizer
from anomaly_detector.network.loss_function import total_loss
from anomaly_detector.utils.evaluation import get_error_patcher
from pretrained.ped1_final_model import test_config

logging.basicConfig()
logger = logging.getLogger('Stream')
logger.setLevel(logging.INFO)

read_event = Event()
predict_event = Event()
model_input = Queue()
model_output = Queue()

dataset = ['ped1', 'ped2']

try:
    is_exit = False

    dataset_index = int(input('Select dataset:\n1. ped1\n2. ped2\n'))
    selected_dataset = dataset[dataset_index - 1]
    model_name = f'{selected_dataset}_final_model'

    logger.info('Initializing model')
    model = load_model(f'pretrained/{model_name}/{model_name}_end_train',
                       custom_objects={'total_loss': total_loss})
    model = Model(inputs=model.input, outputs=model.output[0])
    error_patcher = get_error_patcher((DIM[1], DIM[0], DIM[2]), POOL_SIZE)

    while not is_exit:
        read_event.clear()
        predict_event.clear()

        print('\nList video:')
        print('Good samples', test_config.GOOD_RESULTS)
        print('Intermediate samples', test_config.INTERMEDIATE_RESULTS)
        print('Bad samples', test_config.BAD_RESULTS)
        video = input('Select video: ')

        reader = Reader(model_input, BATCH_SIZE, SEQ_LEN, DIM, read_event)
        predictor = Predictor(model, error_patcher, model_input, model_output, read_event,
                              predict_event)
        visualizer = Visualizer(model_output, predict_event, 20)
        reader.set_video(f'dataset/{selected_dataset}/Test/.videos/{video}.mp4')
        predictor.set_normalization_param(test_config.NORMALIZATION_PARAM[int(video) - 1])

        reader.start()
        predictor.start()
        visualizer.start()

        reader.join()
        predictor.join()
        visualizer.join()
        is_exit = input('Again? y/n: ') == 'n'

except KeyboardInterrupt:
    logger.info('Force stopping all threads')
    read_event.set()
    predict_event.set()
    reader.join()
    predictor.join()
    visualizer.join()