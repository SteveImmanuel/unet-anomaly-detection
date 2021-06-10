# import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import time
import logging
import cv2
import numpy as np
from numpy.core.numeric import full
from tensorflow.keras.models import load_model, Model
from collections import deque

from anomaly_detector.config import DIM, SEQ_LEN
from anomaly_detector.network.loss_function import total_loss
from anomaly_detector.utils.evaluation import get_error_patcher, locate_anomaly, calculate_loss

from pretrained.ped1_final_model import test_config as test_config_1
from pretrained.ped2_final_model import test_config as test_config_2

logging.basicConfig()
logger = logging.getLogger('StreamLive')
logger.setLevel(logging.INFO)

dataset = ['ped1', 'ped2']
test_config = [test_config_1, test_config_2]

try:
    is_exit = False

    dataset_index = int(input('Select dataset:\n1. ped1\n2. ped2\n'))
    selected_dataset = dataset[dataset_index - 1]
    selected_test_config = test_config[dataset_index - 1]
    model_name = f'{selected_dataset}_final_model'
    capturer = None

    logger.info('Initializing model')
    model = load_model(f'pretrained/{model_name}/{model_name}_end_train',
                       custom_objects={'total_loss': total_loss})
    model = Model(inputs=model.input, outputs=model.output[0])
    error_patcher = get_error_patcher((DIM[1], DIM[0], DIM[2]), selected_test_config.POOL_SIZE)

    while not is_exit:
        print('\nList video:')
        print('Good samples', selected_test_config.GOOD_RESULTS)
        print('Intermediate samples', selected_test_config.INTERMEDIATE_RESULTS)
        print('Bad samples', selected_test_config.BAD_RESULTS)
        video = input('Select video: ')

        logger.info('Initializing params')
        pool_size = selected_test_config.POOL_SIZE
        smooth_strength = selected_test_config.SMOOTH_STRENGTH
        norm_param = selected_test_config.NORMALIZATION_PARAM[int(video) - 1]
        video_path = f'dataset/{selected_dataset}/Test/.videos/{video}.mp4'

        logger.info('Initializing result window')
        cv2.namedWindow('Prediction', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Prediction', 160 * 7, 120 * 7)

        capturer = cv2.VideoCapture(video_path)
        success, image = capturer.read()
        sequence = deque(maxlen=SEQ_LEN)
        sequence_batch = []
        target_batch = []

        elapsed = -time.time()

        while success:
            frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (DIM[0], DIM[1]))
            frame = frame / 255.0
            frame = np.expand_dims(frame, axis=frame.ndim)

            if (len(sequence) == SEQ_LEN):
                full_seq = np.array(sequence)
                full_seq = np.expand_dims(full_seq, axis=0)
                target_batch = np.expand_dims(frame, axis=0)

                y = model.predict(full_seq)

                raw_losses, losses = calculate_loss(error_patcher,
                                                    y,
                                                    target_batch,
                                                    min_loss=norm_param[0],
                                                    max_loss=norm_param[1],
                                                    smooth_strength=smooth_strength)
                error_map = locate_anomaly(y, target_batch, raw_losses, losses, pool_size, 0.6)

                g_truth = np.repeat(target_batch, 3, axis=-1)
                prediction = np.repeat(y, 3, axis=-1)

                combined_image = np.zeros(160 * 120 * 4 * 3).reshape(240, 320, 3)
                combined_image += 0.25
                combined_image[:120, :160] = g_truth[0]
                combined_image[:120, 160:] = prediction[0]
                combined_image[120:, 80:240] = error_map[0]

                cv2.imshow('Prediction', combined_image)
                cv2.waitKey(1)

                elapsed += time.time()
                logger.info(f'FPS: {1/elapsed:.4f}')
                elapsed = -time.time()

            sequence.append(frame)
            success, image = capturer.read()

        capturer.release()
        is_exit = input('Again? y/n (y): ') == 'n'

except KeyboardInterrupt:
    logger.info('Force stopping')
finally:
    if capturer is not None:
        capturer.release()
    cv2.destroyAllWindows()
