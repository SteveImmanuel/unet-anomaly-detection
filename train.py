import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

from anomaly_detector.config import DIM, BATCH_SIZE, SEQ_LEN, TRAIN_PATH, VALIDATION_PATH, TEST_PATH, LEARNING_RATE, EPSILON, EPOCHS, MODEL_NAME
from anomaly_detector.network.data_generator import DataGenerator
from anomaly_detector.network.model import get_model
from anomaly_detector.network.loss_function import total_loss
from anomaly_detector.utils.training_callbacks import get_all_callbacks

train_dataset = DataGenerator(TRAIN_PATH, dim=DIM, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
val_dataset = DataGenerator(VALIDATION_PATH,
                            dim=DIM,
                            batch_size=BATCH_SIZE,
                            seq_len=SEQ_LEN,
                            shuffle=False)
test_dataset = DataGenerator(TEST_PATH,
                             dim=DIM,
                             batch_size=BATCH_SIZE,
                             seq_len=SEQ_LEN,
                             shuffle=False)

is_load_model = input('Load model? (y/N): ') == 'y'

if is_load_model:
    latest_model = sorted(os.listdir(f'pretrained/{MODEL_NAME}/checkpoints'))[-1]
    model = load_model(f'pretrained/{MODEL_NAME}/checkpoints/{latest_model}',
                       custom_objects={'total_loss': total_loss})
else:
    model = get_model()

model.compile(loss=[total_loss, total_loss],
              optimizer=Adam(learning_rate=LEARNING_RATE, epsilon=EPSILON),
              run_eagerly=True)
callbacks = get_all_callbacks(val_dataset)

model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks)