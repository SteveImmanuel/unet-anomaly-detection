from anomaly_detector.config import DIM, BATCH_SIZE, SEQ_LEN, TRAIN_PATH, VALIDATION_PATH, TEST_PATH, LEARNING_RATE, EPSILON, EPOCHS
from anomaly_detector.network.data_generator import DataGenerator
from anomaly_detector.network.model import get_model
from anomaly_detector.network.loss_function import total_loss
from anomaly_detector.utils.training_callbacks import get_all_callbacks
from tensorflow.keras.optimizers import Adam

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

model = get_model()
model.compile(loss=[total_loss, total_loss],
              optimizer=Adam(learning_rate=LEARNING_RATE, epsilon=EPSILON),
              run_eagerly=True)
callbacks = get_all_callbacks(val_dataset)
model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, callbacks=callbacks)