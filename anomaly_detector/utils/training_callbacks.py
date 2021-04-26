import os
from typing import List
from anomaly_detector.network.data_generator import DataGenerator
from anomaly_detector.network.visualize_callback import VisualizeCallback
from anomaly_detector.config import MODEL_NAME, DATASET, CHECKPOINT_PERIOD, UPDATE_FREQ, STOP_PATIENCE, REDUCE_PATIENCE, DISPLAY_WHILE_TRAINING, OUTPUT_TRAINING_IMAGES
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback


def get_all_callbacks(val_dataset: DataGenerator) -> List[Callback]:
    tensorboard_cb = TensorBoard(
        log_dir=os.path.join('logs', MODEL_NAME),
        update_freq=UPDATE_FREQ,
    )
    early_stopping_cb = EarlyStopping(monitor='val_loss',
                                      patience=STOP_PATIENCE,
                                      restore_best_weights=True,
                                      mode='min',
                                      verbose=1)
    checkpoint_cb = ModelCheckpoint(filepath=f'pretrained/{MODEL_NAME}/checkpoints' +
                                    '/epoch{epoch:02d}_val_loss{val_loss:.5f}_ckpt',
                                    monitor='val_loss',
                                    mode='min',
                                    verbose=1,
                                    period=CHECKPOINT_PERIOD)
    reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss',
                                     factor=0.5,
                                     patience=REDUCE_PATIENCE,
                                     mode='min',
                                     verbose=1)
    visualize_cb = VisualizeCallback(DATASET,
                                     val_dataset,
                                     n_display_sequence=4,
                                     shuffle=True,
                                     out_dir=OUTPUT_TRAINING_IMAGES,
                                     display_while_training=DISPLAY_WHILE_TRAINING)

    return [tensorboard_cb, early_stopping_cb, checkpoint_cb, reduce_lr_cb, visualize_cb]
