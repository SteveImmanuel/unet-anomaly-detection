#Model configuration
DIM = (160, 120, 1)
SEQ_LEN = 5
BATCH_SIZE = 16
MODEL_NAME = 'ped2_final_model'

#Dataset configuration
DATASET = 'ped2'
TRAIN_PATH = 'dataset/ped2/Train'
VALIDATION_PATH = 'dataset/ped2/Validation'
TEST_PATH = 'dataset/ped2/Test'

#Callback configuration
UPDATE_FREQ = 50
CHECKPOINT_PERIOD = 10
STOP_PATIENCE = 10
REDUCE_PATIENCE = 4
DISPLAY_WHILE_TRAINING = False
OUTPUT_TRAINING_IMAGES = 'training_images'
FRAME_SIZE_MULTIPLIER = 2.5

#Training configuration
LEARNING_RATE = 1e-3
EPSILON = 1e-7
EPOCHS = 100
