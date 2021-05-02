#Model configuration
DIM = (160, 120, 1)
SEQ_LEN = 5
BATCH_SIZE = 32
MODEL_NAME = 'ped1_single_unet_2scale_more_param'

#Dataset configuration
DATASET = 'ped1'
TRAIN_PATH = 'dataset/ped1/Train'
VALIDATION_PATH = 'dataset/ped1/Validation'
TEST_PATH = 'dataset/ped1/Test'

#Callback configuration
UPDATE_FREQ = 50
CHECKPOINT_PERIOD = 3
STOP_PATIENCE = 10
REDUCE_PATIENCE = 4
DISPLAY_WHILE_TRAINING = False
OUTPUT_TRAINING_IMAGES = 'training_images'
FRAME_SIZE_MULTIPLIER = 2.5

#Training configuration
LEARNING_RATE = 1e-3
EPSILON = 1e-7
EPOCHS = 100
