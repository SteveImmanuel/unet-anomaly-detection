from tensorflow.keras.utils import plot_model
from tensorflow.keras import Input
from tensorflow.keras.layers import TimeDistributed, Conv2D, ConvLSTM2D, Conv2DTranspose, LayerNormalization, Concatenate, Lambda, BatchNormalization, AveragePooling2D, UpSampling2D, LeakyReLU
from tensorflow.keras.models import Model
from anomaly_detector.config import SEQ_LEN, DIM


def get_model():
    input_layer = Input(shape=(SEQ_LEN, DIM[1], DIM[0], DIM[2]), name='input_seq')

    enc1 = TimeDistributed(Conv2D(32, (3, 3), strides=1, padding='same'))(input_layer)
    enc1 = TimeDistributed(AveragePooling2D())(enc1)
    enc1 = BatchNormalization()(enc1)
    enc1 = LeakyReLU()(enc1)

    enc2 = TimeDistributed(Conv2D(64, (3, 3), strides=1, padding='same'))(enc1)
    enc2 = TimeDistributed(AveragePooling2D())(enc2)
    enc2 = BatchNormalization()(enc2)
    enc2 = LeakyReLU()(enc2)

    btk = ConvLSTM2D(64, (3, 3), padding='same', return_sequences=True)(enc2)
    btk = BatchNormalization()(btk)
    btk = ConvLSTM2D(64, (3, 3), padding='same')(btk)
    btk = BatchNormalization()(btk)

    dec_pred = UpSampling2D()(btk)
    dec_pred = Conv2DTranspose(32, (3, 3), strides=1, padding='same')(dec_pred)
    dec_pred = BatchNormalization()(dec_pred)
    dec_pred = LeakyReLU()(dec_pred)

    skip_con = Concatenate(axis=-1)([btk, enc2[:, -1, :, :]])
    dec_optf = UpSampling2D()(skip_con)
    dec_optf = Conv2DTranspose(32, (3, 3), strides=1, padding='same')(dec_optf)
    dec_optf = BatchNormalization()(dec_optf)
    dec_optf = LeakyReLU()(dec_optf)

    dec_pred = UpSampling2D()(dec_pred)
    dec_pred = Conv2DTranspose(16, (3, 3), strides=1, padding='same')(dec_pred)
    dec_pred = BatchNormalization()(dec_pred)
    dec_pred = LeakyReLU()(dec_pred)

    skip_con = Concatenate(axis=-1)([dec_optf, enc1[:, -1, :, :]])
    dec_optf = UpSampling2D()(skip_con)
    dec_optf = Conv2DTranspose(16, (3, 3), strides=1, padding='same')(dec_optf)
    dec_optf = BatchNormalization()(dec_optf)
    dec_optf = LeakyReLU()(dec_optf)

    out_pred = Conv2D(DIM[-1], (1, 1), activation='sigmoid', name='prediction')(dec_pred)
    out_optf = Conv2D(3, (1, 1), activation='sigmoid', name='optical_flow')(dec_optf)

    model = Model(inputs=input_layer, outputs=[out_pred, out_optf], name='autoencoder')
    model.summary()
    plot_model(model, show_shapes=True, show_layer_names=False)