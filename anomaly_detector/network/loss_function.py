import tensorflow as tf
from nptyping import NDArray
from tensorflow.keras import backend as K


def l1_loss(y_true: NDArray, y_pred: NDArray):
    error = y_true - y_pred
    error = K.abs(error)
    sum_error = K.sum(error, axis=(1, 2, 3))
    l1_loss = K.mean(sum_error, axis=0)
    return l1_loss


def l2_loss(y_true: NDArray, y_pred: NDArray):
    error = y_true - y_pred
    sqr_error = K.square(error)
    sum_sqr_error = K.sum(sqr_error, axis=(1, 2, 3))
    l2_loss = K.mean(sum_sqr_error, axis=0)
    return l2_loss


def gradient_loss(y_true: NDArray, y_pred: NDArray):
    dy_y_true, dx_y_true = tf.image.image_gradients(y_true)
    dy_y_pred, dx_y_pred = tf.image.image_gradients(y_pred)
    diff = K.abs(dx_y_pred - dx_y_true) + K.abs(dy_y_pred - dy_y_true)
    # sum_diff = K.sum(diff, axis=(1,2,3))
    # gradient_loss = K.mean(sum_diff, axis=0)
    gradient_loss = K.mean(diff)
    return gradient_loss


def total_loss(y_true: NDArray, y_pred: NDArray):
    g_loss = gradient_loss(y_true, y_pred)
    p_loss = l2_loss(y_true, y_pred)
    return p_loss + g_loss


def l2_loss_with_weight(weight: int):
    def l2_loss(y_true: NDArray, y_pred: NDArray):
        error = y_true - y_pred
        sqr_error = K.square(error)
        sum_sqr_error = K.sum(sqr_error, axis=(1, 2, 3))
        l2_loss = K.mean(sum_sqr_error, axis=0)
        return l2_loss * weight

    return l2_loss