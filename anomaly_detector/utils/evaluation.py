import numpy as np
from typing import List, Tuple, Dict
from nptyping import NDArray
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Lambda
from tensorflow.keras.initializers import Constant


def get_error_patcher(
    input_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int] = (32, 32),
    stride_size: Tuple[int, int] = (1, 1)
) -> Model:
    def max_and_log(x):
        maximum_val = K.max(x, axis=(1, 2, 3))
        return K.log(maximum_val)

    input_layer = Input(shape=input_shape)
    o1 = Conv2D(1,
                patch_size,
                strides=stride_size,
                padding='valid',
                input_shape=input_shape,
                kernel_initializer=Constant(1.),
                use_bias=False)(input_layer)
    o2 = Lambda(max_and_log, output_shape=(None, ))(o1)
    error_patcher = Model(inputs=input_layer, outputs=[o1, o2])

    return error_patcher


def smoothen_loss(loss_list: NDArray, strength: float) -> NDArray:
    last_smoothed = loss_list[0]

    smoothed_list = []
    for loss in loss_list:
        smoothed_loss = loss * (1 - strength) + last_smoothed * strength
        smoothed_list.append(smoothed_loss)
        last_smoothed = smoothed_loss

    return np.array(smoothed_list)


def calculate_loss(error_patcher: Model,
                   prediction: NDArray,
                   g_truth: NDArray,
                   smooth_strength: float = 0.5,
                   upper_limit_loss: float = 9999,
                   min_loss: float = None,
                   max_loss: float = None) -> Tuple[NDArray, NDArray]:
    raw_losses, losses = error_patcher.predict(np.square(prediction - g_truth))
    losses = np.clip(losses, a_min=0, a_max=upper_limit_loss)
    if min_loss == None:
        min_loss = np.min(losses)
    if max_loss == None:
        max_loss = np.max(losses)
    losses = smoothen_loss(losses, smooth_strength)
    losses = 1 - (losses - min_loss) / (max_loss - min_loss)
    return raw_losses, losses


def draw_bounding_box(
    image: NDArray,
    index: int,
    pool_size: Tuple[int, int] = (16, 16),
    strides: Tuple[int, int] = (1, 1)) -> None:
    top_left_row = index[0] * strides[0]
    top_left_col = index[1] * strides[1]
    image[top_left_row:top_left_row + pool_size[0], top_left_col] = np.array([1, 0, 0])
    image[top_left_row:top_left_row + pool_size[0],
          top_left_col + pool_size[1] - 1] = np.array([1, 0, 0])
    image[top_left_row, top_left_col:top_left_col + pool_size[1]] = np.array([1, 0, 0])
    image[top_left_row + pool_size[0] - 1,
          top_left_col:top_left_col + pool_size[1]] = np.array([1, 0, 0])


def locate_anomaly(prediction: NDArray,
                   g_truth: NDArray,
                   raw_losses: NDArray,
                   losses: NDArray,
                   pool_size: Tuple[int, int],
                   threshold: float = 0.5) -> NDArray:

    error_map = np.abs(prediction - g_truth)
    if error_map.shape[-1] != 3:
        error_map = np.repeat(error_map, 3, axis=-1)

    for i in range(len(error_map)):
        if losses[i] >= threshold:
            continue

        index = np.unravel_index(np.argmax(raw_losses[i], axis=None), raw_losses[i].shape)
        draw_bounding_box(error_map[i], (index[0], index[1]), pool_size)
    return error_map


def locate_all_anomaly(pred_dict: Dict,
                       g_truth_dict: Dict,
                       raw_losses_dict: Dict,
                       losses_dict: Dict,
                       pool_size: Tuple[int, int],
                       threshold: float = 0.5) -> Dict:
    error_map_dict = {}

    for key in pred_dict.keys():
        error_map_dict[key] = np.abs(pred_dict[key] - g_truth_dict[key])
        if error_map_dict[key].shape[-1] != 3:
            error_map_dict[key] = np.repeat(error_map_dict[key], 3, axis=-1)

        for i in range(len(error_map_dict[key])):
            if losses_dict[key][i] >= threshold:
                continue

            raw_losses = raw_losses_dict[key][i]
            index = np.unravel_index(np.argmax(raw_losses, axis=None), raw_losses.shape)
            draw_bounding_box(error_map_dict[key][i], (index[0], index[1]), pool_size)

    return error_map_dict
