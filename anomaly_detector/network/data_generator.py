import os
import numpy as np

from PIL import Image
from typing import Tuple, List
from nptyping import NDArray
from tensorflow.keras.utils import Sequence

ALLOWED_EXT = ['.tif', '.jpg', '.jpeg', '.png']


class DataGenerator(Sequence):
    def __init__(self,
                 path: str,
                 dim: Tuple[int, int, int] = (128, 128, 1),
                 seq_len: int = 5,
                 batch_size: int = 15,
                 shuffle: bool = True):
        self.path = path
        self.dim = dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.video_dict = {}
        self.batch_ids = self.__generate_batch_ids()

    def __generate_batch_ids(self):
        all_batch_ids = []
        all_videos = next(os.walk(self.path))[1]

        if not self.shuffle:
            all_videos = sorted(all_videos)

        index = 0
        for video_name in all_videos:
            if video_name.endswith('_gt') or video_name.endswith('_optf') or video_name.startswith(
                    '.') or video_name.startswith('Frames'):
                continue
            total_frames = len([
                path for path in os.listdir(f'{self.path}/{video_name}')
                if os.path.splitext(path)[-1] in ALLOWED_EXT
            ])
            total_possible_sequence = total_frames - self.seq_len
            total_batch = total_possible_sequence // self.batch_size

            total_possible_frames = total_batch * self.batch_size
            self.video_dict[video_name] = (index, index + total_possible_frames - 1)
            index += total_possible_frames

            for i in range(total_batch):
                all_batch_ids.append(f'{video_name}-{i}')

        return all_batch_ids

    def __get_frame_sequences(self, all_frames: List[NDArray], optf_frames: List[NDArray]):
        sequences = []
        pred_targets = []
        optf_targets = []

        for i in range(0, len(all_frames) - self.seq_len):
            sequence = []
            for j in range(i, i + self.seq_len):
                sequence.append(all_frames[j])

            sequence = np.array(sequence)
            sequences.append(sequence)
            pred_targets.append(all_frames[i + self.seq_len])
            optf_targets.append(optf_frames[i])

        return np.array(sequences), (np.array(pred_targets), np.array(optf_targets))

    def generate_batch(self, batch_id: str):
        video_name, index = batch_id.split('-')
        index = int(index)
        all_frames = []
        optf_frames = []

        dir_path = f'{self.path}/{video_name}'
        all_frames_path = sorted(
            [path for path in os.listdir(dir_path) if os.path.splitext(path)[-1] in ALLOWED_EXT])
        all_optf_path = sorted([
            path for path in os.listdir(f'{dir_path}_optf')
            if os.path.splitext(path)[-1] in ALLOWED_EXT
        ])

        first_index = index * self.batch_size
        last_index = (index + 1) * self.batch_size + self.seq_len - 1
        for i in range(first_index, last_index + 1):
            frame_array = self.img_to_np_arr(f'{dir_path}/{all_frames_path[i]}')
            all_frames.append(frame_array)

            if i >= (last_index + 1 - self.batch_size):
                optf_array = self.img_to_np_arr(f'{dir_path}_optf/{all_optf_path[i-1]}',
                                                force_rgb=True)
                optf_frames.append(optf_array)

        return self.__get_frame_sequences(all_frames, optf_frames)

    def img_to_np_arr(self, path: str, force_rgb: bool = False):
        frame = Image.open(path).resize((self.dim[0], self.dim[1]), Image.LANCZOS)

        is_rgb = self.dim[-1] == 3
        if is_rgb or force_rgb:
            channel = 3
            frame = frame.convert('RGB')
        else:
            channel = 1
            frame = frame.convert('L')

        frame_array = np.array(frame, dtype='float32') / 255.0
        frame_array = frame_array.reshape(self.dim[1], self.dim[0], channel)
        return frame_array

    def __len__(self):
        return len(self.batch_ids)

    def __getitem__(self, index: int):
        X, y = self.generate_batch(self.batch_ids[index])
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.batch_ids)
