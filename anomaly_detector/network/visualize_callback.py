from tensorflow.keras.callbacks import Callback
from anomaly_detector.network.data_generator import DataGenerator
from typing import Dict
from nptyping import ndarray
from anomaly_detector.config import FRAME_SIZE_MULTIPLIER


class VisualizeCallback(Callback):
    def __init__(self,
                 dataset_name: str,
                 data_generator: DataGenerator,
                 n_display_sequence: int = 3,
                 shuffle: bool = True,
                 out_dir: str = '',
                 display_while_training: bool = False):
        super(VisualizeCallback, self).__init__()
        self.dataset_name = dataset_name
        self.data_generator = data_generator
        self.n_display_sequence = n_display_sequence
        self.shuffle = shuffle
        self.out_dir = out_dir
        self.display_while_training = display_while_training

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        # [nth_batch][input/target][nth_sample][nth_image]
        self.dim = self.data_generator[0][0][0][0].shape

        if not self.shuffle:
            self.data = self.data_generator[0]

    def __visualize(self, epoch: int):
        if self.shuffle:
            index = np.random.randint(0, len(self.data_generator))
            X, y = self.data_generator[index]
        else:
            X, y = self.data
            predictions = self.model.predict(X)
        self.display_pred(epoch, predictions[0], y[0], predictions[1], y[1],
                          self.n_display_sequence)

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        self.__visualize(epoch + 1)

    def on_train_begin(self, logs: Dict = None):
        self.__visualize(0)

    def format_frames(self, frames: ndarray, frame_indexes: ndarray):
        n = len(frame_indexes)
        if frames.shape[-1] == 1:
            frame_seq = frames[frame_indexes, :, :, 0]
            frame_seq = np.swapaxes(frame_seq, 0, 1)
            frame_seq = np.reshape(frame_seq, (frame_seq.shape[0], frame_seq.shape[-1] * n))
        else:
            frame_seq = frames[frame_indexes, :, :, :]
            frame_seq = np.swapaxes(frame_seq, 0, 1)
            frame_seq = np.reshape(
                frame_seq, (frame_seq.shape[0], frame_seq.shape[2] * n, frame_seq.shape[-1]))

        return frame_seq

    def display_pred(self, epoch: int, frame_predictions: ndarray, frame_ground_truth: ndarray,
                     optf_predictions: ndarray, optf_ground_truth: ndarray, n: int):
        height = self.dim[0]
        width = self.dim[1]

        if self.shuffle:
            frame_indexes = np.random.choice(len(frame_predictions), size=n)
        else:
            interval = len(frame_predictions) // n
            frame_indexes = [i for i in range(interval, n * interval + 1, interval)]

        ground_truth_frames = self.format_frames(frame_ground_truth, frame_indexes)
        predictions_frames = self.format_frames(frame_predictions, frame_indexes)
        ground_truth_optf = self.format_frames(optf_ground_truth, frame_indexes)
        predictions_optf = self.format_frames(optf_predictions, frame_indexes)

        if len(predictions_frames.shape) == 3:
            cmap = None
        else:
            cmap = 'gray'

        fig = plt.figure(figsize=(n * FRAME_SIZE_MULTIPLIER * width / height,
                                  4 * FRAME_SIZE_MULTIPLIER))
        fig.suptitle(f'Training {self.dataset_name} | Epoch-{epoch}', fontsize=20)
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)

        ax1 = fig.add_subplot(4, 1, 1)
        ax1.set_title('Ground Truth Frames')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.imshow(ground_truth_frames, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        ax2 = fig.add_subplot(4, 1, 2)
        ax2.set_title('Prediction Frames')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.imshow(predictions_frames, cmap=cmap, aspect='auto', vmin=0, vmax=1)

        ax3 = fig.add_subplot(4, 1, 3)
        ax3.set_title('Ground Truth Optical Flow')
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.imshow(ground_truth_optf, aspect='auto', vmin=0, vmax=1)

        ax4 = fig.add_subplot(4, 1, 4)
        ax4.set_title('Prediction Optical Flow')
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.imshow(predictions_optf, aspect='auto', vmin=0, vmax=1)

        if self.out_dir != '':
            plt.savefig(f'{self.out_dir}/train_{self.dataset_name}_epoch_{epoch:03d}.jpg',
                        format='jpg')
        if self.display_while_training:
            plt.show()
        plt.close(fig)