import matplotlib.pyplot as plt
from typing import Dict, Tuple
from matplotlib.animation import FuncAnimation


class ResultVideoGenerator:
    def __init__(self,
                 pred_dict: Dict,
                 g_truth_dict: Dict,
                 losses_dict: Dict,
                 error_map_dict: Dict,
                 fps: int = 20,
                 outfile: str = 'out.mp4'):
        self.pred_dict = {}
        self.g_truth_dict = {}
        self.losses_dict = losses_dict
        self.error_map_dict = error_map_dict
        self.fps = fps
        self.outfile = outfile

        self.current_video = list(pred_dict.keys())[0]

        for key, item in pred_dict.items():
            if 0 in item.shape:
                continue

            if pred_dict[self.current_video].shape[-1] == 1:
                cmap = 'gray'
                self.pred_dict[key] = pred_dict[key].squeeze()
                self.g_truth_dict[key] = g_truth_dict[key].squeeze()
            else:
                cmap = None
                self.pred_dict[key] = pred_dict[key]
                self.g_truth_dict[key] = g_truth_dict[key]

        height = self.pred_dict[self.current_video][0].shape[0]
        width = self.pred_dict[self.current_video][0].shape[1]

        self.fig = plt.figure(figsize=(3 * 4 * (width / height), 2 * 4))
        self.fig.suptitle('Model Evaluation', fontsize=20)
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.9)

        self.ax1 = self.fig.add_subplot(2, 3, 1)
        self.ax1.set_title('Prediction')
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])

        self.ax2 = self.fig.add_subplot(2, 3, 2)
        self.ax2.set_title('Ground Truth')
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])

        self.ax3 = self.fig.add_subplot(2, 3, 3)
        self.ax3.set_title('Error Map')
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])

        self.ax4 = self.fig.add_subplot(2, 3, (4, 6))
        self.ax4.set_title('Regularity Score')
        self.ax4.set_xlim([0, self.pred_dict[self.current_video].shape[0]])
        self.ax4.set_ylim([0, 1.1])
        self.ax4.set_xlabel('Frames')
        self.ax4.set_ylabel('Score')

        self.current_losses = []
        self.x_data = []

        self.ax1_img = self.ax1.imshow(self.pred_dict[self.current_video][0],
                                       cmap=cmap,
                                       aspect='auto',
                                       vmin=0,
                                       vmax=1)
        self.ax2_img = self.ax2.imshow(self.g_truth_dict[self.current_video][0],
                                       cmap=cmap,
                                       aspect='auto',
                                       vmin=0,
                                       vmax=1)
        self.ax3_img = self.ax3.imshow(self.error_map_dict[self.current_video][0],
                                       cmap='jet',
                                       aspect='auto')
        self.ax4_line = self.ax4.plot(self.current_losses)[0]
        self.text = self.ax4.text(0.5,
                                  -0.3,
                                  f'Video: {self.current_video}',
                                  horizontalalignment='center',
                                  fontsize=13,
                                  transform=self.ax4.transAxes)

        self.parameters = []
        for key, item in self.pred_dict.items():
            total_frames = len(item)
            self.parameters += [(key, i, total_frames) for i in range(total_frames)]

    def update(self, param: Tuple(str, int, int)):
        video_key, index, total_frames = param

        if self.current_video != video_key:
            self.x_data = []
            self.current_losses = []
            self.ax4.set_xlim([0, total_frames])
            self.current_video = video_key
            self.text.set_text(f'Video: {self.current_video}')
            print()

        print(f'\rProcessing {video_key}: {((index + 1)/ total_frames * 100):.2f} %', end='')
        pred_frame = self.pred_dict[video_key][index]
        truth_frame = self.g_truth_dict[video_key][index]
        error_frame = self.error_map_dict[video_key][index]
        self.current_losses.append(self.losses_dict[video_key][index])
        self.x_data.append(index)

        self.ax1_img.set_data(pred_frame)
        self.ax2_img.set_data(truth_frame)
        self.ax3_img.set_data(error_frame)
        self.ax4_line.set_data(self.x_data, self.current_losses)

        return [self.ax1_img, self.ax2_img, self.ax3_img, self.ax4_line, self.text]

    def generate(self):
        generated_video = FuncAnimation(self.fig,
                                        self.update,
                                        frames=self.parameters,
                                        interval=1000 / self.fps,
                                        blit=True)
        generated_video.save(self.outfile, fps=self.fps, extra_args=['-vcodec', 'libx264'])
