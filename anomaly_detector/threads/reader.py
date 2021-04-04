import cv2
import numpy as np
from threading import Thread, Event
from queue import Queue
from collections import deque
from typing import Tuple


class Reader(Thread):
    def __init__(self, video_path: str, model_input: Queue, batch_size: int, seq_len: int,
                 dim: Tuple[int, int, int], read_event: Event):
        Thread.__init__(self)
        self.model_input = model_input
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dim = dim
        self.read_event = read_event
        self.video_capturer = cv2.VideoCapture(video_path)

    def run(self):
        try:
            success, image = self.video_capturer.read()
            sequence = deque(maxlen=self.seq_len)
            sequence_batch = []
            target_batch = []

            while success:
                frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                frame = cv2.resize(frame, (self.dim[0], self.dim[1]))
                frame = frame / 255.0
                frame = frame.reshape(*frame.shape, 1)

                if (len(sequence) == self.seq_len):
                    full_seq = np.array(sequence)
                    sequence_batch.append(full_seq)
                    target_batch.append(frame)

                sequence.append(frame)

                if (len(sequence_batch) == self.batch_size):
                    print('read 1 whole batch')
                    full_sequence_batch = np.array(sequence_batch)
                    full_target_batch = np.array(target_batch)
                    self.model_input.put({'seq': full_sequence_batch, 'target': full_target_batch})
                    sequence_batch = []
                    target_batch = []

                success, image = self.video_capturer.read()

        except Exception as e:
            print(e)
        finally:
            self.video_capturer.release()
            self.read_event.set()