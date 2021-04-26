import math
import shutil
import os
from typing import List
from anomaly_detector.network.data_generator import DataGenerator


def split_test(test_path: str, n: int, **kwargs) -> None:
    all_videos = [
        video for video in next(os.walk(test_path))[1]
        if not (video.endswith('_optf') or video.startswith('Frames'))
    ]
    chunk_size = math.ceil(len(all_videos) / n)

    for i in range(n):
        chunk = all_videos[i * chunk_size:(i + 1) * chunk_size]

        for video in chunk:
            shutil.move(f'{test_path}/{video}', f'{test_path}/Test{i}/{video}')
            shutil.move(f'{test_path}/{video}_optf', f'{test_path}/Test{i}/{video}_optf')


def get_all_test(test_path: str, **kwargs) -> List[DataGenerator]:
    generators = []
    all_test = next(os.walk(test_path))[1]

    for test in all_test:
        data_generator = DataGenerator(f'{test_path}/{test}', **kwargs)
        generators.append(data_generator)

    return generators