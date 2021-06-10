from typing import Mapping
import cv2
import os


def video_to_frames(root_path='.'):
    allowed_ext = ['.avi', '.mp4']

    all_videos = [
        video for video in os.listdir(root_path) if os.path.splitext(video)[-1] in allowed_ext
    ]
    print(f'Found {len(all_videos)} videos')

    for video in all_videos:
        print('Processing', video)

        vid_cap = cv2.VideoCapture(f'{root_path}/{video}')
        out_dir = f'{root_path}/{os.path.splitext(video)[0]}'
        os.makedirs(out_dir, exist_ok=True)

        count = 1
        success, image = vid_cap.read()
        print(success, image.shape)
        while (success):
            print(f'{root_path}/{video}/{count:03}.jpg')
            cv2.imwrite(f'{out_dir}/{count:03}.jpg', image)
            success, image = vid_cap.read()
            count += 1


def frames_to_video(root_path='.'):
    allowed_ext = ['.tif', '.jpg', '.jpeg', '.png']

    all_videos = [video for video in next(os.walk(root_path))][1]
    print(f'Found {len(all_videos)} videos')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 24

    for video in all_videos:
        print('Processing', video)
        all_frames = sorted([
            frame for frame in os.listdir(f'{root_path}/{video}')
            if os.path.splitext(frame)[-1] in allowed_ext
        ])

        if len(all_frames) > 0:
            temp_frame = cv2.imread(f'{root_path}/{video}/{all_frames[0]}')
            height, width, channel = temp_frame.shape
            video_writer = cv2.VideoWriter(f'{root_path}/{video}.mp4', fourcc, fps, (width, height))

            for frame_path in all_frames:
                frame = cv2.imread(f'{root_path}/{video}/{frame_path}')
                video_writer.write(frame)
            video_writer.release()
