import cv2
import os

# Place this file in a directory containing the videos and run the script

ALLOWED_EXT = ['.avi', '.mp4']

all_videos = [video for video in os.listdir() if os.path.splitext(video)[-1] in ALLOWED_EXT]
print(f'Found {len(all_videos)} videos')

for video in all_videos:
    print('Processing', video)
    directory = os.path.splitext(video)[0]
    if not os.path.exists(directory):
        os.mkdir(directory)

    vid_cap = cv2.VideoCapture(video)
    count = 1
    success, image = vid_cap.read()
    while (success):
        cv2.imwrite(f'{directory}/{count}.jpg', image)
        success, image = vid_cap.read()
        count += 1