import cv2
import sys
import tqdm 
import os
import glob

# Create a dir to store all the video dirs(dir labelled the name of a video and contains frames from that video)
if not os.path.exists('output'):
    os.mkdir("output")

# Define the path to the directory of videos
path = '/workspace/ikea_full_data/train_vids/'

for video in tqdm.tqdm(glob.glob(path + '*')):
    vid_name = video.split('/')[-1].split('.')[0]
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    vid_dir = f'output/{vid_name}'
    # Create a video dir for the current video
    if not os.path.exists(vid_dir):
        os.mkdir(vid_dir)
    # If the video dir for this video already exists then continue on to the next video
    else:
        continue
    # If we just made the video dir, then fill it with the frames from the vid
    while success:
        cv2.imwrite(f"{vid_dir}/{count:04d}.jpg", image)
        success,image = vidcap.read()
        count += 1
