import cv2
import os
import shutil


vidcap = cv2.VideoCapture('video.mp4')
success,image = vidcap.read()
count = 0
success = True

shutil.rmtree('test')
os.makedirs('test')

while success:
	success,image = vidcap.read()
	if(count%1==0):

			s = str(count).zfill(5) # max frames in video can't exceed 99999 
			cv2.imwrite(os.path.join("test","frame" + s + ".jpg"), image)     # save frame as JPEG file

	count += 1
