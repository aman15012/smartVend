#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json
from PIL import Image, ImageDraw, ImageFont
import operator

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
	description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
	'-c',
	'--conf',
	default='config.json',
	help='path to configuration file')

argparser.add_argument(
	'-w',
	'--weights',
	default='weights.h5',
	help='path to pretrained weights')

argparser.add_argument(
	'-i',
	'--input',
	default='./test/',
	help='path to an image or an video (mp4 format)')

def common_area(box1, box2):

	xmin1  = ((box1.x - box1.w/2) * 1.0)
	xmax1  = ((box1.x + box1.w/2) * 1.0)
	ymin1  = ((box1.y - box1.h/2) * 1.0)
	ymax1  = ((box1.y + box1.h/2) * 1.0)

	xmin2  = ((box2.x - box2.w/2) * 1.0)
	xmax2  = ((box2.x + box2.w/2) * 1.0)
	ymin2  = ((box2.y - box2.h/2) * 1.0)
	ymax2  = ((box2.y + box2.h/2) * 1.0)

	intersect = [0,0,0,0]
	intersect[0] = max(xmin1, xmin2)
	intersect[1] = max(ymin1, ymin2)
	intersect[2] = min(xmax1, xmax2)
	intersect[3] = min(ymax1, ymax2)

	if(intersect[1] < intersect[3] and intersect[0] < intersect[2]):
		return 1
	else:
		return -1

def _main_(args):
 
	config_path  = args.conf
	weights_path = args.weights
	image_path   = args.input

	with open(config_path) as config_buffer:    
		config = json.load(config_buffer)

	###############################
	#   Make the model 
	###############################

	yolo = YOLO(architecture        = config['model']['architecture'],
				input_size          = config['model']['input_size'], 
				labels              = config['model']['labels'], 
				max_box_per_image   = config['model']['max_box_per_image'],
				anchors             = config['model']['anchors'])

	###############################
	#   Load trained weights
	###############################    

	print weights_path
	yolo.load_weights(weights_path)

	counter = 0
	k = 4 # max possible4 boxes in one frame
	origin = [0 for i in range(k)]
	now = [0 for i in range(k)]
	for image_file in sorted(os.listdir(image_path)):

		# image = Image.open(os.path.join(image_path, image_file))
		image = cv2.imread(os.path.join(image_path, image_file))
		boxes = yolo.predict(image)

####### multiple identification removal
		if(len(boxes)>0):
			mark = [0 for i in range(len(boxes))]
			l = []

			x = []
			for box1 in range(len(boxes)):
				x_ = [0,box1]
				for box2 in range(len(boxes)):
					if(common_area(boxes[box1],boxes[box2])==1 and box1!=box2 and mark[box2]==0):
						x_[0] += 1
						x_.append(box2)
				x.append(x_)
			x.sort()
			x = x[::-1]

			if(x[0][0]>0):

				for box in range(len(x)):

					if(mark[box]==0):

						if(x[box][0]==1 and mark[x[box][2]] == 0):
							if(boxes[x[box][2]].get_score()>boxes[x[box][1]].get_score()):
								mark[x[box][1]] = 1
							else:
								mark[x[box][2]] = 1

						elif(x[box][0]>1):
							avg = 0.0
							for i in range(2,len(x[box])):
								if(mark[x[box][i]] == 0):
									avg += abs(boxes[x[box][i]].get_score() - boxes[x[box][1]].get_score())
							avg /= x[box][0]
							print(avg)
							if(avg <= 0.2):
								mark[x[box][1]] = 1
							else:
								flag = 0
								for i in range(2,len(x[box])):
									if(mark[x[box][i]] == 0):
										if(boxes[x[box][i]].get_score() > boxes[x[box][1]].get_score()):
											mark[x[box][1]] = 1
										else:
											mark[x[box][i]] = 1

				
				for box in range(len(boxes)):
					if(mark[box]==0):
						l.append(boxes[box])

				print(mark)

				boxes = l

########
				

		image = draw_boxes(image, boxes, config['model']['labels'])

		print os.path.join(image_path, image_file)[:-4]
		print len(boxes), 'boxes are found'

		cv2.imwrite(os.path.join(image_path, image_file)[:-4] + '_detected' + os.path.join(image_path, image_file)[-4:], image)

		boxes.sort(key=operator.attrgetter('x'))
		boxes = boxes[::-1]

		for i in range(k):
			if(len(boxes)>i):
 
				if(now[i]==0):
					origin[i] = boxes[i].y

				now[i] = boxes[i].y

			else:
				if(origin[i]!=0):
					if(origin[i]-now[i] <= 0):
						counter += 1
						origin[i]=0
						now[i]=0
					else:
						counter += 1
						origin[i]=0
						now[i]=0

		print(origin)
		print(now)
		print(counter)

	for i in range(k):
		if(now[i]!=0):
			counter+=1
	print(counter)

if __name__ == '__main__':
	args = argparser.parse_args()
	_main_(args)