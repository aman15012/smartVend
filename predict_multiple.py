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

def area(x,y):
	if(x>0.1 and y<0.5):
		return 1
	else:
		return 2

def dist(x1,y1,x2,y2):
	return (x1-x2)**2 + (y1-y2)**2

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

	t = max(((intersect[3]-intersect[1])*(intersect[2]-intersect[0])/float((xmax1 - xmin1)*(ymax1 - ymin1))), ((intersect[3]-intersect[1])*(intersect[2]-intersect[0])/float((xmax2 - xmin2)*(ymax2 - ymin2))))

	if(intersect[1] < intersect[3] and intersect[0] < intersect[2] and t>0.7):
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

	k = 6 # max possible 4 boxes in one frame and 2 buffer
	counter = 0

	flag = [0 for i in range(k)]
	flag2 = [0 for i in range(k)]

	originx = [0 for i in range(k)]
	originy = [0 for i in range(k)]

	nowy = [0 for i in range(k)]
	nowx = [0 for i in range(k)]
	for image_file in sorted(os.listdir(image_path))[:-1]:

		# image = Image.open(os.path.join(image_path, image_file))
		thresh =0.6
		image = cv2.imread(os.path.join(image_path, image_file))
		boxes = yolo.predict(image)
		t = []
		for q in boxes:
			if(q.get_score()>thresh):
				t.append(q)
		boxes = t
####### multiple identification removal
		l = []
		if(len(boxes)==1):
			if(boxes[0].get_score()>thresh):
				l.append(boxes[0])
			boxes = l

		elif(len(boxes)>0):
			mark = [0 for i in range(len(boxes))]
		

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
					if(mark[box]==0 and boxes[box].get_score()>thresh):
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


		mark = [-1 for i in range(len(boxes))]

# stable matching problem!!
		
		match = []
		for box in range(len(boxes)):
			t_match = []
			for i in range(k):
				if(abs(boxes[box].x - nowx[i])<=0.3 and abs(boxes[box].y - nowy[i])<=0.3 and nowx[i]!=0):
					t_match.append((dist(boxes[box].x, nowx[i],boxes[box].y, nowy[i]),i))
			t_match.sort()
			match.append(t_match)

		for box in range(len(boxes)):
			while(len(match[box])>0):
				f = 0
				for i in range(box+1,len(boxes)):
					if(len(match[i])>0):
						if((match[i][0][0]<match[box][0][0] and match[i][0][1]==match[box][0][1]) or match[box][0][1] in mark):
							f = 1
							break
				if(f==0 and match[box][0][1] not in mark):
					mark[box] = match[box][0][1]
					break
				else:
					match[box].pop(0)


		put = 0
		for i in range(len(mark)):
			if(mark[i]==-1):
				while(put in mark or nowx[put]!=0):
					put+=1
				
				mark[i] = put
						


		for i in range(k):

			if(i in mark):
				flag2[i] += 1
 				flag[i] = 0
				if(nowy[i]==0):
					originy[i] = boxes[mark.index(i)].y
					originx[i] = boxes[mark.index(i)].x


				nowy[i] = boxes[mark.index(i)].y
				nowx[i] = boxes[mark.index(i)].x

			else:
				if(originy[i]!=0):

					flag[i] += 1
					if(flag[i]>=10):
						if(flag2[i]>=6):

							if(area(originx[i],originy[i])==1 and area(nowx[i],nowy[i])==2):
								counter += 1
							elif(area(originx[i],originy[i])==2 and area(nowx[i],nowy[i])==1):
								counter -= 1
							else:
								if(abs(originy[i]-nowy[i]) > 0.3):

									if(originy[i]-nowy[i] <= 0 ):
										counter += 1
									else:
										counter -= 1

						nowy[i]=0
						nowx[i] = 0
						originx[i]=0
						originy[i]=0	

						flag[i] = 0
						flag2[i] = 0
		print(mark)
		print(flag)
		print(flag2)
		print(originx)
		print(originy)
		print(nowx)
		print(nowy)		
		print(counter)

	for i in range(k):
		if(originy[i]!=0):

			if(flag2[i]>=6):

				if(area(originx[i],originy[i])==1 and area(nowx[i],nowy[i])==2):
					counter += 1
				elif(area(originx[i],originy[i])==2 and area(nowx[i],nowy[i])==1):
					counter -= 1
				else:
					if(abs(originy[i]-nowy[i]) > 0.3):

						if(originy[i]-nowy[i] <= 0 ):
							counter += 1
						else:
							counter -= 1
	print(counter)

if __name__ == '__main__':
	args = argparser.parse_args()
	_main_(args)
