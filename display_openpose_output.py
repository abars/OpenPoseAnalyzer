# ----------------------------------------------
# Display OpenPose Intermediate Data
# ----------------------------------------------

import cv2
import sys
import numpy as np
import os
import caffe
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------
# Input Data
# ----------------------------------------------

IMAGE_PATH="images/dress.jpg"
CAFFE_MODEL='pose_iter_440000.caffemodel'
PROTOTXT='pose_deploy.prototxt'

IMAGE_WIDTH=368
IMAGE_HEIGHT=368

if not os.path.exists(IMAGE_PATH):
	print(IMAGE_PATH+" not found")
	sys.exit(1)

print IMAGE_PATH
input_img = cv2.imread(IMAGE_PATH)

img = cv2.resize(input_img, (IMAGE_HEIGHT, IMAGE_WIDTH))

img = img[...,::-1]  #BGR 2 RGB

data = np.array(img, dtype=np.float32)
data.shape = (1,) + data.shape

data = data / 255.0

net  = caffe.Net(PROTOTXT, CAFFE_MODEL, caffe.TEST)
data = data.transpose((0, 3, 1, 2))
out = net.forward_all(data = data)

paf = out[net.outputs[0]]
confidence = out[net.outputs[1]]

print("PAF SHAPE : "+str(paf.shape))
print("CONFIDENCE SHAPE : "+str(confidence.shape))

# ----------------------------------------------
# Display output
# ----------------------------------------------

points = []
threshold = 0.1

for i in range(confidence.shape[1]):
	probMap = confidence[0, i, :, :]
	minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

	x = (input_img.shape[1] * point[0]) / confidence.shape[3]
	y = (input_img.shape[0] * point[1]) / confidence.shape[2]
 
	if prob > threshold : 
		cv2.circle(input_img, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
		cv2.putText(input_img, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, lineType=cv2.LINE_AA)
		cv2.putText(input_img, ""+str(prob), (int(x), int(y+16)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)

		points.append((int(x), int(y)))
	else :
		points.append(None)
 
cv2.imshow("Keypoints",input_img)

# ----------------------------------------------
# Display intermediate data
# ----------------------------------------------

def plot_images(title, images, tile_shape):
	assert images.shape[0] <= (tile_shape[0]* tile_shape[1])
	from mpl_toolkits.axes_grid1 import ImageGrid
	fig = plt.figure()
	plt.title(title)
	grid = ImageGrid(fig, 111,  nrows_ncols = tile_shape )
	for i in range(images.shape[1]):
		grd = grid[i]
		grd.imshow(images[0,i])

channels=38
cols=8

plot_images("paf",paf,tile_shape=((channels+cols-1)/cols,cols))
plot_images("confidence",confidence,tile_shape=((channels+cols-1)/cols,cols))
#plot_images("image",data,tile_shape=(1,4))

plt.show()

#cv2.waitKey(0)
cv2.destroyAllWindows()
