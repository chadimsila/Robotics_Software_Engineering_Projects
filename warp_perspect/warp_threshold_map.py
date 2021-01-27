import matplotlib.pyplot as plt 
import numpy as np
import cv2
import math
from matplotlib.patches import Arc
import glob
import pandas as pd



# Read in the sample image
image1 = cv2.imread('angle-example.jpg')
image = cv2.imread('angle-example.jpg')
path='IMG/*'
img_list=glob.glob(path)
idx=np.random.randint(0,len(img_list)-1)
#image = cv2.imread(image,cv2.IMREAD_GRAYSCALE)

# Rover yaw values will come as floats from 0 to 360
# Generate a random value in this range
rover_yaw = np.random.random(1)*360

# Generate a random rover position in world coords
# Position values will range from 20 to 180 to 
# avoid the edges in a 200 x 200 pixel world
rover_xpos = np.random.random(1)*160 + 20
 #rover_ypos=190
rover_ypos = np.random.random(1)*160 + 20
 #rover_ypos=190

# Note: Since we've chosen random numbers for yaw and position, 
# multiple run of the code will result in different outputs each time.


def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    return warped

'''
def color_thresh(warped,thresh,maxval):
    ret,thresh1 = cv2.threshold(warped,thresh,maxval,cv2.THRESH_BINARY)
    return thresh1
'''


def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select


def rover_cord(binaryimage):
    ypos, xpos = binaryimage.nonzero()
    x_pixel=abs(ypos-binaryimage.shape[0]).astype(np.float)
    y_pixel=(-xpos+binaryimage.shape[0]).astype(np.float)
    return x_pixel,y_pixel


# Define a function to apply a rotation to pixel positions
def rotate_pix(xpix, ypix, yaw):
    # TODO:
    # Convert yaw to radians
    yaw_rad=yaw*math.pi/180
    # Apply a rotation
    x_trans=math.cos(yaw_rad)*xpix-math.sin(yaw_rad)*ypix
    y_trans=math.sin(yaw_rad)*xpix+math.cos(yaw_rad)*ypix
    # Return the result  
    return x_trans,y_trans

# Define a function to perform a translation
def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # TODO:
    # Apply a scaling and a translation
    x_rot=xpix_rot/scale+xpos
    y_rot=ypix_rot/scale+ypos
    # Return the result  
    return x_rot,y_rot

def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Clip to world_size
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

def to_polar_coords(xpix, ypix):
    distance=math.sqrt(xpix**2+ypix**2)
    angle=math.atan2(ypix,xpix)
    return distance, angle

source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[ 155 , 154],
                         [ 165 , 154],
                         [ 165 , 144],
                         [ 155 , 144]]
)



warped = perspect_transform(image, source, destination)
thresh1 = color_thresh(warped,rgb_thresh=(160, 160, 160))
x_pixel,y_pixel=rover_cord(thresh1)
y=np.mean(y_pixel)
x=np.mean(x_pixel)
alpha=math.atan2(y,x)


#Display the direction angle of the robot
arrow_length ,angle = to_polar_coords(x,y)
#steering =np.clip(angle,-math.pi/4,math.pi/4)
angle_degree=angle*180/math.pi


world_map=np.zeros([200,200])


xmap,ymap=pix_to_world(x_pixel,y_pixel, rover_xpos, rover_ypos, rover_yaw, world_map.shape[0], 25)




# Draw Source and destination points on images (in blue) before plotting
cv2.polylines(image, np.int32([source]), True, (0, 0, 255), 3)
cv2.polylines(warped, np.int32([destination]), True, (0, 0, 255), 3)
# Display the original image and binary               
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 6), sharey=False)
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(warped, cmap='gray')
ax2.set_title('Result', fontsize=40)

ax3.imshow(thresh1, cmap='gray')
ax3.set_title('map', fontsize=40)

#Display rover_cordinates
ax4.plot(x_pixel,y_pixel,'.')
ax4.set_title('map', fontsize=40)
ax4.set_xlim(0, 160)
ax4.set_ylim(-160, 160)

#Display the direction angle of the robot
arrow_length ,angle = to_polar_coords(x,y)
steering =np.clip(angle,-math.pi/4,math.pi/4)
angle_degree=angle*180/math.pi

ax4.arrow(0, 0, x, y, color='red', zorder=2, head_width=10, width=2)
ax4.arrow(0, 0, x, 0, color='red', zorder=2, head_width=10, width=2)
plt.show()

 

''' UNCOMMENT TO DISPLAY MAP TO WORLD 
fig = plt.figure(figsize=(24, 6))
plt.xlim(0, 200)
plt.ylim(0, 200)
xmap,ymap=pix_to_world(x_pixel,y_pixel, rover_xpos, rover_ypos, rover_yaw, world_map.shape[0], 10)
plt.imshow(world_map,cmap='gray')
plt.plot(xmap,ymap,'.',color='w')
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.5)
plt.show() # Uncomment if running on your local machine '''
