import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from matplotlib.patches import Arc
import glob
import pandas as pd
from celluloid import Camera


df=pd.read_csv('robot_log.csv',delimiter=',',decimal='.')
#Change path of the original dataset
df['Path']=df['Path'].str.slice(start=55)
path_to_list=df['Path'].tolist()


def perspect_transform(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    return warped

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

'''

def rover_cord(binaryimage):
    ypos, xpos = binaryimage.nonzero()
    x_pixel=abs(ypos-binaryimage.shape[0]).astype(np.float)
    y_pixel=(-xpos+binaryimage.shape[0]).astype(np.float)
    return x_pixel,y_pixel


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


# Display map to world 
#fig = plt.figure(figsize=(24, 6))
fig,(ax1,ax2) = plt.subplots(1, 2, figsize=(24, 6))
camera = Camera(fig)
plt.title('Map to World')
plt.xlim(0, 200)
plt.ylim(0, 200)

world_map=np.zeros([200,200])

#map=cv2.imread('map_bw.png',cv2.IMREAD_GRAYSCALE)
#mapx,mapy=map.nonzero()


for i in range(len(path_to_list)) :
    path=df['Path'][i]
    image = cv2.imread(path)
    #ax1.imshow(image)
    image=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    warped = perspect_transform(image, source, destination)
    thresh1 = color_thresh(warped,190,255)
    x_pixel,y_pixel=rover_cord(thresh1)
    xmap,ymap=pix_to_world(x_pixel,y_pixel, df["X_Position"][i], df["Y_Position"][i], df["Yaw"][i], world_map.shape[0], 25)
    #ax2.plot(mapy,mapx,'.',color='c',alpha=0.5)
    ax2.plot(xmap,ymap,'.',color='r',alpha=0.5)
    #camera.snap()
    #plt.pause(1) # Uncomment if running on your local machine'''
    print (i)


##uncomment this if u want gif record
#animation = camera.animate()
#animation.save('celluloid_minimal.gif', writer = 'imagemagick')
#video record
anim = camera.animate(blit=False, interval=10)
anim.save('map_to_world.mp4')
