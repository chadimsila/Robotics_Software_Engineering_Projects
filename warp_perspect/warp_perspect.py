import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

image = cv2.imread('angle-example.jpg',cv2.IMREAD_GRAYSCALE)


# Read in the sample image


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
source = np.float32([[90,590], [700,590],[450,320],[360,320] ])
destination = np.float32([[200,590], [600,590],[600,0],[200,0] ])'''
def color_thresh(warped,thresh,maxval):
    ret,thresh1 = cv2.threshold(warped,thresh,maxval,cv2.THRESH_BINARY)
    return thresh1

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


source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[ 155 , 154],
                         [ 165 , 154],
                         [ 165 , 144],
                         [ 155 , 144]]
)


warped = perspect_transform(image, source, destination)
thresh1 = color_thresh(warped,190,255)
x_pixel,y_pixel=rover_cord(thresh1)

y=np.mean(y_pixel)
x=np.mean(x_pixel)
alpha=math.atan2(y,x)
print(alpha*180/math.pi)

world_map=np.zeros([200,200])








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
ax4.plot(x_pixel,y_pixel)
ax4.set_title('map', fontsize=40)
ax4.set_xlim(0, 160)
ax4.set_ylim(-160, 160)

arrow_length = 10
x_arrow = arrow_length 
y_arrow = -100
ax4.arrow(0, 0, x, y, color='red', zorder=2, head_width=10, width=2)
ax4.arrow(0, 0, x, 0, color='red', zorder=2, head_width=10, width=2)

# Display map to world 
fig = plt.figure(figsize=(24, 6))
plt.title('Map to World')
plt.xlim(0, 200)
plt.ylim(0, 200)
xmap,ymap=pix_to_world(x_pixel,y_pixel, rover_xpos, rover_ypos, rover_yaw, world_map.shape[0], 10)
plt.imshow(world_map,cmap='gray')
plt.plot(xmap,ymap,'.',color='w')




plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show() # Uncomment if running on your local machine'''
