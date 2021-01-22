import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

image = cv2.imread('angle-example.jpg',cv2.IMREAD_GRAYSCALE)



def perspect_transform(img, src, dst):
    # Get transform matrix using cv2.getPerspectivTransform()
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp image using cv2.warpPerspective()
    # keep same size as input image
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    # Return the result
    return warped
'''
source = np.float32([[90,590], [700,590],[450,320],[360,320] ])
destination = np.float32([[200,590], [600,590],[600,0],[200,0] ])'''

dst_size = 5 
# Set a bottom offset to account for the fact that the bottom of the image 
# is not the position of the rover but a bit in front of it
# this is just a rough guess, feel free to change it!
bottom_offset = 6
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])






warped = perspect_transform(image, source, destination)
ret,thresh1 = cv2.threshold(warped,190,255,cv2.THRESH_BINARY)
print thresh1.nonzero()
print thresh1.shape

ypos, xpos = thresh1.nonzero()
x_pixel = np.absolute(ypos - thresh1.shape[0]).astype(np.float)
y_pixel = -(xpos - thresh1.shape[0]).astype(np.float)


y=np.mean(y_pixel)
x=np.mean(x_pixel)
alpha=math.atan2(y,x)
print(alpha*180/math.pi)
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

ax4.plot(x_pixel,y_pixel)
ax4.set_title('map', fontsize=40)
ax4.set_xlim(0, 160)
ax4.set_ylim(-160, 160)
#ax4.plot(0,0,x,y, 'go--', linewidth=2, markersize=12)
#ax4.imshow(ypos, xpos, cmap='gray')
arrow_length = 10
x_arrow = arrow_length 
y_arrow = -100
ax4.arrow(0, 0, x, y, color='red', zorder=2, head_width=10, width=2)
ax4.arrow(0, 0, x, 0, color='red', zorder=2, head_width=10, width=2)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show() # Uncomment if running on your local machine
