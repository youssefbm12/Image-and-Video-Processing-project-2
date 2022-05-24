import cv2 as cv
import numpy as np
# from cvlib.object_detection import draw_bbox
import matplotlib.pyplot as plt
# Read image.
path = r'C:\Users\youss\OneDrive\Documents\IVP\project 2\oranges.jpg'
img = cv.imread(path)
 
def rescaleImage(Image, scaler_factor):
    height = int(Image.shape[0] * scaler_factor)
    width = int(Image.shape[1] * scaler_factor)
    dimension = (width,height)

    return cv.resize(Image, dimension, interpolation=cv.INTER_AREA)
img = rescaleImage(img,0.5)

copy = img

# # Convert to grayscale.
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  
# # Blur using 3 * 3 kernel.
# gray_blurred = cv.blur(gray, (3, 3))
# grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  
# (thresh, blackAndWhiteImage) = cv.threshold(grayImage, 127, 255, cv.THRESH_BINARY)
 
# # cv.imshow('Black white image', blackAndWhiteImage)
# # Apply Hough transform on the blurred image.
# detected_circles = cv.HoughCircles(grayImage, cv.HOUGH_GRADIENT, 1, 20, param1 = 100, param2 = 30, minRadius = 4, maxRadius = 50)

# count = 0
# # Draw circles that are detected.
# if detected_circles is not None:
  
#     # Convert the circle parameters a, b and r to integers.
#     detected_circles = np.uint16(np.around(detected_circles))
  
#     for pt in detected_circles[0, :]:
#         count = count +1
#         a, b, r = pt[0], pt[1], pt[2]
  
#         # Draw the circumference of the circle.
#         cv.circle(img, (a, b), r, (255, 255, 255), -1)
  
#         # Draw a small circle (of radius 1) to show the center.
        
# print(count)
# cv.imshow("Detected Circle", img)
# # cv.imshow("Black & White", blackAndWhiteImage)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

# apply gaussian blur
blur = cv.GaussianBlur(gray, (9,9), 0)


# threshold
thresh = cv.threshold(blur,128,255,cv.THRESH_BINARY)[1]


# apply close and open morphology to smooth
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9,9))
thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

# draw contours and get centroids
circles = img.copy()
contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
for cntr in contours:
    cv.drawContours(circles, [cntr], -1, (255,255,255), -1)
    M = cv.moments(cntr)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    x = round(cx)
    y = round(cy)
    
    circles[y-2:y+3,x-2:x+3] = (0,255,0)
    cv.circle(circles, (x, y), 5, (255, 255, 255), -1)
            

cv.imshow("circles", circles)

# Printing the number of oranges in the picture
print("The number of Oranges in the Image is:",len(contours))

# Question 2 
path = r'C:\Users\youss\OneDrive\Documents\IVP\project 2\jar.jpg'
img2 = cv.imread(path)
img2 = rescaleImage(img2,0.1)
copy = img2
gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

# apply gaussian blur



# threshold
gray = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)

# apply gaussian blur
blur = cv.GaussianBlur(gray, (9,9), 0)


# threshold
thresh = cv.threshold(blur,228,125,cv.THRESH_BINARY)[1]


# apply close and open morphology to smooth
lower = np.array([0, 164, 0])
upper = np.array([179, 255, 255])
mask = cv.inRange(img2,lower,upper)
            

cv.imshow("circles 2", circles)
cv.imshow("Thresh",mask)
cv.imshow("Original Image",img2)
cv.waitKey(0)
cv.destroyAllWindows()