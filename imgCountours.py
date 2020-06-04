import cv2
import numpy as np
import imutils

def detect(c):
    shape = "unidentified"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)
    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"
    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    # if the shape is a pentagon, it will have 5 vertices
    elif len(approx) == 5:
        shape = "pentagon"
    # otherwise, we assume the shape is a circle
    else:
        shape = "circle"
    # return the name of the shape
    return shape

img = cv2.imread('cards/diamond-3.jpg')

# resized = imutils.resize(img, width=300)
ratio = 1.0

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("THresh",thresh)
# canny = cv2.Canny(gray,150,250)
# kernal = np.ones((5,5))
# dailated = cv2.dilate(canny,kernal,iterations=1)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
	shape = detect(c)
	x,y,w,h = cv2.boundingRect(c)
	area = cv2.contourArea(c)
	if area > 5000:
		cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
		cv2.putText(img, shape, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2)
	# show the output image
	cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
