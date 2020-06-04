import cv2
import numpy as np

img = cv2.imread('cards/diamond-3.jpg')


gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray,150,250)
kernal = np.ones((5,5))
dailated = cv2.dilate(canny,kernal,iterations=1)

cv2.imshow("Canny Card",canny)
cv2.imshow("Canny Card",dailated)

# _ , countours = cv2.

cv2.waitKey(0)
cv2.destroyAllWindows()