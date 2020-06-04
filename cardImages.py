import cv2

img = cv2.imread('cards/diamond-3.jpg')

cv2.imshow("Card",img)
cv2.waitKey(0)
cv2.destroyAllWindows()