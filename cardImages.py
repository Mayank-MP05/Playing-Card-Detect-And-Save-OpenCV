import cv2

def readImg():
    img = cv2.imread('cards/diamond-3.jpg')
    return img

img = readImg() 
cv2.imshow("Card",img)
cv2.waitKey(0)
cv2.destroyAllWindows()