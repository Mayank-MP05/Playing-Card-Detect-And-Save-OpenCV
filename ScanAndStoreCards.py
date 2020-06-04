import cv2
import numpy as np

widthImg, heightImg = 480, 640


def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (0, 0, 255), 10)

    # x,y,w,h = cv2.boundingRect(biggest)
    # cv2.rectangle(imgContour,(x,y),(x+w,y+h),(255,0,0),3)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    #print("add", add)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    # print("NewPoints",myPointsNew)
    return myPointsNew


def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32(
        [[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    imgCropped = imgOutput[0:imgOutput.shape[0], 0:imgOutput.shape[1]]
    imgCropped = cv2.resize(imgCropped, (widthImg, heightImg))

    return imgCropped


url = 'http://192.168.43.1:8000/shot.jpg'

s_count = 0

while True:
    vid = cv2.VideoCapture(url)
    _, img = vid.read()
    img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5, 5))
    dialated = cv2.dilate(imgCanny, kernel, iterations=2)
    threshed = cv2.erode(dialated, kernel, iterations=1)

    imgContour = img.copy()
    biggest = getContours(threshed)

    if biggest.size != 0:
        imgWarped = getWarp(img, biggest)
        # imageArray = ([img,imgThres],
        #           [imgContour,imgWarped])
        imageArray = ([imgContour, imgWarped])
        cv2.imshow("Card Only", imgWarped)
        if cv2.waitKey(1) & 0xFF == ord('d'):
            s_count += 1
            cv2.imwrite(f"saved/Card-Saved-{s_count}.jpg",imgWarped)
    else:
        # imageArray = ([img, imgThres],
        #               [img, img])
        imageArray = ([imgContour, img])

    cv2.imshow("Card Dot Detect", imgContour)

    v1 = np.vstack([imgGray, imgGray])
    v2 = np.vstack([imgBlur, imgCanny])
    v3 = np.vstack([dialated, threshed])

    cv2.imshow("Stacked - Gray , Blur , Thresshed", np.hstack([v1, v2, v3]))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
