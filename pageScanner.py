import cv2
import numpy as np

imgWidth = 640
imgHeight = 480
kernel = np.ones((5, 5))
cap = cv2.VideoCapture(0)
cap.set(3, imgWidth)
cap.set(4, imgHeight)

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def Preprocessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgblur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgblur, 100, 100)
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThershold = cv2.erode(imgDial, kernel, iterations=1)
    return imgThershold


def getContours(img):
    maxArea = 0
    biggest = np.array([])
    contours, hiearcy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # cv2.drawContours(imgCnt, cnt, -1, (0, 255, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgCnt, biggest, -1, (0, 255, 0), 10)
    return biggest


def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [imgWidth, 0], [0, imgHeight], [imgWidth, imgHeight]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (imgWidth, imgHeight))
    imgCropped = imgOutput[20:imgOutput.shape[0] - 20, 20:imgOutput.shape[1] - 20]
    imgCropped = cv2.resize(imgCropped, (imgWidth, imgHeight))
    return imgCropped


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myNewPoints = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    myNewPoints[0] = myPoints[np.argmin(add)]
    myNewPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myNewPoints[1] = myPoints[np.argmin(diff)]
    myNewPoints[2] = myPoints[np.argmax(diff)]
    return myNewPoints


while True:
    success, img = cap.read()
    img = cv2.resize(img, (imgWidth, imgHeight))
    imgCnt = img.copy()
    imgThres = Preprocessing(img)
    biggest = getContours(imgThres)
    if biggest.size != 0:
        imgWarp = getWarp(img, biggest)
        imgArray = ([img, imgCnt,
                     imgThres, imgWarp])
    else:
        imgArray = ([img, img,
                     imgThres, img])
    imgStacked = stackImages(0.6, imgArray)
    cv2.imshow("page", imgStacked)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


