import cv2
import numpy as np
# print("Package Imported")
# img=cv2.imread("Resources/lena.jpg")
# cv2.imshow("Output",img)
# cv2.waitKey(0)
# cap=cv2.VideoCapture(0)
# cap.set(3,640)
# cap.set(4,480)
# cap.set(10,100)
# while True:
#     success,img=cap.read()
#     cv2.imshow("Video",img)
#     if cv2.waitKey(1) & 0xFF==ord('q'):
#         break

# kernel=np.ones((5,5),np.uint8)
# imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# imgBlur=cv2.GaussianBlur(imgGray,(7,7),0)
# imgCanny=cv2.Canny(img,150,200)
# imgDialation=cv2.dilate(imgCanny,kernel,iterations=1)
# imgEroded=cv2.erode(imgDialation,kernel,iterations=1)
# cv2.imshow("Gray Image",imgGray)
# cv2.imshow("Blur Image",imgBlur)
# cv2.imshow("Canny Image",imgCanny)
# cv2.imshow("Dialation Image",imgDialation)
# cv2.imshow("Eroded Image",imgEroded)
# cv2.waitKey(0)
# img=cv2.imread("Resources/lena.jpg")
# print(img.shape)
# imgResize=cv2.resize(img,(300,200))
# print(imgResize.shape)
# imgCropped=img[0:90,100:250]
# cv2.imshow("Image",img)
# cv2.imshow("ImageResize",imgResize)
# cv2.imshow("ImageCropped",imgCropped)
# cv2.waitKey(0)

# img=np.zeros((512,512,3),np.uint8)
# # print(img)
# # img[:]=255,0,0
# cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,255,0),3)
# cv2.rectangle(img,(0,0),(250,350),(0,0,255),2)
# cv2.circle(img,(400,50),30,(255,255,0),5)
# cv2.putText(img,"OPENCV",(300,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),1)



# img=cv2.imread("Resources/lena.jpg")
# width,height=250,350
# pts1=np.float32([[111,219],[287,188],[154,482],[352,440]])
# pts2=np.float32([[0,0],[width,0],[0,height],[width,height]])
# matrix=cv2.getPerspectiveTransform(pts1,pts2)
# imgOutput=cv2.warpPerspective(img,matrix,(width,height))
#
#
# cv2.imshow("Image",img)
# cv2.imshow(("Output",imgOutput))
#
# cv2.waitKey(0)






# def stackImages(scale,imgArray):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range ( 0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = np.zeros((height, width, 3), np.uint8)
#         hor = [imageBlank]*rows
#         hor_con = [imageBlank]*rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imgArray[x])
#         ver = np.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
#             if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor= np.hstack(imgArray)
#         ver = hor
#     return ver
# img=cv2.imread('Resources/lena.jpg')
# imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# imgStack=stackImages(0.5,([img,imgGray,img],[img,img,img]))
# # hor=np.hstack((img,img))
# # ver=np.vstack((img,img))
# # cv2.imshow("Horizontal",hor)
# # cv2.imshow("Vertical",ver)
# cv2.imshow("ImageStack",imgStack)
# cv2.waitKey(0)



# def empty(a):
#     pass
#
#
#
# def stackImages(scale,imgArray):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     if rowsAvailable:
#         for x in range ( 0, rows):
#             for y in range(0, cols):
#                 if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
#                 else:
#                     imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
#                 if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         imageBlank = np.zeros((height, width, 3), np.uint8)
#         hor = [imageBlank]*rows
#         hor_con = [imageBlank]*rows
#         for x in range(0, rows):
#             hor[x] = np.hstack(imgArray[x])
#         ver = np.vstack(hor)
#     else:
#         for x in range(0, rows):
#             if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
#                 imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
#             else:
#                 imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
#             if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         hor= np.hstack(imgArray)
#         ver = hor
#     return ver
#
# path='Resources/lena.jpg'
# cv2.namedWindow("TrackBars")
# cv2.resizeWindow("TrackBars",640,240)
# cv2.createTrackbar("Hue Min","TrackBars",3,179,empty)
# cv2.createTrackbar("Hue Max","TrackBars",76,179,empty)
# cv2.createTrackbar("Sat Min","TrackBars",34,255,empty)
# cv2.createTrackbar("Sat Max","TrackBars",109,255,empty)
# cv2.createTrackbar("Val Min","TrackBars",176,255,empty)
# cv2.createTrackbar("Val Max","TrackBars",255,255,empty)
# while True:
#     img = cv2.imread(path)
#
#     imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
#     h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
#     s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
#     s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
#     v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
#     v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
#     print(h_min,h_max,s_min,s_max,v_min,v_max)
#     lower=np.array([h_min,s_min,v_min])
#     upper=np.array([h_max,s_max,v_max])
#
#     mask=cv2.inRange(imgHSV,lower,upper)
#     imgResult=cv2.bitwise_and(img,img,mask=mask)
#     # cv2.imshow("Original", img)
#     # cv2.imshow("HSV", imgHSV)
#     # cv2.imshow("Mask", mask)
#     # cv2.imshow("Result", imgResult)
#     imgStack=stackImages(0.6,([img,imgHSV],[mask,imgResult]))
#     cv2.imshow("StackedImages", imgStack)
#     cv2.waitKey(1)










faceCascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
img=cv2.imread('Resources/lena.jpg')
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces=faceCascade.detectMultiScale(imgGray,1.1,4)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

cv2.imshow("Result",img)
cv2.waitKey(0)














