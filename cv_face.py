import cv2 as cv

face_capture=cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_frontalface_default.xml')
cp=cv.VideoCapture(0)
while True:
    r,frame=cp.read()
    frame=cv.flip(frame,1)
    color=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    fdetect=face_capture.detectMultiScale(
        color,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv.CASCADE_SCALE_IMAGE
        )
    for (x,y,w,h) in fdetect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        

    cv.imshow('live ',frame)
    if cv.waitKey(1)&0xff==ord('q'):
        break
cp.release()    
cv.destroyAllWindows()
            
