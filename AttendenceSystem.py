import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
# importing images from folder
path ='pics'
images = []
personNames = []
imageList=os.listdir(path)
print(imageList)

# this loop iterate through all images
for img in imageList:
    curentImg=cv2.imread(f'{path}/{img}')
    images.append(curentImg)
    personNames.append(os.path.splitext(img)[0])

def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encodeImg=face_recognition.face_encodings(img)[0]
        encodeList.append(encodeImg)
    return encodeList

def AttendenceList(name):
    with open('Attendance.csv','r+') as f:
        dataList=f.readlines()
        nameList=[]
        for line in dataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            time=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{time}')

encodedListKnown=findEncodings(images)
print("Encoding Complete")

cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    
    currentFaces = face_recognition.face_locations(imgS)
    currentEncodedFrame = face_recognition.face_encodings(imgS, currentFaces)
    name=""
    for i, encodeFace in enumerate(currentEncodedFrame):
        matches = face_recognition.compare_faces(encodedListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodedListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = personNames[matchIndex].upper()
            #print(name)
    for i, faceLoc in enumerate(currentFaces):
        y1,x2,y2,x1=faceLoc
        y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        AttendenceList(name)
        
    
    cv2.imshow('webcam', img)
    cv2.waitKey(1)
