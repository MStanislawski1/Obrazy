
import cv2
import os
 
path = "img"
orb = cv2.ORB_create(nfeatures=1000)
 
images = []
classNames = []
myList = os.listdir(path)
 
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
 
def findDes(images):
    desList = []
    for img in images:
        kp,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList
 
def findID(img , desList, thres = 18):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    finalVal = -1
    matchList = []
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
 
            good = []
            for m,n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass 
 
    if len(matchList) != 0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
    return finalVal
 
desList = findDes(images)
 
cap = cv2.VideoCapture(0)
 
while True:
    succes, img2 = cap.read()
    imgOrginal = img2.copy()
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
 
    id = findID(img2, desList)
    if id != -1:
        cv2.putText(imgOrginal, classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
 
    cv2.imshow("", imgOrginal)
    cv2.waitKey(1)