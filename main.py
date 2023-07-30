import os

from cvzone.ClassificationModule import Classifier
import cvzone
import cv2

cap = cv2.VideoCapture(0)
classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')
imgArrow=cv2.imread("Resources/arrow.png",cv2.IMREAD_UNCHANGED)
classIDBin=0

imgWasteList=[]
pathFolderWaste="Resources/Waste"
pathList=os.listdir(pathFolderWaste)
#print(pathList)
for path in pathList:
    imgWasteList.append(cv2.imread(os.path.join(pathFolderWaste,path),cv2.IMREAD_UNCHANGED))

imgBinsList=[]
pathFolderBins="Resources/Bins"
pathList=os.listdir(pathFolderBins)
#print(pathList)
for path in pathList:
    imgBinsList.append(cv2.imread(os.path.join(pathFolderBins,path),cv2.IMREAD_UNCHANGED))

classDic={0:None,1:0,
          2:0,
          3:3 ,
          4:3,
          5:1,
          6:1,
          7:2,
          8:2}


while True:
    _, img = cap.read()
    imgResize=cv2.resize(img,(454,340))
    imgBackground = cv2.imread('Resources/background.png')
    predection = classifier.getPrediction(img)
    print(predection)
    classID=predection[1]

    if classID!=0:
        imgBackground = cvzone.overlayPNG(imgBackground, imgWasteList[classID-1], (989, 127))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (1050, 320))
        classIDBin=classDic[classID]
    imgBackground = cvzone.overlayPNG(imgBackground, imgBinsList[classIDBin], (978, 374))
       # predection=None



    imgBackground[148:148+340,159:159+454]=imgResize
    #cv2.imshow("Image", img)
    cv2.imshow("Ouput", imgBackground)
    cv2.waitKey(1)