import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from random import randrange
import matplotlib.pyplot as plt
import math
from PIL import Image as im

class FaceDetector(object):
    """
    Face detector class
    """

    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def _draw(self, frame, boxes, probs, landmarks, photo):
        """
        Draw landmarks and boxes for each face detected
        """
        i = 0
        img_counter = 0
        try:
            for box, prob, ld in zip(boxes, probs, landmarks):
                #Draw rectangle on frame
                
                if cv2.waitKey(1) % 256 == 32 and len(boxes) == 2:
                     #Space Pressed:
                     newboxes = []
                     newprobs = []
                     for accuracy in range(0,len(probs)):
                          if probs[accuracy] >= .90:
                               newboxes.append(boxes[accuracy])
                               newprobs.append(probs[accuracy])
            
                     boxes = newboxes
                     probs = newprobs
                     print("Number of boxes detected "+ str(len(boxes))+"!")
                     for num in range(0,len(boxes)):
                          photo_xdif = abs(boxes[num][0] - boxes[num][2])
                          photo_ydif = abs(boxes[num][1] - boxes[num][3])
                          print("Detection "+str(num+1)+" Dimensions are: "+ str((photo_xdif, photo_ydif))+"~") 

                     #Start Image Capture
                     img_name = "opencv_frame_{}.png".format(img_counter)
                     cv2.imwrite('./Images/'+str(img_name),frame)
                     print("{} written!".format(img_name))
                     img_counter += 1
                     photo = 1 
                     breakout = 1
                   
               # cv2.putText(frame, str(len(str(i))) + str(round(prob,4)), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
               # cv2.putText(frame, str(i), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

                # Draw landmarks

               #cv2.circle(frame,(box[0],box[1]) , 5, (255,0,0), -1)
                #cv2.circle(frame,(box[0],box[1]) , 5, (255,0,0), -1)

                cv2.circle(frame, tuple(ld[0]), 5, (255,0,0), -1)
                #print(tuple(ld[0]))

                cv2.circle(frame, tuple(ld[1]), 5, (255,0,0), -1)
                #cv2.circle(frame,(ld[1][0],ld[1][1]),(255,0,0), -1)


                cv2.circle(frame, tuple(ld[2]), 5, (255,0,0), -1)
                cv2.circle(frame, tuple(ld[3]), 5, (255,0,0), -1)
                #cv2.circle(frame,(ld[3][0],ld[3][1]),(255,0,0), -1)


                cv2.circle(frame, tuple(ld[4]), 5, (255,0,0), -1)
                #cv2.circle(frame,(ld[4][0],ld[4][1]),(255,0,0), -1)

                
        except:
            pass
                
        return photo, breakout, boxes, probs

    def run(self):
        """
            Run the FaceDetector and draw landmarks and boxes around detected faces
        """
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        photo = 0
        breakout = 0
        while True:
            ret, frame = cap.read()
            
            try:
                # detect face box, probability and landmarks
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)
               
                # draw on frame
                photo, breakout, boxes, probs = self._draw(frame, boxes, probs, landmarks, photo)

            except:
                pass

            # Show the frame
            cv2.imshow('Detector', frame)
            cv2.moveWindow('Detector',0,0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if breakout == 1:
                break

       
        cv2.destroyAllWindows()
        cap.release()
        cv2.waitKey(0)
        return boxes

'''
#Display the image
img = cv2.imread('opencv_frame_0.png')
    #print(img.shape)
    #print("The image size is: "+ str(img.shape)+"~")
muv = 0
cropped_faces = []
xs = []
ys = []
ws = []
hs = []

for person in range(0,len(boxes)):
    y = math.floor(boxes[person][1])
    x = math.floor(boxes[person][0])
    h = math.ceil(abs(boxes[person][1] - boxes[person][3]))
    w = math.ceil(abs(boxes[person][0] - boxes[person][2]))
    facecrop = img[y:y+h,x:x+w]
    cropped_faces.append(facecrop)
    # --- Capture Face Objects -----
    #facecrop2 = np.asarray(facecrop)
    #data = im.fromarray(facecrop2)
    #img_name = "object_{}.png".format(person)
    data.save(img_name)
    #cv2.imwrite(img_name,facecrop)
    print("{} saved!".format(img_name))
    # ------------------------------
    xs.append(x)
    ys.append(y)
    ws.append(w)
    hs.append(h)
#     current_name = "Person: "+str(person+1)
#     cv2.namedWindow(current_name)
#     cv2.imshow(current_name,facecrop)
#     cv2.moveWindow(current_name,muv,0)
#     muv = muv + 50

merged_image = img.copy
cropped_faces2 = []
cv2.namedWindow("Merged Image")

#Resize Images to be eachothers size


cv2.imshow("Merged Image",merged_image)
    #cv2.moveWindow(current_name,0,0)
cv2.waitKey(0)
#cv2.release()
#cv2.destroyAllWindows()
#exit()
'''

