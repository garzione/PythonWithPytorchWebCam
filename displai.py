from torchface import FaceDetector
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from random import randrange
import matplotlib.pyplot as plt
import math
from scipy import stats

def MergeIt(boxes):
     #Display the image
     img = cv2.imread('./Images/opencv_frame_0.png')
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
         #img_name = "object_{}.png".format(person)
         #cv2.imwrite('./Images/'+str(img_name),facecrop2)
         #print("{} saved!".format(img_name))
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

     merged_image = img.copy()
     faceless_image = img.copy()
     cropped_faces2 = []
     #cv2.namedWindow("Merged Image")

     #Resize Images to be eachothers size
     for face in range(0,len(boxes)):
          if face == (len(boxes)-1):
               #Y's & X's are Coordinates, H's & W's are sizes!
               width = ws[0]
               height = hs[0]
               dim = (width, height) 
               cropped_faces[face] = cv2.resize(cropped_faces[face],dim)
               
               #Object Image -> cropped_faces[face]
               #Object Mask -> No idea
               #Background Image -> merged_image
               #bg_ul -> ys[0],xs[0]

               merged_image[ys[0]:ys[0]+hs[0],xs[0]:xs[0]+ws[0]] = cropped_faces[face]
               #------
               avg = np.average(cropped_faces[face], axis=None)
               avg = int(math.floor(avg))
               #mostpixel, __ = stats.mode(cropped_faces[face], axis=None)
               print("Average pixel is: "+ str(avg))
               temp = cropped_faces[face].copy()
               temp.fill(avg)
               faceless_image[ys[0]:ys[0]+hs[0],xs[0]:xs[0]+ws[0]] = temp
               #-------
               facecrop2 = np.asarray(cropped_faces[face])
               img_name = "object_{}.png".format(face)
               cv2.imwrite('./Images/'+str(img_name),facecrop2)
               print("{} saved!".format(img_name))
          
          else:
               width = ws[face+1]
               height = hs[face+1]
               dim = (width, height) 
               cropped_faces[face] = cv2.resize(cropped_faces[face],dim)
               merged_image[ys[face+1]:ys[face+1]+hs[face+1],xs[face+1]:xs[face+1]+ws[face+1]] = cropped_faces[face]
               #------
               avg = np.average(cropped_faces[face], axis=None)
               avg = int(math.floor(avg))
              #mostpixel, __ = stats.mode(cropped_faces[face], axis=None)
               print("Average pixel is: "+ str(avg))
               temp = cropped_faces[face].copy()
               temp.fill(avg)
               faceless_image[ys[face+1]:ys[face+1]+hs[face+1],xs[face+1]:xs[face+1]+ws[face+1]] = temp
               #-------
               facecrop2 = np.asarray(cropped_faces[face])
               img_name = "object_{}.png".format(face)
               cv2.imwrite('./Images/'+str(img_name),facecrop2)
               print("{} saved!".format(img_name))

     img_name = "opencv_frame_{}.png".format(1)
     #print("Frame 1 is: ")
     #print(merged_image)
     cv2.imwrite('./Images/'+str(img_name),merged_image)
     print("{} written!".format(img_name))
     #cv2.imshow("Merged Image",merged_image)
     #------
     img_name = "opencv_frame_{}.png".format(2)
     cv2.imwrite('./Images/'+str(img_name),faceless_image)
     print("{} written!".format(img_name))
     #--------

     return xs, ys, ws, hs

     
     



     #cv2.moveWindow("Merged Image",0,0)
     #waitKey(0)
     #release()
     #destroyAllWindows()
     #exit()