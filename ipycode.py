from torchface import FaceDetector
from displai import MergeIt
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN
from random import randrange
from random import random
import time
import matplotlib.pyplot as plt
import math
import os
import scipy
import scipy.sparse.linalg
import utils



background_img = cv2.cvtColor(cv2.imread('./Images/opencv_frame_0.png'), cv2.COLOR_BGR2RGB).astype('double') / 255.0 
object_img = cv2.cvtColor(cv2.imread('./Images/object_0.png'), cv2.COLOR_BGR2RGB).astype('double') / 255.0 
object_img2 = cv2.cvtColor(cv2.imread('./Images/object_1.png'), cv2.COLOR_BGR2RGB).astype('double') / 255.0 

xs1 = (10, 105, 105, 10)
ys1 = (20, 20, 128, 128)

xs2 = (10, 105, 105, 10)
ys2 = (20, 20, 128, 128)


object_mask = utils.get_mask(ys1, xs1, object_img)
object_mask2 = utils.get_mask(ys2, xs2, object_img2)

#bottom_center = (370, 400)

#object_img, object_mask = utils.crop_object_img(object_img, object_mask)

#bg_ul = utils.upper_left_background_rc(object_mask, bottom_center)

bg_ul = (ys[1],xs[1])
bg_ul2 = (ys[0],xs[0])



#plt.imshow(utils.get_combined_img(background_img, object_img, object_mask, bg_ul))

object_mask[:,-1] = 0
object_mask[-1,:] = 0

object_mask2[:,-1] = 0
object_mask2[-1,:] = 0

def poisson_blend(object_img, object_mask, bg_img, bg_ul):
    #Object Shape
    im_h,im_w = object_mask.shape
    
    #Background upperleft coordinates
    bg_y, bg_x = bg_ul
    
    #Inner Dimensions
    smallh,smallw = (im_h-2,im_w-2)
    
    #Number of variables (Step 1)
    numVar = (object_mask>0).sum()

    #Number of Equations/Constraints = Inner Constraints + Outer Constraints
    neq = (smallh *(smallw-1) + smallw *(smallh-1)) + (2*(smallh+smallw))+1000000

    #Build Sparse Matrix with correct dimensions
    A = scipy.sparse.lil_matrix((neq, smallh*smallw), dtype='double') # init lil
    
    #Build Vector B
    b = np.zeros((neq,1), dtype='double')
    
    #Build im2var
    im2var = -np.ones(object_img.shape[0:2], dtype='int32')
    im2var[object_mask>0] = np.arange(numVar)
    #print(im2var)
    
    e = 0
    #Step 2
    #Inner Constraints - Horizontal
    for y in range(0,im_h):
        for x in range(0,im_w-1):
            if object_mask[y,x] == 1:
                A[e,im2var[y][x+1]] = -1
                A[e,im2var[y][x]] = 1
                b[e] = object_img[y][x] - object_img[y][x+1]
                e = e + 1
            
    #Inner Constraints - Vertical
    for y in range(0,im_h-1):
        for x in range(0,im_w):
            if object_mask[y,x] == 1:
                A[e,im2var[y+1][x]] = -1
                A[e,im2var[y][x]] = 1
                b[e] = object_img[y][x] - object_img[y+1][x] 
                e = e + 1
            
    #Outer Constraints - Vertical North
    for x in range(0,im_w):
        y = 1
        if object_mask[y,x] == 1:
            #A[e,im2var[y-1][x]] = -1
            A[e,im2var[y][x]] = 1
            b[e] = object_img[y][x] - object_img[y-1][x] 
            b[e] = b[e] + bg_img[bg_y+(y-1)][bg_x+x]
            e = e + 1
            
    #Outer Constraints - Vertical South
    for x in range(0,im_w):
        y = im_h - 2
        if object_mask[y,x] == 1:
            #A[e,im2var[y+1][x]] = -1
            A[e,im2var[y][x]] = 1
            b[e] = object_img[y][x] - object_img[y+1][x] 
            b[e] = b[e] + bg_img[bg_y+(y+1)][bg_x+x]
            e = e + 1
            
    #Outer Constraints - Horizontal East
    for y in range(0,im_h):
        x = im_w - 2
        if object_mask[y,x] == 1:
            #A[e,im2var[y][x+1]] = -1
            A[e,im2var[y][x]] = 1
            b[e] = object_img[y][x] - object_img[y][x+1] 
            b[e] = b[e] + bg_img[bg_y+y][bg_x+(x+1)]
            e = e + 1
    
    #Outer Constraints - Horizontal West
    for y in range(0,im_h):
        x = 1
        if object_mask[y,x] == 1:
            #A[e,im2var[y][x-1]] = -1
            A[e,im2var[y][x]] = 1
            b[e] = object_img[y][x] - object_img[y][x-1] 
            b[e] = b[e] + bg_img[bg_y+y][bg_x+(x-1)]
            e = e + 1
        
    v = scipy.sparse.linalg.lsqr(A.tocsr(), b); # solve w/ csr
    v2 = v[0].reshape(smallh,smallw)
    
    output = bg_img
    for y in range(0,smallh):
        for x in range(0,smallw):
            output[bg_y+y][bg_x+x] = v2[y][x]
    
    return output

plt.close('all')
im_blend = np.zeros(background_img.shape)
for b in np.arange(3):
    im_blend[:,:,b] = poisson_blend(object_img[:,:,b], object_mask, background_img[:,:,b].copy(), bg_ul)

im_blend2 = np.zeros(background_img.shape)
for c in nparange(3):
     im_blend2[:,:,b] = poisson_blend(object_img2[:,:,b], object_mask2, im_blend[:,:,:].copy(), bg_ul2)

plt.figure(figsize=(10,10))
plt.imshow(im_blend2)
plt.show()








