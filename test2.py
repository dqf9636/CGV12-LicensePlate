# -*- coding: utf-8 -*-
"""
Created on Fri May  3 18:23:33 2019

@author: BORAM
"""
#엣지
#허프변환

import cv2
import numpy as np
import matplotlib.pyplot as plt

def morphology():
    imgpath = "S03.jpg"
    img_color = cv2.imread(imgpath, 1)
    shrink = cv2.resize(img_color, (400,300), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(shrink,cv2.COLOR_BGR2GRAY)
    img_copy = shrink
    #가우시안 블러링
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    #OTSU 이진 Thresholding
    ret, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    #Mopolozy
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closing = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    
    canny = cv2.Canny(closing,100,175)
    
    lines = cv2.HoughLines(canny,1,np.pi/180,80)

    for i in range(len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0+1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 -1000*(a))
    
            cv2.line(img_copy,(x1,y1),(x2,y2),(0,0,255),2)

    res = np.vstack((shrink,img_copy))
    cv2.imshow('img',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    
    
morphology()