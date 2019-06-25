# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:05:13 2019

@author: BORAM
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def morphology():
    imgpath = "S03.jpg"
    img_color = cv2.imread(imgpath, 1)
    img = cv2.cvtColor(img_color,cv2.COLOR_BGR2GRAY)
    
    #가우시안 블러링
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    
    #OTSU 이진 Thresholding
    ret, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    #Mopolozy
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
    
    opening = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    
    closing_copy = closing
    cimg, contours, hierachy = cv2.findContours(closing_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cimg = cv2.drawContours(img_color, contours, -1, (0, 255, 0), 3)
    
    shrink = cv2.resize(cimg, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    
    cv2.imshow('contours',shrink)
    #cv2.imshow('contours', cimg)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
    titles = ['original', 'gaussian blurred', 'otsu thresholding', 'opening','closing']
    
    output = [img, blur, thr,  opening, closing]
    
    plt.figure(figsize=(50,50))
    for i in range(5):
        plt.subplot(5, 1, i+1)
        plt.imshow(output[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    
    plt.show()
    
morphology()