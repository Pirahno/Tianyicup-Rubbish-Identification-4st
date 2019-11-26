import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import random
import math

class RandomErasing:
    """Random erasing the an rectangle or any region in Image.
    Args:
        sl: min erasing area region 
        sh: max erasing area region
        r1: min aspect ratio range of earsing region
        p: probability of performing random erasing
    """

    def __init__(self, p=0.5, sl=0.3, sh=0.6, r1=0.6, is_ract = True):

        self.p = p
        self.s = (sl, sh)
        self.r = (r1, 1/r1)
        self.is_ract = is_ract
    

    def __call__(self, img):
        """
        perform random erasing
        Args:
            img: opencv numpy array in form of [w, h, c] range 
                 from [0, 255]
        
        Returns:
            erased img
        """
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

        assert len(img.shape) == 3, 'image should be a 3 dimension numpy array'
        
        if  random.random()> self.p:
            return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        
        else:
            while True:
                Se = random.uniform(*self.s) * img.shape[0] * img.shape[1]
                re = random.uniform(*self.r) 

                He = int(round(math.sqrt(Se * re)))
                We = int(round(math.sqrt(Se / re)))

                xe = random.randint(0, int(img.shape[1]*0.65))
                ye = random.randint(0, int(img.shape[0]*0.65))
                
                random_color = (np.random.randint(1,255),np.random.randint(1,255),np.random.randint(1,255))
                
                
        
                if xe + We <= img.shape[1] and ye + He <= img.shape[0]:
                    
                    if self.is_ract:
                        img[ye : ye + He, xe : xe + We, :] = random_color
                    else:
                        poly = []
                        for i in range(7):
                            x = np.random.randint(xe,xe + We)
                            y = np.random.randint(ye,ye + He)
                            poly.append([x, y])
                        poly = np.array(poly)
                        cv2.fillPoly(img,[poly],random_color)

                    return Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

class SelectROI:
    def __init__(self, use_minrect = False):
        self.use_minrect = use_minrect
        
    def __call__(self, img):
        imgcv = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        imgray = cv2.cvtColor(imgcv, cv2.COLOR_BGR2GRAY)
        
        ret, thresh = cv2.threshold(imgray, 1, 255, 0)
        median = cv2.GaussianBlur(thresh,(5,5), 3)
        contours, hierarchy = cv2.findContours(median,cv2.RETR_EXTERNAL ,cv2.CHAIN_APPROX_SIMPLE)
        
        #  未检测到边缘， 返回原图
        if len(contours) == 0:
            return img
        
        # 检测到边缘， 返回最大外接矩形
        
        max_area = 0
        if self.use_minrect:
            # 检测到边缘， 返回最小外接矩形
            for i in range(len(contours)):
                cnt = contours[i]
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                w, h  = int(rect[1][0]), int(rect[1][1])
                if w*h > max_area:
                    max_area = w*h
                    src_pts = box.astype("float32")
                    dst_pts = np.array([[0, h-1],
                                        [0, 0],
                                        [w-1, 0],
                                        [w-1, h-1]], dtype="float32")

                    # the perspective transformation matrix
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

                    # directly warp the rotated rectangle to get the straightened rectangle
                    warped = cv2.warpPerspective(imgcv, M, (w, h))

                    roi = warped[5:-5,5:-5]
                    
        else:
            # 检测到边缘， 返回最大外接矩形
            for i in range(len(contours)):
                cnt = contours[i]
                x,y,w,h = cv2.boundingRect(cnt)
                if w*h > max_area:
                    max_area = w*h
                    roi = imgcv[y+2:y+h-2,x+2:x+w-2]

        image = Image.fromarray(cv2.cvtColor(roi,cv2.COLOR_BGR2RGB))
        return image