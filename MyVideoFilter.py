import cv2
import numpy as np
import pandas as pd
eye_cascade = cv2.CascadeClassifier("./third-party/frontalEyes35x16.xml")
nose_cascade= cv2.CascadeClassifier("./third-party/Nose18x15.xml")
face_cascade= cv2.CascadeClassifier("./third-party/haarcascade_frontalface_alt.xml")

cap = cv2.VideoCapture(0)
specs_ori = cv2.imread('glasses.png',-1)
specs_ori=cv2.cvtColor(specs_ori,cv2.COLOR_BGR2BGRA)
mu = cv2.imread('mustache.png',-1)
mu_ori=cv2.cvtColor(mu,cv2.COLOR_BGR2BGRA)
def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image
 
    # loop over all pixels and apply the blending equation
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src
import itertools
while True:
      ret,frame = cap.read()
      
      faces = face_cascade.detectMultiScale(frame,1.2,5)
      for (x,y,w,h) in faces:
            eyes = eye_cascade.detectMultiScale(frame,1.2,5)
            noses=nose_cascade.detectMultiScale(frame,1.3,5)
            for (nx,ny,nw,nh),(ex,ey,ew,eh) in itertools.product(eyes,noses):
                 #218 348 146 318
                  ww=int((nw*0.2)//1)
                  hh=int((nh*0.125)//1)
                  face_glass_roi_color = frame[ny-ww:ny+2*ww+nw, nx-hh:nx+nh+2*hh]
                  #cv2.rectangle(frame,(nx,ny),(nx+nh,ny+nw),(255,0,0),2)
                  specs = cv2.resize(specs_ori, (nh+2*hh, nw+2*ww),interpolation=cv2.INTER_CUBIC)
                  transparentOverlay(face_glass_roi_color,specs)
                  #312 485 106 128
                  ww=int((ew*0.25)//1)
                  hh=int((eh*0.1)//1)
                  face_glass_roi_color = frame[ey+ew//2:ey+ew+ww, ex-hh:ex+eh+2*hh]
                  #cv2.rectangle(frame,(ex,ey),(ex+eh,ey+ew),(255,0,0),2)
                  mus = cv2.resize(mu_ori, (eh+2*hh, ew//2+ww),interpolation=cv2.INTER_CUBIC)
                  transparentOverlay(face_glass_roi_color,mus)
            
            
                                   
                  
                  
      cv2.imshow("Video Frame",frame)
      keyPressed=cv2.waitKey(1) & 0xFF
      if keyPressed==ord('q'):
            break
cv2.destroyAllWindows()
    
