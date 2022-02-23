from unicodedata import name
import cv2
import json
prex=0
prey=0
cimg=None

cname=0

def drawLine(event,x,y,flags,param):
    global prex,prey,cname,cimg
    if event==cv2.EVENT_LBUTTONDOWN:
        prex=x
        prey=y
    elif event==cv2.EVENT_LBUTTONUP:
        cv2.line(cimg,(prex,prey),(x,y),(0,255,0))
        cv2.imshow('frame',cimg)

        path='./intri.json'
        import os   
        if os.path.exists(path):
            with open(path,'r') as f:
                paras=json.load(f)
        else:
            paras={}
        if cname not in paras:
            paras[cname]={}
        import math
        paras[cname]['K1']= math.sqrt( (y-prey)*(y-prey)+(x-prex)*(x-prex))
        paras[cname]['t1']=cimg.shape[1]
        paras[cname]['t2']=cimg.shape[0]
        with open(path,"w") as f:
            json.dump(paras,f)

def calib(id,name):
    global cname,cimg
    cname=name
    cap=cv2.VideoCapture(id)
    while True:
        # get a frame
        ret, frame = cap.read()
        # show a frame
        cv2.imshow("capture", frame)
        t=(cv2.waitKey(1) & 0xFF)
        if t == ord('q'):
            break
        elif t == ord('c'):
            cimg=frame
            cv2.imshow('frame',cimg)
            cv2.setMouseCallback('frame',drawLine)

if __name__=='__main__':
    with open('./cam.json','r') as f:
        cam=json.load(f)
    for id,name in list(zip(cam['cam_ids'],cam['cam_names'])):
        calib(id,name)