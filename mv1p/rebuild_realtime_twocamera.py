from math import pi
from ssl import ALERT_DESCRIPTION_UNKNOWN_PSK_IDENTITY
from tkinter import N
from tqdm import tqdm
import cv2
import json
import numpy as np
from easymocap.socket.base_client import BaseSocketClient
from detect_server import handleIMG

def main(client):
    from epipolar import calcRT
    from epipolar import lerp,lerp_R


    pts0s=np.zeros((3,0))
    pts1s=np.zeros((3,0))
    preP=None
    for keypoints,annots in handleIMG():

        with open('./cam.json','r') as f:
            cam_names=json.load(f)['cam_names']
        with open('./intri.json','r') as f:
            intris=json.load(f)
        def getK(s):
            K=np.array([
                [s['K1'],0,s['t1']],
                [0,s['K1'],s['t2']],
                [0,0,1],
            ],np.float32)
            return K
        K0=getK(intris[cam_names[0]])
        K1=getK(intris[cam_names[1]])

        pts0=keypoints[0]
        pts1=keypoints[1]
        # pts0=annots[0]
        # pts1=annots[1]


        pts0_c=pts0.T
        pts1_c=pts1.T

        if pts0_c.shape[1]<3:
            pts0_c=np.vstack((pts0_c,np.ones(pts0_c.shape[1])))
            pts1_c=np.vstack((pts1_c,np.ones(pts1_c.shape[1])))
        
        pts0s=np.hstack((pts0s,pts0_c))
        pts1s=np.hstack((pts1s,pts1_c))
        pCahceNum=500
        if pts0s.shape[1]>pCahceNum:
            pts0s=pts0s[:,-pCahceNum:]
        if pts1s.shape[1]>pCahceNum:
            pts1s=pts1s[:,-pCahceNum:]
        import cv2
        P1=calcRT(K0,K1,pts0s,pts1s,cv2.FM_RANSAC)

        print(pts0s.shape,P1 is None)


        if P1 is None:
            continue
        if not preP is None:
            theta=0.01
            P1[:,:3]=lerp_R(preP[:,:3],P1[:,:3],theta)
            P1[:,3]=lerp(preP[:,3],P1[:,3],theta)
        preP=P1    
        #P1=np.hstack((P1[:,:3] @ P0[:,:3],Ts[0]+ np.linalg.norm(Ts[1]-Ts[0])*(P0[:,:3].T @P1[:,3].reshape(3,1))))
        #P1=np.hstack((P1[:,:3] @ P0[:,:3],Ts[1]))
        #P1=np.hstack((Rs[1]  ,Ts[0]+P1[:,3].reshape(3,1)))
        #P1=np.hstack((P1[:,:3]@P0[:,:3]  ,Ts[0]+P1[:,3].reshape(3,1)))
        P0=np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
        ],dtype=np.float32)
        

        P0=K0 @ P0
        P1=K1 @ P1

        annot0=annots[0]
        annot1=annots[1]
        mask=(annot0[:,2]>0.6) & (annot1[:,2]>0.6)
        import cv2
        pts=cv2.triangulatePoints(P0,P1,annots[0].T[:2],annots[1].T[:2])
        pts/=pts[3]
        pts[3][mask==False]=0
        datas=[]
        data={
            'id':0,
            'keypoints3d':pts.T*4
        }
        datas.append(data)
        client.send(datas)
        import time
        time.sleep(0.25)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--body', action='store_false')
    parser.add_argument('--vis2d', action='store_false')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument("--host2d", type=lambda x:(x.split(':')[0], int(x.split(':')[1])), 
        nargs='+')
    args = parser.parse_args()

    client = BaseSocketClient(args.host, args.port)

    main(client)
