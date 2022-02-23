from genericpath import getctime
from random import choices
from re import DEBUG, S, T
from cv2 import norm
from easymocap import dataset
from numpy.lib.nanfunctions import nanpercentile
from numpy.random.mtrand import rand
from tqdm import tqdm
from easymocap.smplmodel import check_keypoints, load_model, select_nf
from easymocap.mytools import simple_recon_person, Timer, projectN3
from easymocap.pipeline import smpl_from_keypoints3d2d
import os
from os.path import join
import numpy as np
from easymocap.socket.base_client import BaseSocketClient
from epipolar import LerpPallCalculator,BrutePallCalculator,MoviePallCalculator,FlipBucketPallCalculator,batch_evaluation,calcRT
from mygeometry import IncrementalStructureByP,GlobalStructureByP
from mytools import time_print,fn_timer
from evaluate import ErrorCalculator


def TestTriangulate(client):
    MIN_CONF_THRES = args.thres2d
    start, end = args.start, min(args.end, len(dataset))
    
    for nf in tqdm(range(start, end), desc='triangulation'):
        images, annots = dataset[nf]
        check_keypoints(annots['keypoints'], WEIGHT_DEBUFF=1, min_conf=MIN_CONF_THRES)
        keypoints3d, kpts_repro = simple_recon_person(annots['keypoints'], dataset.Pall)
        datas=[]
        data={
            'id':0,
            'keypoints3d':keypoints3d
        }
        datas.append(data)
        client.send(datas)
    client.close()

def Rebuild_allRT(client):
    from epipolar import getCandiRT,getPositiveRT,calcRT
    assert len(dataset.cams)==2
    MIN_CONF_THRES = args.thres2d
    start, end = args.start, min(args.end, len(dataset))
    Ks=[dataset.cameras[cam]['K'] for cam in dataset.cams]
    Rs=[dataset.cameras[cam]['R'] for cam in dataset.cams]
    Ts=[dataset.cameras[cam]['T'] for cam in dataset.cams]
    for nf in tqdm(range(start, end), desc='rebuild'):
        images, annots = dataset[nf]
        check_keypoints(annots['keypoints'], WEIGHT_DEBUFF=1, min_conf=MIN_CONF_THRES)
        pts0=annots['keypoints'][0]
        pts1=annots['keypoints'][1]
        P0=np.hstack((Rs[0],Ts[0]))
        if False:
            Ps=getPositiveRT(Ks[0],Ks[1],pts0.T,pts1.T)
        else:
            Ps=getCandiRT(Ks[0],Ks[1],pts0.T,pts1.T)
        datas=[]
        if len(Ps)==0: 
            continue


        #Ps=[min((np.linalg.norm(Rs[0]@P[:,:3]-Rs[1]),i,P) for i,P in enumerate(Ps) )[2]]
        #Ps=[min([(P[0][0],i,P) for i,P in enumerate(Ps)])[2]]
        #Ps=[calcRT(Ks[0],Ks[1],pts0.T,pts1.T)]
        for idx,P1 in enumerate(Ps):
            P1=np.hstack((P0[:,:3] @ P1[:,:3],Ts[1]))
            P1=Ks[1] @ P1
            import cv2
            pts=cv2.triangulatePoints(Ks[0] @ P0,P1,pts0.T[:2],pts1.T[:2])
            pts/=pts[3]
            
            data={
                'id':idx+1,
                'keypoints3d':pts.T
            }
            datas.append(data)
        keypoints3d, kpts_repro = simple_recon_person(annots['keypoints'], dataset.Pall)
        #keypoints3d[:,:2]+=1
        data={
            'id':0,
            'keypoints3d':keypoints3d
        }
        datas.append(data)
        client.send(datas)



def Rebuild(client):
    
    assert len(dataset.cams)==2 or len(dataset.cams)==3
    MIN_CONF_THRES = args.thres2d
    start, end = args.start, min(args.end, len(dataset))

    import random
    k1=random.randint(500,1500)
    def defaultK():
        import random

        
        k2=random.randint(500,1000)
        k2=500
        return np.array([[k1,0,k2],[0,k1,k2],[0,0,1]],dtype=np.float32)

    Ks=[dataset.cameras[cam]['K'] for cam in dataset.cams]
    #Ks=[defaultK() for cam in dataset.cams]
    Rs=[dataset.cameras[cam]['R'] for cam in dataset.cams]
    Ts=[dataset.cameras[cam]['T'] for cam in dataset.cams]


    if False:
        it=tqdm(range(start, end), desc='rebuild')
    else:
        it=range(start, end)
    pcs=[LerpPallCalculator(Ks[0],Ks[1],500,0.01,0.01),FlipBucketPallCalculator(Ks[0],Ks[1],500,0.01,0.01)]
    #pc = LerpPallCalculator(Ks[0],Ks[1],500,0.1)
    #pc = BrutePallCalculator(Ks[0],Ks[1])

    mid =(start+end)//2
    for nf in it:
        images, annots = dataset[nf]
        check_keypoints(annots['keypoints'], WEIGHT_DEBUFF=1, min_conf=MIN_CONF_THRES)
        P0=np.hstack((Rs[0],Ts[0]))
        

        if nf==mid:
            print('changed')
        
        if len(dataset.cams)==3 and nf>=mid:
            chosc=2

        else:
            chosc=1
        for pc in pcs:
            pc.changeK(Ks[0],Ks[chosc])
        pts0=annots['keypoints'][0]    
        pts1=annots['keypoints'][chosc]
        mask=(pts0[:,2]>0.5) & (pts1[:,2]>0.5)
        if False: #add true point for debug
            keypoints3d, kpts_repro = simple_recon_person(annots['keypoints'], dataset.Pall)
            p3d=keypoints3d[keypoints3d[:,3]>0.5]
            P0T=Ks[0] @ np.hstack((Rs[0],Ts[0]))
            P1T=Ks[1] @ np.hstack((Rs[1],Ts[1]))
            if False: #add manual data
                ptsADD=np.random.random((4,100))
                ptsADD[3]=1
                mask=((P0T@ptsADD)[2]>0) & ((P1T@ptsADD)[2]>0)
                p3d=np.vstack((p3d,ptsADD[:,mask].T))
            p3d[:,3]=1

            pts0_c=P0T @ p3d.T
            pts1_c=P1T @ p3d.T
            P1=calcRT(Ks[0],Ks[1],pts0_c,pts1_c)
            if True:#calcualte reproject error for debug
                import matplotlib.pyplot as plt
                pts0_e=pts0[mask].T
                pts1_e=pts1[mask].T      
                acc=pts0_e[2]*pts1_e[2]
                pts0_e[2]=1
                pts1_e[2]=1     
                err=batch_evaluation(P1,pts0_e,pts1_e)
                print(err)
                # plt.scatter(acc,err,alpha=0.1*(nf%10),c='r')
                # if nf%10 ==0:
                #     plt.show()
        else:
            pts0_c=pts0[mask].T
            pts1_c=pts1[mask].T
            pts0_c[2]=1
            pts1_c[2]=1    
            P1s=[pc.calc(pts0_c,pts1_c) for pc in pcs]
            #P1=pc.calc(pts0_c,pts1_c)
        #P0=Ks[0]@np.hstack((Rs[0],Ts[0]))
        #P1=np.hstack((Rs[1],Ts[1]))
        # if P1 is None:
        #     continue

        import cv2
        datas=[]
        for i in range(len(P1s)):
            if P1s[i] is None:
                continue
            P1=P1s[i]
            if True: # fit ground truth
                P1=np.hstack((P1[:,:3] @ P0[:,:3],P1[:,:3]@Ts[0]+ np.linalg.norm(Ts[chosc]-P1[:,:3]@Ts[0])*(P1[:,3].reshape(3,1))))
                #P1=np.hstack((P1[:,:3] @ P0[:,:3],P1[:,:3]@Ts[0]+ np.linalg.norm(Ts[1]-P1[:,:3]@Ts[0])*(P1[:,3].reshape(3,1))))
                #P1=np.hstack((P1[:,:3] @ P0[:,:3],Ts[1]))
                #P1=np.hstack((Rs[1] ,P1[:,:3]@Ts[0]+ np.linalg.norm(Ts[1]-P1[:,:3]@Ts[0])*(P1[:,3].reshape(3,1))))
            else:
                P0=np.array([
                    [1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0],
                ],dtype=np.float32)

            pts=cv2.triangulatePoints(Ks[0] @ P0,Ks[chosc] @ P1,pts0.T[:2],pts1.T[:2])
            pts/=pts[3]
            pts[3][mask==False]=0
            data={
                'id':i+1,
                'keypoints3d':pts.T
            }
            datas.append(data)
        keypoints3d, kpts_repro = simple_recon_person(annots['keypoints'][:2], dataset.Pall[:2])
        #keypoints3d[:,:2]+=1
        data={
            'id':0,
            'keypoints3d':keypoints3d
        }
        datas.append(data)
        client.send(datas)
        import time
        #time.sleep(0.5)

#$data='C:/Users/AAAA/Desktop/ee/data'
#python .\rebuild.py $data --out $data/output/smpl --vis_det --vis_repro --undis --sub 4 11 --vis_smpl

@fn_timer
def Rebuild_MV(client):
    start, end = args.start, min(args.end, len(dataset))
    Ks=[dataset.cameras[cam]['K'] for cam in dataset.cams]
    #Ks=[defaultK() for cam in dataset.cams]
    Rs=[dataset.cameras[cam]['R'] for cam in dataset.cams]
    Ts=[dataset.cameras[cam]['T'] for cam in dataset.cams]
    nc=len(args.sub)
    relatePs=[[None for _ in range(nc)] for __ in range(nc)]
    pcs=[[LerpPallCalculator(Ks[i],Ks[j],500,0.01,0.01) for j in range(nc)] for i in range(nc)]
    annots_pool=np.zeros((len(dataset.cams),0,3))
    annots_cacheNum=500
    err1=ErrorCalculator('calcWithoutBA')
    err2=ErrorCalculator('calcWithtBA')

    for nf in range(start,end):
        images, annots = dataset[nf] 
        datas=[]
        for i in range(nc):
            for j in range(i+1,nc):
                if False:
                    P0T=Ks[i] @ np.hstack((Rs[i],Ts[i]))
                    P1T=Ks[j] @ np.hstack((Rs[j],Ts[j]))
                    ptsADD=np.random.random((4,20))
                    ptsADD[3]=1
                    mask=((P0T@ptsADD)[2]>0) & ((P1T@ptsADD)[2]>0)
                    
                    pts0_c=P0T @ ptsADD[:,mask]
                    pts1_c=P1T @ ptsADD[:,mask]
                else:
                    pts0=annots['keypoints'][i]    
                    pts1=annots['keypoints'][j]
                    mask=(pts0[:,2]>0.5) & (pts1[:,2]>0.5)
                    pts0_c=pts0[mask].T
                    pts1_c=pts1[mask].T
                    pts0_c[2]=1
                    pts1_c[2]=1    

                relatePs[i][j] = pcs[i][j].calc(pts0_c,pts1_c)


        annots_pool=np.concatenate((annots_pool,annots['keypoints']),axis=1)
        if annots_pool.shape[1]>annots_cacheNum:
            annots_pool=annots_pool[:,:annots_cacheNum,:]

        @fn_timer
        def calcWithoutBA():
            return GlobalStructureByP(relatePs,False,Ks,annots_pool)
        @fn_timer
        def calcWithBA():
            return GlobalStructureByP(relatePs,True,Ks,annots_pool)
        Ps1=calcWithoutBA()
        Ps2=calcWithBA()

        keypoints3d_truth, kpts_repro = simple_recon_person(annots['keypoints'], dataset.Pall)
        #keypoints3d[:,:2]+=1
        data={
            'id':2,
            'keypoints3d':keypoints3d_truth
        }
        datas.append(data)

        mask_keypoints=keypoints3d_truth[:,3]>0
        

        def sendPs(Ps,idx,err):
            bigger=1
            if True:
                P0=np.hstack((Rs[0],Ts[0]))
                k=np.linalg.norm(Ts[1]-Ps[1][:,:3]@Ts[0])
                for i in range(len(Ps)):
                    P1=Ps[i]
                    P1=Ks[i] @ np.hstack((P1[:,:3] @ P0[:,:3],P1[:,:3]@Ts[0]+ k*(P1[:,3].reshape(3,1))))
                    Ps[i]=P1
            else:
                bigger=np.linalg.norm(Ts[1]-Ps[1][:,:3]@Ts[0])
                for i in range(len(Ps)):
                    Ps[i]=Ks[i]@Ps[i]

            keypoints3d, kpts_repro = simple_recon_person(annots['keypoints'],np.array(Ps))
            err.add(keypoints3d_truth[mask_keypoints],keypoints3d[mask_keypoints])
            data={
                'id':idx,
                'keypoints3d':keypoints3d*bigger
            }
            datas.append(data)

        sendPs(Ps1,0,err1)
        sendPs(Ps2,1,err2)
        client.send(datas)
    err1.calcAndShow()
    err2.calcAndShow()



if __name__ == "__main__":
    from easymocap.mytools import load_parser, parse_parser
    from easymocap.dataset import CONFIG, MV1PMF
    parser = load_parser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9999)
    args = parse_parser(parser)

    print(args.path)

    dataset = MV1PMF(args.path, annot_root=args.annot, cams=args.sub, out=args.out,
        config=CONFIG[args.body], kpts_type=args.body,
        undis=args.undis, no_img=False, verbose=args.verbose)
    client = BaseSocketClient(args.host, args.port)
    Rebuild_MV(client)
    time_print()
    #Rebuild(client)
    #TestTriangulate(client)
    