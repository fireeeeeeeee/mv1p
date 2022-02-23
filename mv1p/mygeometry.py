from re import I
import numpy as np
from numpy.linalg.linalg import _raise_linalgerror_nonposdef, _raise_linalgerror_svd_nonconvergence, lstsq
from epipolar import lerp,lerp_R,calcRT
import cv2
from mytools import MergeRT
from mytools import SplitP
from mytools import time_print,fn_timer,normal
from mymath import LinearSolver,GetTx
import random
from easymocap.mytools import simple_recon_person
eps=1e-3

def Cross(v0,v1):
    return np.linalg.norm(np.cross(v0,v1))

def lineIntersec(p0,p1,v0,v1):
    if Cross(v0,v1)<=eps:  return None
    t0=Cross(p1-p0,v1)/Cross(v0,v1)
    return p0+t0*v0

def pointProjectToPlane(p0,u,p1):
    '''
        plane: (p-p0)*u=0
        p1: need project point
    '''
    return np.dot((p0-p1),u)*u/np.linalg.norm(u)/np.linalg.norm(u) + p1

def lineIntersecMiddle(p0,p1,v0,v1):
    if Cross(v0,v1)<=eps:  return None
    u=np.cross(v0,v1)
    pp0 = lineIntersec(p0,pointProjectToPlane(p0,u,p1),v0,v1)
    pp1 = lineIntersec(p1,pointProjectToPlane(p1,u,p0),v1,v0)
    return (pp0+pp1)/2 
    





@fn_timer
def DecomposeRelatieRT(relateRs,relateTs,Rs,Ts):
    '''
        n Cameras
        relateRs: n Rij
        relateTs: n Tij
        Rs: n R
        Ts: n T
        Decompose n+1 cameras parameter R,T from n relate, calculate it by mean of every three camera pair's results
    '''
    n=len(relateRs)
    assert len(relateRs)==n and len(relateTs)==n and  len(Rs)==n and len(Ts)==n

    def t_mean(ts):
        return sum(ts)/len(ts)
    def R_mean(Rs):
        rs=[cv2.Rodrigues(R)[0] for R in Rs]
        rm=t_mean(rs)
        return cv2.Rodrigues(rm)[0]
    if n==1: #regard relatet as true value
        if Ts[0] is None or relateTs[0] is None:
            return None,None
        return relateRs[0] @Rs[0],relateRs[0] @ Ts[0]+relateTs[0]
    elif n==2:
        if Ts[0] is None or relateTs[0] is None or Ts[1] is None or relateTs[1] is None:
            return None,None
        return R_mean([relateRs[0] @Rs[0],relateRs[1] @ Rs[1]]) , lineIntersecMiddle(relateRs[0] @ Ts[0],relateRs[1] @ Ts[1],relateTs[0],relateTs[1])
    else:
        sR=[]
        sT=[]
        for i in range(n):
            for j in range(i+1,n):
                R,T=DecomposeRelatieRT([relateRs[i],relateRs[j]],[relateTs[i],relateTs[j]],[Rs[i],Rs[j]],[Ts[i],Ts[j]])
                if T is not None:
                    sR.append(R)
                    sT.append(T)
        if len(sR)==0:return None,None
        return  R_mean(sR),t_mean(sT)
    
                
def randR():
    return cv2.Rodrigues(np.random.random((3,1)))[0]
def randT():
    return np.random.random(3)
def normR():
    return  np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
    ],dtype=np.float32)
def normT():
    return np.array([0,0,0],dtype=np.float32)



def BA(Ks,Rs,Ts,points2d):
    import pycalib
    n = len(Ks)
    Ps=[Ks[i]@MergeRT(Rs[i],Ts[i]) for i in range(n)]
    cameraVec=[{'K':Ks[i],'dist':np.zeros(5,dtype=Ks[i].dtype).reshape(1,5),'Rvec': cv2.Rodrigues(Rs[i])[0] ,'T':Ts[i]} for i in range(n)]
    config = {
        'verb': 1,
        'points': 1
    }
    
    points3d, kpts_repro = simple_recon_person(points2d,np.array(Ps))
    recCamera = pycalib.solveRTP(cameraVec, points3d, points2d, config)
    recRs=[cv2.Rodrigues(recCamera[i]['Rvec'])[0] for i in range(n) ]
    recTs=[recCamera[i]['T'] for i in range(n) ]
    return recRs,recTs





def IncrementalStructure(relateR,relateT,useba=False,Ks=None,points2d=None):
    '''
        n Cameras
        relateR: n*n Rij
        relateT: n*n Tij
        suppose world coordinate is the same as first camera
        and regard front two camera's distance is 1
    '''

    assert len(relateT)==len(relateR)
    n = len(relateR)

    Rs=[]
    Rs.append(np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
    ],dtype=np.float32))
    
    Ts=[]
    Ts.append(np.array([0,0,0],dtype=np.float32))
    cnt=0

    for i in range(1,n):
        rR=[]
        rT=[]
        for j in range(i):
            rR.append(relateR[j][i])
            rT.append(relateT[j][i])
        R,T=DecomposeRelatieRT(rR,rT,Rs,Ts)
        Rs.append(R)
        Ts.append(T)
        cnt+=1
        if cnt % 5==0 and useba:
            Rs,Ts=BA(Ks,Rs,Ts,points2d)
            
    return Rs,Ts

def IncrementalStructureByP(Ps,useba=False,Ks=None,points2d=None):
    from mytools import MergeRT
    relateR=[[P[:,:3] if P is not None else None for P in PP] for PP in Ps ]
    relateT=[[P[:,3] if P is not None else None for P in PP] for PP in Ps ]
    Rs,Ts=IncrementalStructure(relateR,relateT,useba,Ks,points2d)
    Ps=[MergeRT(R,T)  for R,T in list(zip(Rs,Ts))]
    return Ps

@fn_timer
def GlobalStructure(relateR,relateT,useba=False,Ks=None,points2d=None):
    '''
        constrain by R,T equations
        the R's equation is simple use 9 variables
    '''
    assert len(relateT)==len(relateR)
    n = len(relateR)
    LSR=[LinearSolver(n*3-3) for _ in range(3)]

    Rs=[]
    Rs.append(np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
    ],dtype=np.float32))

    Ts=[]
    Ts.append(np.array([0,0,0],dtype=np.float32))

    for i in range(1,n): # first camera constrain
        for j in range(3):
            LSR[j].AddByIndex([(3*i-3,1)],relateR[0][i][0][j])
            LSR[j].AddByIndex([(3*i-2,1)],relateR[0][i][1][j])
            LSR[j].AddByIndex([(3*i-1,1)],relateR[0][i][2][j])
    for i in range(1,n):
        for j in range(i+1,n):
            Rij=relateR[i][j]
            for k in range(3):
                LSR[k].AddByIndex([(3*j-3,1),(3*i-3,-Rij[0][0]),(3*i-2,-Rij[0][1]),(3*i-1,-Rij[0][2])],0)
                LSR[k].AddByIndex([(3*j-2,1),(3*i-3,-Rij[1][0]),(3*i-2,-Rij[1][1]),(3*i-1,-Rij[1][2])],0)
                LSR[k].AddByIndex([(3*j-1,1),(3*i-3,-Rij[2][0]),(3*i-2,-Rij[2][1]),(3*i-1,-Rij[2][2])],0)
    RL=np.zeros((0,3*n-3))
    for i in range(3):
        RL=np.vstack((RL,LSR[i].Solve()))
    for i in range(n-1):
        Rs.append(RL.T[3*i:3*i+3])

    if n==1:
        return Rs,Ts    
    LST=LinearSolver(n*3-3)


    LST.AddByIndex([(0,1)],relateT[0][1][0])
    LST.AddByIndex([(1,1)],relateT[0][1][1])
    LST.AddByIndex([(2,1)],relateT[0][1][2])


    for i in range(1,n):
        Tx=GetTx(relateT[0][i])
        for k in range(3):
            LST.AddByIndex([
            (i*3-3,Tx[k][0]),
            (i*3-2,Tx[k][1]),
            (i*3-1,Tx[k][2])],
            0)        

    for i in range(1,n):
        for j in range(i+1,n):
            Tx=GetTx(relateT[i][j])
            rR=Rs[j] @ Rs[i].T 
            R=Tx @ rR
            L=Tx
            for k in range(3):
                LST.AddByIndex([
                (j*3-3,L[k][0]),
                (j*3-2,L[k][1]),
                (j*3-1,L[k][2]),
                (i*3-3,-R[k][0]),
                (i*3-2,-R[k][1]),
                (i*3-1,-R[k][2])],
                0)
    TL=LST.Solve()
    k= np.linalg.norm(TL[0:3])
    for i in range(n-1):
        Ts.append(TL[3*i:3*i+3]/k)

    if useba:
        Rs,Ts=BA(Ks,Rs,Ts,points2d)
    
    return Rs,Ts

def GlobalStructureByP(Ps,useba=False,Ks=None,points2d=None):
    from mytools import MergeRT
    relateR=[[P[:,:3] if P is not None else None for P in PP] for PP in Ps ]
    relateT=[[P[:,3] if P is not None else None for P in PP] for PP in Ps ]
    Rs,Ts=GlobalStructure(relateR,relateT,useba,Ks,points2d)
    Ps=[MergeRT(R,T)  for R,T in list(zip(Rs,Ts))]
    return Ps

def getRelativeRT(R1,T1,R2,T2,noise=0):
    pts=np.random.random((4,40))
    P1=np.hstack((R1,T1.reshape(3,1)))
    P2=np.hstack((R2,T2.reshape(3,1)))
    K=np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ],dtype=np.float32)

    def interp(pts,a):
        n=pts.shape[1]
        pts[:2]+=a*np.random.random(2*n).reshape(2,n)   
        return pts

    return calcRT(K,K,interp(P1@pts,noise),interp(P2@pts,noise))

@fn_timer
def generateTestdata(n,noise):

    Rs=[randR() for _ in range(n)]
    Ts=[randT() for _ in range(n)]
    Rs[0]=normR()
    Ts[0]=normT()
    Ts[1]=normal(Ts[1])
    relateR=[]
    relateT=[]

    relatePs=[[None for _ in range(n)] for __ in range(n)]

    for i in range(n):
        relateR.append([])
        relateT.append([])
        for j in range(n):
            P=getRelativeRT(Rs[i],Ts[i],Rs[j],Ts[j],noise)
            if j>i:
                relatePs[i][j]=P
            if P is None:
                relateR[i].append(None)
                relateT[i].append(None)
            else:
                relateR[i].append(P[:,:3])
                relateT[i].append(P[:,3].reshape(3))
    return Rs,Ts,relateR,relateT,relatePs


def PrintLoss(Rs,Ts,Ps): 
    calcR=[SplitP(P)[0] for P in Ps] 
    calcT=[SplitP(P)[1] for P in Ps] 

    def removeNone(i,j):
        if i is None or j is None:
            return 'missing'
        return  abs(i-j)
    print([removeNone(i,j) for i,j in list(zip(Rs,calcR))])
    print([removeNone(i,j) for i,j in list(zip(Ts,calcT))])
    print(sum([removeNone(i,j) for i,j in list(zip(Ts,calcT))]))

@fn_timer
def testIncrementalStructure():
    Rs,Ts,relateR,relateT,relatePs=generateTestdata(20,0.01)
    #calcR,calcT=IncrementalStructure(relateR,relateT)
    Ps=IncrementalStructureByP(relatePs)
    PrintLoss(Rs,Ts,Ps)


@fn_timer
def testGlobalStructure():
    Rs,Ts,relateR,relateT,relatePs=generateTestdata(20,0.01)
    Ps=GlobalStructureByP(relatePs)
    PrintLoss(Rs,Ts,Ps)

if __name__=='__main__':
    #testIncrementalStructure()
    testGlobalStructure()
    time_print()