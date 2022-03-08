import re
import cv2
from easymocap.smplmodel.body_param import select_nf
from matplotlib.pyplot import flag
import numpy as np
from numpy.core.arrayprint import printoptions
from numpy.lib.function_base import select
from numpy.linalg import norm
from numpy.random.mtrand import random


USE_ESS=False


def normal(x):
    return x/np.linalg.norm(x)

def getRTfromE(E,pts1,pts2):
    try:
        U,S,V = np.linalg.svd(E)
    except:
        return None
    dig=np.array([[1,0,0],[0,1,0],[0,0,0]],dtype=np.float32)
    E=U.dot(dig).dot(V) 
    U,S,V = np.linalg.svd(E)

    if np.linalg.det(U)<0:
        U=-U
    if np.linalg.det(V)<0:
        V=-V

    W=np.array([[0,-1,0],[1,0,0],[0,0,1]],dtype=np.float32)
    R1=U.dot(W).dot(V)
    R2=U.dot(W.T).dot(V)
    T1=normal(U[:,2])
    T2=-T1

    def checkDir(P1,P2):
        pts=cv2.triangulatePoints(P1,P2,pts1,pts2)
        pts/=pts[3]
        pts1r=P1 @ pts
        pts2r=P2 @ pts
        #print(sum(pts1r[2]>0)/len(pts1r[2]))
        return sum(pts1r[2]>0)>len(pts1r[2])/2 and sum(pts2r[2] >0) > len(pts2r[2])/2
    def getErr(P1,P2):
        pts=cv2.triangulatePoints(P1,P2,pts1,pts2)
        pts/=pts[3]
        pts1r=P1 @ pts
        pts2r=P2 @ pts
        pts1r/=pts1r[2]
        pts2r/=pts2r[2]        
        return np.linalg.norm(pts1r[:2]-pts1)+np.linalg.norm(pts2r[:2]-pts2)
    Ps=[]
    Rs=[R1,R2]
    Ts=[T1,T2]
    for R in Rs:
        for T in Ts:
            Ps.append(np.hstack((R,T.reshape(3,1))))
    P0=np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
    ],dtype=np.float32)

    #return min([(P[0][0],i,P) for i,P in enumerate(Ps)])[2]

    for P in Ps:
        if checkDir(P0,P):
            return P
    
    

def calcRT(K1,K2,pts1,pts2,method=cv2.FM_LMEDS):
    mask=(pts1[2]>0) & (pts2[2]>0)
    pts1=pts1[:,mask]
    pts2=pts2[:,mask]

    pts1=np.linalg.inv(K1) @ pts1
    pts2=np.linalg.inv(K2) @ pts2
    pts1/=pts1[2]
    pts2/=pts2[2]
    E,mask=cv2.findFundamentalMat(pts1[:2].T,pts2[:2].T, method)
    #E=K2.T @ F @ K1
    return getRTfromE(E,pts1[:2],pts2[:2])

def getCandiRT(K1,K2,pts1,pts2):
    mask=(pts1[2]>0) & (pts2[2]>0)
    pts1=pts1[:,mask]
    pts2=pts2[:,mask]

    pts1=np.linalg.inv(K1) @ pts1
    pts2=np.linalg.inv(K2) @ pts2
    pts1/=pts1[2]
    pts2/=pts2[2]
    E,mask=cv2.findFundamentalMat(pts1[:2].T,pts2[:2].T, method=cv2.FM_RANSAC)
    U,S,V = np.linalg.svd(E)
    dig=np.array([[1,0,0],[0,1,0],[0,0,0]],dtype=np.float32)
    E=U.dot(dig).dot(V) 
    U,S,V = np.linalg.svd(E)


    if np.linalg.det(U)<0:
        U=-U
    if np.linalg.det(V)<0:
        V=-V

    W=np.array([[0,-1,0],[1,0,0],[0,0,1]],dtype=np.float32)
    R1=U.dot(W).dot(V)
    R2=U.dot(W.T).dot(V)
    R3=-R1
    R4=-R2
    T1=normal(U[:,2])
    T2=-T1
    Ps=[]
    Rs=[R1,R2]    
    Ts=[T1,T2]
    for R in Rs:
        for T in Ts:
            Ps.append(np.hstack((R,T.reshape(3,1))))
    return Ps

def getPositiveRT(K1,K2,pts1,pts2):
    Ps=getCandiRT(K1,K2,pts1,pts2)
    mask=(pts1[2]>0) & (pts2[2]>0)
    pts1=pts1[:,mask]
    pts2=pts2[:,mask]

    pts1=np.linalg.inv(K1) @ pts1
    pts2=np.linalg.inv(K2) @ pts2
    pts1/=pts1[2]
    pts2/=pts2[2]
    P0=np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
    ],dtype=np.float32)
    def checkDir(P1,P2):
        pts=cv2.triangulatePoints(P1,P2,pts1[:2],pts2[:2])
        pts/=pts[3]
        pts1r=P1 @ pts
        pts2r=P2 @ pts
        return sum(pts1r[2]>0)>len(pts1r[2])/2 and sum(pts2r[2] >0) > len(pts2r[2])/2
    rec=[]
    for P in Ps:
        if checkDir(P0,P):
            rec.append(P)
    return rec

t1=0
t2=0
t3=0
def RTEtest():
    R=cv2.Rodrigues(np.random.random((3,1)))[0]
    t=normal(np.random.random((3))).tolist()
    T=np.array([
        [0,-t[2],t[1]],
        [t[2],0,-t[0]],
        [-t[1],t[0],0]
    ])
    
    P1=np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
    ],dtype=np.float32)
    P2=np.hstack((R,np.array(t).reshape(3,1)))
    #print(P2)


    def getRandK():
        kps=np.random.random((4)).tolist()
        return np.array([
            [kps[0],0,kps[1]],
            [0,kps[2],kps[3]],
            [0,0,1]
        ])

    K1=getRandK()
    K2=getRandK()
    pts=np.random.random((4,20))
    pts[3]=1

    pts1=K1@P1@pts
    pts2=K2@P2@pts
    #pts1=P1@pts
    #pts2=P2@pts
    

    if USE_ESS:
        pts1=np.linalg.inv(K1) @ pts1
        pts2=np.linalg.inv(K2) @ pts2

    mask=pts2[2]>=0

    pts=pts[:,mask]
    pts2=pts2[:,mask]
    pts1=pts1[:,mask]

    pts1/=pts1[2]
    pts2/=pts2[2]
    
    global t3
    E=T.dot(R)

    E,mask=cv2.findFundamentalMat(pts1[:2].T,pts2[:2].T, method=cv2.FM_LMEDS)



    if USE_ESS:
        Pt1=getRTfromE(E,pts1[:2],pts2[:2])
    else:
        Pt1=calcRT(K1,K2,pts1,pts2)

    #print('----')
    

    #Pt2=getRTfromE(E,pts1[:2],pts2[:2])

    #print(((Pt1@pts)/((Pt1@pts)[2]))[:,2])
    #print(((Pt2@pts)/((Pt2@pts)[2]))[:,2])
    #print(((P2@pts)/((P2@pts)[2]))[:,2])

    #print(Pt1)

    global t1,t2
    if Pt1 is None:
        t3+=1
    elif np.linalg.norm(Pt1-P2)<0.1 :
        t1+=1
    else:
        t2+=1

def offset(E1,E2):
    r1,r2=cv2.Rodrigues(E1[:,:3])[0],cv2.Rodrigues(E2[:,:3])[0]
    t1,t2=E1[:,3],E2[:,3]
    return np.linalg.norm(r1-r2),np.linalg.norm(t1-t2)

def batch_evaluation(E,pts0,pts1):
    P0=np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
    ],dtype=np.float32)
    pts=cv2.triangulatePoints(P0,E,pts0[:2],pts1[:2])
    pts/=pts[3]
    pts0r=P0 @ pts
    pts1r=E @ pts
    pts0r/=pts0r[2]
    pts1r/=pts1r[2]
    return np.linalg.norm(pts0r-pts0,axis=0)+np.linalg.norm(pts1r-pts1,axis=0)


def lerp(t1,t2,theta):
    return t1+(t2-t1)*theta

def lerp_R(R1,R2,theta):
    import cv2
    r1=cv2.Rodrigues(R1)[0]
    r2=cv2.Rodrigues(R2)[0]
    return cv2.Rodrigues(lerp(r1,r2,theta))[0]






from abc import  abstractmethod
class PallCalculator:
    def __init__(self,K0,K1) -> None:
        self.K0=K0
        self.K1=K1

    def changeK(self,K0,K1):#for debug
        self.K0=K0
        self.K1=K1

    @abstractmethod
    def calc(self,points0,points1):
        pass

class LerpPallCalculator(PallCalculator):
    '''Use points over a period of time to calculate Pall

    Save some previous points to imporve performence,have a queue with size pCache to save points
    every time user call calc , points will add to queue and calculate P by calcRT ,then will perform
    linear interpolation with theta on the previous P and the current 
    
    '''


    def __init__(self,K0,K1,pCache,theta_T,theta_R) -> None:
        super().__init__(K0, K1)
        self.pCache=pCache
        self.theta_T=theta_T
        self.theta_R=theta_R
        self._pool0=np.zeros((3,0))
        self._pool1=np.zeros((3,0))
        self._preP=None
    def calc(self,points0,points1):
        self._pool0=np.hstack((self._pool0,points0))
        self._pool1=np.hstack((self._pool1,points1))
        if self._pool0.shape[1]>self.pCache:
            self._pool0=self._pool0[:,-self.pCache:]
        if self._pool1.shape[1]>self.pCache:
            self._pool1=self._pool1[:,-self.pCache:]

        P=calcRT(self.K0,self.K1,self._pool0,self._pool1)
        if P is None:
            return self._preP
        if not self._preP is None:
            P[:,:3]=lerp_R(self._preP[:,:3],P[:,:3],self.theta_R)
            P[:,3]=lerp(self._preP[:,3],P[:,3],self.theta_T)
        self._preP=P
        return P    

class BrutePallCalculator(PallCalculator):
    def __init__(self,K0,K1) -> None:
        super().__init__(K0, K1)
    def calc(self,points0,points1):
        P=calcRT(self.K0,self.K1,points0,points0)
        return P    

class MoviePallCalculator(PallCalculator):
    def __init__(self,K0,K1,eCache,pCache,theta) -> None:
        super().__init__(K0, K1)
        self.eCahce=eCache
        self.pCache=pCache
        self.theta=theta
        self._pool0=np.zeros((3,0))
        self._pool1=np.zeros((3,0))
        self._Es=[]
        self._preP=None
    
    def _get_center(self):
        Ts=[E[:,3].reshape(3,1) for E in self._Es]
        Rs=[E[:,:3] for E in self._Es]
        rs=[cv2.Rodrigues(R)[0] for R in Rs]
        #print([t[2][0] for t in sorted(Ts,key=lambda t : t[2])])
        def get_mid_in_3d(pos):
            l=len(pos)
            return [sorted(pos,key= lambda p : p[i])[l//2][i] for i in range(3)]
        def get_center_in_3d(pos):
            l=len(pos)
            def get_legel_pos(x,a):
                distances=[abs(x-i) for i in a]
                #print(sorted(distances))
                thr=sorted(distances)[l//2]  # an easy implement now
                return [abs(x-i)<=thr for i in a]
            def func0():#return the mid point
                return get_mid_in_3d(pos)
            def func1():#make each dimension independent
                x = get_mid_in_3d(pos)
                rec=[]
                for i in range(3):
                    cnt=0
                    ci=0
                    for j,cal in enumerate(get_legel_pos(x[i],[p[i] for p in pos])):
                        if cal:
                            cnt+=1
                            ci+=pos[j][i] 
                    rec.append(ci/cnt)
                return  rec
            def func2():
                x=get_mid_in_3d(pos)
                rec=[0,0,0]
                cnt=0
                for i, cal1,cal2,cal3 in enumerate(list(zip([get_legel_pos(x[i],[p[i] for p in pos]) for i in range(3)]))):
                    if cal1 and cal2 and cal3:
                        cnt+=1
                        for j in range(3):
                            rec[j]+=pos[i][j]
                for j in range(3):
                    rec[j]/=cnt
                return rec
            def func3():
                sums=[]
                for i in range(l):
                    s=[]
                    for j in range(l):
                        if i!=j:
                            s.append(np.linalg.norm(pos[i]-pos[j]))
                    sums.append((sum(sorted(s)[:len(s)//2]),i))
                rec=[0,0,0]
                select_num=(1+l)//2
                for val,i in sorted(sums)[:select_num]:
                    for j in range(3):
                        rec[j]+=pos[i][j]
                rec=[v/select_num for v in rec]
                
                return rec
            if l==1:
                return np.array(pos[0])
            return np.array(func3())
        Tc=get_center_in_3d(Ts)
        rc=get_center_in_3d(rs)
        Rc=cv2.Rodrigues(rc)[0]
        return np.hstack((Rc,Tc))
 
    def calc(self,points0,points1):
        self._pool0=np.hstack((self._pool0,points0))
        self._pool1=np.hstack((self._pool1,points1))
        if self._pool0.shape[1]>self.pCache:
            self._pool0=self._pool0[:,-self.pCache:]
        if self._pool1.shape[1]>self.pCache:
            self._pool1=self._pool1[:,-self.pCache:]
        P=calcRT(self.K0,self.K1,self._pool0,self._pool1)
        if P is  not None:
            self._Es.append(P)
        
        if len(self._Es)>self.eCahce:
            self._Es=self._Es[-self.eCahce:]
        if len(self._Es)==0:
            return None
        P=self._get_center()
        if self._preP is None:
            self._preP=P
        if len(self._Es)==self.eCahce:
            P[:,:3]=lerp_R(self._preP[:,:3],P[:,:3],self.theta)
            P[:,3]=lerp(self._preP[:,3],P[:,3],self.theta)
        self._preP=P
        return P

DEBUG=True

class FlipBucketPallCalculator(PallCalculator):
    def __init__(self,K0,K1,pCache,theta_T,theta_R) -> None:
        super().__init__(K0, K1)
        self.pCache=pCache
        self.theta_T=theta_T
        self.theta_R=theta_R
        self._start()
        if DEBUG:
            self.s=[]

    def _start(self):
        self._pool0=np.zeros((3,0))
        self._pool1=np.zeros((3,0))
        self._preP=None
        self._bucket=0
        self._bucketTime=0
        self._calcCnt=0
    def  _flip(self):
        if DEBUG:
            print('fliping')
        self._start()
        return None

    def calc(self,points0,points1):
        self._pool0=np.hstack((self._pool0,points0))
        self._pool1=np.hstack((self._pool1,points1))
        
        if self._pool0.shape[1]>self.pCache:
            self._pool0=self._pool0[:,-self.pCache:]
        if self._pool1.shape[1]>self.pCache:
            self._pool1=self._pool1[:,-self.pCache:]

        P=calcRT(self.K0,self.K1,self._pool0,self._pool1)
        if P is None:
            return self._preP

        self._calcCnt+=1
        flag=not self._preP is None
        if not self._preP is None and self._pool0.shape[1]==self.pCache and self._calcCnt>20:
            if DEBUG:
                #print(offset(P,self._preP))
                self.s.append(offset(P,self._preP))
            dR,dT=offset(P,self._preP)
            
            if   dR>=0.5 or dT>=0.6: #select the parameter when debug
                print(self._calcCnt-self._bucketTime)
                if self._calcCnt-self._bucketTime >= 100:
                    #flip the bucket if its amount reach pCache
                    return self._flip()
                else:
                    P=self._preP
                self._bucket+=1
                flag=False
            else:
                self._bucket = max(self._bucket-1,0)
                if self._bucket==0:
                    self._bucketTime=self._calcCnt
        if flag:
            P[:,:3]=lerp_R(self._preP[:,:3],P[:,:3],self.theta_R)
            P[:,3]=lerp(self._preP[:,3],P[:,3],self.theta_T)
        self._preP=P
        return P    

    def  __del__(self):
        if DEBUG:
            import matplotlib.pyplot as plt
            xs=[a[0] for a in self.s]
            ys=[a[1] for a in self.s]
            for i,j in self.s:
                plt.scatter(xs,ys)
            plt.show()
        else:
            pass











if __name__=='__main__':
    for i in range(100):
        RTEtest()
    print(t1,t2,t3)

