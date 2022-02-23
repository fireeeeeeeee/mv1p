from re import T
from time import strftime
from cv2 import RETR_CCOMP, data
import numpy as np
import cv2
from numpy.random.mtrand import rand


def transfer(S1, S2,SRT=False):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrustes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    if SRT:
        return S1_hat,scale,R,t
    else:
        return S1_hat

def mpjpe(p1,p2):
    return np.sqrt(np.power(p1 - p2, 2).sum(axis=-1))

def p_mpjpe(p1,p2):
    return mpjpe(transfer(p1,p2),p2)

def addOne(pts):
    return np.vstack((pts,np.ones((1,pts.shape[1]))))



def StandardDeviation(xs):
    if len(xs)<=1 :
        return 0   
    x_hat=sum(xs)/len(xs)
    return   np.sqrt(sum([(x-x_hat)*(x-x_hat) for x in xs ])/(len(xs)))


class ErrorCalculator:
    def __init__(self,name) -> None:
        self.pose_errs=[]
        self.scale_deviations=[]
        self.R_deviations=[]
        self.t_deviations=[]
        self.name=name
        self.nPoints=0
        self.addCnt=1
    def add(self,p_truth,p_estimate):
        n=len(p_truth)
        if n==0:
            return
        self.addCnt+=1
        p_trans,scale,R,t=transfer(p_estimate,p_truth,True)
        self.pose_errs.append(sum(mpjpe(p_trans,p_truth))/n)
        self.scale_deviations.append(scale)
        self.R_deviations.append(R)
        self.t_deviations.append(t)
    def calc(self):
        Devs=[StandardDeviation(xs) for xs in [self.scale_deviations,self.R_deviations,self.t_deviations]]
        return sum(self.pose_errs)/self.addCnt,Devs
    def calcAndShow(self):
        err,devs=self.calc()
        print('error in '+self.name+':')
        print('pose err:')
        print(err)
        print('scale standard deviation:')
        print(devs[0])
        print('R standard deviation:')
        print(devs[1])
        print('t standard deviation:')
        print(devs[2])
        



    






pr=[]

def delayPrint(arr):
    global pr
    out=True
    if len(pr)==0:
        out=False
    pr.extend(arr)
    if out:
        l=len(pr)
        for i in range(l//2):
            print(pr[i])
            print(pr[i+l//2])
            print(pr[i]==pr[i+l//2])
            print('-------')
    


def get_error(calculator,video_points2d,pts_gt,error_func):
    """
        video_points2d:(nFrames,nViews(2),2,nPoints)
    """
    cnt=1
    err=0
    failTime=0
    for points2d,gt in list(zip(video_points2d,pts_gt)) :

        P1=calculator.calc(addOne(points2d[0]),addOne(points2d[1]))
        if P1 is None:
            failTime+=1
            continue
        cnt+=1
        P0=np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
        ],dtype=np.float32)
        pts_e=cv2.triangulatePoints(P0,P1,points2d[0],points2d[1])
        pts_e/=pts_e[3]
        #delayPrint([P0,P1,points2d[0],points2d[1],pts_e[:3],gt])
        err+=error_func(pts_e[:3],gt)
    return err/cnt,failTime


def normal(x):
    return x/np.linalg.norm(x)

def randP():
    R=cv2.Rodrigues(np.random.random((3,1))-0.5)[0]
    T=np.random.random(3).reshape(3,1)
    return np.hstack((R,T))


def randLegelP(pts):
    pts=addOne(pts)
    while True:
        global trycnt

        P=randP()
        pt2d=P@pts
        if min(pt2d[2])>0:
            return P
        
        
        



def randData(nViews,nPoints,noise_len=0.1):
    gt=np.random.rand(3*nPoints).reshape(3,nPoints)
    Ps=[]
    Ps.append(np.array([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
    ],dtype=np.float32))
    for i in range(nViews-1):
        Ps.append(randLegelP(gt))
    gt_o=addOne(gt)
    pts2d=[]
    for P in Ps:
        pts=P@gt_o
        pts/=pts[2]
        noise=(np.random.rand(2*nPoints).reshape(2,nPoints)-0.5)*2*noise_len
        pts2d.append(pts[:2]+noise)

    
    p3d=cv2.triangulatePoints(Ps[0],Ps[1],pts2d[0][:,0:20],pts2d[1][:,0:20])
    p3d/=p3d[3]
    #delayPrint([Ps[0],Ps[1],pts2d[0][:,0:20],pts2d[1][:,0:20],p3d[:3],gt[:,0:20]])
    # print(p3d[:3]-gt[:,0:20])
    #print(pts2d[0][:,0:20])
    #print(gt[:,0:20])
    return pts2d,gt,Ps


def getDataFromDataset(sub):
    from easymocap.mytools import load_parser, parse_parser
    from easymocap.dataset import CONFIG, MV1PMF
    parser = load_parser()
    args = parse_parser(parser)
    datapath='C:/Users/AAAA/Desktop/ee/data'
    out=datapath+"/output/smpl"
    dataset = MV1PMF(datapath, annot_root=args.annot, cams=sub, out=out,
        config=CONFIG[args.body], kpts_type=args.body,
        undis=True, no_img=False, verbose=args.verbose)
    start, end = args.start, min(args.end, len(dataset))
    video_points2d=[]
    video_gt=[]
    for nf in range(start,end):
        images, annots = dataset[nf]
        

if __name__=="__main__":
    from epipolar import LerpPallCalculator,BrutePallCalculator,MoviePallCalculator,FlipBucketPallCalculator,batch_evaluation,calcRT
    pts2d,gt,Ps=randData(2,10000,0.01)
    video_points2d=[]
    video_gt=[]
    for i in range(500):
        video_points2d.append([p[:,i:i+20] for p in pts2d])
        video_gt.append(gt[:,i:i+20])
    K=np.array([
        [1,0,0], 
        [0,1,0],
        [0,0,1],
    ],dtype=np.float32)
    print(get_error(LerpPallCalculator(K,K,500,0.01,0.01),video_points2d,video_gt,p_mpjpe))
    print(get_error(FlipBucketPallCalculator(K,K,500,0.01,0.01),video_points2d,video_gt,mpjpe))
    





