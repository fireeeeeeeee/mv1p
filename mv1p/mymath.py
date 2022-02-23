import numpy as np
from numpy.linalg.linalg import lstsq
import random

def GetTx(t):
    '''
        A:3 dim vector
        B:3 dim vector
        A x B = Ax @ B
        return Ax
    '''
    return np.array([
        [0,-t[2],t[1]],
        [t[2],0,-t[0]],
        [-t[1],t[0],0]
    ])

class LinearSolver:

    def __init__(self,n):
        self.n=n
        self.A=np.zeros((0,n))
        self.b=np.zeros(0)
    def Add(self,A,b):
        self.A=np.vstack((self.A,A))
        self.b=np.append(self.b,b)
    def AddByIndex(self,i_v,b):
        A=np.zeros(self.n)
        for i,v in i_v:
            A[i]=v
        self.Add(A,b)
    def Solve(self):
        return np.linalg.lstsq(self.A, self.b,rcond=-1)[0]



if __name__=='__main__':
    ls = LinearSolver(2)
    for i in range(10):
        for j in range(10):
            x=i
            y=2*j
            z=x+y+random.random()*0.01
            ls.Add(np.array([i,j],dtype=np.float32),z)
    print(ls.Solve())
