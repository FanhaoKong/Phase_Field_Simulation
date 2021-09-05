#phase field simulation (prototype)
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab
from scipy import interpolate

info={'alpha':1.0,'beta':1.0,'gamma':1.0,'LP':1.0,'K':0.5,'nump':4,'size':50, \
      'timestep':0.1,'spacestep':1.0,'rounds':2000}
ratio=5
#nump: 序参量个数  size:模拟点阵大小 

def space():
    nump,size=info['nump'],info['size']
    #初始化序参量存储空间
    orderinitial=np.zeros((size,size,nump))
    for i in range(size):
        for j in range(size):
            q=np.random.randint(0,nump)
            orderinitial[i][j][q]=np.random.randn()*0.001
    #print(orderinitial)
    return orderinitial

def ordergrad2(orderfield):
    #对序参量场求两次梯度
    nump,size=info['nump'],info['size']
    ordergrad2=np.zeros((size,size,nump))
    for k in range(nump):
        for i in range(size):
            for j in range(size):
                left=i-1;right=i+1;up=j+1;down=j-1;
                if i==0:
                    left=size-1
                if i==size-1:
                    right=0
                if j==0:
                    down=size-1
                if j==size-1:
                    up=0
                ordergrad2[i][j][k]=(orderfield[right][j][k]+orderfield[left][j][k]-2*orderfield[i][j][k]) \
                    +(orderfield[i][up][k]+orderfield[i][down][k]-2*orderfield[i][j][k])
    return ordergrad2


def nextfield(currentfield,ordergrad2):
    nump,size=info['nump'],info['size']
    alpha,beta,gamma,LP,K=info['alpha'],info['beta'],info['gamma'],info['LP'],info['K']
    timestep=info['timestep']
    dndt=np.zeros((size,size,nump))
    nextfield=np.zeros((size,size,nump))
    suma=0.0
    for k in range(nump):
        for i in range(size):
            for j in range(size):
                AA=beta*(-1.0*(alpha**2)*currentfield[i][j][k]+currentfield[i][j][k]**3)
                for q in range(nump):
                    if q==k:
                        continue
                    suma=suma+currentfield[i][j][q]**2
                BB=2*gamma*currentfield[i][j][k]*suma
                suma=0.0
                CC=K*ordergrad2[i][j][k]
                dndt[i][j][k]=-LP*(AA+BB-CC)
                nextfield[i][j][k]=currentfield[i][j][k]+dndt[i][j][k]*timestep
    return nextfield

def display(currentfield):
    nump,size=info['nump'],info['size']
    x,y=np.mgrid[0:size:1,0:size:1]
    Value=np.zeros((size,size))
    Value2=np.zeros((ratio*size,ratio*size))
    base=np.zeros((ratio*size,ratio*size))
    for i in range(size):
        for j in range(size):
            for k in range(nump):
                Value[i][j]=Value[i][j]+currentfield[i][j][k]**2
    Value=Value*size/10.0
    newfunc=interpolate.interp2d(x,y,Value,kind='cubic')
    x1,y1=np.mgrid[0:size:(1.0/ratio),0:size:(1.0/ratio)]
    for i in range(ratio*size):
        for j in range(ratio*size):
            Value2[i][j]=newfunc(x1[i][0],y1[0][j])
            if Value2[i][j]>size/10.0:
                Value2[i][j]=size/10.0
            if Value2[i][j]<0.0:
                Value2[i][j]=0.0
    mlab.clf()
    mlab.mesh(x1,y1,base,colormap='autumn')
    mlab.mesh(x1,y1,Value2,colormap='autumn')
    f=mlab.gcf()
    f.scene._lift()
    return mlab.screenshot()
    
def main():
    Currentfield=space()
    display(Currentfield)
    for m2 in range(info['rounds']):
        print(m2,end='    ')
        Ordergrad2=ordergrad2(Currentfield)
        print(np.max(Currentfield), end='   ')
        print(np.min(Currentfield))
        Currentfield=nextfield(Currentfield,Ordergrad2)
        if np.max(Currentfield)>10.0:
            break
        if m2%100==0 and m2!=0:
            display(Currentfield)
    
main()


#orderinitial=space()
#print(ordergrad2(orderinitial))















    


