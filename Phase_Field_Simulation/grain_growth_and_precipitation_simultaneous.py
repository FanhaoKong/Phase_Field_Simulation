import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

#nump：序参量个数  size：分辨率
nump=10;size=50;rounds=4000
#fbe：平衡时第二相的体积分数
fbe=0.3;#0.3
spacestep=2.0;timestep=0.37;Calpha=0.05;Cbeta=0.95;
A=1.0;B=7.29;Dalpha=1.23;Dbeta=1.23;delta=1.0;epsilon=3.0;
Cm=(Calpha+Cbeta)/2.0
ki=1.0;kc=2.0;Li=1;D=0.5;
gamma=delta/(pow((Cbeta-Calpha),2))

def initial():
    C0=Calpha*(1-fbe)+Cbeta*fbe
    coninitial=np.full((size,size),C0)
    orderinitial=np.zeros((size,size,nump))
    for i in range(size):
        for j in range(size):
            q=np.random.randint(0,nump)
            orderinitial[i][j][q]=np.random.randn()*0.01
            #coninitial[i][j]=coninitial[i][j]+np.random.randn()*0.01
    return coninitial,orderinitial

def nextfield(currentcon,currentorder):
    dcdt=np.zeros((size,size))
    dfdc=np.zeros((size,size))
    cgrad2=np.zeros((size,size))
    cgrad4=np.zeros((size,size))
    dndt=np.zeros((size,size,nump))
    suma=0.0
    for q in range(nump+2):
        for i in range(size):
            for j in range(size):
                k=q-2
                left=i-1;right=i+1;up=j+1;down=j-1;
                if i==0:
                    left=size-1
                if i==size-1:
                    right=0
                if j==0:
                    down=size-1
                if j==size-1:
                    up=0
                if k>=0:
                    ograd2=(0.5*(currentorder[right][j][k]+currentorder[left][j][k]+ \
                            currentorder[i][up][k]+currentorder[i][down][k]-4*currentorder[i][j][k]) \
                        +0.25*(currentorder[right][up][k]+currentorder[left][up][k]+ \
                        currentorder[left][down][k]+currentorder[right][down][k]-4*currentorder[i][j][k]))/(spacestep**2)
                if q==0:
                    cgrad2[i][j]=(0.5*(currentcon[right][j]+currentcon[left][j]+ \
                            currentcon[i][up]+currentcon[i][down]-4*currentcon[i][j]) \
                        +0.25*(currentcon[right][up]+currentcon[left][up]+ \
                        currentcon[left][down]+currentcon[right][down]-4*currentcon[i][j]))/(spacestep**2)
                if q==1:
                    cgrad4[i][j]=(0.5*(cgrad2[right][j]+cgrad2[left][j]+ \
                            cgrad2[i][up]+cgrad2[i][down]-4*cgrad2[i][j]) \
                        +0.25*(cgrad2[right][up]+cgrad2[left][up]+ \
                        cgrad2[left][down]+cgrad2[right][down]-4*cgrad2[i][j]))/(spacestep**2)
                if k>=0:
                    AA=-gamma*((currentcon[i][j]-Calpha)**2+(currentcon[i][j]-Cbeta)**2)*currentorder[i][j][k] \
                        +delta*pow(currentorder[i][j][k],3)
                    CC=ki*ograd2
                    for m1 in range(nump):
                        suma=suma+currentorder[i][j][m1]**2
                    BB=epsilon*currentorder[i][j][k]*(suma-currentorder[i][j][k]**2)
                    dndt[i][j][k]=-Li*(AA+BB-CC)
                
                if k==0:
                    DD=-gamma*(2*currentcon[i][j]-Calpha-Cbeta)*suma
                    dfdc[i][j]=-A*(currentcon[i][j]-Cm)+B*pow((currentcon[i][j]-Cm),3) \
                        +Dalpha*pow((currentcon[i][j]-Calpha),3)+Dbeta*pow((currentcon[i][j]-Cbeta),3)+DD
                if k==1:
                    dfdcgrad2=(0.5*(dfdc[right][j]+dfdc[left][j]+ \
                            dfdc[i][up]+dfdc[i][down]-4*dfdc[i][j]) \
                        +0.25*(dfdc[right][up]+dfdc[left][up]+ \
                        dfdc[left][down]+dfdc[right][down]-4*dfdc[i][j]))/(spacestep**2)
                    dcdt[i][j]=D*(dfdcgrad2-kc*cgrad4[i][j])
                suma=0.0
    currentcon=currentcon+timestep*dcdt
    currentorder=currentorder+timestep*dndt
    return currentcon,currentorder
                    
def display(currentcon,currentorder,step):
    x,y=np.mgrid[0:size:1,0:size:1]
    #Valuecon=np.zeros((size,size))
    Valueorder1=np.zeros((size,size))
    Valueorder2=np.zeros((size,size))
    base=np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            for k in range(nump):
                Valueorder1[i][j]=Valueorder1[i][j]+currentorder[i][j][k]**2
            if currentcon[i][j]>0.7:
                Valueorder2[i][j]=Valueorder1[i][j]
    for i in range(size):
        for j in range(size):
            minimum=np.min(Valueorder1)
            if currentcon[i][j]>0.7:
                Valueorder1[i][j]=minimum
            else:
                Valueorder2[i][j]=minimum
    Valueorder1=Valueorder1*size/10.0;Valueorder2=Valueorder2*size/10.0
    cover=np.full((size,size),minimum*size/10.0+0.1)
    mlab.clf()
    mlab.mesh(x,y,Valueorder1,colormap='autumn')
    mlab.mesh(x,y,Valueorder2,colormap='cool')
    mlab.mesh(x,y,base,colormap='autumn')
    mlab.mesh(x,y,cover,colormap='autumn')
    f=mlab.gcf()
    f.scene._lift()
    return mlab.screenshot()

def main():
    currentcon,currentorder=initial()
    display(currentcon,currentorder,0)
    for m2 in range(rounds):
        print(m2,end='    ')
        print(np.max(currentorder), end='   ')
        print(np.min(currentorder), end='   ')
        print(np.max(currentcon), end='   ')
        print(np.min(currentcon))
        currentcon,currentorder=nextfield(currentcon,currentorder)
        if (m2+1)%100==0:
            display(currentcon,currentorder,m2)
    
main()




'''
con,order=initial()
con2,order2=nextfield(con,order)
display(con2,order2)
'''












