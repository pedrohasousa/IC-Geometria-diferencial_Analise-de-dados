# Calculo de geodésicas de um cilindro através do método de Euler

import matplotlib.pyplot as plt
import numpy as np


## Parametrização do cilindro
raio=1
altura=6
precisao=100
u=np.linspace(0,2*np.pi,precisao)
v=np.linspace(0,altura,precisao)
x=raio*np.outer(np.cos(u),np.ones(precisao))
y=raio*np.outer(np.sin(u),np.ones(precisao))
z=np.outer(np.ones(precisao),v)


## Aplicação do método
divisoes=100
t=np.linspace(0,300,divisoes)
p0=(-0.5*np.pi,1)     #u de 0 a 2pi, v de 0 até 'altura'
direcao=(1,1)

h=(t[1]-t[0])/divisoes
u0,v0=p0
Du,Dv=direcao
x1=[np.cos(u0)]
y1=[np.sin(u0)]
z1=[v0]
for ti in t:
    u0=u0+h*Du
    v0=v0+h*Dv
  # aqui usamos as eqs das geodésicas,mas D_2u=D_2v=0, então Du e Dv são constantes 
  # como as eqs são identicamente nulas, o método fornece as geodésicas exatas

    x1.append(np.cos(u0))
    y1.append(np.sin(u0))
    z1.append(v0)


## Geodésica exata para verificação
u1=p0[0]
v1=p0[1]
p1=(np.cos(u1),np.sin(u1),v1)
u2=u0
v2=v0
p2=(np.cos(u2),np.sin(u2),v2)
t=np.linspace(0,1,100)
x2=np.cos((u2-u1)*t+u1)
y2=np.sin((u2-u1)*t+u1)
z2=(v2-v1)*t+v1


## Plot
ax=plt.axes(projection='3d')
ax.plot_surface(x,y,z,alpha=0.7, color='white')
ax.plot(x1,y1,z1, color='y')
ax.plot(x2,y2,z2, color='r')
ax.scatter(p1[0],p1[1],p1[2],color='black')
ax.scatter(p2[0],p2[1],p2[2],color='orange')

ax.set_xlabel('x',color='r')
ax.set_ylabel('y',color='g')
ax.set_zlabel('z',color='b')
ax.set_aspect('equal')
plt.show()
