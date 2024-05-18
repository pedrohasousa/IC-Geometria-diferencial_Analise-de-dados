# Plano e suas geodésicas (segmentos de reta)

import matplotlib.pyplot as plt
import numpy as np

## Parametrização do plano
a=3
b=1
u=np.linspace(0,10,100)
v=np.linspace(0,10,100)
x=np.outer(u,np.ones(100))
y=np.outer(np.ones(100),v)
z=np.outer(a*u*np.ones(100),np.ones(100))+np.outer(np.ones(100),b*v*np.ones(100))

## Parametrização da geodésica
p1=(1,3) # pontos no domínio
p2=(4,6)
x1=[p1[0],p2[0]]
y1=[p1[1],p2[1]]
z1=[a*p1[0]+b*p1[1],a*p2[0]+b*p2[1]]


## Plot
ax=plt.axes(projection='3d')
ax.plot_surface(x,y,z,alpha=0.7, color='white')
ax.plot(x1,y1,z1,color='r')
ax.scatter(p1[0],p1[1],z1[0],color='black')
ax.scatter(p2[0],p2[1],z1[1],color='orange')

ax.set_xlabel('x',color='r')
ax.set_ylabel('y',color='g')
ax.set_zlabel('z',color='b')
#ax.set_aspect('equal')
plt.show()