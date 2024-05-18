# Cálculo do disco geodésico de um cilindro atraves do método de Euler
# Como esse método fornece as geodésicas exatas,não há motivo para aplicar Runge-Kutta

import matplotlib.pyplot as plt
import numpy as np


## Parametrização do cilindro
altura=6
precisao=100
u=np.linspace(0,2*np.pi,precisao)
v=np.linspace(0,altura,precisao)
x=np.outer(np.cos(u),np.ones(precisao))
y=np.outer(np.sin(u),np.ones(precisao))
z=np.outer(np.ones(precisao),v)


## Plot
ax=plt.axes(projection='3d')
ax.plot_surface(x,y,z,alpha=0.7, color='white')




## Criação do disco geodésico
iteracoes=200
p0=(-0.5*np.pi,3)     #u de 0 a 2pi, v de 0 até 'altura'
h=0.01
numero_geo=25

for i in range(numero_geo):
  direcao=(np.cos((2*i*np.pi)/numero_geo),np.sin((2*i*np.pi)/numero_geo))
  t=-h
  u0,v0=p0
  Du,Dv=direcao
  x1=[np.cos(u0)]
  y1=[np.sin(u0)]
  z1=[v0]
  for k in range(iteracoes):
    t+=h
    u0=u0+h*Du
    v0=v0+h*Dv
  # aqui usamos as eqs das geodésicas,mas D_2u=D_2v=0, então Du e Dv são constantes 
  # como as eqs são identicamente nulas, o método fornece as geodésicas exatas

    x1.append(np.cos(u0))
    y1.append(np.sin(u0))
    z1.append(v0)

  ax.plot(x1,y1,z1, color='red')
  if k==iteracoes-1:
    ax.scatter(x1[iteracoes-1],y1[iteracoes-1],z1[iteracoes-1],color='black')


ax.set_xlabel('x',color='g')
ax.set_ylabel('y',color='b')
ax.set_zlabel('z',color='r')
ax.set_aspect('equal')
plt.show()