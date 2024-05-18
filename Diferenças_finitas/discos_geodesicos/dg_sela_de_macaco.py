# Cálculo do disco geodésico da 'sela de macaco' através do método de Runge-Kutta de quarta ordem

import numpy as np
import matplotlib.pyplot as plt

## Equações das Geodésicas da sela de macaco
def D_2u(t,u,v,Du,Dv):
  return -((Du**2-Dv**2)*18*(u**3-u*v**2) + Du*Dv*36*(v**3-u**2*v)) / (9*(u**2+v**2)**2 + 1)
def D_2v(t,u,v,Du,Dv):
  return -((Du**2-Dv**2)*36*u**2*v + Du*Dv*72*u*v**2) / (9*(u**2+v**2)**2 + 1)


## Parametrização da sela de macaco
u=np.linspace(-1,1,100)
v=np.linspace(-1,1,100)
x=np.outer(u,np.ones(100))
y=np.outer(np.ones(100),v)
z=np.outer(u**3,np.ones(100))-np.outer(3*u,v**2)


## Plot
ax=plt.axes(projection='3d')
ax.plot_surface(x,y,z,alpha=0.7,)


## Criação do disco geodésico
iteracoes=600
p0=(0.5,0.5)
h=0.001
numero_geo=15

for i in range(numero_geo):
  direcao=(np.cos(2*i*np.pi/numero_geo),np.sin(2*i*np.pi/numero_geo))
  t=-h
  u0,v0=p0[0],p0[1]
  Du,Dv=direcao[0],direcao[1]
  x1=[u0]
  y1=[v0]
  z1=[u0**3-3*u0*v0**2]
  for k in range(iteracoes):
    t+=h

    u0=u0+h*Du+(h**2)/2*D_2u(0,u0,v0,Du,Dv)
    v0=v0+h*Dv+(h**2)/2*D_2v(0,u0,v0,Du,Dv)
    
    k1_u=D_2u(0,u0,v0,Du,Dv)
    k1_v=D_2v(0,u0,v0,Du,Dv)
    k2_u=D_2u(0,u0+h/2,v0+h/2,Du+h/2*k1_u,Dv+h/2*k1_v)
    k2_v=D_2v(0,u0+h/2,v0+h/2,Du+h/2*k1_u,Dv+h/2*k1_v)
    k3_u=D_2u(0,u0+h/2,v0+h/2,Du+h/2*k2_u,Dv+h/2*k2_v)
    k3_v=D_2v(0,u0+h/2,v0+h/2,Du+h/2*k2_u,Dv+h/2*k2_v)
    k4_u=D_2u(0,u0+h,v0+h,Du+h*k3_u,Dv+h*k3_v)
    k4_v=D_2v(0,u0+h,v0+h,Du+h*k3_u,Dv+h*k3_v)

    Du=Du+(h/6)*(k1_u + 2*k2_u + 2*k3_u + k4_u)
    Dv=Dv+(h/6)*(k1_v + 2*k2_v + 2*k3_v + k4_v)

    x1.append(u0)
    y1.append(v0)
    z1.append(u0**3-3*u0*v0**2)

    ax.plot(x1,y1,z1,color='red')
    if k==iteracoes-1:
      ax.scatter(x1[iteracoes-1],y1[iteracoes-1],z1[iteracoes-1],color='black')


#ax.set_aspect('equal')
plt.show()