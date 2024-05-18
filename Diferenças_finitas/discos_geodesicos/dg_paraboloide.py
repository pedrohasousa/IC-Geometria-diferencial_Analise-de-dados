# Cálculo do disco geodésico do parabolóide elíptico através do método de Runge-Kutta de quarta ordem

import numpy as np
import matplotlib.pyplot as plt


## Equações das Geodésicas do Parabolóide
def D_2u(t,u,v,Du,Dv):
  return (-Du**2*4*u+Dv**2*u)/(4*u**2+1)  #COLOCAR a E b

def D_2v(t,u,v,Du,Dv):
  return -2*Du*Dv/u


## Parametrização do parabolóide
a=1
b=1
u=np.linspace(0,5,100)
v=np.linspace(0,2*np.pi,100)
x=np.outer(a*u,np.cos(v))
y=np.outer(b*u,np.sin(v))
z=np.outer(u**2,np.ones(100))




## Plot
ax=plt.axes(projection='3d')
ax.plot_surface(x,y,z,alpha=0.7,color='white')




## Criação do Disco Geodésico
iteracoes=100
p0=(3,0.5)
h=0.01
numero_geo=36
for i in range(numero_geo):
  direcao=(np.sin(2*i*np.pi/numero_geo),np.cos(2*i*np.pi/numero_geo))

  t=-h
  u0,v0=p0
  Du,Dv=direcao
  x1=[u0*np.cos(v0)]
  y1=[u0*np.sin(v0)]
  z1=[u0**2]
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

    x1.append(u0*np.cos(v0))
    y1.append(u0*np.sin(v0))
    z1.append(u0**2)

    ax.plot(x1,y1,z1,color='red')
    
    if k==iteracoes-1:
      ax.scatter(x1[k],y1[k],z1[k],color='black')
    


#ax.set_aspect('equal')
plt.show()