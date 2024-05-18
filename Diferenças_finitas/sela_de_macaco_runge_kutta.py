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





## Aplicação do Runge-Kutta de quarta ordem
# aplica Runge-Kutta para determinar as primeiras derivadas e aproximação de Taylor de segunda ordem para determinar os pontos da geodésica
iteracoes=600
p0=(0.2,0.1)
h=0.001
direcao=(1,2)

t=-h
u0,v0=p0
Du,Dv=direcao
x1=[u0]
y1=[v0]
z1=[u0**3-3*u0*v0**2]
for _ in range(iteracoes):
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
    
    



## Método de Euler para comparação
t=-h
u0,v0=p0
Du,Dv=direcao
x2=[u0]
y2=[v0]
z2=[u0**3-3*u0*v0**2]
for _ in range(iteracoes):
    t+=h
    u0, v0=u0+h*Du, v0+h*Dv
    Du, Dv=Du+h*D_2u(t,u0,v0,Du,Dv), Dv+h*D_2v(t,u0,v0,Du,Dv)

    x2.append(u0)
    y2.append(v0)
    z2.append(u0**3-3*u0*v0**2)




## Plot
ax=plt.axes(projection='3d')
ax.plot_surface(x,y,z,alpha=0.7,) #surperfície
ax.plot(x2,y2,z2, color='purple') #Euler
ax.plot(x1,y1,z1,color='red') #runge-kutta
ax.scatter(x1[0],y1[0],z1[0],color='black') #ponto inicial
ax.scatter(x1[iteracoes-1],y1[iteracoes-1],z1[iteracoes-1],color='black') #ponto final
#ax.set_aspect('equal')
plt.show()