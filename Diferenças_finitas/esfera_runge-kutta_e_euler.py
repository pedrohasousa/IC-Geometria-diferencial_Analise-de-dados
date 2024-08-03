## Aproximação de geodésicas de uma esfera através do método de Runge-Kutta de quarta ordem

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors

## Função para aplicar o Runge-Kutta de quarta ordem na esfera
# aplica Runge-Kutta para determinar as primeiras derivadas e aproximação de Taylor de segunda ordem para determinar os pontos da geodésica
def runge_kutta(iteracoes,p0,direcao,h):
  t=-h
  u0,v0=p0
  Du,Dv=direcao
  x1=[np.cos(u0)*np.cos(v0)]
  y1=[np.cos(u0)*np.sin(v0)]
  z1=[np.sin(u0)]
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

    x1.append(np.cos(u0)*np.cos(v0))
    y1.append(np.cos(u0)*np.sin(v0))
    z1.append(np.sin(u0))
  
  return (x1,y1,z1)



## Função para aplicar o método de Euler na esfera
def euler(iteracoes,p0,direcao,h):
  t=-h
  u0,v0=p0
  Du,Dv=direcao
  x2=[np.cos(u0)*np.cos(v0)]
  y2=[np.cos(u0)*np.sin(v0)]
  z2=[np.sin(u0)]
  for _ in range(iteracoes):
    t+=h
    u0, v0=u0+h*Du, v0+h*Dv
    Du, Dv=Du+h*D_2u(t,u0,v0,Du,Dv), Dv+h*D_2v(t,u0,v0,Du,Dv)

    x2.append(np.cos(u0)*np.cos(v0))
    y2.append(np.cos(u0)*np.sin(v0))
    z2.append(np.sin(u0))

  return (x2,y2,z2)  


## Equações geodésicas da esfera
def D_2u(t,u,v,Du,Dv):
    return 2*np.sin(u)*np.cos(u)*Dv**2

def D_2v(t,u,v,Du,Dv):
    return 2*np.tan(u)*Du*Dv



## Parametrização da esfera
precisao=100
u=np.linspace(0,2*np.pi,precisao)
v=np.linspace(0,np.pi,precisao)
x=np.outer(np.cos(u),np.cos(v))
y=np.outer(np.cos(u),np.sin(v))
z=np.outer(np.sin(u),np.ones(precisao))




## Aplicação dos métodos
iteracoes=500
p0=(np.pi,4)
direcao=(2,1)
h=0.001

x1,y1,z1 = runge_kutta(iteracoes,p0,direcao,h)
x2,y2,z2 = euler(iteracoes,p0,direcao,h)





## Geoésica exata
# Calculada através da interseção da esfera com o plano que contém o ponto inicial e o ponto gerado na primeira iteração do Runge-Kutta

# verifica se um ponto está na esfera
def esfera(x,y,z):
    return ((x**2+y**2+z**2)-1<=10**(-2) and (x**2+y**2+z**2)-1>0)

# domínio do plano (produto cartesiano de u1 consigo mesmo)
exatidao=1000
u1=np.linspace(-1.5,1.5,exatidao)

# base ortonormal usada para parametrizar o plano contendo p1 e p2
p1=np.array([x1[0],y1[0],z1[0]])
p2=np.array([x1[1],y1[1],z1[1]])
p2=(1/h)*(p2-np.inner(p1/((p1[0]**2+p1[1]**2+p1[2]**2)**(0.5)),p2)*p1/((p1[0]**2+p1[1]**2+p1[2]**2)**(0.5)))

# listas contendo os pontos da interseção
x3=[]
y3=[]
z3=[]
for i in range(exatidao):
    for j in range(exatidao):
        px=u1[i]*p1[0]+u1[j]*p2[0]
        py=u1[i]*p1[1]+u1[j]*p2[1]
        pz=u1[i]*p1[2]+u1[j]*p2[2]
        if esfera(px,py,pz):
          x3.append(px)
          y3.append(py)
          z3.append(pz)


# uso do KNN para auxiliar no plot da geodésica exata
X=[]
for i in range(len(x3)):
  X.append([x3[i],y3[i],z3[i]])

numero_vizinhos = 30
knn = NearestNeighbors(n_neighbors=numero_vizinhos)
knn.fit(X)
vizinhos = knn.kneighbors(X,return_distance=False)

x4,y4,z4 = [x3[0]],[y3[0]],[z3[0]]
vizinho = [vizinhos[0,0],vizinhos[0,1]]
usados = [vizinho[0]]
for i in range(len(x3)):
  x4.append(x3[vizinho[1]])
  y4.append(y3[vizinho[1]])
  z4.append(z3[vizinho[1]])
  
  vizinho = [vizinhos[vizinho[1],0],vizinhos[vizinho[1],1]]
  for i in range(numero_vizinhos-1):
    if vizinho[1] in usados:
      vizinho[1] = vizinhos[vizinho[0],i+1]

  usados.append(vizinho[0])




## Plot
ax=plt.axes(projection='3d')
ax.plot_surface(x,y,z, alpha=0.5, color='white')
ax.plot(x4,y4,z4,alpha=0.7,color='orange')  #Exata
ax.plot(x2,y2,z2,color='purple')  #Euler
ax.plot(x1,y1,z1,color='red')  #Runge-Kutta
ax.scatter(x1[0],y1[0],z1[0],color='black')
ax.scatter(x1[iteracoes-1],y1[iteracoes-1],z1[iteracoes-1],color='orange')

ax.set_xlabel('x',color='r')
ax.set_ylabel('y',color='g')
ax.set_zlabel('z',color='b')
ax.set_aspect('auto')
plt.show()
