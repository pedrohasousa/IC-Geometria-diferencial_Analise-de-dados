# Seleciona um conjunto de pontos aleatórios sobre uma esfera, digamos p_1,...,p_n , projeta-os em 2 dimensões utilizando o método MDS, obtendo
# q_1,...,q_n em R2 (q_i é a projeção de p_i), e define uma triangulação de Delaunay para os pontos projetados, extendendo-a também aos pontos da esfera.

# Então, seleciona um ponto aleatório, p, no dominio estimado pelo MDS; determina dentro de qual triângulo p está, digamos 
# q_i, q_j, q_k; e cacula sua imagem pelo plano que leva q_i, q_j, q_k em seus correspondentes p_i, p_j, p_k na esfera.
# Se o ponto cair fora de todos os triângulos sua imagem não é calculada



import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay


## Funções para fazer os plots (ver 'plt.title' para informações sobre cada um)
def plot_2d(pontos, cor, p = ['nda'], cor_p = 'orange', triangulo = ['nda']):
  _, ax = plt.subplots()
  ax.scatter(pontos[:,0], pontos[:,1], color=cor)
  ax.scatter(p[0], p[1], color = cor_p)

  if triangulo[0] != 'nda':
    ax.triplot([pontos[triangulo[0]][0], pontos[triangulo[1]][0], pontos[triangulo[2]][0]], [pontos[triangulo[0]][1], pontos[triangulo[1]][1], pontos[triangulo[2]][1]], [[0,1,2]], color = 'green')
    ax.scatter(p[0], p[1], color = cor_p)
  plt.xlabel('Eixo X')
  plt.ylabel('Eixo Y')  
  plt.title('Projeção MDS/PCA/ISOMAP e ponto p em seu triângulo')  
  plt.show()


def plot_3d(pontos, cor, ponto = ['nda'], cor_ponto = 'orange', triangulo = ['nda']):
  ax = plt.axes(projection='3d')
  ax.scatter(pontos[:,0], pontos[:,1], pontos[:,2], color=cor)
  
  if ponto[0] != 'nda':
    ax.scatter(ponto[0], ponto[1], ponto[2], color = cor_ponto)
    ax.plot_trisurf([pontos[triangulo[0]][0], pontos[triangulo[1]][0], pontos[triangulo[2]][0]], [pontos[triangulo[0]][1], pontos[triangulo[1]][1], pontos[triangulo[2]][1]], [pontos[triangulo[0]][2], pontos[triangulo[1]][2], pontos[triangulo[2]][2]], triangles=[[0,1,2]], color='green', alpha = 0.5)
  plt.xlabel('Eixo X')
  plt.ylabel('Eixo Y')
  ax.set_zlabel('Eixo Z')
  plt.title('Pontos da esfera e imagem de p em seu triângulo')
  plt.show()


def plot_triangulacao_2d(pontos, triangulacao):
  plt.plot(pontos[:,0], pontos[:,1], 'o', color = 'blue')
  plt.triplot(pontos[:,0], pontos[:,1], triangulacao, color = 'green')
  plt.xlabel('Eixo X')
  plt.ylabel('Eixo Y')
  plt.title('Projeção e triangulação de Delaunay')
  plt.show()


def plot_triangulacao_3d(pontos, triangulacao):
  fig = plt.figure(figsize=plt.figaspect(0.5))
  ax = fig.add_subplot(1, 2, 1, projection='3d')
  ax.plot_trisurf(pontos[:,0], pontos[:,1], pontos[:,2], triangles=triangulacao)
  plt.xlabel('Eixo X')
  plt.ylabel('Eixo Y')
  ax.set_zlabel('Eixo Z')
  plt.title('Reconstrução da esfera a partir da projeção e da triangulação')
  plt.show()




## Definição dos pontos sobre a esfera x² + y² + z² = 1
random = np.random.default_rng()

# u = random.uniform(0, 1.2*np.pi, size = (400,2))

# []
# Pontos = []
# for ponto in u:
#   Pontos.append([np.sin(ponto[0])*np.cos(ponto[1]), np.sin(ponto[0])*np.sin(ponto[1]), np.cos(ponto[0])])
#Pontos = np.array(Pontos)

Pontos = []
for _ in range(400):
  u = random.uniform(0, 0.4*np.pi)
  v = random.uniform(0, 2*np.pi)
  Pontos.append([np.sin(u)*np.cos(v), np.sin(u)*np.sin(v), np.cos(u)])
Pontos = np.array(Pontos)



## Aplicação do MDS, PCA ou ISOMAP

# mds = MDS(n_components=2, max_iter=50, verbose = 0, random_state=None)
# proj = mds.fit_transform(Pontos)

# pca = PCA(n_components=2, iterated_power=50, random_state=None)
# proj = pca.fit_transform(Pontos)

isomap = Isomap(n_components=2, n_neighbors=7, max_iter=50)
proj = isomap.fit_transform(Pontos)







## Aplicação da triangulação de Delaunay
triang = Delaunay(proj)
triangulos = triang.simplices



## Determinação do domíno estimado pela redução de dimensionalidade
x_min = x_max = proj[0][0]
y_min = y_max = proj[0][1]

for ponto in proj:
  if ponto[0] < x_min:
    x_min = ponto[0]
  elif ponto[0] > x_max:
    x_max = ponto[0]  

  if ponto[1] < y_min:
    y_min = ponto[1]
  elif ponto[1] > y_max:
    y_max = ponto[1]    


print("\nDomínio da projeção: x:["+str(x_min)+", "+str(x_max)+"]   y:["+str(y_min)+", "+str(y_max)+"]")





## Escolha de um ponto aleatório no domínio
#p = [random.normal((x_max+x_min)/2, (x_max-x_min)/8), random.normal((y_max+y_min)/2, (y_max-y_min)/8)]
p = np.array([random.uniform(x_min, x_max), random.uniform(y_min, y_max)])
print("p =",p)





## Localização do ponto p
# determinamos as coordenadas baricêntricas de p em relação a cada triângulo da triangulação
# e paramos quando acharmos o triângulo que o contém

# Dado o triângulo abc e um ponto p, usaremos a notação p = alfa[0]*a + alfa[1]*b + alfa[2]*c para suas coordenadas baricêntricas
triangulo_p = ['nda']
a = b = c = 0
for triangulo in triangulos:
  a = proj[triangulo[0]]
  b = proj[triangulo[1]]
  c = proj[triangulo[2]]
  
  T = np.linalg.inv(np.array([[a[0], a[1], 1], [b[0], b[1], 1], [c[0], c[1], 1]]))
  alfa = np.matmul(np.array([p[0], p[1], 1]), T)

  #print("p_bari =", alfa[0]*a + alfa[1]*b + alfa[2]*c)

  if alfa[0] >= 0 and alfa[1] >= 0 and alfa[2] >= 0:
    triangulo_p = triangulo
    print("triangulo contendo p:", triangulo_p)
    break


## Determinação da imagem de p
# usamos as coordenadas baricêntricas de p
Im_p = ['nda']

if triangulo_p[0] != 'nda':
  Im_p = alfa[0]*Pontos[triangulo_p[0]] + alfa[1]*Pontos[triangulo_p[1]] + alfa[2]*Pontos[triangulo_p[2]]
else:
  print('p está fora de todos os triângulos')






## Plot
plot_triangulacao_2d(proj, triangulos)
plot_2d(proj, 'blue', p, 'orange', triangulo_p)
plot_3d(Pontos, 'blue', Im_p, 'orange', triangulo_p)

# o plot abaixo pode ter problemas a depender da parte da esfera escolhida
plot_triangulacao_3d(Pontos, triang.simplices)
