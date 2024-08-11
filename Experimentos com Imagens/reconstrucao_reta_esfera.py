# Define uma triangulação de Delanunay sobre uma esfera (ver arquivo projecao_triangulacao_esfera) e
# escolhe dois pontos aleatórios no interior da triangulação. Então, traça a reta que liga esses pontos e
# toma uma amostra de pontos sobre essa reta. Por fim, transfere essa reta para a esfera.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from scipy.spatial import Delaunay


## Funções para fazer os plots (ver 'plt.title' para informações sobre cada um)
def plot_2d(pontos_projecao, cor_projecao, pontos_reta, triangulos_reta, cor_reta = 'orange'):
  _, ax = plt.subplots()
  ax.scatter(pontos_projecao[:,0], pontos_projecao[:,1], color=cor_projecao)
  ax.triplot(pontos_projecao[:,0], pontos_projecao[:,1], triangulos_reta, color = 'green')
  ax.scatter(pontos_reta[:,0], pontos_reta[:,1], color = cor_reta)
  plt.xlabel('Eixo X')
  plt.ylabel('Eixo Y')
  plt.title('Projeção MDS/PCA/ISOMAP e pontos em seus triângulos')  
  plt.show()

def plot_3d(pontos_projecao, cor_projecao, pontos_reta, triangulos_reta, cor_reta = 'orange'):
  ax = plt.axes(projection='3d')
  ax.scatter(pontos_projecao[:,0], pontos_projecao[:,1], pontos_projecao[:,2], color=cor_projecao)
  ax.plot_trisurf(pontos_projecao[:,0], pontos_projecao[:,1], pontos_projecao[:,2], triangles=triangulos_reta, color='green', alpha = 0.5)
  ax.scatter(pontos_reta[:,0], pontos_reta[:,1], pontos_reta[:,2], color = cor_reta)
  plt.xlabel('Eixo X')
  plt.ylabel('Eixo Y')
  ax.set_zlabel('Eixo Z')
  plt.title('Pontos da esfera e imagem dos pontos em seus triângulos')
  plt.show()

def plot_triangulacao_2d(pontos, triangulacao):
  plt.plot(pontos[:,0], pontos[:,1], 'o', color = 'blue')
  plt.triplot(pontos[:,0], pontos[:,1], triangulacao, color = 'green')
  plt.xlabel('Eixo X')
  plt.ylabel('Eixo Y')
  plt.title('Projeção e triangulação de Delaunay')
  plt.show()


## Definição dos pontos sobre a esfera x² + y² + z² = 1
random = np.random.default_rng()
Pontos = []
for _ in range(400):
  u = random.uniform(0, 0.4*np.pi)
  v = random.uniform(0, 2*np.pi)
  Pontos.append([np.sin(u)*np.cos(v), np.sin(u)*np.sin(v), np.cos(u)])
Pontos = np.array(Pontos)


## Aplicação do MDS, PCA ou ISOMAP

mds = MDS(n_components=2, max_iter=50, verbose = 0, random_state=None)
proj = mds.fit_transform(Pontos)

# pca = PCA(n_components=2, iterated_power=50, random_state=None)
# proj = pca.fit_transform(Pontos)

# isomap = Isomap(n_components=2, n_neighbors=7, max_iter=50)
# proj = isomap.fit_transform(Pontos)


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







## Escolha de dois pontos aleatórios no interior da triangulação
triangulo_p1 = ['nda']
alfa1 = a = b = c = 0

while triangulo_p1[0] == 'nda':
  p1 = np.array([random.uniform(x_min, x_max), random.uniform(y_min, y_max)])

  for triangulo in triangulos:
    a = proj[triangulo[0]]
    b = proj[triangulo[1]]
    c = proj[triangulo[2]]
  
    T = np.linalg.inv(np.array([[a[0], a[1], 1], [b[0], b[1], 1], [c[0], c[1], 1]]))
    alfa1 = np.matmul(np.array([p1[0], p1[1], 1]), T)

    if alfa1[0] >= 0 and alfa1[1] >= 0 and alfa1[2] >= 0:
      triangulo_p1 = triangulo
      print("triangulo contendo p1:", triangulo_p1)
      break


triangulo_p2 = ['nda']
alfa2 = a = b = c = 0

while triangulo_p2[0] == 'nda':
  p2 = np.array([random.uniform(x_min, x_max), random.uniform(y_min, y_max)])

  for triangulo in triangulos:
    a = proj[triangulo[0]]
    b = proj[triangulo[1]]
    c = proj[triangulo[2]]
  
    T = np.linalg.inv(np.array([[a[0], a[1], 1], [b[0], b[1], 1], [c[0], c[1], 1]]))
    alfa2 = np.matmul(np.array([p2[0], p2[1], 1]), T)

    if alfa2[0] >= 0 and alfa2[1] >= 0 and alfa2[2] >= 0:
      triangulo_p2= triangulo
      print("triangulo contendo p2:", triangulo_p2)
      break

print("p1 =",p1)
print("p2 =",p2)



## Determinação dos pontos sobre a reta que liga p1 a p2 e dos triângulos que os contêm
pontos_intermediarios = 20
T = np.linspace(0, 1, pontos_intermediarios+2)
reta = np.array([(1-t)*p1 + t*p2 for t in T])

triangulos_reta = [triangulo_p1]
alfas_reta = [alfa1]

for ponto in reta[1:len(reta)-1]:
  for triangulo in triangulos:
    a = proj[triangulo[0]]
    b = proj[triangulo[1]]
    c = proj[triangulo[2]]
  
    T = np.linalg.inv(np.array([[a[0], a[1], 1], [b[0], b[1], 1], [c[0], c[1], 1]]))
    alfa = np.matmul(np.array([ponto[0], ponto[1], 1]), T)

    if alfa[0] >= 0 and alfa[1] >= 0 and alfa[2] >= 0:
      alfas_reta.append(alfa)
      triangulos_reta.append(triangulo)
      break

alfas_reta.append(alfa2)
triangulos_reta.append(triangulo_p2)
triangulos_reta = np.array(triangulos_reta)

if len(triangulos_reta) < len(reta):
  print("Existe algum ponto da reta fora da triangulação")
  exit(0) 



## Determinação das imagens dos pontos sobre a reta
Im_reta = []
for i in range(len(reta)):
  Im_reta.append(alfas_reta[i][0]*Pontos[triangulos_reta[i][0]] + alfas_reta[i][1]*Pontos[triangulos_reta[i][1]] + alfas_reta[i][2]*Pontos[triangulos_reta[i][2]])
Im_reta = np.array(Im_reta)





## Plot
# plot_triangulacao_2d(proj, triangulos)
plot_2d(proj, 'blue', reta, triangulos_reta, 'orange')
plot_3d(Pontos, 'blue', Im_reta, triangulos_reta, 'orange')
