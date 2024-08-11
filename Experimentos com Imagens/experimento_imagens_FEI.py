# Aplica uma técnica de redução de dimensionalidade (MDS ou ISOMAP) em um banco de imagens de faces
# (reduz para dimensão 2) e cria uma triangulação (Delaunay) sobre os pontos projetados.
# Então, determina uma reta na triangulação e recupera as imagens referentes a uma amostra de pontos sobre a reta.



import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from scipy.spatial import Delaunay
from PIL import Image
import os

## Funcões para os plots

def plot_triangulacao_2d(pontos, triangulacao):
  plt.plot(pontos[:,0], pontos[:,1], 'o', color = 'blue')
  plt.triplot(pontos[:,0], pontos[:,1], triangulacao, color = 'green')
  plt.xlabel('Eixo X')
  plt.ylabel('Eixo Y')
  plt.title('Projeção e triangulação de Delaunay')
  plt.show()

def plot_2d(pontos_projecao, cor_projecao, pontos_reta, triangulos_reta, cor_reta = 'orange'):
  _, ax = plt.subplots()
  ax.scatter(pontos_projecao[:,0], pontos_projecao[:,1], color=cor_projecao)
  ax.triplot(pontos_projecao[:,0], pontos_projecao[:,1], triangulos_reta, color = 'green')
  ax.scatter(pontos_reta[:,0], pontos_reta[:,1], color = cor_reta)
  plt.xlabel('Eixo X')
  plt.ylabel('Eixo Y')
  #plt.title('Projeção MDS/ISOMAP e pontos em seus triângulos')  
  plt.show()





## Coleta das imagens 
pasta = 'imagens_normalizadas_FEI'

imagens = []
caminhos = [os.path.join(pasta, arquivo) for arquivo in os.listdir(pasta)]

for caminho in caminhos:
  imagem = Image.open(caminho)
  imagem = np.ndarray.flatten(np.array(imagem))
  imagens.append(imagem)

imagens = np.array(imagens)

print('\nNúmero de imagens =', len(imagens))


## Redução de dimensionalidade (para duas dimensões)
# mds = MDS(n_components=2, max_iter=50, verbose = 0, random_state=None)
# proj = mds.fit_transform(imagens)

isomap = Isomap(n_components=2, n_neighbors=10, max_iter=50)
proj = isomap.fit_transform(imagens)


## Triangulação de Delaunay
triang = Delaunay(proj)
triangulos = triang.simplices

print(proj)
print(triangulos)

## Domíno estimado pela redução de dimensionalidade
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
random = np.random.default_rng()
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
      break

print("\np1 =",p1)
print("p2 =",p2)



## Determinação dos pontos sobre a reta que liga p1 a p2 e dos triângulos que os contêm
pontos_intermediarios = 7
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
  Im_reta.append(alfas_reta[i][0]*imagens[triangulos_reta[i][0]] + alfas_reta[i][1]*imagens[triangulos_reta[i][1]] + alfas_reta[i][2]*imagens[triangulos_reta[i][2]])
Im_reta = np.array(Im_reta)




## Transformação dos pontos em imagens
for i in range(len(Im_reta)):
  imagem = np.around(np.reshape(Im_reta[i], (193,162)))
  imagem = np.array(imagem, np.uint8)
  imagem = Image.fromarray(imagem)
  imagem.save('imagens_experimento/imagem'+str(i+1)+'.jpg')



## Plots
plot_triangulacao_2d(proj, triangulos)
plot_2d(proj, 'blue', reta, triangulos_reta)