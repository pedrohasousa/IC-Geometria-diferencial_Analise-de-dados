# Aplica uma técnica de redução de dimensionalidade (MDS ou ISOMAP) em um banco de imagens de faces
# e cria uma triangulação (Delaunay) sobre os pontos projetados.
# Então, determina uma reta na triangulação e recupera as imagens referentes a uma amostra de pontos sobre a reta.

# Não possui os plots, pois a dimensçao não permite visualização

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from scipy.spatial import Delaunay
from PIL import Image
import os


## Coleta das imagens 
pasta = 'imagens_normalizadas_FEI'

imagens = []
caminhos = [os.path.join(pasta, arquivo) for arquivo in os.listdir(pasta)]

for caminho in caminhos:
  imagem = Image.open(caminho)
  print(np.shape(imagem))
  imagem = np.ndarray.flatten(np.array(imagem))
  imagens.append(imagem)

imagens = np.array(imagens)

print('\nNúmero de imagens =', len(imagens))


## Redução de dimensionalidade
dimensoes = 5
mds = MDS(n_components=dimensoes, max_iter=100, verbose = 0, random_state=None)
proj = mds.fit_transform(imagens)

# isomap = Isomap(n_components=dimensoes, n_neighbors=7, max_iter=50)
# proj = isomap.fit_transform(imagens)


## Triangulação de Delaunay
# em dimensẽs acima de dois gera polígonos com mais dimensões
triang = Delaunay(proj)
politopos = triang.simplices
num_vertices = len(politopos[0])



## Domíno estimado pela redução de dimensionalidade
# Dom = [[x1_min, x1_max], [x2_min, x2_max], ..., [xn_min, xn_max]]
Dom = []

for i in range(dimensoes):
  Dom.append([proj[0][i], proj[0][i]])

for ponto in proj:
  for i in range(dimensoes):
    if ponto[i] < Dom[i][0]:
      Dom[i][0] = ponto[i]
    elif ponto[i] > Dom[i][1]:
      Dom[i][1] = ponto[i]

Dom = np.array(Dom)



## Escolha de dois pontos aleatórios no interior da triangulação
# escolhemos um ponto aleatório de 'proj', p, e tomamos um ponto aleatório de uma hiperesfera centrada em p com raio 'raio'
# assim, determinamos p1 e p2. Então, traçamos o segmento de reta que liga p1 a p2. Se esse segmento possuir algum ponto fora
# da triangulação, repetimos todo o procedimento
random = np.random.default_rng()

politopos_reta = []
reta = [1,1]
while len(politopos_reta) < len(reta):
  
  politopo_p1 = ['nda']
  alfa1 = a = b = c = 0
  vertices_politopo = np.zeros((num_vertices, dimensoes))
  Transformacao = np.zeros((num_vertices, dimensoes+1)) #supondo num_vertices == dimensoes+1
  raio = abs(np.max(np.ndarray.flatten(Dom)))/2

  while politopo_p1[0] == 'nda':
    # ponto no domínio
    p1 = np.copy(proj[np.random.randint(0, len(proj))])
    for i in range(dimensoes):
      p1[i] += random.uniform(-raio, raio)
    p1 = np.append(p1, 1)
  
    # busca do politopo que contém p1
    for politopo in politopos:
      for i in range(num_vertices):
        for j in range(dimensoes):
          vertices_politopo[i][j] = proj[politopo[i]][j]
    
      for i in range(num_vertices):
        for j in range(dimensoes):
          Transformacao[i][j] = vertices_politopo[i][j]
        Transformacao[i][dimensoes] = 1
      Transformacao = np.linalg.inv(Transformacao)
    
      alfa1 = np.matmul(p1, Transformacao)

      # verificação se o politopo atual contém p1
      for i in  range(num_vertices):
        if alfa1[i] < 0:
          break

        if i == num_vertices-1:
          politopo_p1 = politopo

      if politopo_p1[0] != 'nda':
        break



  politopo_p2 = ['nda']
  alfa2 = a = b = c = 0

  while politopo_p2[0] == 'nda':
    # ponto no domínio
    p2 = np.copy(proj[np.random.randint(0, len(proj))])
    for i in range(dimensoes):
      p2[i] += random.uniform(-raio, raio)
    p2 = np.append(p2, 1)  
  
    # busca do politopo que contém p2
    for politopo in politopos:
      for i in range(num_vertices):
        for j in range(dimensoes):
          vertices_politopo[i][j] = proj[politopo[i]][j]
    
      for i in range(num_vertices):
        for j in range(dimensoes):
          Transformacao[i][j] = vertices_politopo[i][j]
        Transformacao[i][dimensoes] = 1
      Transformacao = np.linalg.inv(Transformacao)
    
      alfa2 = np.matmul(p2, Transformacao)

      # verificação se o politopo atual contém p2
      for i in  range(num_vertices):
        if alfa2[i] < 0:
          break

        if i == num_vertices-1:
          politopo_p2 = politopo

      if politopo_p2[0] != 'nda':
        break



  ## Determinação dos pontos sobre a reta que liga p1 a p2 e dos triângulos que os contêm
  pontos_intermediarios = 98
  T = np.linspace(0, 1, pontos_intermediarios+2)
  reta = np.array([(1-t)*p1 + t*p2 for t in T])

  politopos_reta = [politopo_p1]
  alfas_reta = [alfa1]
  flag = 0

  for ponto in reta[1:len(reta)-1]:
    for politopo in politopos:
      for i in range(num_vertices):
        for j in range(dimensoes):
          vertices_politopo[i][j] = proj[politopo[i]][j]
  
      for i in range(num_vertices):
        for j in range(dimensoes):
          Transformacao[i][j] = vertices_politopo[i][j]
        Transformacao[i][dimensoes] = 1
      Transformacao = np.linalg.inv(Transformacao)

      alfa = np.matmul(ponto, Transformacao)

      for i in  range(num_vertices):
        if alfa[i] < 0:
          break

        if i == num_vertices-1:
          politopos_reta.append(politopo)
          alfas_reta.append(alfa)
          flag = 1

      if flag == 1:
        flag = 0
        break

  alfas_reta.append(alfa2)
  politopos_reta.append(politopo_p2)
  politopos_reta = np.array(politopos_reta)


  ###TIRAR###
  if len(politopos_reta) < len(reta):
    print("ERRO")
    exit(0)


print("p1 =", p1[0:dimensoes])
print("p2 =", p2[0:dimensoes])
# print("politopos reta =", politopos_reta)
# print("\nalfas reta =", alfas_reta)



# ## Determinação das imagens dos pontos sobre a reta
Im_reta = []
auxiliar = np.zeros(np.shape(imagens[0])[0])
for i in range(len(reta)):
  for j in range(dimensoes+1):
    auxiliar += alfas_reta[i][j]*imagens[politopos_reta[i][j]]
  Im_reta.append(auxiliar)
  auxiliar = np.zeros(np.shape(imagens[0])[0])
Im_reta = np.array(Im_reta)



## Transformação dos pontos em imagens
for i in range(len(Im_reta)):
  imagem = np.around(np.reshape(Im_reta[i], (193,162)))
  imagem = np.array(imagem, np.uint8)
  imagem = Image.fromarray(imagem)
  imagem.save('imagens_experimento/imagem'+str(i+1)+'.jpg')


print(politopos_reta, "\n")
print(alfas_reta)
