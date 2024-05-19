# Define um conjunto aleatório de pontos de R2, X, e cacula suas imagens pela parametrização de uma esfera, Im_X.

# Então, aplica o algoritmo KNN em Im_X e constrói um grafo onde cada nó é um ponto de Im_X e arestas existem
# entre os vizinhos determinados pelo KNN, com custo dado pela distância entre os respectivos pontos.

# Por fim, escolhe dois nós do grafo aplica o algoritmo de Djikstra, a fim de utilizar o caminho encontrado como
# aproximação para a geodésica que liga esses pontos da esfera.

from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

## Pontos aleatorios do R2
rng = np.random.default_rng()
amostra = 100
X = rng.normal(0.0,1.0,size=(amostra,2))

## Imagem de X pela parametrização (de uma esfera)
Im_X = []
for ponto in X:
  Im_X.append([np.cos(ponto[0])*np.cos(ponto[1]), np.cos(ponto[0])*np.sin(ponto[1]), np.sin(ponto[0])])
Im_X = np.array(Im_X)  
  


## Aplicação do KNN Im_X
numero_vizinhos = 7
knn = NearestNeighbors(n_neighbors=numero_vizinhos)
knn.fit(Im_X)
distanciasR3, vizinhosR3 = knn.kneighbors(Im_X,return_distance=True)



## Construção do grafo
grafoR3 = np.zeros((amostra,amostra))

for i in range(len(X)):
  for j in range(len(X)):

    if i>j:
      grafoR3[i,j] = grafoR3[j,i]
      continue

    if j in vizinhosR3[i]:
      for k in range(numero_vizinhos):
        if vizinhosR3[i][k] == j:
          grafoR3[i,j] = distanciasR3[i,k]






## Plot de Im_X
ax = plt.axes(projection='3d')

for ponto in Im_X:
  x = ponto[0]
  y = ponto[1]
  z = ponto[2]

  ax.scatter(x,y,z,color='black')


## Plot da esfera para referência
precisao=100
u=np.linspace(0,2*np.pi,precisao)
v=np.linspace(0,np.pi,precisao)
x=np.outer(np.cos(u),np.cos(v))
y=np.outer(np.cos(u),np.sin(v))
z=np.outer(np.sin(u),np.ones(precisao))

ax.plot_surface(x,y,z, alpha=0.5, color='white')






## Plot dos caminhos mínimos sobre a esfera
inicio = 5
fim = 7

# Dijkstra da NetworkX
G_R3 = nx.from_numpy_array(grafoR3)
caminhoR3 = nx.dijkstra_path(G_R3,inicio,fim)

# Plot do caminho
for i in range(len(caminhoR3)-1):
  x = [Im_X[caminhoR3[i],0], Im_X[caminhoR3[i+1],0]]
  y = [Im_X[caminhoR3[i],1], Im_X[caminhoR3[i+1],1]]
  z = [Im_X[caminhoR3[i],2], Im_X[caminhoR3[i+1],2]]

  ax.plot(x,y,z,color='blue')


# Trocando as cores do inicio e do fim do caminho
ax.scatter(Im_X[inicio,0], Im_X[inicio,1], Im_X[inicio,2], color = 'orange')
ax.scatter(Im_X[fim,0], Im_X[fim,1], Im_X[fim,2], color = 'green')

#ax.set_aspect('equal')
plt.show()