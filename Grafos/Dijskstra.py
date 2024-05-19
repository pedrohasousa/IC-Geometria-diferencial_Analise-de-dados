# Implementação do algoritmo de Dijskstra para encontrar o caminho mínimo entre dois vértices de um grafo simples orientado.
# Feita apenas como exercício. No projeto foi utilizada a implementação da biblioteca NetworkX

# grafo (simples e orientado) dado por sua matriz de custos (que devem ser não-negativos), com -1 se não há aresta entre os respectivos nós
grafo = [[0,35,3,-1,-1,-1],[16,0,6,-1,-1,21],[-1,-1,0,2,-1,7],[4,-1,2,0,-1,-1],[15,-1,-1,9,0,-1],[-1,5,-1,-1,-1,0]]
#grafo=[[0,50,30,100,10],[50,0,5,-1,-1],[30,5,0,50,-1],[100,-1,50,0,10],[10,-1,-1,10,0]]

vertices = len(grafo)

# o caminho sera calculado entre os vertices 'inicio' e 'fim', dados pelo seu indice na matriz do grafo
inicio = 4
fim = 5

if inicio >= vertices or fim >= vertices:
  print("Ponto fora do grafo")
  exit(0)

permanente = inicio
permanentes = [inicio]
rotulo_permanente = 0

lista_obtencao = []
rotulos = []
for i in range(vertices):
  rotulos.append(grafo[inicio][i])
  lista_obtencao.append(-1)


while permanente != fim:
  
  # lista dos vertices a serem testados
  temporarios = [vertice for vertice in range(vertices) if vertice not in permanentes and (rotulos[vertice] >= 0 or grafo[permanente][vertice] >= 0)]

  if temporarios == []:
    print("Caminho não encontrado")
    exit(0)

  #inicialização do vertice que será transformado em permanente e seu rotulo (vertice_minimo e rotulo_minimo)
  vertice_minimo = temporarios[0]
  
  if rotulos[temporarios[0]] < 0:
    rotulo_temporario = rotulo_permanente + grafo[permanente][temporarios[0]]
    lista_obtencao[temporarios[0]] = permanente

  elif grafo[permanente][temporarios[0]] < 0:
    rotulo_temporario = rotulos[temporarios[0]]
  else:
    rotulo_temporario = min(rotulos[temporarios[0]], rotulo_permanente + grafo[permanente][temporarios[0]])  
    
    if min(rotulos[temporarios[0]], rotulo_permanente + grafo[permanente][temporarios[0]]) == rotulo_permanente + grafo[permanente][temporarios[0]]:
      lista_obtencao[temporarios[0]] = permanente

  rotulos[temporarios[0]] = rotulo_temporario
  rotulo_minimo = rotulo_temporario


  # atualização do vertice que será transformado em permanente e da lista de rotulos
  for i in temporarios[1:]:
    if rotulos[i] < 0:
      rotulo_temporario = rotulo_permanente + grafo[permanente][i]
      lista_obtencao[i] = permanente
      
    elif grafo[permanente][i] < 0:
      rotulo_temporario = rotulos[i]
    else:
      rotulo_temporario = min(rotulos[i], rotulo_permanente + grafo[permanente][i])

      if min(rotulos[i], rotulo_permanente + grafo[permanente][i]) == rotulo_permanente + grafo[permanente][i]:
        lista_obtencao[i] = permanente

    rotulos[i] = rotulo_temporario

    if rotulo_temporario < rotulo_minimo:
      rotulo_minimo = rotulo_temporario
      vertice_minimo = i
  

  permanente = vertice_minimo
  permanentes.append(permanente)
  rotulo_permanente = rotulo_minimo


#Recuperação do caminho
caminho = []
vertice_atual = fim

while vertice_atual != inicio:
  caminho.append(lista_obtencao[vertice_atual])
  vertice_atual = lista_obtencao[vertice_atual]


caminho = [caminho[len(caminho)-1-i] for i in range(len(caminho))]
caminho.append(fim)  

print("Caminho:",caminho)
print("Custo do caminho:",rotulos[fim])