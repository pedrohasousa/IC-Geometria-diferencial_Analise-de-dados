# Fixa duas listas de pontos, A = [a1,...,a10] e B = [b1,...,b10] sobre uma esfera e as geodésicas que ligam cada a_i a b_i.
# Então, aplica o método de Runge-Kutta adaptado para aproximar essas geodésicas e compara com as exatas.



from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


## Equações geodésicas da esfera
def D_2u(u,v,Du,Dv):
  return 2*np.sin(u)*np.cos(u)*Dv**2

def D_2v(u,v,Du,Dv):
  return 2*np.tan(u)*Du*Dv




# Plot da esfera
ax = plt.axes(projection="3d")

precisao=100
u=np.linspace(0,2*np.pi,precisao)
v=np.linspace(0,np.pi,precisao)
x=np.outer(np.cos(u),np.cos(v))
y=np.outer(np.cos(u),np.sin(v))
z=np.outer(np.sin(u),np.ones(precisao))

ax.plot_surface(x,y,z, alpha=0.5, color='white')




# Função para aplicar Runge-Kutta dados pontos inicial e final da esfera
def runge_kutta(P1, P2):

  #Ponto inicial, ponto final e direção inicial em R2, recuperados pela inversa da parametrização da esfera supondo cos(a1) e np.arcsin(P2[2] não-nulos.
  a1 = np.arcsin(P1[2])
  b1 = np.arccos(P1[0]/np.cos(a1))

  a2 = np.arcsin(P2[2]) - a1
  b2 = np.arccos(P2[0]/np.cos(np.arcsin(P2[2]))) - b1

  direcao = np.array([a2,b2]) / np.linalg.norm(np.array([a2,b2]))   # direção inicial, liga os pontos inicial e final
  angulo_inicial = np.arctan2(direcao[1],direcao[0])
  
  #Ponto inicial, ponto final e direção que os liga, em R3
  ponto_inicial= P1
  ponto_final = P2
  dir_R3 = (ponto_final - ponto_inicial) / np.linalg.norm(ponto_final - ponto_inicial)

  # Primeira iteração do runge-kutta. Irá determinar o ajuste inicial para o angulo
  X=np.array([a1,b1,np.cos(angulo_inicial),np.sin(angulo_inicial)])
  cos_parada=2
  h=0.001
  dist_pontofinal_geo=1
  erro_maximo=0.01

  listax=np.array([])
  listay=np.array([])
  listaz=np.array([])

  while abs(cos_parada)>0.1 and dist_pontofinal_geo>erro_maximo:
    xi=np.array([np.cos(X[0])*np.cos(X[1])])
    yi=np.array([np.cos(X[0])*np.sin(X[1])])
    zi=np.array([np.sin(X[0])])

    listax=np.append(listax,xi)                  
    listay=np.append(listay,yi)
    listaz=np.append(listaz,zi)

    ponto_geo=np.array([xi[0],yi[0],zi[0]])

    v2 = ponto_geo-ponto_final

    dist_pontofinal_geo=np.linalg.norm(v2) 

    cos_parada = np.inner(dir_R3,v2) / dist_pontofinal_geo


    k1=h*np.array([X[2],X[3],D_2u(X[0],X[1],X[2],X[3]),D_2v(X[0],X[1],X[2],X[3])])
    Y=X+0.5*k1
    k2=h*np.array([Y[2],Y[3],D_2u(Y[0],Y[1],Y[2],Y[3]),D_2v(Y[0],Y[1],Y[2],Y[3])])
    Y=X+0.5*k2
    k3=h*np.array([Y[2],Y[3],D_2u(Y[0],Y[1],Y[2],Y[3]),D_2v(Y[0],Y[1],Y[2],Y[3])])
    Y=X+k3
    k4=h*np.array([Y[2],Y[3],D_2u(Y[0],Y[1],Y[2],Y[3]),D_2v(Y[0],Y[1],Y[2],Y[3])])
    X=X+(1/6)*(k1+2*k2+2*k3+k4)

  v1 = ponto_geo - ponto_inicial
  norma_v1 = np.linalg.norm(v1)
  vetor_referencia = np.cross(dir_R3,v1) #vetor que será usado como referência para checar se o ajuste deve ser somado ou subtraído do angulo 

  #primeiro ajuste
  cos_ajuste = np.inner(dir_R3,v1) / norma_v1
  ajuste = np.arccos(cos_ajuste)

  #Segunda e Terceira iterações para determinar quando o angulo deve ser somado ou subtraido
  for iter in range(2):  #em uma iteração o ajuste é somado e na outra subtraído. O melhor resultado é escolhido e usado para decidir o que fazer com os próximos ajustes
    if iter==0:
      X=np.array([a1,b1,np.cos(angulo_inicial+ajuste),np.sin(angulo_inicial+ajuste)])

      listax1=np.array([])
      listay1=np.array([])   #listas para salvar a geodésica
      listaz1=np.array([])
    else:
      X=np.array([a1,b1,np.cos(angulo_inicial-ajuste),np.sin(angulo_inicial-ajuste)])
      listax2=np.array([])
      listay2=np.array([])   #listas para salvar a geodésica
      listaz2=np.array([])

    cos_parada=2
    h=0.001
    dist_pontofinal_geo=1

    while abs(cos_parada)>0.1 and dist_pontofinal_geo>erro_maximo:
      xi=np.array([np.cos(X[0])*np.cos(X[1])])
      yi=np.array([np.cos(X[0])*np.sin(X[1])])
      zi=np.array([np.sin(X[0])])                  

      ponto_geo=np.array([xi[0],yi[0],zi[0]])

      v2 = ponto_final - ponto_geo
      dist_pontofinal_geo=np.linalg.norm(v2)

      cos_parada = np.inner(dir_R3,v2) / dist_pontofinal_geo

      if iter==0:
        listax1=np.append(listax1,xi)
        listay1=np.append(listay1,yi)
        listaz1=np.append(listaz1,zi)
      if iter==1:
        listax2=np.append(listax2,xi)
        listay2=np.append(listay2,yi)
        listaz2=np.append(listaz2,zi)

      k1=h*np.array([X[2],X[3],D_2u(X[0],X[1],X[2],X[3]),D_2v(X[0],X[1],X[2],X[3])])
      Y=X+0.5*k1
      k2=h*np.array([Y[2],Y[3],D_2u(Y[0],Y[1],Y[2],Y[3]),D_2v(Y[0],Y[1],Y[2],Y[3])])
      Y=X+0.5*k2
      k3=h*np.array([Y[2],Y[3],D_2u(Y[0],Y[1],Y[2],Y[3]),D_2v(Y[0],Y[1],Y[2],Y[3])])
      Y=X+k3
      k4=h*np.array([Y[2],Y[3],D_2u(Y[0],Y[1],Y[2],Y[3]),D_2v(Y[0],Y[1],Y[2],Y[3])])
      X=X+(1/6)*(k1+2*k2+2*k3+k4)

    if iter==0:
      dist_pontofinal_geo_1=dist_pontofinal_geo
    else:
      dist_pontofinal_geo_2=dist_pontofinal_geo

  
  # decisão de somar ou subtrair o primeiro ajuste e da orientação, que será usada para decidir como realizar os ajustes seguintes
  if dist_pontofinal_geo_1<dist_pontofinal_geo_2:
    orientaçao=1
    angulo_inicial+=ajuste
  else:
    orientaçao=-1
    angulo_inicial-=ajuste

  #Aplicando Runge kutta com ajuste da direção inicial
  erro_maximo=0.01
  dist_pontofinal_geo=1
  while dist_pontofinal_geo>erro_maximo:
    #print(dist_pontofinal_geo)
    # Salvando pontos da geodésica para plot e calculo de erro
    X=np.array([a1,b1,np.cos(angulo_inicial),np.sin(angulo_inicial)])
    cos_parada=2
    passos=0
    h=0.001
    listax=np.array([])
    listay=np.array([])
    listaz=np.array([])

    dist_pontofinal_geo=1
    matriz_geo = np.array([[0,0,0]])

    while (cos_parada==2 or abs(cos_parada)>0.15) and dist_pontofinal_geo>erro_maximo:
      xi=np.array([np.cos(X[0])*np.cos(X[1])])
      yi=np.array([np.cos(X[0])*np.sin(X[1])])
      zi=np.array([np.sin(X[0])])               

      listax=np.append(listax,xi)                  
      listay=np.append(listay,yi)
      listaz=np.append(listaz,zi)

      ponto_geo=np.array([xi[0],yi[0],zi[0]])
      matriz_geo = np.append(matriz_geo,[ponto_geo],axis=0)

      v2 = ponto_final - ponto_geo
      dist_pontofinal_geo=np.linalg.norm(v2)

      cos_parada = np.inner(dir_R3,v2) / dist_pontofinal_geo

      k1=h*np.array([X[2],X[3],D_2u(X[0],X[1],X[2],X[3]),D_2v(X[0],X[1],X[2],X[3])])
      Y=X+0.5*k1

      k2=h*np.array([Y[2],Y[3],D_2u(Y[0],Y[1],Y[2],Y[3]),D_2v(Y[0],Y[1],Y[2],Y[3])])
      Y=X+0.5*k2

      k3=h*np.array([Y[2],Y[3],D_2u(Y[0],Y[1],Y[2],Y[3]),D_2v(Y[0],Y[1],Y[2],Y[3])])
      Y=X+k3

      k4=h*np.array([Y[2],Y[3],D_2u(Y[0],Y[1],Y[2],Y[3]),D_2v(Y[0],Y[1],Y[2],Y[3])])
      X=X+(1/6)*(k1+2*k2+2*k3+k4)

      passos+=1
    
    if dist_pontofinal_geo>erro_maximo:
      v1 = ponto_geo - ponto_inicial
      norma_v1 = np.linalg.norm(v1)

      vetor_normal=np.cross(dir_R3,v1)
      ref = np.inner(vetor_referencia,vetor_normal) #o sinal desse valor, juntamente com a orientação, decide se o ajuste será somado ou subtraído

      #próximos ajustes por bissecção
      cos_ajuste = np.inner(dir_R3,v1) / norma_v1
      ajuste = 0.5*np.arccos(cos_ajuste)

      angulo_inicial+=np.sign(ref)*orientaçao*ajuste

  matriz_geo = np.delete(matriz_geo,0,0)

  return matriz_geo



# verifica se um ponto está na esfera
def esfera(x,y,z):
  return ((x**2+y**2+z**2)-1<=10**(-2) and (x**2+y**2+z**2)-1>0)



# Função para calcular geodésica exata
# Calculada através da interseção da esfera com o plano que contém o ponto inicial e o ponto gerado na primeira iteração do Runge-Kutta
def geodesica_exata(p_inicial, p_final):
  # parametrização do plano que contém a geodésica
  divisoes=1000
  u=np.linspace(-1.5,1.5,divisoes)
  p1=np.array(p_inicial)
  p2=np.array(p_final)
  p3=p2-np.inner(p1/((p1[0]**2+p1[1]**2+p1[2]**2)**(0.5)),p2)*p1/((p1[0]**2+p1[1]**2+p1[2]**2)**(0.5))
  norma_p3 = (p2[0]**2 + p2[1]**2 + p2[2]**2)**0.5
  p3=2*p3/norma_p3


  x2=[p1[0],p2[0]]
  y2=[p1[1],p2[1]]
  z2=[p1[2],p2[2]]
  for i in range(divisoes):
    for j in range(divisoes):
      px=u[i]*p1[0]+u[j]*p3[0]
      py=u[i]*p1[1]+u[j]*p3[1]
      pz=u[i]*p1[2]+u[j]*p3[2]
      if esfera(px,py,pz):
        x2.append(px)
        y2.append(py)
        z2.append(pz)



  # Plot da geodésica usando um grafo
  grande_circulo=[]
  for i in range(len(x2)):
    grande_circulo.append([x2[i],y2[i],z2[i]])

  numero_vizinhos = 30
  knn = NearestNeighbors(n_neighbors=numero_vizinhos)
  knn.fit(grande_circulo)
  distancias, vizinhos = knn.kneighbors(grande_circulo,return_distance=True)


  grafo_geo = np.zeros((len(grande_circulo),len(grande_circulo)))

  for i in range(len(grande_circulo)):
    for j in range(len(grande_circulo)):

      if i>j:
        grafo_geo[i,j] = grafo_geo[j,i]
        continue

      if j in vizinhos[i]:
        for k in range(numero_vizinhos):
          if vizinhos[i][k] == j:
            grafo_geo[i,j] = distancias[i,k]



  G = np.array(grafo_geo)
  G = nx.from_numpy_array(grafo_geo)
  caminho_geo = nx.dijkstra_path(G,0,1)
  return (grande_circulo, caminho_geo)






## INÍCIO DA COMPARAÇÃO
erro_relativo_medio = 0
erro_relativo_max = 0
erro_relativo_min = 0

erro_absoluto_medio = 0
erro_absoluto_max = 0
erro_absoluto_min = 0

erros_relativos = []
erros_absolutos = []

for i in range(1):
  # Pontos aleatórios de inicio e fim da geodésicas
  u = np.random.uniform(-0.45*np.pi, 0.45*np.pi, size=(2,))
  v = np.random.uniform(0.0*np.pi, 0.1*np.pi, size=(2,))

  #u = [-0.53206167, -1.06735093]
  #v = [0.26834216, 0.0383948 ]
  print("u =",u)
  print("v =",v)
  inicio = np.array([np.cos(u[0])*np.cos(v[0]), np.cos(u[0])*np.sin(v[0]), np.sin(u[0])])
  fim = np.array([np.cos(u[1])*np.cos(v[1]), np.cos(u[1])*np.sin(v[1]), np.sin(u[1])])
  




  geodesica_rk = runge_kutta(inicio, fim)
  grande_circulo, caminho_geo = geodesica_exata(inicio, fim)

  # Cálculo do comprimento da geodésica gerada pelo Runge-Kutta
  comprimento_geodesica = 0
  comprimento_geodesica_rk = 0
  for i in range(1,len(geodesica_rk)):
    comprimento_geodesica_rk += np.linalg.norm(geodesica_rk[i] - geodesica_rk[i-1])



  # Plot das geodésicas e comprimento da exata
  ax.plot(geodesica_rk[:,0], geodesica_rk[:,1], geodesica_rk[:,2], color='red')

  for i in range(len(caminho_geo)-1):
    x = [grande_circulo[caminho_geo[i]][0], grande_circulo[caminho_geo[i+1]][0]]
    y = [grande_circulo[caminho_geo[i]][1], grande_circulo[caminho_geo[i+1]][1]]
    z = [grande_circulo[caminho_geo[i]][2], grande_circulo[caminho_geo[i+1]][2]]
    comprimento_geodesica += np.linalg.norm([x[1]-x[0], y[1]-y[0], z[1]-z[0]])

    ax.plot(x,y,z,color='blue')


  ## Erros
  erro_absoluto = comprimento_geodesica_rk - comprimento_geodesica
  erro_relativo = erro_absoluto/comprimento_geodesica

  erros_absolutos.append(erro_absoluto)
  erros_relativos.append(erro_relativo)


  print("comprimento exata", comprimento_geodesica)
  print("comprimento rk", comprimento_geodesica_rk)


## Erros médios
erro_absoluto_medio = sum(erros_absolutos)/10
erro_relativo_medio = sum(erros_relativos)/10
print('Erro absoluto médio =', erro_absoluto_medio)
print('Erro relativo médio =', erro_relativo_medio,'\n')


## Determinação dos erros da maior e menor módulo
erro_absoluto_max = np.max(np.absolute(erros_absolutos))
erro_absoluto_min = np.min(np.absolute(erros_absolutos))
if erro_absoluto_max not in erros_absolutos:
  erro_absoluto_max = -erro_absoluto_max
if erro_absoluto_min not in erros_absolutos:
  erro_absoluto_min = -erro_absoluto_min

erro_relativo_max = np.max(np.absolute(erros_relativos))
erro_relativo_min = np.min(np.absolute(erros_relativos))
if erro_relativo_max not in erros_relativos:
  erro_relativo_max = -erro_relativo_max
if erro_relativo_min not in erros_relativos:
  erro_relativo_min = -erro_relativo_min  

print("Erro absoluto máximo = ", erro_absoluto_max)
print("Erro absoluto mínimo = ", erro_absoluto_min)
print("Erro relativo máximo = ", erro_relativo_max)
print("Erro relativo mínimo = ", erro_relativo_min)  
  

ax.set_xlabel("Eixo X")
ax.set_ylabel("Eixo Y")
ax.set_zlabel("Eixo Z")
plt.show()