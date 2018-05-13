#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 

#Para rodar: pip install numpy, sudo apt-get install python-tk, pip install matplotlib

#Duvidas: 

'''
funcao np.amax
'''


#Entradas
'''
Serie
Filme
Dia da semana ou fim de semana
Criancas ou nao
Ferias ou nao
Notas para cada genero
'''


#Parametros
'''
Numero de camadas ocultas = 1
Numero de saidas de cada camada = 
Funcao de ativacao = 
Batch ou online = 
Numero de iteracoes de treinamento =
Criterio de parada = 
'''


X = np.array(([2,9],[1,5],[3,6]), dtype = float)
Y = np.array(([92],[86],[89]), dtype = float)
teste = np.array(([4,8]), dtype = float)


X = X/np.amax(X, axis=0)
Y = Y/100
teste = teste/np.amax(teste, axis = 0)



def leitor():
	with open('metade dos dados.csv','r') as arquivo:
		for linha in arquivo:
			linha_lida = linha.split(",")
			trata_linha(linha_lida)

def trata_linha(linha):
	Entrada_NN = []
	#Tratamento primeiro valor: generos do filme assistido
	byte = trata_generos(linha[0])
	Entrada_NN.append(byte)

	#Tratamento do segundo valor: filhos 1 se sim 0 se nao
	Entrada_NN.append(linha[1])
	
	#Tratamento do terceiro valor: Ferias 1 se sim 0 se nao
	Entrada_NN.append(linha[2])

	#Tratamento do quarto valor: Filme
	byte = trata_generos(linha[3])
	Entrada_NN.append(byte)

	#Tratamento do quinto valor: fim de semana 0 se sim 1 se nao
	Entrada_NN.append(linha[4])

	#Tratamento do sexto valor ao 13 notas de cada genero
	Entrada_NN.append(linha[5])
	Entrada_NN.append(linha[6])
	Entrada_NN.append(linha[7])
	Entrada_NN.append(linha[8])
	Entrada_NN.append(linha[9])
	Entrada_NN.append(linha[10])
	Entrada_NN.append(linha[11])
	Entrada_NN.append(linha[12])

	with open("dados_tratados.csv","a") as arquivo:
		arquivo.write(str(Entrada_NN)+"\n")



def trata_generos(lista_generos):
	byte = [0,0,0,0,0,0,0,0]
	if ("Acao" in lista_generos):
		byte[0] = 1
	if ("Aventura" in lista_generos):
		byte[1] = 1
	if ("Comedia" in lista_generos):
		byte[2] = 1
	if ("Romance" in lista_generos):
		byte[3] = 1
	if ("Terror" in lista_generos):
		byte[4] = 1
	if ("Suspense" in lista_generos):
		byte[5] = 1
	if ("Animacao" in lista_generos):
		byte[6] = 1
	if ("Drama" in lista_generos):
		byte[7] = 1
	byte =int(''.join(str(e) for e in byte),2)
	return byte

class Rede_neural(object):
	def __init__(self):
		#Parametros da rede
		self.inputSize = 2
		self.outputSize = 1
		self.hiddenSize = 3

		#Pesos inicializados aleatoriamente
		self.W1 = np.random.randn(self.inputSize,self.hiddenSize) #Camada de entrada
		self.W2 = np.random.randn(self.hiddenSize,self.outputSize) #Camada oculta

		#Passo para frente

	def foward(self,X):
		self.z = np.dot(X,self.W1) # Multiplicando entrada pelos pesos
		self.z2 = self.sigmoid(self.z) # Passando pela funcao de ativacao
		self.z3 = np.dot(self.z2,self.W2) #Passando pela camada oculta
		o = self.sigmoid(self.z3) #Funcao de ativacao final
		return o

	def sigmoid(self, s):
		return 1/(1+np.exp(-s))

		#Passo para Trás
	def sigmoid_derivada(self,s):
		return s*(1-s)

	def backward(self, X, Y, o):
		self.o_erro = Y - o #Erro
		self.o_delta = self.o_erro * self.sigmoid_derivada(o) # Derivada da sigmoide

		self.z2_error = self.o_delta.dot(self.W2.T) #Erro na camada oculta
		self.z2_delta = self.z2_error * self.sigmoid_derivada(self.z2) #Aplicando derivada da sigmoide ao erro em z2

		self.W1 += X.T.dot(self.z2_delta) #ajustando os pesos da primeira camada
		self.W2 += self.z2.T.dot(self.o_delta) #ajustando os pesos da camada oculta

	#Treinamento
	def treinamento(self,X,Y):
		o = self.foward(X)
		self.backward(X,Y,o)

	def predicao(self):
		print ('Resultado previsto')
		print ('Entrada '+str(teste))
		print ('Saida '+str(self.foward(teste)))

if __name__ == '__main__':
	leitor()



#Teste da rede neural
"""
	RN = Rede_neural()
	erro = []
	for i in range(1000):
		print ('Entrada: '+ str(X))
		print ('Saida: '+ str(RN.foward(X)))
		print ('Esperado: '+str(Y))
		print ('Erro: '+ str(np.mean(np.square(Y-RN.foward(X)))))
		erro.append(np.sum((np.mean(np.square(Y-RN.foward(X))))))
		RN.treinamento(X,Y)
	RN.predicao()
	plt.plot(erro)
	plt.ylabel('erro')
	plt.show()
"""