#!/usr/bin/env python
# encoding: utf-8
import numpy as np
from numpy import array
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
O que fazer com o bias??
Numero de camadas ocultas = 1
Numero de saidas de cada camada = 5,6,8
Funcao de ativacao = sigmoide
Batch ou online =  online
Numero de iteracoes de treinamento = 1000
Criterio de parada = erro < 10^-2 na nota
'''

Entradas = []
Saida_esperada = []

def leitor():
	with open('metade dos dados.csv','r') as arquivo:
		for linha in arquivo:
			linha_lida = linha.split(",")
			trata_linha(linha_lida)


def trata_linha(linha):
	Entrada_NN = []
	#Bias ?
	#Entrada_NN.append(-1)

	#Tratamento primeiro valor: generos do filme assistido
	byte = trata_generos(linha[0])
	Entrada_NN.append(byte)

	#Tratamento do segundo valor: filhos 1 se sim 0 se nao
	Entrada_NN.append(float(linha[1]))
	
	#Tratamento do terceiro valor: Ferias 1 se sim 0 se nao
	Entrada_NN.append(float(linha[2]))

	#Tratamento do quarto valor: Filme
	byte = trata_generos(linha[3])
	Entrada_NN.append(byte)

	#Tratamento do quinto valor: fim de semana 0 se sim 1 se nao
	Entrada_NN.append(float(linha[4]))

	#Tratamento do sexto valor ao 13 notas de cada genero
	temp_list =[]
	for x in range(5,13):
		temp_list.append(float(linha[x]))

	Saida_esperada.append(temp_list)
	Entradas.append(Entrada_NN)



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
		#Tamanho das camadas
		self.TamEntrada = 5
		self.TamOculta = 6
		self.TamSaida = 8

		#Pesos inicializados aleatoriamente
		self.W1 = np.random.randn(self.TamEntrada,self.TamOculta) #Camada de entrada
		self.W2 = np.random.randn(self.TamOculta,self.TamSaida) #Camada oculta

	#Passo para frente
	def Passo_Frente(self,X):
		self.z = np.dot(X,self.W1)
		self.z2 = self.func_at(self.z)
		self.z3 = np.dot(self.z2,self.W2)
		saida = self.func_at(self.z3)
		return saida

	def func_at(self, x):
		#sigmoide
		return 1/(1+np.exp(-x))

	def func_at_derivada(self,x):
		return x*(1-x)

	def Passo_tras(self,X,Y,saida):
		self.saida_erro = Y - saida
		self.saida_derivada = self.func_at_derivada(saida)

		self.gradiente = saida_erro*saida_derivada
		self.variacao_peso = Passo_aprendizado*gradiente*z2 + self.W2



if __name__ == '__main__':
	leitor()
	X = array(Entradas)
	Y = array(Saida_esperada)
	Rede_TOP = Rede_neural()
	saida = Rede_TOP.Passo_Frente(X)

	print(X)
	print(Y)
	print(saida)


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