#!/usr/bin/python
# -*- coding: utf-8 -*-

# Esse script serve para testar a generalização da rede neural treinada no
# script Multicamadas.py
#
# OBS: É necessário executar o Multicamadas.py ANTES de executar esse aqui!
#
# Author: João Marcos Meirelles da Silva
# creation date	: jan, 23th, 2019
# updated	: jan, 23th, 2019

# Carrega os dados de treinamento
peso = np.array([110, 113, 120,  125, 97])
pH   = np.array([6.0, 4.4, 3.5, 5.5, 5.0])

# Vetor de classificação desejada.
d = np.array([-1, -1, 1, 1, 1])

# ===============================================================
# TESTE DA REDE.
# ===============================================================

Error_Test = np.zeros(5)

for i in range(5):
    # Insere o bias no vetor de entrada.
    Xb = np.hstack((bias, X[:,i]))

    # Saída da Camada Escondida.
    O1 = np.tanh(W1.dot(Xb))            # Equações (1) e (2) juntas.      

    # Incluindo o bias. Saída da camada escondida é a entrada da camada
    # de saída.
    O1b = np.insert(O1, 0, bias)

    # Neural network output
    Y = np.tanh(W2.dot(O1b))            # Equações (3) e (4) juntas.

    Error_Test[i] = d[i] - Y
    
print(Error_Test)
print(np.round(Error_Test) - d)
