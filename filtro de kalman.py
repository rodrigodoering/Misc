# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:18:35 2019

@author: rodri

Funções para computar filtro de kalman
"""

import math
import numpy as np

def fNormal(mu, sigma2, x):
    coeficiente = 1.0/ math.sqrt(2.0 * math.pi * (sigma2))
    expoente = math.exp(-0.5 * (x-mu)**2 / sigma2)
    return coeficiente * expoente

def atualizar_parametros(mu1, var1, mu2, var2):
    mu_ = (var2*mu1 + var1*mu2)/(var1 + var2)
    var_ = 1 / (1/var2 + 1/var1)
    return mu_, var_

def predict(mu1, var1, mu2, var2):
    mu_ = mu1 + mu2
    var_ = var1 + var2
    return mu_, var_

def filtro(medidas, motion, incerteza_medida, incerteza_motion, mu_inicial, incerteza_inicial):
    mu_ = mu_inicial
    var_ = incerteza_inicial
    for n in range(len(medidas)):
        mu_, var_ = atualizar_parametros(mu_, var_, medidas[n], incerteza_medida)
        print('Update: [{}, {}]'.format(mu_, var_))
        mu_, var_ = predict(mu_, var_, motion[n], incerteza_motion)
        print('Predict: [{}, {}]'.format(mu_, var_))
        