# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:56:43 2020

@author: rodri
"""

import pandas as pd
import os
from textdistance import jaro_winkler as jk
from ppt_reader import ppt

vocab = pd.read_excel('vocabulário_final.xlsx')
os.chdir('C:\\Users\\rodri\\Desktop\\Relacionados a Código\\Projeto Live Rodrigo Raissa\\Noturno\Sala')
arquivos = [file for file in os.listdir() if 'pptx' in file and 'aula' in file.lower() and '~' not in file]

strings = []

for arquivo in arquivos:
    print(arquivo)
    ppt_ = ppt(arquivo)
    texto = ppt_.processar_texto()
    
    representacao_final = []
    
    for termo in vocab.termo:
        distancias = [jk.distance(termo, termo_ppt) for termo_ppt in texto]
        busted = [termo_ppt for termo_ppt, dist in zip(texto, distancias) if dist <= .1]
        
        if len(busted) == 0:
            pass
        
        else:
            for termo_ in busted:
    
                representacao_final.append(termo)
    
    representacao = list(set(representacao_final))
    strings.append(','.join(representacao))

df_final = pd.DataFrame({'Arquivo':arquivos, 'Descrição':strings})
df_final.to_excel('df.xlsx')



        
