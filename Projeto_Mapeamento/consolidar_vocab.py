# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:14:48 2020

@author: rodri
"""

print('(SCRIPT): Importando módulos')
import os
import docx
import string
import nltk
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from SQLServer import SQLServer

os.chdir('C:\projeto_filtro\Programas Cursos')
print(os.getcwd())
    
arquivos = [arquivo for arquivo in os.listdir() if 'docx' in arquivo]
vocabulario = []
stopwords = nltk.corpus.stopwords.words("portuguese")

input_ = input('Começar?')

def processar_texto(array_strings):
    texto = list(map(lambda x: x.lower().strip(string.punctuation).replace(",", ""), array_strings[1:]))
    return [word_tokenize(sentenca, language="portuguese") for sentenca in texto if len(sentenca) > 0]

for arquivo in arquivos:
    doc = docx.Document(arquivo)
    texto = [p.text for p in doc.paragraphs]
    array_termos = processar_texto(texto)
    vocabulario += [termo for array in array_termos for termo in array if termo not in stopwords and len(termo) > 1]

os.chdir('C:\projeto_filtro')
print(os.getcwd())

sheets = ['Design de Projetos', 'Gestão de Startup', 'Finanças 2.0', 'Mkt Estrat. e Com. Corp','Liderança e Gestão de Equipes', 'Compras']

df = pd.read_excel('Estrutura Pós Ondemand - Todas.xlsx', sheet_name=sheets)

for dataframe in df.keys():
    texto = df[dataframe].AULA.dropna().values.tolist()
    array_termos = processar_texto(texto)
    vocabulario += [termo for array in array_termos for termo in array if termo not in stopwords and not bool(re.search(r"\d", termo)) and len(termo) > 1]

vocabulario = list(set(vocabulario))

database = SQLServer(dsn='DSN_TEST', auth='windows')
database.connect()
database.set_database('DB_POJETO_FILTRO_AULA')

vocabulario_final = []

def get_info_termo(termo): 
    return (termo[:2], termo[-2:], str(len(termo)))

for termo in vocabulario:
    inicio, fim, caracteres = get_info_termo(termo)
    filtro = "where caracteres = {} and inicio = '{}' and fim = '{}'".format(caracteres, inicio, fim)
    df_verbos = database.select('tb_verbos', columns='verbo', condition=filtro, verbose=False)
    
    if df_verbos is None:
        vocabulario_final.append(termo)

    elif termo not in df_verbos.verbo.values.tolist():
        vocabulario_final.append(termo)
    
    else:
        pass

def split_terms(array_termos):
    novos_termos = []
    remover = []
    
    for i, termo in enumerate(array_termos):
        if '/' in termo:
            remover.append(termo)
            splitted = termo.split('/')
            
            for termo_novo in splitted:
                novos_termos.append(termo_novo)
            
    array_termos = [termo for termo in array_termos if termo not in remover]
    
    for termo_novo in novos_termos:
        array_termos.append(termo_novo)
    
    return [termo for termo in array_termos if termo not in stopwords and len(termo) > 1]

vocabulario_final = split_terms(vocabulario_final)

print('(SCRIPT): Vocabulário consolidado com %d termos' % len(vocabulario_final))

TB_VOCAB = pd.DataFrame({'termo':vocabulario_final,
                         'inicio_termo':[termo[:2] for termo in vocabulario_final],
                         'tamanho_termo':[len(termo) for termo in vocabulario_final]}
)

database.query('TRUNCATE TABLE TB_VOCAB', commit=True)
database.insert(TB_VOCAB, 'TB_VOCAB')

print('Fim')

