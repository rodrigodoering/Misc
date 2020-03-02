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
from nltk.tokenize import word_tokenize
from SQLServer import SQLServer

os.chdir('C:\projeto_filtro')
print(os.getcwd())

input_1 = str(input('(INPUT): Iniciar ingestão do vocabulário [y/n]:  '))

if input_1.lower() == 'y':
    
    arquivos = [arquivo for arquivo in os.listdir() if 'docx' in arquivo]
    vocabulario = []
    stopwords = nltk.corpus.stopwords.words("portuguese")
    
    for arquivo in arquivos:
        doc = docx.Document(arquivo)
        texto = [p.text for p in doc.paragraphs]
        sentenca_normalizada = lambda x: x.lower().strip(string.punctuation).replace(",", "")
        texto = list(map(sentenca_normalizada,texto[1:]))
        array_termos = [word_tokenize(sentenca, language="portuguese") for sentenca in texto if len(sentenca) > 0]
        vocabulario += [termo for array in array_termos for termo in array if termo not in stopwords and len(termo) > 1]
    
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
    
    print('(SCRIPT): Vocabulário consolidado com %d termos' % len(vocabulario_final))

else:
    pass
    
def filtrar_lista(final):
    return [termo for termo in vocabulario_final if termo[-len(final):] == final]

def procurar(char):
    termos = [termo for termo in vocabulario_final if char in termo]
    print('(SCRIPT): %d termos retornados na pesquisa' % len(termos))
    for i, termo in enumerate(termos):
        print(i+1, termo)

input_2 = 'y'

while input_2 is not 'n':
    input_3 = str(input('(INPUT): Procurar no vocabulário:  '))
    print(procurar(input_3))
    input_2 = str(input('(INPUT): Buscar novo termo [y/n]:  '))

print('(SCRIPT): Fim do script')
