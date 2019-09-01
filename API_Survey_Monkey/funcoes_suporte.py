import requests
import pandas as pd


def make_request(url):
    '''
    Faz o request
    Retorna os dados em json
    '''
    client = requests.session()
    headers = {
        "Authorization": "bearer %s" % "<TOKEN DE ACCESSO - CADA APLICATIVO DO SM TEM UM TOKEN DE ACESSO A API>",
        "Content-Type": "application/json"
    }
    return client.get(url, headers=headers).json()
 
    
 def get_resposta(url):
    '''
    Recebe o caminho do questionário em específico
    retorna um dict id pergunta: resposta
    '''
    # extraí os dados 
    dados_survey_ = make_request(url)
    # armazena apenas a informação desejada dentro do arquivo JSON completo
    respostas = [pagina['questions'] for pagina in dados_survey_['pages']]
    respostas_final = {}
    # Percorre as respostas, armazenando o id de pergunta e o id de resposta
    for pagina in respostas:
        for questao in pagina:
            # filtra por tipo de pergunta
            if 'text' in questao['answers'][0].keys():
                respostas_final[questao['id']] = questao['answers'][0]['text']
            else:
                respostas_final[questao['id']] = questao['answers'][0]['choice_id']
    # retorna o dicionário
    return respostas_final 
    
    
def traduzir(lista_respostas, id_resposta):
    '''
    Recebe a lista de respostas e as opções das respostas com suas respectivas representações no formulário 
    Retorna uma lista contendo a representação de formulário da resposta
    Essa função é criada para ser usada pela função 'gerar_dataframe'
    '''
    return [tupla[1] for tupla in lista_respostas if id_resposta in tupla][0]


def gerar_dataframe(questionarios_respondidos, dados_questionario):
    '''
    Recebe as respostas do questionário e os dados do questionário
    Retorna um dataframe contendo as respostas em sua representação de formulário
    Cada coluna do dataframe representa uma pergunta existente no formulário
    Cada registro representa um questionário preenchido e os valores do dataframe são as respostas (em sua representação real)
    Extraídas com a função 'traduzir'
    '''
    # Cria um dicionário para armazenar as respostas
    respostas_traduzido = {id_:[] for id_ in respostas_final}
    # intera sobre cada um dos questionários preenchidos
    for id_ in questionarios_respondidos:
        # isola as respostas
        respostas = questionarios_respondidos[id_] 
        # intera sobre todas as questões existentes no formulário
        for question_id in dados_questionario.keys():
            # verifica se a questão foi respondida naquele questionário em específico
            # se ela não foi respondida, o seu ID não estará presente e será representado como None
            if question_id not in respostas.keys():
                respostas_traduzido[id_].append(None)
            else:
                resposta = respostas[question_id]
                if not dados_questionario[question_id]:
                    # adiciona diretamente a resposta cado seja uma pergunta aberta
                    respostas_traduzido[id_].append(resposta)
                else:
                    lista_respostas = dados_questionario[question_id]
                    # adiciona a representação real da resposta
                    respostas_traduzido[id_].append(traduzir(lista_respostas, resposta))  
                    
    # retorna o dataframe              
    return pd.DataFrame(respostas_traduzido).T
