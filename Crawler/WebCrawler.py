from urllib.request import urlopen
import pandas as pd
from bs4 import BeautifulSoup


class WebMining:

	def __init__(self, to_craw):
		# inicializa o objeto, como atributos o arquivo contendo o conjunto de links e uma lísta contendo as keywords para busca
		self.to_craw = to_craw
		self.estados = ['Acre','Alagoas','Amapá','Amazonas','Bahia','Ceará','Distrito Federal (Brasil)','Espírito Santo (estado)','Goiás',
           'Maranhão','Mato Grosso','Mato Grosso do Sul','Minas Gerais','Paraná','Paraíba','Pará','Pernambuco',
           'Piauí','Rio de Janeiro (estado)','Rio Grande do Norte','Rio Grande do Sul','Rondônia','Roraima',
           'Santa Catarina','Sergipe','São Paulo (estado)','Tocantins']


	def Crawler(self, delimitador):
		# Essa função lê o dataframe contendo os links e armazena o conteúdo HTML na variável self.html
		self.guia_links = pd.read_csv(self.to_craw, sep=delimitador, header=0)
		self.html = []
		for link in self.guia_links.link:
			get = urlopen(link)
			self.html.append(get.read())


	def Scraper(self):
		# Esse bloco extrai os dados de taxa de homicídio por 100 mil habitantes
		# A indexação dos valores é feito na ordem alfabética dos estados
		conteudo_html = BeautifulSoup(self.html[0], 'html.parser')
		conteudo = str(conteudo_html('tr'))
		texto_alvo = [[] for i in range(len(self.estados))]
		for i, estado in enumerate(self.estados):
			start = conteudo.index('title="'+estado+'"')
			end = conteudo[start:].index('</tr>')
			texto_alvo[i] = str(conteudo[start:start+end])
		texto_alvo = [termo.split() for termo in texto_alvo]    
		self.valores_taxa_homicidio = [float(termo[len(termo)-2][4:].replace(',','.')) for termo in texto_alvo] 

		# Esse bloco extrai as informações da incidência de pobreza
		# A indexação é feita de acordo com a ordem em que os estados e seus respectivos valores estão dispostos no site
		# Por isso também é gerado uma lista contendo a ordem exata dos estados para poder gerar o dataset final
		conteudo_html = BeautifulSoup(self.html[1], 'html.parser')
		filtro = conteudo_html.find_all('td', attrs={'align':'right'})
		self.valores_pobreza = [float(tag.text.replace(',','.').replace('%','')) for tag in filtro]
		filtro = conteudo_html.find_all('td', attrs={'align':'left'})
		self.ordem_estados_pobreza = [termo.find('a').attrs['title'] for termo in filtro if termo.find('a').attrs['title'] in self.estados]

		# Esse bloco extrai as informações da página da Exame sobre investimento em segurança pública
		# A indexação é feita de acordo com a ordem em que os estados e seus respectivos valores estão dispostos no site
		# Por isso também é gerado uma lista contendo a ordem exata dos estados para poder gerar o dataset final
		conteudo_html = BeautifulSoup(self.html[2], 'html.parser')
		filtro = conteudo_html.find_all('strong', attrs={'class':'gallery-title'})
		temp = [termo.text.replace('com segurança pública por pessoa','').replace('- investe R$ ','')[8:] for termo in filtro]
		temp = [termo.replace('-','') for termo in temp]
		temp = [termo[1:] if termo[0] == ' ' else termo for termo in temp]
		temp.pop()
		self.valores_seguranca = [int(termo[-4:]) for termo in temp]
		self.ordem_estados_seguranca = [termo[:-4] for termo in temp]

		# Padroniza o nome de determinados estados de acordo com o padrão dos três demais listas de estados
		for i, estado in enumerate(self.ordem_estados_seguranca):
			if estado == 'São Paulo ':
				self.ordem_estados_seguranca[i] = 'São Paulo (estado)'
			if estado == 'Rio de Janeiro ':
				self.ordem_estados_seguranca[i] = 'Rio de Janeiro (estado)'
			if estado == 'Espírito Santo ':
				self.ordem_estados_seguranca[i] = 'Espírito Santo (estado)'
			if estado == 'Distrito Federal ':
				self.ordem_estados_seguranca[i] = 'Distrito Federal (Brasil)'

		# Esse bloco extrai as informações do índice GINI
		# A indexação é feita de acordo com a ordem em que os estados e seus respectivos valores estão dispostos no site
		# Por isso também é gerado uma lista contendo a ordem exata dos estados para poder gerar o dataset final
		conteudo_html = BeautifulSoup(self.html[3], 'html.parser')
		filtro = conteudo_html.find_all('td', attrs={'align':'right'})
		self.valores_gini = [float(tag.text.replace(',','.')) for tag in filtro]
		filtro = conteudo_html.find_all('td', attrs={'align':'left'})
		self.ordem_estados_gini = [termo.find('a').attrs['title'] for termo in filtro]


	def ConstruirDataframe(self):
		# Esse bloco remove qualquer espaço residual que tenha ficado do processo de Scrap e padroniza todas as strings
		estados = [estado.replace(' ','') for estado in self.estados]
		ordem_estados_pobreza = [estado.replace(' ','') for estado in self.ordem_estados_pobreza]
		ordem_estados_seguranca = [estado.replace(' ','') for estado in self.ordem_estados_seguranca]
		ordem_estados_gini = [estado.replace(' ','') for estado in self.ordem_estados_gini]

		# Três matrizes são geradas para facilitar o processo de construção do Data Frame uma vez que os valores vem em ordens de estado diferentes
		matriz_estados = [estados, ordem_estados_pobreza, ordem_estados_seguranca, ordem_estados_gini]
		matriz_valores = [self.valores_taxa_homicidio, self.valores_pobreza, self.valores_seguranca, self.valores_gini]
		matriz_vetores = []

		# Construção dos vetores dos estados contendo seus respectivos valores para cada indicador
		for estado in estados:
			vetor = [matriz_valores[i][matriz_estados[i].index(estado)] for i in range(4)]
			matriz_vetores.append(vetor)

    	# Construção do DataFrame que será utilizado na modelagem
		self.indicadores = ['TaxaHomicidio','IndicePobreza','InvestimentoSeguranca','GINI']
		self.df = pd.DataFrame(matriz_vetores, columns = self.indicadores, index = estados)


	def Normalizar(self):
		# Inverte os valores de InvestimentoSeguranca, para que todos os indicadores sigam a ordem de quanto maior, pior
		self.df.InvestimentoSeguranca = [(valor*-1) for valor in self.df.InvestimentoSeguranca]

		# Normaliza todos os valores com a fórmula MinMax Scale
		for coluna in self.indicadores:
			self.df[coluna] = [(x - self.df[coluna].min())/(self.df[coluna].max() - self.df[coluna].min()) for x in self.df[coluna]]
		return self.df


	def ExecutarCrawler(self, delimitador, normalizar=False):
		# Executa as funções como um pipeline
		self.Crawler(delimitador)
		self.Scraper()
		self.ConstruirDataframe()
		if not normalizar:
			return self.df
		else:
			return self.Normalizar()




