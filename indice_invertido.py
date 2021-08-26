import nltk
import sys
import pickle
import os

if (len(sys.argv) <= 1):
    print('Arquivo base não encontrado')
    sys.exit()

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('rslp')
nltk.download('mac_morpho')

sentencas_etiquetadas = nltk.corpus.mac_morpho.tagged_sents()
etiquetador_unigram = None

if(os.path.isfile('mac_morpho.pkl')):
    arquivo = open("mac_morpho.pkl", "rb")
    etiquetador_unigram = pickle.load(arquivo)
    arquivo.close()
else:
    etiq0 = nltk.DefaultTagger('N')
    etiquetador_unigram = nltk.UnigramTagger(sentencas_etiquetadas, backoff=etiq0)
    output = open('mac_morpho.pkl', 'wb')
    pickle.dump(etiquetador_unigram, output, -1)
    output.close()

# Lista de stopwords
stopwords = nltk.corpus.stopwords.words('portuguese')
# Extrai radicais
stemmer = nltk.stem.RSLPStemmer()

"""
Adicionalmente, seu programa deve
utilizar o pacote nltk para considerar qualquer palavra classificada como preposição(PREP), conjunção(KC, KS) ou
artigo(ART) como sendo stopword
"""

base = sys.argv[1]
arquivos_base = []
indice_invertido = {}
conjunto_radicais = set()
lista_classes_stopwords = ['PREP', 'KC', 'KS', 'ART']
lista_caracteres = ['.', '...', '..', ',', '!', '?']

# Lê o arquivo de base e salva os arquivos a serem lidos
with open(sys.argv[1], 'r') as reader:
    arquivos_base = [file_name.strip() for file_name in reader.readlines()]

for file in arquivos_base:
    with open(file, 'r') as reader:
        conteudo_arquivo = reader.read()
        palavras = nltk.word_tokenize(conteudo_arquivo)
        palavras = list(filter(
            lambda word: word not in stopwords and word not in lista_caracteres, palavras))
        etiquetas = etiquetador_unigram.tag(palavras)
        palavras = [palavra[0] for palavra in etiquetas if palavra[1] not in lista_classes_stopwords]
        for word in palavras:
            conjunto_radicais.add(stemmer.stem(word))

for radical in conjunto_radicais:
    indice_invertido[radical] = ''

for radical in indice_invertido:
    for file in arquivos_base:
        with open(file, 'r') as reader:
            ocorrencias = 0
            numero_arquivo = arquivos_base.index(file) + 1

            conteudo_arquivo = reader.read()

            palavras = nltk.word_tokenize(conteudo_arquivo)
            palavras = list(filter(
                lambda word: word not in stopwords and word not in lista_caracteres, palavras))
            palavras = [palavra[0] for palavra in etiquetador_unigram.tag(
                palavras) if palavra[1] not in lista_classes_stopwords]
            for word in palavras:
                if (stemmer.stem(word) == radical):
                    ocorrencias += 1
            if (ocorrencias > 0):
                indice_invertido[radical] = indice_invertido[radical] + \
                    f"{numero_arquivo},{ocorrencias} "

with open('indice.txt', 'w') as reader:
    for radical in indice_invertido:
        reader.writelines(f"{radical}: {indice_invertido[radical]}\n")
