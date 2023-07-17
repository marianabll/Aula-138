# Bibliotecas de pré-processamento de dados de texto
import nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import json
import pickle
import numpy as np
import random

words=[] #lista de palavras-raiz únicas nos dados
classes = [] #lista de tags únicas nos dados
#lista dos pares de (['palavras', 'da', 'frase'], 'tags')
pattern_word_tags_list = [] 
ignore_words = ['?', '!',',','.', "'s", "'m"]

train_data_file = open('intents.json').read()
intents = json.loads(train_data_file)

def get_stem_words(words, ignore_words):
    stem_words = []
    for word in words:
        if word not in ignore_words:
            w = stemmer.stem(word.lower())
            stem_words.append(w)
    return stem_words
 
def create_bot_corpus(words, classes, pattern_word_tags_list, ignore_words):

    for intent in intents['intents']:

        # Adicione todos os padrões e tags a uma lista
        for pattern in intent['patterns']:            
            pattern_word = nltk.word_tokenize(pattern)            
            words.extend(pattern_word)                        
            pattern_word_tags_list.append((pattern_word, intent['tag']))
              
        # Adicione todas as tags à lista classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
            
    stem_words = get_stem_words(words, ignore_words) 
    stem_words = sorted(list(set(stem_words)))
    classes = sorted(list(set(classes)))

    return stem_words, classes, pattern_word_tags_list


# Conjunto de Dados de Treinamento: 
# Texto de Entrada ----> como Saco de Palavras 
# Tags-----------------> como Etiqueta

def bag_of_words_encoding(stem_words, pattern_word_tags_list):  
    bag = []
    for word_tags in pattern_word_tags_list:

        pattern_words = word_tags[0] 
        bag_of_words = []
        stem_pattern_words= get_stem_words(pattern_words, ignore_words)
        
        for word in stem_words:            
            if word in stem_pattern_words:              
                bag_of_words.append(1)
            else:
                bag_of_words.append(0)
        bag.append(bag_of_words)
    return np.array(bag)

