################### INSTALL LIBRARIES AND MODELS ###################################
import os
import pandas as pd
import csv
from IPython.display import clear_output

from torch import Tensor
!!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
!! pip install spacy
!! python -m spacy download en_core_web_sm
!!pip install sentence-transformers

from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer, util
from typing import List
from transformers import BertTokenizer

import stanza
stanza.download('kk')
nlp = stanza.Pipeline('kk')
clear_output()

import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
model_2 = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)

stop_words=['басқалай', 'сенің', 'бірнеше', 'қазіргі', 'егерде', 'соншалықты', 'жылдардың', 'осы', 'неше', 'себебі', 'секілді', 'осылайша', 'бірақ та', 'қайда', 'кездесі', 'ешқандай', 'қашан', 'жоғары', 'өте', 'бірде', 'өз', 'де',
            'анау', 'сияқты', 'біреу', 'қалай', 'қайталай', 'егер', 'мынау', 'олар', 'арасында', 'сен', 'ешқашан', 'бірдеңе', 'не', 'оң', 'тағы да', 'бар', 'дегенмен', 'ешкім', 'қайта', 'алайда', 'қазір', 'да', 'барлық', 'әркім',
            'бәрі', 'байлынысты', 'жоқ', 'жиі', 'сондықтан да', 'біз', 'кейін', 'сол', 'соңғы', 'мүмкін', 'олай болса', 'айналасында', 'төмен', 'ішінде', 'болуы мүмкін', 'қаншалықты', 'бәрібір', 'соңында', 'дейін', 'сіз', 'осында',
            'туралы', 'олардікі', 'әрдайым', 'қандай', 'қалайша', 'мен', 'бірге', 'осылай', 'оның', 'ал', 'болатын', 'әр', 'және', 'алыс', 'әрі', 'кез-келген', 'сондықтан', 'кейбіреулер', 'бұрын', 'неге', 'кейінірек', 'арнайы', 'басқа',
            'байланысты', 'ертең', 'ғана', 'кеше', 'сіздің', 'сонда', 'кім', 'тек', 'әлдеқайда', 'жылы', 'тамаша', 'сирек', 'барлығы', 'бірақ', 'кезде', 'бастап', 'бұл', 'қай жерде', 'кезінде', 'үшін', 'ол', 'болып табылады', 'сондай',
            'біздің', 'мұнда', 'менің', 'кейде', 'арқылы', 'болды', 'тағы', 'жылдың', 'сыртында', 'әрқашан', 'жақын', 'олардың', 'онда', 'сондай-ақ', 'қанша', 'біздікі', 'бәріміз', 'бүгін', 'ештеңе', 'көптеген']

import string
punc = list(string.punctuation)+["»","«"]
punc.remove('-')
punc.remove("'")
punc2=list(string.punctuation)+["»","«"]


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF
from sklearn import decomposition
from sklearn.preprocessing import normalize

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer, RegexpTokenizer
from nltk.util import ngrams
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk import stem

import numpy as np, re, pandas as pd
import os, glob, csv
import string
import nltk
import nltk, re, string, collections

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
#################### ------- ####################################


#################### MATCHA CORPUS ####################################
# Connecting the Match case
path_to_corpus="path to corpus" #write  path to corpus

langs=["en","kaz"]
domains=["block","material_sci"]
language=langs[1]                # Language selection    set kazakh , since this version is for the Kazakh language
domain=domains[0]                # domain selection

folder_path=path_to_corpus+"/Matcha-main/"+language+"/"+domain+"/annotated/texts"          # Path to domain texts in the corresponding language
files_name=[]
texts=[]
true_terms_by_doc=dict()
all_texts=""

all_true_terms=[]
ann_path = path_to_corpus+"/Matcha-main/"+language+"/"+domain+"/annotated/annotations/unique_annotation_lists/"+domain+"_"+language+"_terms.csv"    #  Path to domain terms in the corresponding language,    extract from csv file
df = pd.read_csv(ann_path, delimiter=';')
data_list = df.values.tolist()
all_true_terms=[i[0].lower() for i in data_list]

file_list = os.listdir(folder_path)

for filename in file_list:
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            text = file.read().replace("  ", " ").replace("- ","-")
            all_texts+=text+". "
            true_terms_by_doc["text_"+file_path[-6:]]=[i for i in all_true_terms if i in text]
            texts.append(text)
            files_name.append("text_"+file_path[-6:])

true_terms_mwe=[w for w in all_true_terms if (len(w.split(" "))>1)]+ [w for w in all_true_terms if len(w.split("-"))>1 ]
true_terms_uni=[w for w in all_true_terms if w not in true_terms_mwe ]
print('True terms all: ', len(set(all_true_terms)))
print('True terms uni: ', len(set(true_terms_uni)))
print('True terms mwe: ', len(set(true_terms_mwe)))

#################### ------- ####################################

#################### EXTRACT UNIGRAMS ####################################

unigrams=[]
abb=[]
sents=[]
for tex in texts:
        text_token=nlp(tex)
        se=[sent.text.lower() for sent in text_token.sentences if len(sent.text) > 0]
        sents+=se
        for_abb=[i.text for sent in text_token.sentences for i in sent.words]
        for_abb=[w for w in for_abb if w.lower() not in stop_words and len(set(w).intersection(set(list("0123456789")+list(punc2))))<len(set(w))]
        for_abb=[w  for w in for_abb if w not in punc2 and len(set(w).intersection(set(punc2)))==0]
        for_abb=[i for i in for_abb if sum(1 for char in i if char.isupper())>1 and len(i)<30]
        abb+=for_abb

        text_token=[i.text.lower() for sent in text_token.sentences for i in sent.words  if i.pos in ["PROPN","NOUN","ADJ"]]
        text_token=[w for w in text_token if w.lower() not in stop_words and len(set(w).intersection(set(list("0123456789")+list(punc2))))<len(set(w))]
        text_token=[w for w in text_token if w not in punc2 and len(set(w).intersection(set(punc)))==0]
        unigrams+=text_token
abb1=[i.lower() for i in abb]

#################### ------- ####################################

#################### EXTRACT PHRATHES ####################################

all_tetxts_lemms=" ".join([word.lemma for sent in nlp(all_texts).sentences for word in sent.words])
!pip install git+https://github.com/term-extraction-project/multi_word_expressions.git

from multi_word_expressions import PhraseExtractor
clear_output()
characters="аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя 0123456789'-()’abcdefghijklmnopqrstuvwxyz"
candidate_list=[]
for text in texts:
    doc = nlp(text)
    for sent in doc.sentences:
          extractor = PhraseExtractor(
                        text=sent,
                        lang="kk",
                        #stop_words=custom_stop_words_en,   # Пользовательские стоп-слова, по умолчанию установлены
                        #list_seq=custom_pos_patterns_en,   # Пользовательские POS-шаблоны, по умолчанию установлены
                        cohision_filter=True,               # Фильтрация по когезии
                        additional_text=all_tetxts_lemms,    # Дополнительный текст (если требуется)
                        f_raw_sc=9,                         # Частотный фильтр для сырого текста
                        f_req_sc=3)                         # Частотный фильтр для отобранных кандидатов
          candidates = extractor.extract_phrases()
          candidate_list+=candidates
candidate_list=[i for i in candidate_list if set(i.lower()).intersection(set(characters))==set(i.lower())]
#################### ------- ####################################


#################### ENCODING ####################################
abb_i=[i[0].lower() for i in data_list if i[1]=="Abb"]

sents_en=[[i,model_2.encode(i, normalize_embeddings=True)] for i in sents]
cos_uni=[]
for i in set(unigrams):
  i_en= model_2.encode(i, normalize_embeddings=True)
  for s in sents_en:
     if i.lower() in s[0].lower():
       s_en=s[1]
       topic_score=model_2.similarity(i_en, s_en).tolist()[0][0]
       cos_uni.append([i,topic_score])
cos_mwe=[]
for i in set(candidate_list):
  i_en= model_2.encode(i, normalize_embeddings=True)
  for s in sents_en:
     if i.lower() in s[0].lower():
       s_en=s[1]
       topic_score=model_2.similarity(i_en, s_en).tolist()[0][0]
       cos_mwe.append([i,topic_score])
uni=[i[0] for i in cos_uni if i[1]>0.8]
mwe=[i[0] for i in cos_mwe if i[1]>0.8]

#################### ------- ####################################


#################### LEMMATIZATION ####################################

tt_lemma_u=[i.lemma for tt in set(true_terms_uni)-set(abb_i)  for s in nlp(tt).sentences  for i in s.words]
ext_lemma_u=[i.lemma for tt in uni  for s in nlp(tt).sentences  for i in s.words]

tt_lemma_m=[]
for i in set(true_terms_mwe):
  temp=[]
  for s in nlp(i).sentences:
     for i, w in enumerate(s.words):
        if i==len(s.words)-1:
          temp.append(w.lemma)
        else:
            temp.append(w.text)
  tt_lemma_m.append(" ".join(temp))

ext_lemma_m=[]
for i in set(mwe):
  temp=[]
  for s in nlp(i).sentences:
     for i, w in enumerate(s.words):
        if i==len(s.words)-1:
          temp.append(w.lemma)
        else:
            temp.append(w.text)
  ext_lemma_m.append(" ".join(temp))
#################### ------- ####################################


#################### NMF ####################################

# Customize the number of topics and terms
num_components = 5   # Selecting the number of topics
num_terms = 200         # Selecting the number of terms to be extracted from each topic

# Create a custom_tokenizer lambda function that accepts a string x and returns a list containing that string
custom_tokenizer = lambda x: [x]

# Initializing the NMF (Non-Negative Matrix Factorization) model with specified parameters
vectorizer = TfidfVectorizer( tokenizer = custom_tokenizer,        # Using custom_tokenizer for tokenization
                              lowercase = False,                   # Not convert text to lower case
                              use_idf = False,                     # Disabling the use of inverse document frequency (IDF)
                              ngram_range = (1,3),                 # Use of unigrams, bigrams and trigrams
                              max_df = 0.5,                        # Ignoring tokens that occur in more than 50% of documents
                              token_pattern = None)                # Using custom_tokenizer, thus a token template is not required
X = vectorizer.fit_transform(ext_lemma_m+ext_lemma_u+abb1)    # Converting all_ngrams text data list to sparse TF-IDF matrix

# Initialization of NMF (Non-Negative Matrix Factorization) model object
model = NMF(n_components = num_components,     # Number of components (topics) in the factorization
            init = 'nndsvdar',                 # Method for initializing matrices W and H
            random_state = 27,                 # Setting the initial state for reproducibility
            max_iter = 1500,                   # Maximum number of iterations for optimization
            alpha_H = 0.0,                     # Regularization parameter for matrix H
            l1_ratio = 0.5,                    # Ratio of L1 to L2 regularization
            shuffle = False)                   # Disabling shuffling of data at each iteration

# Applying the NMF model to fit and transform the input sparse TF-IDF matrix (X)
W = model.fit_transform(X)                      # Matrix W - weights of topics in documents
H = model.components_                           # Matrix H - topics in terms
terms = vectorizer.get_feature_names_out()      # Extracting the list of terms (words) from the vectorizer

def calculate_metrics(true_terms, extracted_terms):
    true_positives = len(true_terms.intersection(extracted_terms))
    false_positives = len(extracted_terms.difference(true_terms))
    false_negatives = len(true_terms.difference(extracted_terms))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score


def get_topics_terms(terms, H, num_terms):
    top_term_indices = np.argsort(-H, axis=1)[:, :num_terms]         # Sorting term indices in each topic based on their weights in descending order
    topics_terms = [                                                 # Creating a list of terms for each topic using the sorted indices
           [terms[term_index] for term_index in topic]
           for topic in top_term_indices
          ]
    return topics_terms

def get_top_topic(topics_terms, W):
    topic_sums = np.sum(W, axis=0)                        # Calculating the sum of weights for each topic across all documents
    top_topic_index = np.argmax(topic_sums)               # Finding the index of the topic with the highest sum of weights
    top_topic_terms = topics_terms[top_topic_index]       # Extracting the terms for the topic with the highest sum of weights
    return top_topic_terms

def final_metrics(num_components, num_terms):
    topics_terms = get_topics_terms(terms, H, num_terms)    # Obtaining terms for each topic using the specified number of terms
    top_topic_terms = get_top_topic(topics_terms, W)        # Obtaining terms for the top topic with the highest sum of weights

    precision, recall, f1_score = calculate_metrics(set(tt_lemma_u+tt_lemma_m+abb_i), set(top_topic_terms))    # Calculating precision, recall, and F1-score for true and extracted terms
    print(len(set(top_topic_terms)))
    print(len(set(all_true_terms).intersection(set(top_topic_terms))))
    return precision, recall, f1_score,top_topic_terms

precision, recall, f1_score,top_topic_terms = final_metrics(num_components, num_terms)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)







