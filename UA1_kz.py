import os
import pandas as pd
import csv

#################### MATCHA ####################################
# Connecting the Match case

langs=["en","kaz"]               
domains=["block","material_sci"]

language=langs[1]                # Language selection    set kazakh , since this version is for the Kazakh language
domain=domains[0]                # domain selection

folder_path="path to texts of corpus"          # Path to domain texts in the corresponding language
files_name=[]
texts=dict()
true_terms_by_doc=dict()
all_texts=""

file_list = os.listdir(folder_path)

for filename in file_list:
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            text = file.read().replace("  ", " ").lower().replace("- ","-")
            all_texts+=text+". "

            texts["text_"+file_path[-6:]] = text
            files_name.append("text_"+file_path[-6:])

all_true_terms=[]

ann_path = "path to terms of corpus"    #  Path to domain terms in the corresponding language,    extract from txt file
with open(ann_path, 'r') as file:
    all_true_terms = [line.lower().replace("  "," ").replace("- ","-").strip() for line in file]  # Strips newline characters from each line

print('True terms all: ', len(set(all_true_terms)))

#################### ------- ####################################

! pip install stanza     # Download the Kazakh model for preprocessing text
import stanza
stanza.download('kk')
nlp = stanza.Pipeline('kk')   # Initialize the Kazakh pipeline

from torch import Tensor
!!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
!!pip install sentence-transformers

# Load the language model
# Remove hyphen from the default infix patterns
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

from sentence_transformers import SentenceTransformer, util
from typing import List
from transformers import XLMRobertaTokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

from collections import namedtuple
CandidateMWE = namedtuple('CandidateMWE',['text','head', 'sentence','self_encode', 'sent_encode'])
CandidateW=namedtuple('CandidateW',['text','lemma', 'self_encode' ])

all_tetxts_lemms=" ".join([word.lemma for sent in nlp(all_texts).sentences for word in sent.words])

import nltk
from nltk.util import ngrams
nltk.download("punkt")
import string
import spacy
import requests

from spacy.lang.char_classes import ALPHA, ALPHA_LOWER, ALPHA_UPPER
from spacy.lang.char_classes import CONCAT_QUOTES, LIST_ELLIPSES, LIST_ICONS
from spacy.util import compile_infix_regex
from operator import itemgetter
import pandas as pd


punc = list(string.punctuation)
punc.remove('-')
punc.remove("'")
punc2=list(string.punctuation)

punc_without = list(string.punctuation)+["»","«"]
punc_without.remove('-')
punc_without.remove("'")
punc_all=list(string.punctuation)+["»","«"]


def tokinizer(sent):                    # split texts on tokens
  index=0
  list_tok=[]
  for i in sent.words:
        list_tok.append((i.text.lower(), i.upos, index, i.lemma))
        index+=1
  return list_tok

def concatenate_ngrams(candidate):        # using for phrases, concatenate unigrams to one phrase
  cand_temp= []
  temp=''
  if type(candidate) !=type(str()):
     for w in candidate:
        if (w not in punc_without) and (len(temp)>0) and ((temp[-1]=="'") or ((w[0] not in punc_without) and (temp[-1] not in punc_without))):
             temp=temp+" "+str(w)
        else:
             temp=temp+str(w)
  else:
    temp=candidate
  temp = temp.lower()
  return str(temp)

list_seq_2=  [[["PROPN","NOUN"],"*"],
              ["ADJ",'*', ["PROPN","NOUN"], '*']]


def filter_ngrams_by_pos_tag(sentence, sequense):                  # extract multiword expressions candidates using pos-tag templates
    filtered_ngrams=[]
    t=0
    for seq in sequense:
        for i in range(len(sentence)):
            temp=[]
            temp_index=[]
            temp_pos=[]
            temp_lemma=[]
            checker=True


            if ((sentence[i][1] in seq[0]) or (sentence[i][1] == seq[0])) and (sentence[i][0] not in punc) :
               seq_index=0
               sent_index=0

               while seq_index<len(seq) and i+sent_index<len(sentence) and checker==True:
                   if (sentence[i+sent_index][1] in seq[seq_index]) or (sentence[i+sent_index][0] in seq[seq_index]) :
                       temp.append(sentence[i+sent_index][0])
                       temp_pos.append(sentence[i+sent_index][1])
                       temp_index.append(sentence[i+sent_index][2])
                       temp_lemma.append(sentence[i+sent_index][3])
                       seq_index += 1
                       sent_index += 1


                   elif seq[seq_index]=="*" and (sentence[i+sent_index][1] in seq[seq_index-1]):
                       if seq_index<len(seq)-1:
                              temp.append(sentence[i+sent_index][0])
                              temp_pos.append(sentence[i+sent_index][1])
                              temp_index.append(sentence[i+sent_index][2])
                              temp_lemma.append(sentence[i+sent_index][3])
                              sent_index += 1

                       elif seq_index==len(seq)-1:
                             if (len(temp)>1  or "-" in "".join(temp)) and (len(set("".join(temp)).intersection(set(punc)-set("-'")))==0):
                                 temp_2=temp.copy()
                                 temp_pos2=temp_pos.copy()
                                 temp_index2=temp_index.copy()
                                 temp_lemma2=temp_lemma.copy()
                                 if [temp_2, seq, temp_pos2, temp_index2,len(temp_2), len(concatenate_ngrams(temp_2)), temp_lemma2] not in filtered_ngrams and temp[-1] not in punc:
                                     filtered_ngrams.append([temp_2, seq, temp_pos2, temp_index2,len(temp_2),len(concatenate_ngrams(temp_2)),temp_lemma2])


                             if (i+sent_index)<len(sentence):
                                temp.append(sentence[i+sent_index][0])
                                temp_pos.append(sentence[i+sent_index][1])
                                temp_index.append(sentence[i+sent_index][2])
                                temp_lemma.append(sentence[i+sent_index][3])
                                sent_index += 1

                   elif seq[seq_index]=="*" and (sentence[i+sent_index][1] not in seq[seq_index-1]):
                      seq_index += 1

                   else:
                      checker=False
               #
               if seq_index==len(seq) and (len(temp)>1  or "-" in "".join(temp)) and len(set("".join(temp)).intersection(set(punc)-set("-'")))==0:
                    if [temp, seq, temp_pos, temp_index,len(temp),len(concatenate_ngrams(temp)),temp_lemma] not in filtered_ngrams and temp[-1] not in punc:
                         filtered_ngrams.append([temp, seq, temp_pos, temp_index,len(temp),len(concatenate_ngrams(temp)),temp_lemma])

    return  filtered_ngrams


def f_req_calc(mwe, all_txt, f_raw_req_list): # calculate frequency raw and rectified 
  temp=all_txt
  mwe_c=concatenate_ngrams(mwe)
  for i in f_raw_req_list:
    i_c=concatenate_ngrams(i[0])
    if mwe_c in i_c and mwe_c!=i_c and len(mwe)!=len(i[0]):
      temp=temp.replace(i_c," ")
  f=temp.count(mwe_c)
  return f

import string


class UnionFind:       # group candidates by same positions of their words in text
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] =  self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def group_items(lst):
    n = len(lst)
    uf = UnionFind(n)

    # Union sets with intersecting numbers
    for i in range(n):
        for j in range(i + 1, n):
            if set(lst[i][3]).intersection(lst[j][3]):
                uf.union(i, j)

    # Assign group numbers
    group_map = {}
    group_number = 1
    for i in range(n):
        root = uf.find(i)
        if root not in group_map:
            group_map[root] = group_number
            group_number += 1
        lst[i].append(group_map[root])

    return lst

class PhraseExtractor: # class for extract phrases
    def __init__(self, text, list_seq=list_seq_2,  cohision_filter=True, additional_text="1", f_raw_sc=9, f_req_sc=3):
        self.text = text
        self.cohision_filter=cohision_filter
        self.additional_text=additional_text
        self.f_req_sc=f_req_sc
        self.f_raw_sc=f_raw_sc
        self.list_seq=list_seq

    def extract_phrases(self):
        sent=tokinizer(self.text)
        mwe_list = filter_ngrams_by_pos_tag(sent, self.list_seq)

        mwe_list_n = [tuple(i[0]) for i in mwe_list]
        candidates = []
        for i in set(mwe_list_n):
            candidates.append(concatenate_ngrams(i))

        candidates = [i for i in candidates if ((i[-1] not in punc) and (i[-1] not in string.punctuation))]
        candidates = [i for i in candidates if len(set('1234567890').intersection(set(i))) == 0]



        #text for cohision filter and calculate frequency
        all_txt=self.additional_text

        #Filter based on frequency
        if  self.cohision_filter==True:

            f_raw_req_list=[]
            all_cand_r=[tuple(i[6]) for i in mwe_list]
            possible_mwe=sorted(set(all_cand_r), key=len, reverse=True)

            for mwe in possible_mwe:
                f_raw=all_txt.count(concatenate_ngrams(mwe))
                f_req=f_req_calc(mwe,all_txt, f_raw_req_list)
                f_raw_req_list.append([mwe,f_raw,f_req])

            mwe_f=[]
            f_raw_req_list_ind=[concatenate_ngrams(i[0]) for i in f_raw_req_list]

            for mwe in mwe_list:
                k=f_raw_req_list_ind.index(concatenate_ngrams(mwe[6]))
                mwe_f.append([mwe[0],f_raw_req_list[k][1], f_raw_req_list[k][2],mwe[3]])


            grouped_data = group_items(mwe_f)

            candidates=[]
            remover=[]
            df=pd.DataFrame(grouped_data, columns=["mwe","f raw","f req","index","group"])

            for i in range(1,len(set(df["group"]))+1):
                df_temp=df[df['group'] == i]
                while len(df_temp)>0:
                      max=df_temp["f req"].max()
                      cand=df_temp[df_temp["f req"]==max].values.tolist()[0]
                      if df_temp["f raw"].max()>0:
                            candidates.append(cand)

                      index=df_temp["index"][df_temp["f req"]==max].values.tolist()[0]
                      drop=df_temp.index[df_temp['index'].apply(lambda x: any(i in index for i in x))].tolist()
                      dd=df_temp.loc[drop].values.tolist()
                      df_temp=df_temp[~df_temp.index.isin(drop+index)]
                      remover+=dd

            data1=df[df["f req"]>=self.f_req_sc].values.tolist()
            data2=df[df["f raw"]>=self.f_raw_sc].values.tolist()

            cand_mwe=[concatenate_ngrams(i[0]) for i in candidates]
            cand_mwe1=[concatenate_ngrams(i[0]) for i in data1]
            cand_mwe2=[concatenate_ngrams(i[0]) for i in data2]

            cand=cand_mwe+cand_mwe1+cand_mwe2
            candidates = [i for i in cand if ((i[-1] not in punc) and (i[-1] not in string.punctuation))]

        return candidates


characters="аәбвгғдеёжзийкқлмнңоөпрстуұүфхһцчшщъыіьэюя 0123456789'-()’abcdefghijklmnopqrstuvwxyz"       # valid character values

def parse_candidates(text:str):
        # Remove the default rule that splits on hyphens
    # Process the text with spaCy
    doc = nlp(text)

    # Extract MWEs (noun phrases) from the text
    mwe_list = []
    single_noun_list=dict()
    candidate_list=[]
    temp_noun=[]
    for sent in doc.sentences:
        sent_encode=model.encode(sent.text, convert_to_tensor=True).to(device)

        extractor = PhraseExtractor(text=sent,
                            cohision_filter=True,
                            additional_text=all_tetxts_lemms,
                            list_seq=list_seq_2,
                            f_raw_sc=9,
                            f_req_sc=3)
        candidates = extractor.extract_phrases()


        for chunk in candidates:
                if len(set(chunk).intersection(set(characters)))==len(set(chunk)):
                    head=[word.text for word in nlp(chunk).sentences[0].words if word.head==0][-1]
                    candidate_list.append(CandidateMWE(chunk, head, sent.text, model.encode(chunk, convert_to_tensor=True).to(device), sent_encode))
                # print(f'Added candicate expression: {cleared_candidate}')

        for word in sent.words:             # extract single noun camdidates 
                    if word.upos in ['NOUN', 'PROPN']:
                       if len(set(word.text).intersection(set(characters)))==len(set(word.text)) and word.text not in temp_noun:
                           single_noun_list[word.text]=  CandidateW(word.text, word.lemma, model.encode(word.text))
                           temp_noun.append(word.text)

    return list(set(candidate_list)), single_noun_list.values()

def dist(wi_encode, wj_encode):  
    return util.pytorch_cos_sim(
        wi_encode,
        wj_encode
    )
def calculate_topic_score(expression_embedding, sentence_embedding)->float:           # calculate topic score
    """
    Calculate the topic score between a multiword expression and a sentence.

    Args:
        multiword_expression (str): The multiword expression.
        sentence (str): The sentence containing the expression.

    Returns:
        float: The topic score (cosine similarity) between the two embeddings.
    """
    # Load the distilbert-base-nli-mean-tokens model

    # Encode the multiword expression and sentence into embeddings
    # expression_embedding = model.encode(multiword_expression, convert_to_tensor=True)
    # sentence_embedding = model.encode(sentence, convert_to_tensor=True)

    # Calculate cosine similarity between the two embeddings
    similarity_score = util.pytorch_cos_sim(expression_embedding, sentence_embedding)

    # Extract the cosine similarity value from the tensor
    topic_score = similarity_score[0].item()

    return topic_score


def calculate_specificity_score(mw:CandidateMWE, full_encode:Tensor)->float:              # calculate specific score
    """
    Calculate the specificity score (SP) between a multiword expression (mw) and a list of words/multiword expressions (w).

    Args:
        mw (str): The multiword expression.
        w (list of str): The list of words/multiword expressions in the context.

    Returns:
        float: The specificity score (SP).
    """
    # Load the distilbert-base-nli-mean-tokens model
    # Calculate distances between mw and each word/phrase in w
    distances = dist(mw.self_encode,full_encode)

    # Calculate the mean of the distances
    specificity_score = distances.mean().item()


    return specificity_score


def detect_mw_terms(candidate_list:List[CandidateMWE], TSP:float = 0.2, Ttopic:float = 0.7)->List[CandidateMWE]:        # filter phrases by topic and specific score

    full_encode= torch.stack([wi.self_encode for wi in candidate_list], dim=0)

    temp_candidate = []
    for candidate in candidate_list:
        topic_score = calculate_topic_score(candidate.self_encode, candidate.sent_encode)
        sp_score=calculate_specificity_score(candidate, full_encode)
        if topic_score > Ttopic and sp_score > TSP:
            temp_candidate.append(candidate)
            # print(f'Added "{candidate.text}" topic score {topic_score}, specifity score {sp_score}')

    return temp_candidate

def detect_single_noun_terms(term_mws:List[CandidateMWE], single_noun_list:List[CandidateW], subtoken_threshold:int=4)->List[CandidateW]:        # filter single nouns
    term_nouns=[]
    for candidate in single_noun_list:
        #Check if the lemma of the noun is the same as any of the heads of the multiword expressions.
        is_term=False
        lemma_is_head=False
        for term_mw in term_mws:
            if term_mw.head==candidate.lemma:
                is_term=True
                term_nouns.append(candidate)
                # print(f'"{candidate.text}" is added by lemma: "{candidate.lemma}" is head of "{term_mw}"')
                break
        if is_term:
            continue
        #segment the word using a subword-unit segmentation and a vocabulary trained over a large general purpose corpus.
        subtokens=tokenizer.tokenize(candidate.text)
        if len(subtokens)>subtoken_threshold:
            term_nouns.append(candidate)
            # print(f'{candidate.text} is added by subtokens count: {len(subtokens)}')
    return term_nouns

def test_file(text):        # monitoring the stage of term extraction process

    print('_____________________________FIRST STEP_________________________________________')
    candidate_list, single_noun_list=parse_candidates(text)
    print(len(set(candidate_list)))
    print('_____________________________SECOND STEP___________________________________________________')
    term_mws=detect_mw_terms(candidate_list)
    print('_____________________________THIRD STEP______________________________________________________________')
    term_nouns=detect_single_noun_terms(term_mws, single_noun_list)
    print('____________________________________TESTING__________________________________________________________________')
    extracted_terms=set([txt.text.lower() for txt in term_mws+term_nouns])
    print(len(set(term_mws)))
    print(set(extracted_terms))

    return extracted_terms

def calculate_metrics(true_terms, extracted_terms):     # calculate metrics precision, recall, f1 score
    true_positives = len(true_terms.intersection(extracted_terms))
    false_positives = len(extracted_terms.difference(true_terms))
    false_negatives = len(true_terms.difference(extracted_terms))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score

def compare_sets(extracted_terms, true_terms):               
    true_detections = extracted_terms.intersection(true_terms)
    false_gaps = true_terms.difference(extracted_terms)
    false_detections = extracted_terms.difference(true_terms)

    return true_detections, false_gaps, false_detections


def report(true_terms, extracted_terms):
    # Расчет метрик
    precision, recall, f1_score = calculate_metrics(true_terms, extracted_terms)

    print("Precision:", precision) #Точность
    print("Recall:", recall) #Полнота
    print("F1 Score:", f1_score)
    # Выводит списки истинных обнаружений, ложных пропуски и ложных обнаружений
    true_detections, false_gaps, false_detections = compare_sets(extracted_terms, true_terms)

    print("Истинные обнаружения:", true_detections)
    print("Ложные пропуски:", false_gaps)
    print("Ложные обнаружения:", false_detections)
    return precision, recall, f1_score, true_detections, false_gaps, false_detections



f1_score_set=[]
whole_extracted_terms=[]
results=[]
kk=[]
stop_words=["туралы","кезде","кездесі","кезінде","арасында","арқылы","жылы","соңғы","басқалай","жылдың","арнайы","жылдардың","тамаша","байлынысты","бар","байланысты","басқа"]  # kazakh stop-words 

for file_name in files_name:
    print('____________________________________________________________________\n')
    print(f"checking file: {file_name}")

    extracted_terms=test_file(texts[file_name])
    results.append(extracted_terms)
    extracted_terms=[i for i in extracted_terms if i[0]!="-" and  i[-1]!="-"]
    extracted_terms=set(extracted_terms) - set([i for i in extracted_terms for w in stop_words if w in i])   # cleaning phrases with stop words

    print('____________________________________________________________________\n')

    true_terms_per_text=[i for i in set(all_true_terms) if i in texts[file_name]]

    precision, recall, f1_score, true_detections, false_gaps, false_detections=report(set(true_terms_per_text), set(extracted_terms))
    whole_extracted_terms+=extracted_terms
print('___________________________TOTAL_________________________________________\n')
whole_extracted_terms=set(whole_extracted_terms)
print('total result')
print(len(set(whole_extracted_terms)))
print(set(whole_extracted_terms))

precision, recall, f1_score, true_detections, false_gaps, false_detections=report(set(all_true_terms), whole_extracted_terms)

# lemmatization extracted terms and true terms for check the efficiency of UA1
tt_lemma=[]
for i in set(all_true_terms):
  temp=[]
  for s in nlp(i).sentences:
     for i, w in enumerate(s.words):
        if i==len(s.words)-1:
          temp.append(w.lemma)
        else:
            temp.append(w.text)
  tt_lemma.append(" ".join(temp))

ext_lemma=[]
for i in set(whole_extracted_terms):
  temp=[]
  for s in nlp(i).sentences:
     for i, w in enumerate(s.words):
        if i==len(s.words)-1:
          temp.append(w.lemma)
        else:
            temp.append(w.text)
  ext_lemma.append(" ".join(temp))

# print results for lematize true terms and extracted terms
precision, recall, f1_score, true_detections, false_gaps, false_detections=report(set(tt_lemma), set(ext_lemma))



