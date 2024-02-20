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

!git clone https://github.com/AylaRT/ACTER.git

# Extracting texts and terms from the ACTER corpus
all_text = ""          # Contains all texts from the specified folder
given_terms = []       # Contain all the correct terms

domains = ["corp", "equi", "wind", "htfl"]   # List of ACTER corpus topics
langs   = ["en", "fr", "nl"]                 # List of languages in the ACTER corpus

domain   = domains[0]   # Selecting the ACTER corpus topic
language = langs[0]     # Selecting the language of the ACTER corpus

# Extract texts from a specified folder and convert from txt to string format
folder_path   = '/content/ACTER/' + language + "/" + domain + "/annotated/texts_tokenised"
file_list = os.listdir(folder_path)
for filename in file_list:
    if filename.endswith('.txt'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            file_content = file.read()
            all_text += file_content

# Extract true terms from the specified folder and convert from tsv to list format
ann_path = '/content/ACTER/' + language + "/" + domain + "/annotated/annotations/unique_annotation_lists/" + domain + "_en_terms_nes.tsv"
with open(ann_path, 'r', newline='') as tsv_file:
    reader = csv.reader(tsv_file, delimiter='\t')
    for row in reader:
        given_terms.append(row[0].lower())

# Connecting stop-words and adding additional stop-words
add_stop = ['said', 'say', '...', 'like', 'cnn', 'ad', 'etc', 'aforementioned', 'accordance', 'according', 'do', 'did', 'does', 'done', 'possible',
            'consider', 'concern', 'concerning', 'conсerned', 'regard', 'regarding', 'regards', 'have', 'has', 'had', 'having', 'refer', 'referred', 'shall']
stop_words = ENGLISH_STOP_WORDS.union(add_stop)

# Saving stop-words in txt format
text_file = open('stop_words.txt', "w")
n = text_file.write(str(stop_words))
text_file.close()

punc = list(string.punctuation)   # Connecting the punctuation list
wnl = stem.WordNetLemmatizer()    # Connecting a module for lemmatizing words
lemmatizer = WordNetLemmatizer()

# Function for filtering trigrams
def filtering_trigrams(trigrams):
    filt = []
    for trigram in trigrams:
        first_word, second_word, third_word = trigram                      # Separating the trigrams into words
        trigram_tags = pos_tag(word_tokenize(' '.join(trigram)))           # Tokenize and tag words in the trigram
        if first_word in stop_words or third_word in stop_words:           # Check if 1st or 3rd word in the trigram is a stop word
            continue
        first_pos = trigram_tags[0][1]                                     # Check if the first or third word is a preposition, pronoun, or conjunction
        third_pos = trigram_tags[2][1]
        if first_pos in ['IN', 'PRP', 'CC'] or third_pos in ['IN', 'PRP', 'CC', 'JJ', 'VBD']:
            continue
        if len(trigram) == 3 and second_word == 'of':                       # Keep trigrams with a length of 3 and the second word is 'of'
            filt.append(trigram)
        else:
            words = [word for word in trigram if word not in stop_words]    # Removing stop words from the trigram and append
            filt.append(tuple(words))
    return [tple for tple in filt if len(tple) == 3]

# Split text into tokens
tokenizer = TweetTokenizer()                     # Connecting a module for tokenizing text
text_tokens = tokenizer.tokenize(all_text)       # Tokenization of text
text_tokens = [ngram for ngram in text_tokens if not any(char.isdigit() for char in ngram)]   # Removing digits from the text
text_tokens = [each.lower() for each in text_tokens if not each.isdigit()]                    # Lowercasing and removing numbers

# Creating unigrams and filtering out stop-words and punctuation
unigrams = [word for word in text_tokens
            if (word not in stop_words) and             # Removing stop-words
               (word not in punc) and                   # Removing punctuation
               (word not in string.punctuation)]

# Creating bigrams and filtering out stop-words and punctuation
bigrams_not_filtred = list(ngrams(text_tokens, 2))              # Splitting text into bigrams
filtered_bigrams = [bi for bi in bigrams_not_filtred
                    if not any((word in stop_words) or          # Removing stop-words
                               (word in punc) or                # Removing punctuation
                               (word in string.punctuation)
                               for word in ' '.join(bi).split())]
bigrams = [' '.join(word_tuple) for word_tuple in filtered_bigrams]

# Creating trigrams and filtering out stop-words and punctuation
trigrams_not_filtred = list(ngrams(text_tokens, 3))
filtered_trigrams = [tri for tri in trigrams_not_filtred
                     if not any(
                         (word in punc) or             # Removing punctuation
                         (word in string.punctuation)
                         for word in ' '.join(tri).split())]
filtered_trigrams = filtering_trigrams(filtered_trigrams)  # Calling the function for filtering trigrams
trigrams = [' '.join(word_tuple) for word_tuple in filtered_trigrams]

# Combining all n-grams into one list
n_grams = unigrams + bigrams + trigrams
print(len(n_grams))

# Function for text processing with lemmatization
def process_text(text):
    text = [ngram for ngram in text if not any(char in string.punctuation for char in ngram)]
    text = [wnl.lemmatize(each) if wnl.lemmatize(each) != each else each for each in text]      # Lemmatizing (but keeping original form for non-dictionary words)
    text = [w for w in text if (w not in punc) and (w not in stop_words)]                       # Removing punctuations and stop words
    text = [each for each in text if len(each) > 1 and ' ' not in each]                         # Removing single letters and words with spaces
    return text

# Function for processing trigrams with lemmatization
def process_trigram(text):
    text = [ngram for ngram in text if not any(char in string.punctuation for char in ngram)]
    text = [wnl.lemmatize(each) for each in text]             # Lemmatizing
    text = [w for w in text if w not in punc]                 # Removing punctuations
    text = [each for each in text if len(each) > 1]           # Removing single letters
    text = [each for each in text if ' ' not in each]
    return text

# Creating unigrams with lemmatization
lem_text = process_text(text_tokens)             # Calling the function for text processing with lemmatization
lem_grams = list(set(lem_text) - set(n_grams))   # Find unique unigrams which are absent in the list of n_grams

# Creating bigrams with lemmatization
lem_bigrams = [bi for bi in bigrams_not_filtred if len(process_text(bi)) == 2]    # Creating bigrams and calling the function for processing with lemmatization
lem_bigrams = [' '.join(word_tuple) for word_tuple in lem_bigrams]
new_lem_bigrams = list(set(lem_bigrams) - set(bigrams))                           # Find the bigrams in lem_bigrams that are not present in bigrams

# Creating trigrams with lemmatization
lem_trigrams = [word for word in trigrams_not_filtred if len(process_trigram(word)) == 3]  # Creating trigrams and calling the function for processing with lemmatization
lem_trigrams = filtering_trigrams(lem_trigrams)                                            # Calling the function to filter trigrams
lem_trigrams = [' '.join(word_tuple) for word_tuple in lem_trigrams]
new_lem_trigrams = list(set(lem_trigrams) - set(trigrams))                                 # Find the trigrams in lem_trigrams that are not present in trigrams

# Combining all n-grams into one list
all_ngrams = n_grams + lem_grams + new_lem_trigrams + new_lem_bigrams
print("Total number of n-grams:", len(all_ngrams))

# Function for calculating precision, recall and F1 score
def calculate_metrics(true_terms, extracted_terms):
    true_positives = len(true_terms.intersection(extracted_terms))
    false_positives = len(extracted_terms.difference(true_terms))
    false_negatives = len(true_terms.difference(extracted_terms))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1_score

# Customize the number of topics and terms
num_components = 5       # Selecting the number of topics
num_terms = 3500         # Selecting the number of terms to be extracted from each topic

# Create a custom_tokenizer lambda function that accepts a string x and returns a list containing that string
custom_tokenizer = lambda x: [x]

# Initializing the NMF (Non-Negative Matrix Factorization) model with specified parameters
vectorizer = TfidfVectorizer( tokenizer = custom_tokenizer,        # Using custom_tokenizer for tokenization
                              lowercase = False,                   # Not convert text to lower case
                              use_idf = False,                     # Disabling the use of inverse document frequency (IDF)
                              ngram_range = (1,3),                 # Use of unigrams, bigrams and trigrams
                              max_df = 0.5,                        # Ignoring tokens that occur in more than 50% of documents
                              token_pattern = None)                # Using custom_tokenizer, thus a token template is not required
X = vectorizer.fit_transform(all_ngrams)    # Converting all_ngrams text data list to sparse TF-IDF matrix

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

    precision, recall, f1_score = calculate_metrics(set(given_terms), set(top_topic_terms))    # Calculating precision, recall, and F1-score for true and extracted terms
    return precision, recall, f1_score

precision, recall, f1_score = final_metrics(num_components, num_terms)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
