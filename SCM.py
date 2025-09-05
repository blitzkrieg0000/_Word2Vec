"""
    !=> Soft Cosine Measure (SCM)
    Kelime benzerliklerini dikkate alarak iki metin arasındaki benzerliği ölçen bir tekniktir. 
    Geleneksel Cosine Similarity'de, iki metni temsil eden vektörler arasındaki açı hesaplanırken,
    Soft Cosine Similarity kelimelerin birbirine benzer olup olmadığını da dikkate alır. 
    Bu, özellikle semantik olarak benzer kelimeler arasında daha yüksek bir benzerlik skoru elde edilmesini sağlar.

    ?=> https://radimrehurek.com/gensim/auto_examples/tutorials/run_scm.html
"""
import logging
from nltk.corpus import stopwords
from nltk import download
import gensim.downloader as api
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import SparseTermSimilarityMatrix, WordEmbeddingSimilarityIndex

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
download("stopwords")  # Download stopwords list.

sentence_obama = "Obama speaks to the media in Illinois"
sentence_president = "The president greets the press in Chicago"
sentence_orange = "Oranges are my favorite fruit"


stop_words = stopwords.words("english")

def preprocess(sentence):
    return [w for w in sentence.lower().split() if w not in stop_words]

sentence_obama = preprocess(sentence_obama)
sentence_president = preprocess(sentence_president)
sentence_orange = preprocess(sentence_orange)


documents = [sentence_obama, sentence_president, sentence_orange]
dictionary = Dictionary(documents)

sentence_obama = dictionary.doc2bow(sentence_obama)
sentence_president = dictionary.doc2bow(sentence_president)
sentence_orange = dictionary.doc2bow(sentence_orange)


documents = [sentence_obama, sentence_president, sentence_orange]
tfidf = TfidfModel(documents)
sentence_obama = tfidf[sentence_obama]
sentence_president = tfidf[sentence_president]
sentence_orange = tfidf[sentence_orange]


model = api.load("word2vec-google-news-300")
termsim_index = WordEmbeddingSimilarityIndex(model)
termsim_matrix = SparseTermSimilarityMatrix(termsim_index, dictionary, tfidf)


# Calculate similarity
similarity = termsim_matrix.inner_product(sentence_obama, sentence_president, normalized=(True, True))
print("similarity = %.4f" % similarity)

similarity = termsim_matrix.inner_product(sentence_obama, sentence_orange, normalized=(True, True))
print("similarity = %.4f" % similarity)


