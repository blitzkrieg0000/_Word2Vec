"""
    ?=> https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
"""


import warnings
warnings.filterwarnings(action = "ignore")

import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


import gensim
from gensim import downloader
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# wv = downloader.load('word2vec-google-news-300') # 2GB
# wv.index_to_key
# print(wv.similarity(w1, w2))
# print(wv.most_similar(positive=['car', 'minivan'], topn=5))
# print(wv.doesnt_match(['fire', 'water', 'land', 'sea', 'air', 'car']))


stopWords = set(stopwords.words("turkish"))
sentence= """
    Önümüzdeki yıl Almanya'da seçim var. Seçim sürecinde Eurofighter konusunun siyasete malzeme olmaması için
    yılbaşından önce konunun çözülmesi yönündeki beklenti Ankara'da yüksek. Türkiye'nin Eurofighter'larla ilgili
    yürüttüğü görüşmeler birebir Almanya ile değil, İngiltere üzerinden görüşmeler yürütülüyor. İngiltere, Almanya
    ile konuşuyor. Almanya da şu anda 'önce bir müzakere edilsin' pozisyonunda. Anlaşma sağlanırsa 40'a kadar uçak
    alımı söz konusu olacak. Parti parti gelecek olan bu uçaklar, ilk aşamada İngiltere'de üretilip Türkiye'ye
    gönderilecek. Bazı haberlerde 2. el de alınabileceği yazıldı ancak o şu anda gündemde değil. Öncelik üretimden
    çıkacak sıfır uçakların alınmasında. İhtiyaca göre 40'a kadar alınabilir. İhtiyaç listesine göre 35-36 uçak da
    alınabilir, bu değerlendirilecek.
"""

data = [] 
# Paragrafı cümle cümle tokenize et
for w in sent_tokenize(sentence):
    temp = [] 

    # Cümleleri kelime kelime tokenize et
    words = word_tokenize(sentence) 
    for w in words:
        if w not in stopWords:
            temp.append(w)
    data.append(temp)


# Görselleştir
text = []
for i in sentence:
    text.append(i)
text = "".join(map(str, text)) 
wordcloud = WordCloud(width=6000, height=1000, max_font_size=300,background_color="white").generate(text)
plt.figure(figsize=(20,17))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



# Create CBOW model 
model1 = gensim.models.Word2Vec(data, min_count=1, vector_size=200, window=10, alpha=0.25) 
print("""Cosine similarity between "ilgili" """ + """ve  "Türkiye" - CBOW : """, model1.wv.similarity("ilgili", "görüşmeler")) 
print("""Cosine similarity between "önce" """ + """ve  "edilsin" - CBOW : """, model1.wv.similarity("önce", "edilsin")) 


# Create Skip Gram model 
model2 = gensim.models.Word2Vec(data, min_count=1, vector_size=300, alpha=0.025, window=10, sg=1) 
print("""Cosine similarity between "ilgili" """ + """ve  "görüşmeler" - Skip Gram : """, model2.wv.similarity("ilgili","görüşmeler")) 
print("""Cosine similarity between "önce" """ + """ve "edilsin" - Skip Gram : """, model2.wv.similarity("önce", "edilsin")) 