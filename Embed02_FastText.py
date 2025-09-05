"""
    Benzerlik işlemleri word2vec ile aynı şekilde çalışır. 
    Eğitim verilerinde en az bir karakter ngramı bulunması koşuluyla, kelime dağarcığı dışındaki kelimeler de kullanılabilir.

    ?=> https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html
"""

from gensim.models import FastText
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

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


# data = [["bugün", "hava", "çok", "güzel"], ["yarın", "yağmur", "yağacak"], ["hava", "bugün", "güzel"]]

model = FastText(sentences=data, vector_size=100, window=5, min_count=1, sg=1)

similar_words = model.wv.most_similar("Türkiye", topn=5)
print(similar_words)

# vektor = model.wv["uçak"]
# print(vektor)
