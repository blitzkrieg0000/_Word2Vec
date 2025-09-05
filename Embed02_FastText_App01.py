"""
    Kelimelerin vektör temsillerini(embedding) kullanarak anlamsal olarak birbirine en yakın kelimeler bulunabilir.
"""

from gensim.models import FastText
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# Eğitim verisi (örnek cümleler)
metin = [
    ["elma", "yemek", "meyve"],
    ["elma", "tatlı", "yapmak"],
    ["muz", "tatlı", "yemek"],
    ["meyve", "sağlıklı"],
    ["portakal", "meyve", "sıkmak"],
    ["limon", "ekşi", "meyve"],
    ["çilek", "meyve", "tatlı"],
    ["araba", "otomobil", "motosiklet"],
    ["kamyon", "yük", "araç"],
    ["uçak", "hava", "havaalanı"],
    ["tren", "ray", "istasyon"],
    ["gemi", "deniz", "rıhtım"],
    ["bisiklet", "pedal", "spor"],
    ["kedi", "köpek", "hayvan"],
    ["aslan", "orman", "hayvan"],
    ["kuş", "uçmak", "hayvan"],
    ["bilgisayar", "yazılım", "donanım"],
    ["internet", "tarayıcı", "web"],
    ["telefon", "mobil", "iletişim"],
    ["televizyon", "izlemek", "ekran"],
    ["giysi", "giyim", "kıyafet"],
    ["giysi", "tekstil", "moda"],
    ["kıyafet", "ütü", "çamaşır"],
    ["çanta", "aksesuar", "moda"],
    ["spor", "futbol", "basketbol"],
    ["futbol", "maç", "gol"],
    ["basketbol", "oyun", "nba"],
    ["voleybol", "smaç", "saha"],
    ["tenis", "raket", "top"],
    ["yüzme", "havuz", "spor"],
    ["bilim", "araştırma", "deney"],
    ["matematik", "geometri", "denklem"],
    ["fizik", "kuvvet", "enerji"],
    ["tarih", "savaş", "medeniyet"],
    ["sanat", "resim", "müzik"],
    ["edebiyat", "şiir", "kitap"],
    ["teknoloji", "yeni", "cihaz"],
    ["telefon", "teknoloji", "mobil"],
    ["teknoloji", "gelişme", "inovasyon"]
]

# Modeli eğitme
model = FastText(
    sentences=metin, 
    vector_size=300,
    window=7,
    min_count=1,
    sg=1,
    seed=72,
    min_n=3,
    max_n=6,
    epochs=500
)

#  OOV (Out of Vocabulary) kelimelerde benzerlik bulma
print("OOV: ", model.wv.similarity("tekno", "teknoloji"))

# Benzer kelimeleri bulma
similar_words = model.wv.most_similar("çamaşır", topn=3)
print(similar_words)


# Plot
words = model.wv.index_to_key
word_vectors = [model.wv[word] for word in words]
pca = PCA(n_components=2)
pcavectors = pca.fit_transform(word_vectors)

plt.scatter(pcavectors[:, 0], pcavectors[:, 1])

for i, kelime in enumerate(words):
    plt.annotate(kelime, (pcavectors[i, 0], pcavectors[i, 1]))

plt.show()