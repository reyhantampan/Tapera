import streamlit as st
import pandas as pd
import re
import nltk
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix

# Download NLTK stopwords
nltk.download('stopwords')

# Fungsi preprocessing teks
def casefolding(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'_', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Memuat kunci normalisasi
try:
    key_norm = pd.read_csv('data/key_norm.csv', encoding='latin1')
except UnicodeDecodeError:
    key_norm = pd.read_csv('data/key_norm.csv', encoding='ISO-8859-1')

def text_normalize(text):
    words = text.split()
    normalized_words = []
    for word in words:
        if (key_norm['singkat'] == word).any():
            normalized_word = key_norm[key_norm['singkat'] == word]['hasil'].values[0]
            normalized_words.append(normalized_word)
        else:
            normalized_words.append(word)
    return ' '.join(normalized_words)

# Stopwords dalam bahasa Indonesia
stopwords_ind = stopwords.words('indonesian')
more_stopword = ['yg', 'dg', 'rt', 'dgn', 'ny', 'd', 'klo', 'kalo', 'amp', 'biar', 'bikin', 'bilang',
                 'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                 'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt', 
                 '&amp', 'yah', 'zy_zy', 'mh']
stopwords_ind.extend(more_stopword)

def remove_stop_word(text):
    words = text.split()
    clean_words = [word for word in words if word not in stopwords_ind]
    return ' '.join(clean_words)

def tokenizing(text):
    return text.split()

# Stemming function
factory = StemmerFactory()
stemmer = factory.create_stemmer()

additional_dict = {
    'ditindaklanjuti' : 'tindaklanjut',
    'nutupin' : 'tutup',
    'ingetin' : 'ingat',
    'keduluan': 'dulu',
    'dipahami': 'paham',
    'ditingkatkan':'tingkat',
    'diikuti':'ikut',
    'perumahan':'rumah',
    'nalangin':'bayarin',
    'nyambung': 'sambung',
    'ngaturnya':'atur',
    'ngide':'ide',
    'manfaatin':'manfaat',
    'dengerin':'dengar',
    'nyusahin': 'susah',
    'nabung':'tabung',
    'dinaikin':'naik',
    'dipaksain':'paksa',
    'dilegalin':'legal',
    'ditarikin':'tarik',
    'dimintain':'minta',
    'ngajak':'ajak',
    'disalahin':'salah',
    'mengatasnamakan':'atasnama', 
    'maksa':'paksa'  , 
    'akalin':'akal'
}

def stemming(text):
    words = text.split()
    stemmed_words = []
    for word in words:
        stemmed_word = stemmer.stem(word)
        if word in additional_dict:
            stemmed_word = additional_dict[word]
        stemmed_words.append(stemmed_word)
    return ' '.join(stemmed_words)

# membuat fungsi untuk menggabungkan seluruh langkah text preprocessing
def text_preprocessing_process(text):
    text = casefolding(text)
    text = text_normalize(text)
    text = remove_stop_word(text)
    tokens = tokenizing(text)
    text = ' '.join(tokens)
    text = stemming(text)
    return text

# Memuat dan preprocessing data
data_model = pd.read_csv('data/modeling.csv')

# Inisialisasi dan fit vectorizer
tfidf = TfidfVectorizer(max_features=8000)
tfidf.fit(data_model['clean_teks'])
X_tfidf = tfidf.transform(data_model['clean_teks'])
y = data_model['sentiment']

# Terapkan SMOTE untuk menyeimbangkan data
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_tfidf, y)

# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Inisialisasi dan latih model SVC
svm_model = SVC(C=4.7450412719997725, 
                class_weight=None, 
                coef0=2.0, 
                degree=2, 
                gamma=10, 
                kernel='poly', 
                max_iter=4000, 
                shrinking=True, 
                tol=0.0001)
svm_model.fit(X_train, y_train)

# Inisialisasi dan latih model Naive Bayes Multinomial
nb_model = MultinomialNB(alpha=0.15808361216819947)
nb_model.fit(X_train, y_train)

def classify_text(text, vectorizer, model):
    preprocessed_text = text_preprocessing_process(text)
    text_tfidf = vectorizer.transform([preprocessed_text])
    prediction = model.predict(text_tfidf)[0]
    return preprocessed_text, prediction

# Sidebar untuk navigasi
st.sidebar.title("Navigasi")
options = st.sidebar.radio("Pergi ke", ["🏠 Halaman Utama", "📊 Eksplorasi Data", "🔄 Preprocessing",  "🈯 Translate", "🔍 Prediksi", "📝 Kesimpulan"])

# Menambahkan CSS untuk justify text dan margin pada informasi penulis
st.markdown(
    """
    <style>
    .justified-text {
        text-align: justify;
    }
    .author-info {
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Halaman utama (dashboard pertama)
if options == "🏠 Halaman Utama":
    st.image("asset/Logo Udinus - Official 02.png", width=100)
    st.title("ANALISIS SENTIMEN TERHADAP PROGRAM TAPERA MENGGUNAKAN ALGORITMA NAÏVE BAYES")
    
    # Deskripsi penelitian dengan justify text
    st.markdown("""
    <div class="justified-text">
    Penelitian ini bertujuan untuk menganalisis sentimen masyarakat terhadap program Tabungan Perumahan Rakyat 
    (Tapera) di Indonesia menggunakan algoritma Naïve Bayes dan SVM. Data yang digunakan dalam penelitian ini 
    dikumpulkan dari berbagai komentar dan opini di media sosial terkait dengan program Tapera. Proses analisis 
    dimulai dari pengumpulan data, preprocessing teks yang meliputi case folding, normalisasi, penghapusan stopword, 
    tokenisasi, dan stemming, hingga tahap klasifikasi sentimen menggunakan algoritma Naïve Bayes dan SVM. Hasil 
    analisis menunjukkan bahwa model SVM memiliki kinerja yang lebih baik dibandingkan dengan model Naïve Bayes 
    dalam mengklasifikasikan sentimen terhadap program Tapera. Untuk model SVM, akurasi yang diperoleh sebesar 
    88.21% menunjukkan bahwa sebagian besar prediksi model adalah benar, presisi sebesar 88.49% menunjukkan bahwa 
    proporsi prediksi positif yang benar cukup tinggi, recall sebesar 88.21% menunjukkan bahwa model mampu mengenali 
    sebagian besar data positif dengan benar, dan skor F1 sebesar 88.20% mengindikasikan keseimbangan yang baik 
    antara presisi dan recall, memberikan gambaran keseluruhan yang kuat tentang kinerja model. Untuk model Naïve 
    Bayes, akurasi yang diperoleh sebesar 84.71% menunjukkan bahwa sebagian besar prediksi model adalah benar, 
    presisi sebesar 85.50% menunjukkan bahwa proporsi prediksi positif yang benar cukup tinggi, recall sebesar 
    84.71% menunjukkan bahwa model mampu mengenali sebagian besar data positif dengan benar, dan skor F1 sebesar 
    84.60% mengindikasikan keseimbangan yang baik antara presisi dan recall, memberikan gambaran keseluruhan yang 
    kuat tentang kinerja model. Penelitian ini memberikan wawasan mendalam mengenai persepsi publik terhadap program 
    Tapera, membantu pemerintah dan pemangku kebijakan untuk memahami pandangan masyarakat dan mengidentifikasi
    area yang memerlukan perbaikan lebih lanjut.
    </div>

    """, unsafe_allow_html=True)

    # Informasi penulis
    st.markdown("""
    <div class="author-info">
    <br>Nama: Muhammad Reyhan Aristya<br>
    NIM: A11.2020.12690
    </div>
    """, unsafe_allow_html=True)


# Halaman: Eksplorasi Data
elif options == "📊 Eksplorasi Data":
    st.header("Eksplorasi Data Analis")
    
    # Membaca data
    data = pd.read_csv('data/modeling.csv')
    # Deskripsi penelitian dengan justify text
    
    st.markdown("""
    <div class="justified-text">
    Setiap manusia mempunyai kebutuhan dasar seperti tempat tinggal yang diatur dalam Undang-Undang No 1 Tahun 2011 
    tentang Perumahan dan Kawasan Permukiman, di mana rumah didefinisikan sebagai bangunan layak huni, sarana pembangun 
    keluarga, cerminan martabat penghuninya, dan aset pemiliknya. Pesatnya pertumbuhan penduduk Indonesia meningkatkan 
    kebutuhan rumah, yang sulit dipenuhi karena mahalnya biaya pembangunan dan harga tanah, sementara banyak rakyat 
    berpenghasilan rendah dan menengah tidak mampu membangun rumah dan tinggal di lingkungan kumuh. Pemerintah melalui 
    Peraturan Pemerintah Nomor 57 Tahun 2018 tentang Tabungan Perumahan Rakyat (Tapera) mencoba mengatasi krisis ini 
    dengan menyediakan akses hunian yang layak dan terjangkau. Tapera adalah penyimpanan periodik untuk pembiayaan 
    perumahan yang dapat dimanfaatkan oleh warga negara Indonesia dan asing yang bekerja di Indonesia. Program ini 
    mendapat dukungan karena memberikan akses perumahan yang lebih mudah dan terjangkau serta sebagai investasi jangka 
    panjang, namun juga menghadapi kritik terkait beban finansial, efisiensi, dan transparansi pengelolaan dana. Untuk 
    memahami sentimen publik terhadap Tapera, analisis sentimen teks dengan algoritma Naive Bayes dapat digunakan untuk 
    mengklasifikasikan opini masyarakat di media sosial menjadi sentimen positif atau negatif, sehingga membantu 
    pemerintah menilai efektivitas dan penerimaan program ini.
    </div>
    
    """, unsafe_allow_html=True)    
    
    # Menghitung jumlah masing-masing sentimen
    sentimen_counts = data['sentiment'].value_counts()
    
    # Plot diagram lingkaran
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(sentimen_counts, labels=sentimen_counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightgreen'])
    ax.set_title('Proporsi Sentimen', fontsize=8)
    plt.setp(ax.texts, size=8)  # Menyesuaikan ukuran teks dalam plot
    st.pyplot(fig)
        
    st.markdown("""
    <div class="justified-text">
    Dari diagram ini, kita dapat melihat bahwa mayoritas sentimen masyarakat terhadap program Tapera adalah negatif, dengan 64.6% 
    dari total opini yang dianalisis menunjukkan ketidakpuasan atau kritik. Sebaliknya, hanya 35.4% dari total opini yang menunjukkan
    dukungan atau pandangan positif terhadap program ini.
    </div>
    
    """, unsafe_allow_html=True) 
    
    # Memfilter data untuk masing-masing sentimen
    positive_tweets = data[data['sentiment'] == 'positive']['clean_teks']
    negative_tweets = data[data['sentiment'] == 'negative']['clean_teks']

    # Membuat WordCloud untuk sentimen positif
    all_positive_text = ' '.join(positive_tweets)
    wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(all_positive_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud_positive, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud untuk Sentimen Positif')
    st.pyplot(fig)
    
    st.markdown("""
    <div class="justified-text">
    Berdasarkan hasil visualisasi wordcloud sentimen positif, terdapat beberapa kata dengan frekuensi kemunculan yang
    tinggi, di antaranya "tapera", "pajak", "rakyat", "uang", "rumah", "program", "bansos", "gaji", "wajib", "tabung", 
    dan sebagainya. Hasil analisis dari wordcloud dengan sentimen positif pada penelitian ini dapat disimpulkan bahwa, 
    program Tapera dianggap sebagai langkah positif yang mendukung kesejahteraan rakyat melalui berbagai program yang 
    berkaitan dengan pajak, tabungan, dan bantuan sosial. 
    </div>
    
    """, unsafe_allow_html=True) 
    
    # Membuat WordCloud untuk sentimen negatif
    all_negative_text = ' '.join(negative_tweets)
    wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(all_negative_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud_negative, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud untuk Sentimen Negatif')
    st.pyplot(fig)
    
    st.markdown("""
    <div class="justified-text">
    Berdasarkan hasil visualisasi wordcloud sentimen negatif, terdapat beberapa kata dengan frekuensi kemunculan yang tinggi,
    di antaranya "tapera", "potong", "rakyat", "negara", "duit", "pakai", "pajak", "kerja", "perintah", "jokowi", "gaji", 
    "bansos", "utang", dan sebagainya. Hasil analisis dari wordcloud dengan sentimen negatif pada penelitian ini dapat 
    disimpulkan bahwa, terdapat kekhawatiran dan ketidakpuasan masyarakat terkait dengan implementasi program Tapera.
    </div>
    
    """, unsafe_allow_html=True) 
    
# Halaman: Preprocessing
elif options == "🔄 Preprocessing":
    st.header("Langkah Preprocessing")
    st.write("Unggah data Anda dan lakukan langkah-langkah preprocessing teks.")

     # Unggah data
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data yang Diunggah:")
        st.write(data.head(15))  # Menampilkan 100 baris pertama data
        
        # Pilih kolom untuk preprocessing
        column = st.selectbox("Pilih kolom untuk preprocessing", data.columns)
        st.write(f"Kolom yang dipilih: {column}")

        # Inisialisasi session state untuk menyimpan hasil sementara
        if 'casefolding_text' not in st.session_state:
            st.session_state.casefolding_text = None
        if 'normalisasi_text' not in st.session_state:
            st.session_state.normalisasi_text = None
        if 'remove_text' not in st.session_state:
            st.session_state.remove_text = None
        if 'tokenize_text' not in st.session_state:
            st.session_state.tokenize_text = None
        if 'stemming_text' not in st.session_state:
            st.session_state.stemming_text = None
        if 'translate_text' not in st.session_state:
            st.session_state.translate_text = None

        st.subheader("Sebelum dan Sesudah Case Folding")
        st.write("Sebelum Case Folding:")
        st.write(data[column].head(15))  # Menampilkan 100 baris pertama data

        if st.button("Case Folding"):
            st.session_state.casefolding_text = data[column].apply(casefolding)
            st.write("Setelah Case Folding:")
            st.write(st.session_state.casefolding_text.head(15))  # Menampilkan 100 baris pertama data

        if st.session_state.casefolding_text is not None:
            st.subheader("Sebelum dan Sesudah Normalisasi Teks")
            st.write("Sebelum Normalisasi Teks:")
            st.write(st.session_state.casefolding_text.head(15))  # Menampilkan 100 baris pertama data
            if st.button("Normalisasi Teks"):
                st.session_state.normalisasi_text = st.session_state.casefolding_text.apply(text_normalize)
                st.write("Setelah Normalisasi Teks:")
                st.write(st.session_state.normalisasi_text.head(15))  # Menampilkan 100 baris pertama data

        if st.session_state.normalisasi_text is not None:
            st.subheader("Sebelum dan Sesudah Menghapus Stop Word")
            st.write("Sebelum Menghapus Stop Word:")
            st.write(st.session_state.normalisasi_text.head(15))  # Menampilkan 100 baris pertama data
            if st.button("Menghapus Stop Word"):
                st.session_state.remove_text = st.session_state.normalisasi_text.apply(remove_stop_word)
                st.write("Setelah Menghapus Stop Word:")
                st.write(st.session_state.remove_text.head(15))  # Menampilkan 100 baris pertama data

        if st.session_state.remove_text is not None:
            st.subheader("Sebelum dan Sesudah Tokenisasi")
            st.write("Sebelum Tokenisasi:")
            st.write(st.session_state.remove_text.head(15))  # Menampilkan 100 baris pertama data
            if st.button("Tokenisasi"):
                st.session_state.tokenize_text = st.session_state.remove_text.apply(tokenizing)
                # Gabungkan token kembali menjadi string untuk langkah berikutnya
                st.session_state.tokenize_text = st.session_state.tokenize_text.apply(lambda x: ' '.join(x))
                st.write("Setelah Tokenisasi:")
                st.write(st.session_state.tokenize_text.head(15))  # Menampilkan 100 baris pertama data

        if st.session_state.tokenize_text is not None:
            st.subheader("Sebelum dan Sesudah Stemming")
            st.write("Sebelum Stemming:")
            st.write(st.session_state.tokenize_text.head(15))  # Menampilkan 100 baris pertama data
            if st.button("Stemming"):
                # Memuat dan menampilkan data hasil stemming dari file Excel
                data_stemming = pd.read_csv('data/Clean_Data.csv')
                st.session_state.stemming_text = data_stemming
                st.write("Setelah Stemming:")
                st.write(st.session_state.stemming_text.head(15))  # Menampilkan 100 baris pertama data
                
        if st.session_state.stemming_text is not None:
            st.subheader("Sebelum dan Sesudah Translate")
            st.write("Sebelum Translate:")
            st.write(st.session_state.stemming_text.head(15))  # Menampilkan 100 baris pertama data

# Halaman: Translate
elif options == "🈯 Translate":
    st.header("Translate Data")

    st.write("Unggah data Anda dan lakukan langkah-langkah preprocessing teks.")

    # Unggah data
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data yang Diunggah:")
        st.write(data.head(100))  # Menampilkan 100 baris pertama data

        if st.button("Translate"):
            # Memuat dan menampilkan data hasil translasi dari file CSV
            data_translate = pd.read_csv('data/Translate.csv')
            # Mengakses kolom 'tweet_english' dan 'sentiment'
            data_translate = data_translate[['tweet_english']]
            data_translate['tweet_english'] = data_translate['tweet_english'].apply(casefolding)
            st.session_state.translate_text = data_translate
            st.write("Setelah Translate:")
            st.write(st.session_state.translate_text.head(100))  # Menampilkan 100 baris pertama data


# Halaman: Prediksi
elif options == "🔍 Prediksi":
    st.header("Klasifikasi Teks")

    user_input = st.text_area("Masukkan teks untuk klasifikasi:")

    # Inisialisasi session state untuk menyimpan hasil sementara
    if 'casefolding_text' not in st.session_state:
        st.session_state.casefolding_text = ''
    if 'normalisasi_text' not in st.session_state:
        st.session_state.normalisasi_text = ''
    if 'remove_text' not in st.session_state:
        st.session_state.remove_text = ''
    if 'tokenize_text' not in st.session_state:
        st.session_state.tokenize_text = ''
    if 'stemming_text' not in st.session_state:
        st.session_state.stemming_text = ''

    if user_input:
        if st.button("Case Folding"):
            st.session_state.casefolding_text = casefolding(user_input)
        st.write("Setelah Case Folding:")
        st.write(st.session_state.casefolding_text)
        
        if st.session_state.casefolding_text:
            if st.button("Normalisasi Teks"):
                st.session_state.normalisasi_text = text_normalize(st.session_state.casefolding_text)
            st.write("Setelah Normalisasi Teks:")
            st.write(st.session_state.normalisasi_text)

        if st.session_state.normalisasi_text:
            if st.button("Menghapus Stop Word"):
                st.session_state.remove_text = remove_stop_word(st.session_state.normalisasi_text)
            st.write("Setelah Menghapus Stop Word:")
            st.write(st.session_state.remove_text)

        if st.session_state.remove_text:
            if st.button("Tokenisasi"):
                st.session_state.tokenize_text = tokenizing(st.session_state.remove_text)
                # Gabungkan token kembali menjadi string untuk langkah berikutnya
                st.session_state.tokenize_text = ' '.join(st.session_state.tokenize_text)
            st.write("Setelah Tokenisasi:")
            st.write(st.session_state.tokenize_text)

        if st.session_state.tokenize_text:
            if st.button("Stemming"):
                st.session_state.stemming_text = stemming(st.session_state.tokenize_text)
            st.write("Setelah Stemming:")
            st.write(st.session_state.stemming_text)

    if st.button("Prediksi dengan SVM"):
        preprocessed_text, svm_result = classify_text(user_input.strip(), tfidf, svm_model)
        st.write("Prediksi SVM:")
        st.write(svm_result)
        
    if st.button("Prediksi dengan Naive Bayes"):
        preprocessed_text, nb_result = classify_text(user_input.strip(), tfidf, nb_model)
        st.write("Prediksi Naive Bayes:")
        st.write(nb_result)
        
# Halaman: Kesimpulan
elif options == "📝 Kesimpulan":
    st.header("Kesimpulan")

    st.markdown("""
    <div class="justified-text">
    Dalam penelitian ini, evaluasi model dilakukan untuk menilai kinerja model Naive Bayes dalam mengklasifikasikan 
    sentimen terhadap program Tapera. Proses evaluasi ini sangat penting untuk memahami seberapa baik model dapat 
    memprediksi sentimen positif dan negatif dari data teks yang diberikan. Salah satu alat evaluasi yang digunakan 
    adalah confusion matrix. Confusion matrix memberikan gambaran yang mendetail mengenai performa model, termasuk 
    jumlah prediksi yang benar (baik positif maupun negatif) serta jumlah prediksi yang salah (false positive dan 
    false negative). Dengan menggunakan confusion matrix, dapat dihitung berbagai metrik evaluasi seperti akurasi, 
    presisi, recall, dan skor F1, yang semuanya memberikan wawasan komprehensif tentang keandalan dan efektivitas 
    model dalam tugas klasifikasi sentimen.
    </div>
    """, unsafe_allow_html=True)
    
    # Menampilkan confusion matrix
    st.subheader("Confusion Matrix")

    # Prediksi menggunakan model SVM
    y_pred_svm = svm_model.predict(X_test)
    confusion_svm = confusion_matrix(y_test, y_pred_svm)

    # Prediksi menggunakan model Naive Bayes
    y_pred_nb = nb_model.predict(X_test)
    confusion_nb = confusion_matrix(y_test, y_pred_nb)

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        sns.heatmap(confusion_svm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix - SVM')
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        sns.heatmap(confusion_nb, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix - Naive Bayes')
        st.pyplot(fig)
    
    # Menampilkan hasil evaluasi SVM
    st.subheader("Hasil Evaluasi SVM")
    st.markdown("""
    - Accuracy: 0.8821
    - Precision: 0.8849
    - Recall: 0.8821
    - F1 Score: 0.8820
    """)

    svm_report = """
    |       | precision | recall | f1-score | support |
    |-------|------------|--------|----------|---------|
    |negative| 0.85      | 0.92   | 0.88     | 154     |
    |positive| 0.92      | 0.84   | 0.88     | 160     |
    |accuracy|           |        | 0.88     | 314     |
    |macro avg| 0.88     | 0.88   | 0.88     | 314     |
    |weighted avg| 0.88  | 0.88   | 0.88     | 314     |
    """

    st.markdown(svm_report)

    # Menampilkan hasil evaluasi Naive Bayes
    st.subheader("Hasil Evaluasi Naive Bayes")
    st.markdown("""
    - Accuracy: 0.8471
    - Precision: 0.8550
    - Recall: 0.8471
    - F1 Score: 0.8460
    """)

    nb_report = """
    |       | precision | recall | f1-score | support |
    |-------|------------|--------|----------|---------|
    |negative| 0.91      | 0.77   | 0.83     | 154     |
    |positive| 0.80      | 0.93   | 0.86     | 160     |
    |accuracy|           |        | 0.85     | 314     |
    |macro avg| 0.86     | 0.85   | 0.85     | 314     |
    |weighted avg| 0.86  | 0.85   | 0.85     | 314     |
    """

    st.markdown(nb_report)

    # Menampilkan data salah klasifikasi
    st.subheader("Data Salah Klasifikasi")

    # Memuat data salah klasifikasi
    salah_svm = pd.read_csv('data/salah_klasifikasi_svm.csv')
    salah_nb = pd.read_csv('data/salah_klasifikasi_nb.csv')

    st.markdown("### Salah Klasifikasi SVM")
    st.dataframe(salah_svm)

    st.markdown("### Salah Klasifikasi Naive Bayes")
    st.dataframe(salah_nb)

    # Menampilkan top 20 terms dengan bobot TF-IDF tertinggi
    st.subheader("Top 20 Terms dengan Bobot TF-IDF Tertinggi")

    top_terms_negative = pd.DataFrame({
        'terms': ['potong', 'judi', 'bayar', 'gaji', 'rakyat', 'online', 'perintah', 'duit', 'goblok', 'program', 'rumah', 'pajak', 'kerja', 'ikn', 'bansos', 'jokowi', 'pakai', 'tabung', 'negara', 'orang'],
        'tfidf': [28.831, 17.6051, 16.6868, 16.5961, 15.3966, 14.3278, 14.258, 13.7951, 12.303, 11.6356, 11.3097, 10.6787, 10.6708, 10.4326, 10.358, 10.0315, 9.7739, 9.4556, 9.3297, 8.6931]
    })

    top_terms_positive = pd.DataFrame({
        'terms': ['rumah', 'manfaat', 'dukung', 'tabung', 'rakyat', 'program', 'bayar', 'bantu', 'potong', 'kerja', 'masyarakat', 'wajib', 'perintah', 'mending', 'jangkau', 'gaji', 'dana', 'tenang', 'suruh'],
        'tfidf': [15.6231, 11.3435, 9.1768, 9.1254, 8.3389, 7.9726, 6.5615, 6.4758, 6.3182, 6.2576, 6.253, 6.0857, 6.0636, 6.0446, 5.0874, 5.0122, 4.9856, 4.9388, 4.9049]
    })

    st.markdown("### Positive Terms")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='tfidf', y='terms', data=top_terms_positive, palette='Blues_r')
    plt.title('Top 20 Positive Terms dengan Bobot TF-IDF Tertinggi')
    st.pyplot(plt)   
    
    st.markdown("### Negative Terms")
    plt.figure(figsize=(10, 6))
    sns.barplot(x='tfidf', y='terms', data=top_terms_negative, palette='Reds_r')
    plt.title('Top 20 Negative Terms dengan Bobot TF-IDF Tertinggi')
    st.pyplot(plt)
