import streamlit as st
import spacy
import nltk
from nltk.util import ngrams
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import spacy.displacy as displacy
from io import StringIO
import base64
import PyPDF2
import docx
from textblob import TextBlob
from nltk.corpus import stopwords

# Downloads
nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")

st.title("📚 Advanced Corpus Analysis Tool")
st.write("Upload any document to perform a detailed linguistic analysis.")

uploaded_file = st.file_uploader("Upload File", type=["txt", "pdf", "docx", "csv"])

def extract_text(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "txt":
        return uploaded_file.read().decode("utf-8")

    elif file_type == "pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for p in reader.pages:
            content = p.extract_text()
            if content:
                text += content
        return text

    elif file_type == "docx":
        d = docx.Document(uploaded_file)
        return "\n".join([p.text for p in d.paragraphs])
    
    elif file_type == "csv":
        df = pd.read_csv(uploaded_file)
        return " ".join(df.astype(str).values.flatten())

    return ""

# User options
remove_stop = st.checkbox("Remove Stopwords")
lemmatize = st.checkbox("Apply Lemmatization")

if uploaded_file:
    text = extract_text(uploaded_file)
    
    doc = nlp(text)

    if remove_stop:
        stop_words = set(stopwords.words("english"))
        words = [token.lemma_.lower() if lemmatize else token.text.lower()
                 for token in doc if token.is_alpha and token.text.lower() not in stop_words]
    else:
        words = [token.lemma_.lower() if lemmatize else token.text.lower()
                 for token in doc if token.is_alpha]

    sentences = list(doc.sents)

    st.subheader("📊 Basic Statistics")
    st.write(f"Total Sentences: {len(sentences)}")
    st.write(f"Total Words: {len(words)}")
    st.write(f"Unique Words: {len(set(words))}")
    st.write(f"Lexical Diversity: {(len(set(words))/len(words)):.2f}")

    # Sentiment
    st.subheader("😊 Sentiment Analysis")
    sentiment = TextBlob(text).sentiment
    st.write(f"Polarity Score: {sentiment.polarity:.2f}")
    st.write(f"Subjectivity Score: {sentiment.subjectivity:.2f}")

    # POS Tags
    st.subheader("🧩 POS Tag Distribution")
    pos_counts = Counter([token.pos_ for token in doc])
    fig, ax = plt.subplots()
    ax.bar(pos_counts.keys(), pos_counts.values())
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Named Entity Recognition
    st.subheader("🏷 Named Entity Recognition")
    ner_html = displacy.render(doc, style='ent')
    st.markdown(ner_html, unsafe_allow_html=True)

    # Frequent Words
    st.subheader("🔠 Top Frequent Words")
    freq = Counter(words).most_common(20)
    df_freq = pd.DataFrame(freq, columns=["Word", "Frequency"])
    st.dataframe(df_freq)

    # Download button
    csv = df_freq.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Word Frequency CSV", csv, "word_freq.csv")

    # Word Cloud
    st.subheader("☁ Word Cloud")
    wc = WordCloud(width=800, height=400, background_color='white').generate(" ".join(words))
    fig2, ax2 = plt.subplots()
    ax2.imshow(wc)
    ax2.axis("off")
    st.pyplot(fig2)

    # Bigrams
    st.subheader("🔗 Top 20 Bigrams")
    bigrams = Counter(ngrams(words, 2)).most_common(20)
    df_bi = pd.DataFrame(bigrams, columns=["Bigram", "Frequency"])
    st.dataframe(df_bi)

    # CSV download for Bigrams
    csv_bi = df_bi.to_csv(index=False).encode('utf-8')
    st.download_button("⬇ Download Bigrams CSV", csv_bi, "bigrams.csv")

    # Keyword Search
    st.subheader("🔍 Keyword in Context (KWIC)")
    keyword = st.text_input("Enter keyword")

    if keyword:
        st.write("Context Results:")
        tokens = text.split()
        window = 5
        for i, w in enumerate(tokens):
            if keyword.lower() == w.lower():
                left = " ".join(tokens[max(0, i-window):i])
                right = " ".join(tokens[i+1:i+1+window])
                st.write(f"{left} 👉 **{w}** 👈 {right}")
