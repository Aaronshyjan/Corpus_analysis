import streamlit as st
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.util import ngrams
import nltk

# Download NLTK data
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

st.title("📊 Corpus Analysis Tool")
st.write("Upload a text or CSV corpus and perform NLP analysis")

# File upload
uploaded_file = st.file_uploader("Upload a text file (.txt) or CSV (.csv)", type=["txt", "csv"])

text = ""

# Load file content
if uploaded_file:
    if uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")

    elif uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        if "text" in df.columns:
            text = " ".join(df["text"].astype(str))
        else:
            st.error("CSV must contain a 'text' column.")
            st.stop()

    st.success("File uploaded successfully!")

    # Process text using spaCy
    doc = nlp(text)

    st.header("📌 Text Statistics")
    words = [token.text for token in doc if token.is_alpha]
    sentences = list(doc.sents)

    st.write("**Total Sentences:**", len(sentences))
    st.write("**Total Words:**", len(words))
    st.write("**Unique Words:**", len(set(words)))
    st.write("**Lexical Diversity:**", round(len(set(words)) / len(words), 4))


    # POS Tag Distribution
    st.header("📌 POS Tag Distribution")
    pos_counts = Counter([token.pos_ for token in doc])

    fig, ax = plt.subplots()
    ax.bar(pos_counts.keys(), pos_counts.values())
    ax.set_title("POS Tag Distribution")
    st.pyplot(fig)


    # Named Entity Recognition
    st.header("📌 Named Entity Recognition (NER)")
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    ent_df = pd.DataFrame(ents, columns=["Entity", "Label"])
    st.dataframe(ent_df)


    # Word Cloud
    st.header("📌 Word Cloud")
    wc = WordCloud(width=800, height=400).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)


    # Word Frequency
    st.header("📌 Top 20 Most Frequent Words")
    freq = Counter(words).most_common(20)
    freq_df = pd.DataFrame(freq, columns=["Word", "Frequency"])
    st.table(freq_df)


    # N-grams
    st.header("📌 Bigrams (Top 20)")
    bigrams = Counter(ngrams(words, 2)).most_common(20)
    bigram_df = pd.DataFrame(bigrams, columns=["Bigram", "Count"])
    st.table(bigram_df)


    # KWIC Search
    st.header("📌 KWIC (Keyword in Context) Search")
    keyword = st.text_input("Enter keyword")

    if keyword:
        tokens = text.split()
        window = 5
        results = []

        for i, w in enumerate(tokens):
            if w.lower() == keyword.lower():
                left = " ".join(tokens[max(0, i-window):i])
                right = " ".join(tokens[i+1:i+1+window])
                results.append([left, w, right])

        if results:
            st.write(f"### Results for '{keyword}':")
            st.table(pd.DataFrame(results, columns=["Left Context", "Keyword", "Right Context"]))
        else:
            st.write("No occurrences found.")
