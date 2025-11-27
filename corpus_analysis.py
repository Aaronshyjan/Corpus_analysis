import streamlit as st
import spacy
from spacy import displacy
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import re

# Load English NLP model
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="Corpus Analysis Tool", layout="wide")
st.title("📊 Corpus Analysis Tool")

# Input options
option = st.radio("Upload text file or paste text:", ("Upload", "Paste Text"))

text = ""
if option == "Upload":
    file = st.file_uploader("Upload .txt file", type=["txt"])
    if file:
        text = file.read().decode("utf-8")
else:
    text = st.text_area("Paste your corpus text here:", height=200)

if st.button("Analyze") and text.strip():
    doc = nlp(text)

    # 📌 Statistical Analysis
    tokens = [token.text.lower() for token in doc if token.is_alpha]
    word_freq = Counter(tokens)
    vocab_size = len(word_freq)

    st.subheader("🧮 Statistical Analysis")
    st.write(f"Total Words: **{len(tokens)}**")
    st.write(f"Vocabulary Size: **{vocab_size}**")
    st.write(f"Total Sentences: **{len(list(doc.sents))}**")

    # Show top words
    st.write("🔝 Most Common Words")
    common_df = pd.DataFrame(word_freq.most_common(10), columns=["Word", "Frequency"])
    st.dataframe(common_df)

    # 📌 POS Tag Distribution
    st.subheader("📌 POS Tag Distribution")
    pos_counts = Counter([token.pos_ for token in doc])

    pos_df = pd.DataFrame(pos_counts.items(), columns=["POS", "Count"])
    st.bar_chart(pos_df.set_index("POS"))

    # 📌 Named Entity Recognition
    st.subheader("🏷 Named Entity Visualization")

    html = displacy.render(doc, style="ent", jupyter=False)
    st.markdown(html, unsafe_allow_html=True)

else:
    st.info("Upload or paste text then click Analyze!")
