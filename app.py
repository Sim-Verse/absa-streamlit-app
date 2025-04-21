"""
import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Set up page
st.set_page_config(page_title="Aspect-Based Sentiment Analysis", layout="wide")
st.title("🧠 Aspect-Based Sentiment Analysis (ABSA) App")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "yangheng/deberta-v3-base-absa-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    absa_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return absa_pipeline

absa_pipeline = load_model()

# Upload CSV file
st.sidebar.header("📁 Upload your CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV with 'review_text' column", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()  # remove extra spaces from column names

        if 'review_text' not in df.columns:
            st.error("❌ Your CSV must have a column named 'review_text'.")
            st.write("📋 Available columns:", list(df.columns))
        else:
            st.success("✅ File uploaded successfully!")
            st.write("### 📄 Sample Reviews")
            st.dataframe(df.head())

            if st.button("🔍 Run ABSA on Reviews"):
                results = []
                for review in df['review_text']:
                    result = absa_pipeline(review)
                    results.append(result)

                df['aspects_sentiment'] = results
                st.write("### 🔎 ABSA Results")
                st.dataframe(df[['review_id', 'review_text', 'aspects_sentiment']])

                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download results as CSV",
                    data=csv,
                    file_name='absa_results.csv',
                    mime='text/csv'
                )
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("👈 Upload a CSV file from the sidebar.")




import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter

# Set up page
st.set_page_config(page_title="Aspect-Based Sentiment Analysis", layout="wide")
st.title("🧠 Aspect-Based Sentiment Analysis (ABSA) App")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "yangheng/deberta-v3-base-absa-v1.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    absa_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return absa_pipeline

absa_pipeline = load_model()

# Upload CSV file
st.sidebar.header("📁 Upload your CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV with 'review_text' column", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()  # remove extra spaces from column names

        if 'review_text' not in df.columns:
            st.error("❌ Your CSV must have a column named 'review_text'.")
            st.write("📋 Available columns:", list(df.columns))
        else:
            st.success("✅ File uploaded successfully!")
            st.write("### 📄 Sample Reviews")
            st.dataframe(df.head())

            if st.button("🔍 Run ABSA on Reviews"):
                results = []
                sentiment_labels = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
                for review in df['review_text']:
                    result = absa_pipeline(review)
                    sentiment = result[0]['label']
                    sentiment_labels[sentiment] += 1
                    results.append(result)

                # Store results in dataframe
                df['aspects_sentiment'] = results
                st.write("### 🔎 ABSA Results")
                st.dataframe(df[['review_id', 'review_text', 'aspects_sentiment']])

                # Visualize sentiment distribution
                st.write("### 📊 Sentiment Distribution")
                sentiment_counts = Counter(sentiment_labels)
                sentiment_names = list(sentiment_counts.keys())
                sentiment_values = list(sentiment_counts.values())

                fig, ax = plt.subplots()
                ax.bar(sentiment_names, sentiment_values, color=['green', 'yellow', 'red'])
                ax.set_xlabel('Sentiment')
                ax.set_ylabel('Count')
                ax.set_title('Sentiment Distribution of Reviews')
                st.pyplot(fig)

                # Download results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download results as CSV",
                    data=csv,
                    file_name='absa_results.csv',
                    mime='text/csv'
                )
    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("👈 Upload a CSV file from the sidebar.")
"""
import streamlit as st
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Review Sentiment Analyzer", layout="wide")

# Load sentiment analysis pipeline
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis")

sentiment_pipeline = load_model()

# Sidebar
st.sidebar.title("📂 Upload Review CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with a 'review_text' column", type="csv")

st.title("📝 Customer Review Sentiment Analyzer")
st.markdown("Analyze product reviews and visualize their sentiment using a transformer model.")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if "review_text" not in df.columns:
            st.error("CSV must contain a 'review_text' column.")
        else:
            # Analyze sentiment
            st.success("Sentiment analysis in progress...")
            with st.spinner("Analyzing reviews..."):
                results = sentiment_pipeline(df["review_text"].tolist())
                df["label"] = [res["label"] for res in results]
                df["confidence"] = [round(res["score"], 3) for res in results]

            st.subheader("🔍 Analyzed Reviews")
            st.dataframe(df[["review_text", "label", "confidence"]])

            # Sentiment count chart
            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = df["label"].value_counts()
            fig, ax = plt.subplots()
            ax.bar(sentiment_counts.index, sentiment_counts.values, color=["tomato", "gold", "mediumseagreen"])
            ax.set_xlabel("Sentiment")
            ax.set_ylabel("Number of Reviews")
            ax.set_title("Sentiment Distribution")
            st.pyplot(fig)

            # Option to download results
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results", data=csv, file_name="analyzed_reviews.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error reading file: {e}")
else:
    st.info("Please upload a CSV file to begin.")

