import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Title
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
uploaded_file = st.sidebar.file_uploader("Choose a CSV file with a 'review' column", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if 'review_id' not in df.columns:
            st.error("CSV must contain a 'review' column.")
        else:
            st.success("✅ File uploaded successfully!")
            st.write("### Sample Reviews")
            st.dataframe(df.head())

            if st.button("🔍 Run ABSA on Reviews"):
                results = []
                for review in df['review']:
                    result = absa_pipeline(review)
                    results.append(result)

                df['aspects_sentiment'] = results
                st.write("### 🔎 ABSA Results")
                st.dataframe(df[['review', 'aspects_sentiment']])

                # Download button
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
    st.info("👈 Upload a CSV file to get started.")
