# Customer Product Feedback And Reviews Analysis (ABSA) – Streamlit App

This Streamlit app performs **Aspect-Based Sentiment Analysis** on customer product reviews using a pre-trained transformer model from Hugging Face 🤗. It allows users to upload review data, analyze sentiments at the aspect level (e.g., "battery", "camera"), and visualize the distribution of sentiments (positive, negative, neutral) using interactive charts.

## Features

-  Analyze customer feedback at the **aspect level**
-  Uses pre-trained transformer model: `yangheng/deberta-v3-base-absa-v1.1`
-  Upload CSV files with product reviews
-  Visualize sentiment distribution with a **pie chart**
-  Download results as CSV

## Tech Stack

- Python
- [Streamlit](https://streamlit.io/)
- [Transformers (Hugging Face)](https://huggingface.co/)
- PyTorch
- Pandas
- Matplotlib

##  File Structure

<pre> ## 📂 File Structure <code> absa-streamlit-app/ │ ├── app.py # Main Streamlit application ├── requirements.txt # Python dependencies ├── reviews.csv # Sample input file (optional) ├── README.md # Project documentation └── .gitignore # Git rules </code> </pre>


