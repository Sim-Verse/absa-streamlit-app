# Customer Product Feedback And Reviews Analysis (ABSA) – Streamlit App

This project is a Streamlit web application that performs Aspect-Based Sentiment Analysis on customer product reviews using a state-of-the-art transformer model. Instead of merely labeling a review as positive or negative, this app identifies sentiments for specific aspects (e.g., battery, screen, design) within each review, providing more granular insights for businesses and analysts.

🎯 Features
✅ Upload a CSV file containing customer product reviews.

🔍 Automatically detects sentiments (Positive, Neutral, Negative) for various aspects in each review.

📊 Real-time visualization of sentiment distribution using a pie chart.

💾 Download the analysis results as a CSV file.

⚡ Powered by Hugging Face's yangheng/deberta-v3-base-absa-v1.1 transformer model.

🧰 Tech Stack
Python 3

Streamlit – For building interactive web UI

Hugging Face Transformers – For pre-trained ABSA model

PyTorch – Deep learning backend

Pandas – Data processing

Matplotlib – Data visualization

##  File Structure
absa-streamlit-app/
│
├── app.py                # Main Streamlit application
├── requirements.txt      # Project dependencies
├── reviews.csv           # Sample CSV file with reviews
├── README.md             # Project documentation


