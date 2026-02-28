import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import string

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background: #f0f2f6;
        color: #333333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        height: 3em;
        width: 100%;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        margin-top: 10px;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .big-font {
        font-size:28px !important;
        font-weight: 700 !important;
        color: #1F2937;
    }
    .small-font {
        font-size:14px !important;
        color: #4B5563;
    }
    .result-spam {
        background-color: #fde2e2;
        color: #b72121;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        display: inline-block;
        margin: 10px 0;
        font-size: 20px;
    }
    .result-not-spam {
        background-color: #e2f4de;
        color: #2a6f32;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        display: inline-block;
        margin: 10px 0;
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="big-font">Spam Email Detector with AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="small-font">Detect spam messages easily using a trained Naive Bayes classifier. Paste a message or upload a dataset for batch prediction.</p>', unsafe_allow_html=True)

# Sidebar info and options
with st.sidebar:
    st.header("About This App")
    st.markdown("""
    - Uses Naive Bayes classification on message texts
    - Dataset: SMS Spam Collection (Ensure your CSV has columns: 'Category' and 'Message')
    - Model accuracy > 95%
    - Preprocessing includes stopword removal & vectorization
    - Upload CSV files with a 'Message' column for batch detection
    """)
    st.markdown("---")
    st.header("User Controls")
    preprocess_lower = st.checkbox("Convert to lowercase", value=True)
    preprocess_remove_punct = st.checkbox("Remove punctuation", value=True)
    st.markdown("---")
    st.header("Sample Messages")
    st.write("Some example spam messages you can try:")
    st.markdown("""
    - Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/12345 now.
    - Free entry in 2 a weekly competition to win FA Cup final tickets.
    - Call this number to claim your prize.
    - Text 'WIN' to 80085 for a chance to win a phone.
    - Your account has been suspended. Click to verify.
    """)

# Load dataset with caching for speed
@st.cache_data
def load_data():
    # Update this path if your spam.csv location is different
    data = pd.read_csv("C:/Users/mohit/OneDrive/Desktop/spam/spam.csv", encoding='latin-1')
    data.drop_duplicates(inplace=True)

    # Check columns; adjust these lines if your CSV columns are differently named
    if set(['Category', 'Message']).issubset(data.columns):
        data = data[['Category', 'Message']]
    else:
        # If original dataset uses 'v1' and 'v2', rename accordingly
        data = data[['v1', 'v2']]
        data.columns = ['Category', 'Message']

    data['Category'] = data['Category'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])
    return data

data = load_data()

# Show dataset insights section
if st.checkbox("Show Dataset Overview and Insights"):
    st.subheader("Dataset Snapshot")
    st.dataframe(data.sample(10))

    st.subheader("Class Distribution")
    class_counts = data['Category'].value_counts()
    fig, ax = plt.subplots()
    colors = ['#2a6f32', '#b72121']
    sns.barplot(x=class_counts.index, y=class_counts.values, palette=colors, ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("Category")
    ax.set_title("Spam vs Not Spam Message Counts")
    st.pyplot(fig)

    st.subheader("Message Length Analysis")
    data['Message_Length'] = data['Message'].apply(len)
    fig2, ax2 = plt.subplots()
    sns.histplot(data=data, x='Message_Length', hue='Category', bins=50, multiple="stack", palette=colors, ax=ax2)
    ax2.set_title("Message Length Distribution by Category")
    ax2.set_xlabel("Length of Message (characters)")
    st.pyplot(fig2)

# Text preprocessing helper
def preprocess_text(text, lower=True, remove_punct=True):
    if lower:
        text = text.lower()
    if remove_punct:
        text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Prepare features and model training with caching
@st.cache_data(show_spinner=False)
def train_model(data, lower, remove_punct):
    data['Processed'] = data['Message'].apply(lambda x: preprocess_text(x, lower, remove_punct))
    X = data['Processed']
    y = data['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cv = CountVectorizer(stop_words='english')
    X_train_vec = cv.fit_transform(X_train)
    X_test_vec = cv.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    clf_report = classification_report(y_test, preds, output_dict=True)
    conf_mat = confusion_matrix(y_test, preds)

    return cv, model, acc, clf_report, conf_mat

cv, model, accuracy, clf_report, conf_mat = train_model(data, preprocess_lower, preprocess_remove_punct)

# Show model performance summary
if st.checkbox("Show Model Performance"):
    st.markdown(f"**Model Accuracy:** {accuracy:.3f}")
    st.subheader("Classification Report")
    report_df = pd.DataFrame(clf_report).transpose()
    st.dataframe(report_df.style.background_gradient(cmap='PuRd', axis=1))

    st.subheader("Confusion Matrix")
    fig3, ax3 = plt.subplots(figsize=(5,4))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'], ax=ax3)
    ax3.set_xlabel("Predicted Label")
    ax3.set_ylabel("True Label")
    st.pyplot(fig3)

# Prediction function using trained vectorizer and model
def predict_message(message):
    proc_message = preprocess_text(message, preprocess_lower, preprocess_remove_punct)
    vect_message = cv.transform([proc_message])
    pred = model.predict(vect_message)[0]
    return pred

# Main input area
st.markdown("---")
st.header("Test Your Message for Spam")

user_input = st.text_area("Enter your email or SMS message here:", height=150)
if user_input:
    prediction = predict_message(user_input)
    if prediction == "Spam":
        st.markdown('<div class="result-spam">⚠️ This message is likely <b>SPAM</b>.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-not-spam">✅ This message is <b>NOT SPAM</b>.</div>', unsafe_allow_html=True)

# Batch processing for user-uploaded CSV file
st.markdown("---")
st.header("Batch Prediction for CSV Upload")

uploaded_file = st.file_uploader("Upload a CSV file with a 'Message' column", type=['csv'])
if uploaded_file:
    try:
        batch_data = pd.read_csv(uploaded_file)
        if 'Message' not in batch_data.columns:
            st.error("CSV must contain a 'Message' column.")
        else:
            # Preprocess messages
            batch_data['Processed'] = batch_data['Message'].astype(str).apply(lambda x: preprocess_text(x, preprocess_lower, preprocess_remove_punct))
            batch_vect = cv.transform(batch_data['Processed'])
            batch_preds = model.predict(batch_vect)
            batch_data['Prediction'] = batch_preds

            st.write("Batch prediction results:")
            st.dataframe(batch_data[['Message', 'Prediction']])

            # Downloadable results
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')

            csv_data = convert_df_to_csv(batch_data[['Message', 'Prediction']])
            st.download_button(label="Download results as CSV", data=csv_data, file_name='spam_predictions.csv', mime='text/csv')
    except Exception as e:
        st.error(f"Error reading the file: {e}")
