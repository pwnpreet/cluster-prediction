import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
import matplotlib.pyplot as plt
import nltk


from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams, pos_tag


ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Instering image  -----------------------------------
st.markdown(
        f"""
        <style>
        .stApp {{
        background-image: url("https://i.imgur.com/33BvndE.jpeg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Loading data   --------------------------------------
df = pd.read_csv("cl_hier.csv")
mean_shift = joblib.load("ms_model.pkl")
kmeans = joblib.load("kmean_model.pkl")
cluster = joblib.load("dbscan_model.pkl")
corpus = joblib.load("corpus.pkl")


# heading and subheading
st.title("✨ Clustering Application")
try:
    df= pd.read_csv("cl_hier.csv")
    st.success("✅ Data loaded successfuly")
except:
    st.error("😕 Data not found")  

# sidebare section 
section = st.sidebar.radio("Select an option", ["Dataset Preview", "Dataset Information", "Numerical Summary"])

if section == "Dataset Preview":
    view_options= st.sidebar.radio("Select Show to view dataset", ["Hide", "Show"])
    if view_options == "Show":
        st.sidebar.subheader("📁 Dataset preview")
        st.sidebar.write(df.head())
elif section == "Dataset Information":
    st.sidebar.subheader("🌟 Dataset Information")
    col1, col2= st.sidebar.columns(2)
    col1.metric(label="Number of Rows", value=df.shape[0])
    col2.metric(label="Number of Columns", value=df.shape[1])
elif section == "Numerical Summary":
    with st.sidebar.expander("📊 Numerical Summary"):
        st.write(df.describe())    

# catbot section  ----------------------------------------
st.markdown("---")
st.subheader("👀 Chatbot Assistent")

user_input = st.text_input("Ask your Queries")

def chatbot_response(user_text):
    text = user_text.lower()
    for q, a in corpus:
        words = q.lower().split()
        if all (w in text for w in words):
            return a
    return "sorry did not understand"

if st.button("Ask"):
    response = chatbot_response(user_input)
    st.session_state["last_response"]= response
    st.success(f"🤖: {response}")

# Login section 
credencials = {
    "admin": "a",
    "manager": "m",
    "trainer": "t"
}
if "logged_in" not in st.session_state:
    st.session_state.logged_in= False

if not st.session_state.logged_in:
    st.subheader("🔒 Login Required")
    username= st.text_input("Enter your Username")
    password= st.text_input("Enter your password", type= "password")
    btn= st.button("Login")

    if btn:
        if username in credencials and credencials[username] == password:
            st.success(f"😀 Welcome, {username}")
            st.session_state.logged_in= True
        else:
            st.error("🙄 Invalid Username or Password")

if st.session_state.logged_in:
    st.write("🎉 You have access to complete data")

    # st.subheader("📝 Data Analysis FAQ's")
    # ques1


    # ques2


    # ques3


    # ques4



    st.subheader("🐾 Predictions")

    # adding columns
    age = st.number_input("Age")
    annualIncome = st.number_input("Annual Income(k$)")
    score = st.number_input("Spending score", min_value=1, max_value=100)

    features = np.array([[age, annualIncome, score]])

    if st.button("Predict"):
        meanshift_pred = mean_shift.predict(features)[0]
        kmean_pred = kmeans.predict(features)[0]

        col1, col2 = st.columns(2)
        with col1:
            st.success(f"Cluster by Mean Shift: {meanshift_pred}")
        with col2:    
            st.success(f"Clusters by KMeans: {kmean_pred}")
    
        # plots highlighing user inputs
        st.subheader('Scatter Plots Highlighting User Input: K-Means Vs Mean-Shift')
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.scatter(df["Age"], df["Annual Income (k$)"],c = mean_shift.labels_, cmap= "viridis", marker="o")
            ax.scatter(age, annualIncome, color = "red", marker="*")
            ax.set_title("Mean-Shift Clustering")
            st.pyplot(fig)

        with col2:
            fig, ax =plt.subplots()
            ax.scatter(df['Age'], df['Annual Income (k$)'], c=kmeans.labels_, cmap='viridis', marker='o')
            ax.scatter(age, annualIncome, color= 'red', marker='*')
            ax.set_title("KMeans Clustering")
            st.pyplot(fig)



# nlp techniques  -------------------------------
st.sidebar.title("🔍 NLP Techniques")
options = st.sidebar.multiselect(
    "Select technique to display:",
    ["Stemmed", "Word Tokenize", "Lemmatized", "Stopword", "Ngrams", "POS Tags"],
)
nlp_btn = st.sidebar.button("Show")

def process_nlp(text):
    words = word_tokenize(text.lower())
    results = {
        "Stemmed": " ".join(words), 
        "Word Tokenize": " ".join([f"'{ps.stem(w)}'" for w in words]), 
        "Lemmatized": " ".join([f"'{lemmatizer.lemmatize(w)}'" for w in words]), 
        "Stopword": " ".join(f"'{w}'" for w in words if w not in stop_words), 
        "Ngrams": " ".join([" ".join(gram) for gram in ngrams(words, 2)]), 
        "POS Tags": " ".join([f"[{w:5}->{t}]" for w, t in pos_tag(words)])
    }
    return results
    


if nlp_btn:
    if 'last_response' in st.session_state:
        response = st.session_state['last_response']
        result = process_nlp(response)

        if options:
            st.sidebar.subheader("🧠 NLP Output")
            for technique in options:
                st.sidebar.info(f"**{technique}:** {result[technique]}")
        else:
            st.sidebar.warning("Please select at least one NLP technique.")
    else:
        st.sidebar.error("❗ Please ask something in chatbot first.")   
