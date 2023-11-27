import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import nltk
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')
from wordcloud import WordCloud
# from cleaning_dictionaries import contractions, stopwords_list, tagalog_bad_words

warnings.filterwarnings("ignore")


# page setting
st.set_page_config(layout="wide", page_title="Reviews Dashboard")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Reviews Dashboard")


# datasets
semi = pd.read_csv("https://raw.githubusercontent.com/Junnie-FTWB8/files/main/semi_dashboard_play_store_reviews_6_months.csv")
clean = pd.read_csv("https://raw.githubusercontent.com/Junnie-FTWB8/files/main/clean_dashboard_play_store_reviews_6_months.csv")



# SIDEBAR
with st.sidebar:
    st.header("Parameters")
    
    month = st.multiselect(
    'Select Month/s (2023)',
    ['All Months', 'May', 'June', 'July', 'August', 'September', 'October', 'November'],
    default=['All Months'])

    bin = st.selectbox(
    'Select Rating Category',
    ['All Ratings','Low Ratings', 'High Ratings'])

# querying the dataframes
if 'All Months' in month:
    data_semi = semi.copy()  # For 'All Months', copy the entire dataframe
    data_clean = clean.copy()  # For 'All Months', copy the entire dataframe
else:
    data_semi = semi[semi['month'].isin(month)]  # Filter semi dataframe by selected months
    data_clean = clean[clean['month'].isin(month)]  # Filter clean dataframe by selected months

if bin == 'Low Ratings':
    data_semi = data_semi[data_semi['bin_label'] == 'Low']
    data_clean = data_clean[data_clean['bin_label'] == 'Low']
elif bin == 'High Ratings':
    data_semi = data_semi[data_semi['bin_label'] == 'High']
    data_clean = data_clean[data_clean['bin_label'] == 'High']
elif bin != 'All Ratings':
    data_semi = data_semi[data_semi['bin_label'] == bin]



# MAIN DASHBOARD AREA

# ROW A (Number of Reviews, Average Rating, Rating Distribution)
# Sample data (Replace these with your actual data)
num_reviews = len(data_clean) - 1
avg_rating = data_clean["rating"].mean()

col1, col2 = st.columns(2)

# Card 1: Number of Reviews
with col1:
    st.metric("Number of Reviews", num_reviews)

# Card 2: Average Rating
with col2:
    st.metric("Average Rating", avg_rating)
        


# ROW B (Topic Frequency - with unigram and bigram tabs, and Dataframe)
col1, col2, = st.columns([1,2])

with col1:
    st.subheader('Themes/Topics')
    selected_tab = st.tabs(["Unigram", "Bigram"])

    # Display content based on selected tab
    if selected_tab == "Unigram":
        st.write("Content for Unigram")
    elif selected_tab == "Bigram":
        st.write("Content for Bigram")
with col2:
    # Display the 'clean' dataframe
    st.subheader('Filtered Dataframe')
    st.dataframe(data_clean)
        


# ROW C (Word Cloud - with unigram and bigram tabs)
st.subheader("Word Cloud")
selected_tab = st.tabs(["Unigram", "Bigram"])

# Content based on selected tab
if selected_tab == "Unigram":
    combined_text = ' '.join(data_clean['review_clean'])
    wordcloud = WordCloud(background_color='white').generate(combined_text)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
elif selected_tab == "Bigram":
    # Your code for generating word cloud from bigrams here
    st.write("Code for generating word cloud from bigrams")
    
    
    
# ROW C (Word Cloud - with unigram and bigram tabs)
st.subheader("Word Cloud")
selected_tab = st.sidebar.radio("Select Tab", ["Unigram", "Bigram"])

# Content based on selected tab
if selected_tab == "Unigram":
    combined_text = ' '.join(data_clean['review_clean'])
    wordcloud = WordCloud(background_color='white').generate(combined_text)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
elif selected_tab == "Bigram":
    # Your code for generating word cloud from bigrams here
    st.write("Code for generating word cloud from bigrams")



# ROW D (Coefficients)
col1, col2 = st.columns(2)

with col1:
    st.write("Positive Coefficient")

# Contents for the second column
with col2:
    st.write("Negative Coefficient")


