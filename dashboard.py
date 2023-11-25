import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from wordcloud import WordCloud

warnings.filterwarnings("ignore")



# page setting
st.set_page_config(layout="wide", page_title="Reviews Dashboard")
st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Reviews Dashboard")


# datasets
semi = pd.read_csv("https://raw.githubusercontent.com/Junnie-FTWB8/files/main/semi_dropped_play_store_reviews_6_months.csv")
clean = pd.read_csv("https://raw.githubusercontent.com/Junnie-FTWB8/files/main/clean_dropped_play_store_reviews_6_months.csv")



# parameters (group into a container - side)
with st.sidebar:
    st.header("Parameters")
    
    month = st.multiselect(
    'Select Month/s (2023)',
    ['May', 'June', 'July', 'August', 'September', 'October', 'November'])

    bin = st.selectbox(
    'Select Rating Category',
    ('All Ratings','Low Ratings', 'High Ratings'))

    ngrams = st.selectbox(
    'Select Grouping',
    ('Unigram', 'Bigram'))

    # add conditionals to update the dashboard content once the button is clicked
    st.button("Generate", type="primary")



# MAIN DASHBOARD AREA

# rating distribution and average rating
def filter_ratings(selected_option):
    if selected_option == 'Low Ratings':
        filtered_df = clean[clean['rating'] <= 3]
    elif selected_option == 'High Ratings':
        filtered_df = clean[clean['rating'] > 3]
    else:
        filtered_df = clean  # Show all ratings by default
    return filtered_df

def main():
    st.title('Rating Distribution and Mean')

    # Select box for choosing low or high ratings
    selected_option = st.selectbox(
        'Select Rating Category',
        ('Low Ratings', 'High Ratings')
    )
    
    filtered_data = filter_ratings(selected_option)
    
    # Display the distribution of ratings using a bar chart
    st.subheader('Rating Distribution')
    rating_counts = filtered_data['rating'].value_counts().sort_index()
    plt.bar(rating_counts.index, rating_counts.values)
    plt.xlabel('Rating')
    plt.ylabel('Count')
    st.pyplot()

    # Calculate and display mean rating
    mean_rating = filtered_data['rating'].mean()
    st.subheader('Mean Rating')
    st.write(f"The mean rating is: {mean_rating:.2f}")

    # Display color indicator based on mean rating
    if mean_rating > 3:
        st.markdown("<p style='color: green;'>Above 3</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: red;'>Below 3</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()



# word cloud
st.subheader("Word Cloud")

text = st.text_input("Enter text")
if text:
    w = WordCloud().generate(text)
    plt.imshow(w)
    plt.axis("off")
    st.pyplot()
