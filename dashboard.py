import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
nltk.download('wordnet')
nltk.download('stopwords')
from cleaning_dictionaries import *

from sklearn.exceptions import FitFailedWarning
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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
        default=['All Months']
    )

    if not month:
        st.error("Please select at least one month.")

    bin = st.selectbox(
    'Select Rating Category',
    ['All Ratings','Low Ratings', 'High Ratings'])

    ngrams = st.radio("Select Grouping", ["Unigram", "Bigram"])


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

# ROW A (Number of Reviews, Average Rating)
num_reviews = len(data_clean) - 1
avg_rating = data_clean["rating"].mean()

col1, col2 = st.columns(2)

# Card 1: Number of Reviews
with col1:
    st.metric("Number of Reviews", num_reviews)

# Card 2: Average Rating
with col2:
    st.metric("Average Rating", avg_rating)



# ROW B (Rating Distribution and Daily Distribution)
col1, col2 = st.columns(2)

# Card 1: Number of Reviews
with col1:
    st.subheader("Rating Distribution")
    rating_counts = data_clean['rating'].value_counts().sort_index()
    st.bar_chart(rating_counts)

# Card 2: Average Rating
with col2:
    st.subheader(f'Daily Rating Distribution - {month} ({bin})')
    # Group by date and calculate rating distribution
    daily_rating_counts = data_clean.groupby(['date', 'rating']).size().unstack(fill_value=0)
    st.area_chart(daily_rating_counts)
        


# ROW C (Topic Frequency - with unigram and bigram tabs, and Dataframe)
col1, col2, = st.columns([1,2])

with col1:
    st.subheader('Themes/Topics')
    # Display content based on selected tab
    if ngrams == "Unigram":
        st.write("Content for Unigram")
    elif ngrams == "Bigram":
        st.write("Content for Bigram")

with col2:
    # Display the 'clean' dataframe
    st.subheader('Filtered Dataframe')
    st.dataframe(data_clean)
        


# ROW D (Word Cloud - with unigram and bigram tabs)
st.subheader("Word Cloud")

# Content based on selected tab
if ngrams == "Unigram":
    combined_text = ' '.join(data_clean['review_clean'])
    wordcloud = WordCloud(background_color='white').generate(combined_text)
    plt.figure()
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
elif ngrams == "Bigram":
    # Your code for generating word cloud from bigrams here
    st.write("Code for generating word cloud from bigrams")
    


# ROW E (Coefficients)
# Mapping 1 to low rating and 2, 3 to high rating
data_clean['rating_category'] = data_clean['bin_rating'].map({1: 'Low', 2: 'Low', 3: 'High'})

# Mapping Low rating as 0 and High rating as 1
data_clean['binary_rating'] = data_clean['rating_category'].map({'Low': 0, 'High': 1})

# Drop the intermediate column 'rating_category' if you don't need it
df2 = data_clean.drop('rating_category', axis=1)

X = df2.review_clean
y = df2.binary_rating
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

warnings.filterwarnings("ignore", category=FitFailedWarning)

# Creating a pipeline with TfidfVectorizer and Logistic Regression
pipeline = Pipeline([
    ('vect', TfidfVectorizer(ngram_range=(2, 2))),
    ('clf', LogisticRegression(max_iter=1000))  # Increase max_iter if necessary
])

# Define the parameters you want to search through for unigrams and bigrams
parameters = {
    'clf__C': [1.0, 10.0, 100.0],  # Regularization parameter
    'clf__solver': ['lbfgs', 'liblinear'],  # Solvers for Logistic Regression
    'vect__use_idf': [True]
}

# Create GridSearchCV to search for best parameters
grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, error_score='raise')

try:
    # Fit the model
    grid_search.fit(X_train, y_train)

    # Predict with the best model
    y_pred = grid_search.predict(X_test)

except Exception as e:
    print(f"An error occurred during grid search:\n{str(e)}")

# Get feature names after fitting the vectorizer
feature_names = grid_search.best_estimator_.named_steps['vect'].get_feature_names_out()

# Assuming 'clf' in your pipeline represents the Logistic Regression model
logistic_regression_coef = grid_search.best_estimator_.named_steps['clf'].coef_

# Bigrams coefficients
# bigrams_coef = logistic_regression_coef[0][len(feature_names):]
bigrams_coef = logistic_regression_coef[0]    # No need to use Len since the set tfidvectorizer is for Bigrams only. It isn't considering both unigrams and bigrams like in the previous rating prediction

# Display coefficients with corresponding features for bigrams
bigrams_coef_with_features = list(zip(feature_names, bigrams_coef))
sorted_bigrams_coef_with_features = sorted(bigrams_coef_with_features, key=lambda x: x[1], reverse=True)

top_n = 3  # Set the number of top coefficients to display

top_features = []
top_coeffs = []

bottom_features = []
bottom_coeffs = []

for feature, coef in sorted_bigrams_coef_with_features[:top_n]:
    top_features.append(feature)
    top_coeffs.append(coef)

for feature, coef in sorted_bigrams_coef_with_features[-top_n:]:
    bottom_features.append(feature)
    bottom_coeffs.append(coef)

col1, col2 = st.columns(2)

# for unigrams, not functional yet
if ngrams == "Unigram":
    with col1:
        st.subheader("Top Features")
        # Create a bar chart for top coefficients
        st.bar_chart(pd.DataFrame({'Top Features': top_features, 'Top Coefficients': top_coeffs}).set_index('Top Features'), color = "#B2FF66")

    with col2:
        st.subheader("Bottom Features")
        # Create a bar chart for bottom coefficients
        st.bar_chart(pd.DataFrame({'Bottom Features': bottom_features, 'Bottom Coefficients': bottom_coeffs}).set_index('Bottom Features'), color = "#FF6666")

if ngrams == "Bigram":
    with col1:
        st.subheader("Top Features")
        # Create a bar chart for top coefficients
        st.bar_chart(pd.DataFrame({'Top Features': top_features, 'Top Coefficients': top_coeffs}).set_index('Top Features'), color = "#B2FF66")

    with col2:
        st.subheader("Bottom Features")
        # Create a bar chart for bottom coefficients
        st.bar_chart(pd.DataFrame({'Bottom Features': bottom_features, 'Bottom Coefficients': bottom_coeffs}).set_index('Bottom Features'), color = "#FF6666")


