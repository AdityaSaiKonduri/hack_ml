import torch
import proselint
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Load the model

def preprocess(title, text, subject, date):
    # Step 1 : Checking punctuation
    def check_punctuation(headline):
        issues = proselint.tools.lint(headline)
        print("checking")
        if(not len(issues)):
            return 1
        else:
            return 0
    
    # Step 2 : title length, text length, text sentence length
    stop_words = set(stopwords.words('english'))  # Get English stop words

    def title_length(title):
        # remove stop words and count the length of the title
        title = title.split()
        title = [word for word in title if word.lower() not in stop_words]
        # count number of chars in title without stop words
        return [sum(len(word) for word in title), len(title)]


    def text_length(text):
        # remove stop words and count the length of the text
        text = text.split()
        text = [word for word in text if word.lower() not in stop_words]
        return [sum(len(word) for word in text), len(text)]
    
    def text_sentence_count(text):
        # count number of sentences in the text
        return len(text.split('.'))
    
    # Step 3 : Sentiment Analysis
    def sentiment_analyzer_scores(title):
        analyser = SentimentIntensityAnalyzer()
        score = analyser.polarity_scores(title)
        return [score['neg'], score['neu'], score['pos'], score['compound']]
    
    # Step 4 : Cateory ID
    def category_id(subject):
        categories = ['politics', 'worldnews', 'News', 'politicsNews', 'worldnews']
        return categories.index(subject) + 1
    
    # Step 5 : Keyword Density
    def title_keyword_density(title):
        headline = title
        tokens = word_tokenize(headline)
        tags = pos_tag(tokens)

        jj_count = sum(1 for word, tag in tags if tag == 'JJ')
        vbg_count = sum(1 for word, tag in tags if tag == 'VBG')
        rb_count = sum(1 for word, tag in tags if tag == 'RB')
        total_count = len(tokens)

        jj_density = (jj_count / total_count) * 100 if total_count > 0 else 0
        vbg_density = (vbg_count / total_count) * 100 if total_count > 0 else 0
        rb_density = (rb_count / total_count) * 100 if total_count > 0 else 0

        return [jj_density, vbg_density, rb_density]

    def text_keyword_density(text):
        headline = text
        tokens = word_tokenize(headline)
        tags = pos_tag(tokens)

        jj_count = sum(1 for word, tag in tags if tag == 'JJ')
        vbg_count = sum(1 for word, tag in tags if tag == 'VBG')
        rb_count = sum(1 for word, tag in tags if tag == 'RB')
        total_count = len(tokens)

        jj_density = (jj_count / total_count) * 100 if total_count > 0 else 0
        vbg_density = (vbg_count / total_count) * 100 if total_count > 0 else 0
        rb_density = (rb_count / total_count) * 100 if total_count > 0 else 0

        return [jj_density, vbg_density, rb_density]
    
    return [category_id(subject), *title_length(title), *text_length(text), text_sentence_count(text), *sentiment_analyzer_scores(title), check_punctuation(title), *title_keyword_density(title), *text_keyword_density(text)]

model = torch.load('hack_ml/hack/model.pth', weights_only=True)

test_title = input("Enter the title of the article: ")
test_text = input("Enter the text of the article: ")
test_subject = input("Enter the subject of the article: ")
test_date = input("Enter the date of the article: ")

# Preprocess the input
test_input = preprocess(test_title, test_text, test_subject, test_date)

print(test_input)

# Convert the input to a tensor
test_input = torch.tensor(test_input).float()

# Make a prediction
output = model.predict(test_input)
print(output)
output = output.detach().numpy()
print(output)