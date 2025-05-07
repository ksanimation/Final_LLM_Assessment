import torch
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import torch.optim as optim
from pprint import pprint  # Importing pprint for pretty-printing
import pandas as pd

# ___TASK 1___ Sentiment Labeling

# Import module and load model
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="MarieAngeA13/Sentiment-Analysis-BERT")

# Load data
df = pd.read_csv('./test_rev.csv', usecols=["Subject","body","date","from"], parse_dates=["date"])

# Convert useable data to list
data = df['body'].tolist()
## Check print(data[1])

# Store output in variable
sentiment = sentiment_pipeline(data)
##print(sentiment[1]["label"])

# Change labels to more readable format
final_sentiment = []


for i in range(len(sentiment)):
    if sentiment[i]["label"] == "positive":
        final_sentiment.append("Positive")
    if sentiment[i]["label"] == "neutral":
        final_sentiment.append("Neutral")
    elif sentiment[i]["label"] == "negative":
        final_sentiment.append("Negative")

## Check print(final_sentiment[1])

# Store sentiment in dataframe
dfSentiment = df.copy()
dfSentiment['Sentiment'] = final_sentiment

## Check print(dfSentiment.head(3)) 


""" sentiment_labels = []

    if sentiment[i]["label"] not in sentiment_labels:
        sentiment_labels.append(sentiment[i]["label"])
        continue
print(sentiment_labels)
    
for i in range(len(sentiment)):
if sentiment[i]["label"] == "NEUTRAL":
        print("Neutral Found") """
