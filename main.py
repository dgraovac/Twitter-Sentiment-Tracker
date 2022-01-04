
import pandas as pd
import numpy as np
import pickle
import spacy
import tensorflow as tf
from tensorflow.keras import layers
import sys
import requests
import json
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# CONSTANTS
N_DATA_POINTS = 15
MAX_RESULTS = 100
BEARER_TOKEN = "<INSERT BEARER TOKEN HERE>"
SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"

# List of queries to be made and plotted
queries = []
queries.append("Manchester United")
queries.append("Liverpool FC")
queries.append("Chelsea FC")


# Load model.
# Model is deep neural network that has input size 300 and output size 1.
# Each word in a text is converted into a vector and then they are averaged into
# one vector (the input). The output is a number in [0,1] that is positive if > 0.5
# and negative otherwise.
model = pickle.load(open('model.pkl', 'rb'))

# Load pretained language model for text -> vector conversion.
nlp = spacy.load("en_core_web_lg")

# Authorization function
def auth(r):
    r.headers["Authorization"] = "Bearer " + BEARER_TOKEN
    r.headers["User-Agent"] = "v2RecentSearchPython"
    return r

# Make request to Twitter API and return JSON reponse
def get_response(url, params):
    response = requests.get(url, auth=auth, params=params)
    if response.status_code != requests.codes.ok:
        sys.exit(response.text)
    return response.json()


# JSON reponse and model are input. JSON reponse contains a list of tweets.
# The sentiment of each tweet is calculated and the average is returned
def calculate_sentiment(json_response, model):
    sentiments = []
    try:
        for tweet in json_response["data"]:
            text_vec = np.array([nlp(tweet["text"]).vector])
            sentiments.append(model.predict(text_vec)[0][0])
        classes = [1 if i > 0.5 else 0 for i in sentiments]
        return sum(classes)/len(classes)
    except:
        return 0.5



# Converts datetime object into suitable string format for Twitter API
def datetime_to_string(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

# Constructs string query
def construct_query(query, max_results, start_time, end_time):
    return {'query': query,'tweet.fields': 'text', 'max_results': max_results, "start_time": datetime_to_string(start_time), "end_time": datetime_to_string(end_time)}


# Produces n points, from which we sample the API between.
def produce_time_ranges(n):
    time_ranges = []
    start_point = datetime.today()-timedelta(days=6, hours=23)
    end_point = datetime.today()-timedelta(hours=1)
    if n < 2:
        raise Exception("Invalid value of number of data points")
    else:
        delta = (end_point-start_point)/(n-1)
        for i in range(0,n-1):
            t1 = start_point + delta*i
            t2 = start_point + delta*(i+1)
            time_ranges.append([t1,t2])
        return time_ranges


# Takes the query, max_results to be returned by API and number of slices.
# Produces the average sentiment value.
def produce_sentiments(query, max_results, data_points):
    time_ranges = produce_time_ranges(data_points)
    results = pd.DataFrame(columns=["date", "sentiment"])
    if max_results < 10 or max_results >100:
        raise Exception("Invalid number of max results. Must be between 10 and 100.")
    for i,dates in enumerate(time_ranges):
        date_mid_point = (dates[1]-dates[0])/2 + dates[0]
        full_query = construct_query(query, max_results, dates[0], dates[1])
        json_response = get_response(SEARCH_URL, full_query)
        sentiment = calculate_sentiment(json_response, model)
        results.loc[i] = [date_mid_point, sentiment]
    return results


# Produces plot
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())

# Loops through the queries and plots each sentiment over time.
for q in queries:
    results = produce_sentiments('"'+ q + '" -(is:retweet) lang:en', MAX_RESULTS, N_DATA_POINTS)
    plt.plot(results.date, results.sentiment, label=q, marker='.')


plt.legend()
plt.ylabel("Sentiment")
plt.xlabel("Date")
plt.title("Sentiment over time")
plt.gcf().autofmt_xdate()
plt.show()
