# Twitter-Sentiment-Tracker
A program that utilises Twitter API and a pre-trained model to produce plots of sentiment trends from the past week.
The model was trained using the dataset generated from: Go, A., Bhayani, R. and Huang, L., 2009. Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(2009), p.12.

## Usage
First a Twitter Developer account must be made. With this account, a project must be made. This will yield a bearer token, which will be used to authorise use of the API. In the main.py file, place the bearer token in the BEARER_TOKEN variable. The file should now run perfectly.

To use the program, place all queries in the 'queries' list. Queries are made of strings. See Twitter API for more details. The N_DATA_POINTS and MAX_RESULTS variables can also be altered.

The program will iterate through the queries and do the following:
1. Divide the last week up into N_DATA_POINTS regions.
2. Make request to Twitter API with query to extract tweets in each range.
3. Compute the sentiment of each tweet and produce an average sentiment.
4. Plot the sentiment of the query over the past week.

Once completed, you should have a plot of the average sentiment vs date for all queries.
