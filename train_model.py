
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import spacy
from sklearn.model_selection import train_test_split
import pickle

# Information for reading data file
input_dir = 'tweet_dataset.csv'
encoding = 'ISO-8859-1'
column_names = ['target', 'id', 'date', 'flag', 'user', 'text']


df = pd.read_csv(input_dir, names=column_names, encoding=encoding)

# Randomly sample 50,000 each of positive and negative sentiment tweets.
data_negative = df[df.target==0.0].sample(n=50000)
data_positive = df[df.target==4.0].sample(n=50000)
data=pd.concat([data_positive, data_negative])

# shuffle the data points
data=data.sample(frac=1).reset_index(drop=True)

X=data['text']
y=(data['target']/4).astype(int) # convert sentiment values to binary integers.


# Load language model.
nlp = spacy.load('en_core_web_lg')

# Split data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.1, random_state=1)

# Produce document vectors for text in the training and validation sets
with nlp.disable_pipes():
    X_t = np.array([nlp(text).vector for text in X_train])
    X_v = np.array([nlp(text).vector for text in X_valid])

y_valid = y_valid.reset_index()
y_valid = y_valid.drop('index', axis=1)


# Define model to be used
model = keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(600, activation='relu', input_shape=[300]),
    layers.Dropout(0.2),
    layers.Dense(400, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(200, activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile model with appropriate loss funcition and evaluation metric
model.compile(
    optimizer='SGD',
    loss='binary_crossentropy',
    metrics=["binary_accuracy"]
)

# Train model
history=model.fit(
    X_t, y_train,
    batch_size=50,
    epochs=100,
)

# get predictions from validation set
preds = model.predict(X_v)
preds = pd.DataFrame(preds)

# Classifies sentiment responses
preds[0] = preds[0].apply(lambda x: 0 if x < 0.5 else 1)

# Collect predictions and actual values in dataframe
new_df = pd.concat([preds, y_valid], axis=1, names=['preds', 'actual'])

new_df.loc[new_df[0] == new_df['target'], 'correct'] = 1
new_df.fillna(0)

# Print accuracy of model
print(new_df.correct.value_counts()/10000)

# Dump model into pickle file
pickle.dump(model, open('model.pkl', 'wb'))
