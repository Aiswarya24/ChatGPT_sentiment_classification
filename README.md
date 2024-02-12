LSTM Sentiment Analysis

This repository contains code for sentiment analysis using LSTM (Long Short-Term Memory) neural networks. The model is trained to classify tweets into three sentiment categories: Positive, Neutral, and Negative.
Dataset

The dataset consists of two CSV files:
train_dataset.csv: Contains the training data.
test_dataset.csv: Contains the test data.
Each CSV file has two columns: Tweets (text of the tweet) and Label (sentiment label).

Preprocessing

Tokenization: Tokenize the text using NLTK's word_tokenize function.

Stopwords Removal: Remove stopwords using NLTK's stopwords corpus.

Lemmatization: Lemmatize the words to their base forms using WordNet lemmatizer.

Model Architecture

The LSTM model consists of the following layers:
Embedding layer: Converts input sequences into dense vectors of fixed size.
SpatialDropout1D layer: Dropout layer for input sequences.
LSTM layer: Long Short-Term Memory layer with dropout.
Dense layer: Fully connected layer with softmax activation for classification.

Training

The model is trained using the training data and evaluated on the test data. Early stopping is applied to prevent overfitting.

Hyperparameters

embedding_dim: Dimensionality of the embedding space.
lstm_out: Number of LSTM units.
batch_size: Size of mini-batch.
epochs: Number of training epochs.

Results

The model achieves an accuracy of 0.88 on the test dataset.
