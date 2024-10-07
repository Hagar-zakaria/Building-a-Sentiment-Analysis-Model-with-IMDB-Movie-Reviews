In this project, we aim to create a deep learning model that classifies IMDB movie reviews as either positive or negative. We will use techniques like text vectorization, embeddings, and a Bidirectional LSTM network to process the text data and predict sentiment.

# 1. Loading the IMDB Dataset
We'll use TensorFlow's tensorflow_datasets module to load the IMDB dataset, which contains 50,000 movie reviews classified into positive and negative sentiments. We'll also preprocess the data to make it ready for model input.

## Code : 
```python
import tensorflow_datasets as tfds

# Loading the IMDB dataset
dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
```

## Explanation:
The dataset is loaded using TensorFlow's datasets library. We split it into training and testing sets for model development and evaluation.


## Output (Dataset Info):
'''yaml
Splits: 2 (train and test)
Number of Examples: 50,000
'''
