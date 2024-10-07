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
```yaml
Splits: 2 (train and test)
Number of Examples: 50,000
```



# 2. Preparing the Data for Training
We need to vectorize the text data and ensure all sequences have the same length using padding.

## Code:

```python
import tensorflow as tf
BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Displaying a sample from the dataset

```python
for text, label in train_dataset.take(1):
    print(text.numpy())
    print(label.numpy())
```

## Explanation:
We shuffle the data and split it into batches to improve training efficiency.
The AUTOTUNE option automatically tunes the number of batches to ensure smooth training.


# 3. Text Vectorization and Embedding
Before feeding text into our model, we need to convert it into numerical format. We'll use TextVectorization for this.

```python
VOCAB_SIZE = 1000
encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train_dataset.map(lambda text, label: text))

# Displaying the first 20 words in the vocabulary
vocab = np.array(encoder.get_vocabulary())
print(vocab[:20])
```


## Explanation:
The TextVectorization layer creates a vocabulary from the training dataset and maps each word to a unique integer.
We then print the first 20 words in the vocabulary to see how the text data is processed.


## Output (First 20 Words):
```css
['' '[UNK]' 'the' 'and' 'a' 'of' 'to' 'is' 'it' 'in' 'i' 'this' 'that' 'was' 'for' 'you' 'movie' 'with' 'but' 'film']
```
