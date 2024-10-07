In this project, we aim to create a deep learning model that classifies IMDB movie reviews as either positive or negative. We will use techniques like text vectorization, embeddings, and a Bidirectional LSTM network to process the text data and predict sentiment.

![image](https://github.com/user-attachments/assets/e8da15f2-ca6c-4701-8e8c-9b77a09a8702)

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


# 4. Building the Model
We build a deep learning model using Bidirectional LSTM, which reads the input text data in both directions to understand the context better.

## Code:
```python
model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(input_dim=len(encoder.get_vocabulary()), output_dim=64, mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
```

## Explanation:
The model begins with a text encoder, followed by an embedding layer that converts each word into a dense vector.
We then use a Bidirectional LSTM to process the text in both forward and backward directions, followed by two dense layers for classification.

# 5. Training the Model
We train the model using the IMDB dataset and evaluate its performance on unseen test data.

## Code:

```python
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)
```

## Explanation:
We train the model for 10 epochs (passes through the entire training data) and validate its performance on the test data after each epoch.


## Output (Model Accuracy and Loss):

```yaml
Epoch 1/10
Train Accuracy: 85.81%, Test Accuracy: 85.8%
Loss: 0.32 on both training and validation datasets.
```


# 6. Evaluating the Model
After training, we evaluate the model's performance using the test dataset.

## Code:

```python
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")
```

## Output (Test Loss and Accuracy):

```yaml
Test Loss: 0.32
Test Accuracy: 85.80%
```

# 7. Visualizing Model Performance

We plot the accuracy and loss over each epoch for both training and validation datasets to check if the model is learning correctly and avoiding overfitting.

## Code:
```python
import matplotlib.pyplot as plt

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])
    plt.show()

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plot_graphs(history, 'accuracy')
plt.subplot(1, 2, 2)
plot_graphs(history, 'loss')
```

![image](https://github.com/user-attachments/assets/9bd5b64e-8781-48d7-b660-ef7c30a490b3)

## Explanation:
We create two subplots: one for accuracy and one for loss, to visualize the model's performance over the 10 epochs.
Output (Accuracy and Loss Graphs):

## Conclusion:
The model's accuracy and loss both improve over time and stabilize, showing good generalization and minimal overfitting.

# 8. Predicting on New Text Samples
Finally, we make predictions on new IMDB reviews using our trained model.

## Code:
```python
sample_text = ("The movie was cool. The animation and the graphics were out of this world. "
               "I would recommend this movie.")
predictions = model.predict(np.array([sample_text]))
print(f"Prediction for the sample text: {predictions}")
```

## Explanation:
We pass a new text review through the model and predict its sentiment. If the output is greater than 0, the sentiment is positive, and if it’s less than 0, the sentiment is negative.

## Output (Sample Prediction):

```java
Prediction: Positive sentiment (score: 0.98)
```


# Conclusion
In this project, we successfully built and trained a sentiment analysis model using the IMDB dataset. The model was able to predict movie review sentiments with over 85% accuracy on the test data. Through techniques like text vectorization, padding, and Bidirectional LSTM, the model can capture complex relationships in the text and generalize well on unseen data.

Future improvements could involve hyperparameter tuning or experimenting with different architectures such as Transformer models.



