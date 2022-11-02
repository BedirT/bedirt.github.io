---
author: "Bedir Tapkan"
title: "Sentiment Analysis A to B: Episode 1"
date: 2022-11-01
description: "Sentiment Analysis Experiments, Data Prep, Representations and Logistic Regression"
tags: ["NLP", "Machine Learning", "Sentiment Analysis"]
ShowToc: true
---
# Sentiment Analysis A to B: Episode 1

In this series, I will work my way into different Sentiment Analysis methods and experiment with other techniques. I will use the data from the [IMDB review dataset](https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis) acquired from Kaggle. The series is called A to B since I need to cover all the methods and the best, for that matter. I am covering some I find exciting and test-worthy.

In this episode, I will be examining/going over the following:

- Data preprocessing for sentiment analysis
- 2 different feature representations:
    - Sparse vector representation
    - Word frequency counts
- Comparison using logistic regression

## Feature Representation

Your model will be, at most, as good as your data, and your data will be only as good as you understand them to be, hence the features. I want to see the most useless or naive approaches and agile methods and benchmark them for both measures of prediction success and for training and prediction time. 

Before anything else, let's load, organize and clean our data really quick:

```python
import CSV

def get_data():
    with open('data/kaggle_data/movie.csv', 'r') as f:
        data = list(CSV.reader(f, delimiter=','))
    return data

def split_data(data):
    # split 80/10/10
    train_split = int(0.8 * len(data))
    val_split = int(0.9 * len(data))
    return data[:train_split], data[train_split:val_split], data[val_split:]

def save_data(data, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)

def main():
    data = get_data()
    train, val, test = split_data(data[1:])
    save_data(train, 'data/train.csv')
    save_data(val, 'data/val.csv')
    save_data(test, 'data/test.csv')
```

Let's start with creating a proper and clean vocabulary that we will use for all the representations we will examine.

### Clean Vocabulary

We just read all the words as a set, to begin with,

```python
# Get all the words
words = [w for s in train_data for w in s[0].split()]
# len(words) = 7391216

# Get the vocabulary
dirty_vocab = set(words)
# len(dirty_vocab) = 331056
```

So for the beginning of the representation, we have 331.056 words in our vocabulary. This number is every non-sense included, though. We also didn't consider any lowercase - uppercase conversion. So let's clean these step by step. 

```python
# Convert to lowercase
vocab = set([w.lower() for w in dirty_vocab])
# len(vocab) = 295827

# Remove punctuation
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
vocab = set([w.lower() for w in tokenizer.tokenize(' '.join(vocab))])
# len(vocab) = 84757
```

We reduced the number from 331.056 to 84.757. We can do more. With this method, we encode every word we see in every form possible. So, for example, "called," "calling," "calls," and "call" will all be a separate words. Let's get rid of that and make them reduce to their roots. Here we start getting help from the dedicated NLP library NLTK since I don't want to define all these rules myself (nor could I):

```python
# Reduce words to their stems
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
vocab = set([stemmer.stem(w) for w in vocab])
# len(vocab) = 58893
```

The last step towards cleaning will be to get rid of stopwords. These are 'end,' 'are,' 'is,' etc. words in the English language.

```python
# Remove connectives
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = set(stopwords.words('English))
vocab = vocab - stop_words
# len(vocab) = 58764
```

Now that we have good words, we can set up a lookup table to keep encodings for each word.

```python
# Vocabulary dictionary
vocab_dict = {w: i for i, w in enumerate(vocab)}
```

Now we have a dictionary for every proper word we have in the data set. Therefore, we are ready to prepare different feature representations.

Since we will convert sentences in this clean form, again and again, later on, let's create a function that combines all these methods:

```python
# Function to combine all the above to clean a sentence
def clean_sentence(sentence):
    # Convert to lowercase
    sentence = sentence.lower()
    # Words
    words = sentence.split()
    # Remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(' '.join(words))
    # Remove stop words
    stop_words = set(stopwords.words('English))
    words = [w for w in words if w not in stop_words]
    # Reduce words to their stems
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]
    # remove repeated words
    return set(words)
```

Ideally, we could initialize `tokenizer` `stemmer` and `stop_words` globally (or as a class parameter), so we don't have to keep initializing.

### Sparse Vector Representation

This will represent every word we see in the database as a feature… Sounds unfeasible? Yeah, it should be. I see multiple problems here. The main one we all think about is this is a massive vector for each sentence with a lot of zeros (hence the name). This means most of the data we have is telling us practically the same thing as the minor part; we have these words in this sentence vs. we don't have all these words. Second, we are not keeping any correlation between words (since we are just examining word by word).

We go ahead and create a function for encoding every word for a sentence:

```python
# function to convert a sentence to a vector encoding
def encode_sparse(sentence):
    words = sentence.split()
    vec = np.zeros(len(vocab))
    clean_words = clean_sentence(sentence)
    for w in clean_words:
        if w in vocab_dict:
            vec[vocab_dict[w]] += 1
    return vec
```

We then convert all the data we have using this encoding (in a single matrix):

```python
train_data_sparse = np.array([encode_sparse(s[0]) for s in train_data]), np.array([int(s[1]) for s in train_data])
val_data_sparse = np.array([encode_sparse(s[0]) for s in val_data]), np.array([int(s[1]) for s in val_data])
```

That's it for this representation.

### Word Frequency Representation

This version practically reduces the 10.667 dimensions to 3 instead. We are going to count the number of negative sentences a word passes in as well as positive sentences. This will give us a table indicating how many positive and negative sentences a word has found in:

```python
# Counting frequency of words
freqs = np.zeros((len(vocab), 2)) # [positive, negative]
for i, s in enumerate(train_data):
    words = clean_sentence(s[0])
    for w in words:
        if w in vocab_dict:
            freqs[vocab_dict[w], int(s[1])] += 1
```

The next thing to do is to convert these enormous numbers into probabilities. There are multiple points to add here: First, we are getting the probability of this single word being in many positive and negative sentences, so the values will be minimal. Hence we need to use a log scale to avoid floating point problems. Second is, we might get words that don't appear in our dictionary, which will have a likelihood of 0. Since we don't want a 0 division, we add laplacian smoothing, like normalizing all the values with a small initial. Here goes the code:

```python
# Convert to log probabilities with Laplace smoothing
freqs = np.log((counts + 1) / (np.sum(counts, axis=0) + len(vocab)))
```

After getting the frequencies and fixing the problems we mentioned, we now define the new encoding method for this version of the features

```python
def encode_freq(sentence):
    words = clean_sentence(sentence)
    vec = np.array([1., 0., 0.]) # [bias, positive, negative]
    for word in words:
        if word in vocab_dict:
            vec[1] += freqs[vocab_dict[word], 0]
            vec[2] += freqs[vocab_dict[word], 1]
    return vec
```

We end by converting our data as before

```python
train_data_pos_neg = np.array([encode_freq(s[0]) for s in train_data]), np.array([int(s[1]) for s in train_data])
val_data_pos_neg = np.array([encode_freq(s[0]) for s in val_data]), np.array([int(s[1]) for s in val_data])
```

Let's take a sneak peek at what our data looks like:

```python
# Visualize the data with PCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
%matplotlib inline

# Create a PCA instance. 
# This will reduce the data to 2 dimensions 
# as opposed to 3, where we have 2 features and a bias, more on that in next episode.
PCA = PCA(n_components=2)

# Fit the PCA instance to the training data
x_data = train_data_pos_neg[0]
PCA.fit(x_data)

# Transform the training data to 2 dimensions ignoring the bias. This is due to the fact that the bias is a constant and will not affect the PCA
x_data_2d = PCA.transform(x_data)
plt.scatter(x_data_2d[:, 0], x_data_2d[:, 1], c=train_data_pos_neg[1], cmap='bwr')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

# Setup legend
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='Negative')
blue_patch = mpatches.Patch(color='blue', label='Positive')
plt.legend(handles=[red_patch, blue_patch])
```

![Untitled](graph.png)

A better would be to use PCA for this kind of representation, but for now, we will ignore that fact since we want to explore that in episode 2.

## Model Development

This episode mainly focuses on cleaning the data and developing decent representations. This is why I will only include a single model to test everything; Logistic Regression.

### Logistic Regression

Logistic regression is a simple single-layer network with sigmoid activation. This is an excellent baseline as it is one of the simplest binary classification methods. I am not explaining this method in depth, so if you want to learn more, please do so. I will use a simple `PyTorch` implementation.

```python
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
```

We then define the loss function and the optimizer to use. I am using Binary Cross Entropy for the loss function and Adam for the optimization with a learning rate of `0.01`.

```python
device = torch.device('cuda' if torch.Cuda.is_available() else 'CPU')
model = LogisticRegression(x_data.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.BCELoss()

# Train the model
num_epochs = 100
train_loss = []
val_loss = []
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train.unsqueeze(1))
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_loss.append(loss.item())
    
    # Validation
    with torch.no_grad():
        outputs = model(X_val)
        loss = criterion(outputs, y_val.unsqueeze(1))
        val_loss.append(loss.item())
    
    if (epoch) % 100 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}' 
               .format(epoch, num_epochs, loss.item()))
```

**Sparse Representation Training** We first start with training the sparse representation. I trained for `100` epochs and reached `0.614` training accuracy and `0.606` validation accuracy. Here is the learning curve

![Untitled](graph1.png)

**Word Frequency Representation** **Training** I trained using the same parameter settings above, reaching `0.901` training accuracy and `0.861` validation accuracy. Here is the learning curve in the log scale

![Untitled](graph2.png)