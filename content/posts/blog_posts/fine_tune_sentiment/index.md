---
author: "Bedir Tapkan"
title: "Tackling Unconventional Data: A Guide to Optimizing Sentiment Analysis Models for Atypical Text"
date: 2023-04-13
description: "Talking about my research and details on Sentiment Analysis"
tags: ["NLP", "Machine Learning", "Sentiment Analysis", "Transfer Learning", "HuggingFace", "Transformers"]
ShowToc: true
---

I was recently working on a sentiment analysis tool for [my company](https://www.avalancheinsights.com/). Having worked
on it myself before, I was confident that I could get it done in a few days. However, I was wrong. I ended up going 
through quite a bit of research and experimentation to get the results we were happy with. In this post, I will
talk about my research, thought process, and the methods I used to get the results we wanted.

# The Problem & Starting Point

My company is a startup that provides high-quality qualitative coding services to businesses. After coding, we 
display the results in a dashboard for our clients. We want to accomodate our clients' needs by providing them
with a dashboard that has the features they want. One of the features we wanted to add was sentiment analysis.

Sentiment analysis is a common NLP task that involves classifying text into the sentiments. This can be done in
a variety of ways. For example, you can classify the text into positive, negative, or neutral. You can also classify
the text into more fine-grained sentiments such as very positive, positive, neutral, negative, and very negative.
There are also other sentiment analysis tasks such as emotion classification where you classify the text into
emotions such as anger, joy, sadness, etc. If we go even further, there is an interest over finding where the
sentiment is in the text. For example, you can find the sentiment of a sentence in a paragraph. This is called
aspect-based sentiment analysis. You can read more about these tasks [here](https://www.surveymonkey.co.uk/mp/what-customers-really-think-how-sentiment-analysis-can-help/).

With that being said, we decided to go with the most common sentiment analysis task which is classifying the text
into positive, negative, or neutral. This gives us the most flexibility in terms of the types of data we can use
for training, or models we can test that are already available.

So now, we have a sentiment analysis task that will classify the text into positive, negative, or neutral. The
next step is to find a pre-trained sentiment analysis model to have a baseline to compare our results to. But
we have a problem prior to that. Our data is not in the format that the models are trained on, or any publicly
available data. So we need to have some labelled data to test these models on to compare their results (to know,
which model performs the best for our data). 

# Data Labeling

The first step is to label our data. We have a lot of data, but we don't have the time to label all of it. So
we decided to label a very small subset of it. We used [Doccano](https://github.com/doccano/doccano) to
label. It is a very simple tool that is built to easily label your data. You can read more about it on their
github page.

After labeling, we had a small dataset that we can use to test our models. We had 200 samples that were
selected via stratified sampling. The initial plan was to label 1000 data samples, but we decided to go with
200 samples to save time.

# Pre-trained Models

Now that we have our data, we can start testing our models. The easiest thing to do is to use the models from
[HuggingFace's Transformers](https://huggingface.co/models?pipeline_tag=text-classification). It's no mystery
that the Attention-based Transformer models have been dominating the NLP tasks for the past few years. They have
been the state-of-the-art for many NLP tasks. So it is no surprise that they are also the state-of-the-art for
sentiment analysis. Later in the post I will talk about some specific base-models that I used, what are their
differences and why did I select them. 

For the first pass, I selected couple high ranked models from HuggingFace's Transformers. I also used a base 
model 'VADER' that is a rule-based sentiment analysis tool. I used the base model to compare the results of the
Transformer models. And, of course with all the success of GPT-3.5 and GPT-4, we needed to include some few-shot
and zero-shot models using GPT using [OpenAI](https://openai.com/) framework.

So let's list out all the models I used:
1. [VADER](https://github.com/cjhutto/vaderSentiment)
2. [Huggingface "sbcBI/sentiment_analysis_model"](https://huggingface.co/sbcBI/sentiment_analysis_model)
3. [Huggingface "cardiffnlp/twitter-xlm-roberta-base-sentiment"](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment)
4. [Huggingface "Seethal/sentiment_analysis_generic_dataset"](https://huggingface.co/Seethal/sentiment_analysis_generic_dataset)
5. [Huggingface "LiYuan/amazon-review-sentiment-analysis"](https://huggingface.co/LiYuan/amazon-review-sentiment-analysis)
6. [Huggingface "ahmedrachid/FinancialBERT-Sentiment-Analysis"](https://huggingface.co/ahmedrachid/FinancialBERT-Sentiment-Analysis)
7. [Huggingface "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)
8. [PySentimento](https://github.com/pysentimiento/pysentimiento")
5. GPT-3.5 (zero-shot, few-shot)
6. GPT-4 (zero-shot, few-shot)

Let's go over some basic examples on how to use each type of models, and jump into our initial results.

### VADER
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sample = "I love this product. It is the best product I have ever used."

# Create a SentimentIntensityAnalyzer object.
analyzer = SentimentIntensityAnalyzer()

# Sentiment scores
score = sentiment_task.polarity_scores(row['text'])['compound']

# The scoring for VADER is different than the other models. Please read about it in the documentation.
if score >= 0.05:
    sentiment = 'positive'
elif score <= -0.05:
    sentiment = 'negative'
else:
    sentiment = 'neutral'
```

### HuggingFace Transformers
```python
from transformers import pipeline

sample = "I love this product. It is the best product I have ever used."

model_name = "sbcBI/sentiment_analysis_model"
sentiment_task = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
label = sentiment_task(row['text'])[0]['label']
```

### PySentimento
```python
from pysentimiento import create_analyzer

sample = "I love this product. It is the best product I have ever used."

sentiment_task = create_analyzer(task='sentiment', lang='en')
label = sentiment_task.predict(row['text']).output
```

### GPT-3.5/4
```python
import openai
import os

# Get your api key loaded
openai.api_key = os.environ.get("OPENAI_API_KEY")

sample = "I love this product. It is the best product I have ever used."

messages = [
    {"role": "system", "content": "Specific Sentiment Instructions"},
    # if you want few-shot, samples go here
    {"role": "user", "content": f"{sample}\nSentiment:"},
]
# Send the request
response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=messages,
    max_tokens=20,
)

sentiment = response['choices'][0]['message']['content']
```

# Evaluation Metrics

To be able to judge the performance of our models, we need to have some evaluation metrics.
Commonly used metrics for sentiment analysis are:
1. Accuracy
2. Precision
3. Recall
4. F1 Score

It is a good idea to use a combination of these metrics to get a better understanding of the performance of
the model. This is especially important when we have an imbalanced dataset. For example, if we have 1000
samples, and 900 of them are positive, and 100 of them are negative. Then we can get a very high accuracy
score by always predicting positive. But that doesn't mean that our model is good. So we need to use other
metrics to evaluate the performance of our model.

F1 score is a combination of precision and recall. So we decided to use F1 score and accuracy as our
evaluation metrics.

Here is the function that we will use to calculate the accuracy and F1 score.

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
```

# Initial Results

Now that we have our models, and we have our evaluation metrics, we can start testing the pre-trained models.
We will use the 200 samples that we labeled to test the models. Since there is no training involved, we will
use all the data for testing. 

Let's not forget that these results are more of a sanity check, and a general evaluation of how close our
data is to the ones used to train the models. If by luck our data is very similar to the data used to train
the models, then we can expect to get good results and stop there. But if the results are not good, then we
need to do some more work to get better results, or try to find a better model.

Here is the accuracy plot including all the models.

![Accuracy Plot](img/accuracy_plot.png)

Here is the F1 score plot including all the models.

![F1 Score Plot](img/f1_score_plot.png)

As you can see, the VADER model is the worst performing model. Best performing model is the GPT-4 model. Other
than that, GPT-3.5 is performing close. As we can see the huggingface models are not really performing well.
Best open-source model is the PySentimento model, but it still isn't at the level we want. 

One thing to note is that the labelling of our data is pretty complex and is even hard for humans to label. So
there might be some bias in the data. But we will not go into that in this post since I am not revealing the 
data itself. 

So we can see that the GPT-3.5 and GPT-4 models are performing well. These are zero-shot models, we could get
even better results if we do few-shot training. 

After seeing the potential of GPT models (and the poor performance of the pre-trained sentiment analysis models),
we decided to first investigate GPT-3.5 and GPT-4 models, and then try to train our own sentiment analysis model
using GPT as the labeler. This will give us a smaller open-source model that we can use for our system, that
performs similar to GPT models but doesn't cost us anything.

# Evaluating GPT-3.5 and GPT-4

Starting with the same small dataset, we first test some different prompting methods to see how can we get the
best results. This will guide us as to which method we should use as the labeling method for our sentiment
analysis model.

One thing we tested aside from the prompts was the
general prompting technique. For these kind of individually dependent tasks, we can introduce a parameter
called `sample batch size`. This parameter controls how many samples are sent to the model at once. This
parameter is important because if we send all the samples at once. This will result in the model trying to
generate all the labels at once, which is a harder task. Pros however is the cost since we do not have to
repeat the same pre-prompt (or instructions) for each sample.

I am not going into too much detail as to what prompts we used. But to give general direction;
We include clear instructions for the model. GPT models gives us a flexibility to explain what exactly we 
want from the model. We can describe how we perceive the tasks, and what we expect from the model. For this
we have clear definitions as to what is considered positive, negative, and neutral. 

Here are the results of the different prompting methods.

![GPT Prompting Results](img/gpt_prompting_results.png)

We included 4 different metrics in the plot:
1. Accuracy: As we already discussed, this is the main measure of how good our model is predicting the labels.
We can see that both GPT-3.5 and GPT-4 are performing very well with `sample batch size` of 1, the 
`sample batch size` of 10 is performing significantly worse.
2. F1 Score: This is a combination of precision and recall. We can see that F1 score is following the same 
pattern as accuracy.
3. Price: This is the cost of the model. This is important as we might end up using this model in production.
We can see that the `sample batch size` of 1 is more expensive than the `sample batch size` of 10. 
4. Time: This is the time it takes to generate the labels. Again this is important if we end up using this
model in production.

As we can see, both GPT-3.5 and GPT-4 are performing very well. We can see that the `sample batch size` of 1
is performing better than the `sample batch size` of 10. Even though GPT-4 is performing slightly better, we
decided to go with GPT-3.5 as it is way cheaper and way faster. 

For training an open-source model we will use GPT-3.5 to generate the bulk of the labels (120000 data points). 
We then use GPT-4 to generate the labels for an extra 10000 data points. This way we can see how close we can
get to GPT-4 performance with a smaller model.

# Training a Sentiment Analysis Model

Now that we have the labels for our data, we can start training our sentiment analysis model. We will discuss
in order of the following steps:
1. Data Preprocessing and Preparation
2. Model Selection
3. Model Training
4. Model Evaluation

## Data Preprocessing and Preparation

Let's start by splitting the data into train and test sets. We will use 80% of the data for training, and 
20% of the data for validation. We will use the extra 10000 data points we generated with GPT-4 to test the 
performance of our model (test set). 

Right before that, we need to do some preprocessing. We need to convert the labels to numbers. We will use
the following mapping:
1. Positive: 2
2. Neutral: 1
3. Negative: 0

Aside from that, the labels by GPT-3.5 is not always correct. We can either drop these data points, or we can
manually correct them. We decided to manually correct most, and leave some of the outliers alone. These examples
include dots, different phrasings, etc. I will include couple of examples below.

```python
data.loc[data["label"] == "Negative (with a hint of frustration) ", "label"] = "Negative"
data.loc[data["label"] == "Negative.", "label"] = "Negative"
data.loc[data["label"] == "Mixed/Neutral.", "label"] = "Neutral"
# ...

# Convert labels to numbers
label_mapping = {"Positive": 2, "Neutral": 1, "Negative": 0}
data['label'] = data['label'].map(label_mapping)
data.dropna(inplace=True)

# Split the data into train and validation sets
train, val = train_test_split(data, test_size=0.2, random_state=42)
```

We can check the distribution of the labels in the train and test sets. This will give us an idea of how
balanced our data is.

```python
data["label"].value_counts()
```

```text
0    62002
1    34350
2    23820
```

```python
train["label"].value_counts()
```

```text
0    49773
1    27366
2    18998
```

```python
val["label"].value_counts()
```

```text
0    12229
1     6984
2     4822
```

We can see that the distribution of the labels is not very balanced. We can see that the negative labels are
the most common, and the positive labels are the least common. This is something we considered when we were
deciding on how we will evaluate our model. But it is generally a good idea to know the distribution of the
labels in the data.

Next is to create data loader for pytorch. We prepare the data for training by creating a `Dataset` class.

```python
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, data_file, tokenizer):
        self.data = pd.read_csv(data_file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.loc[idx, 'text']
        label = self.data.loc[idx, 'label']
        inputs = self.tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors="pt")
        inputs['input_ids'] = inputs['input_ids'].squeeze()
        inputs['attention_mask'] = inputs['attention_mask'].squeeze()
        inputs['labels'] = torch.tensor(label, dtype=torch.long)  # Change this line
        return inputs
```

## Model Selection

Next step is to select a model to train. We will use the `transformers` library to train our models. We are 
going to build a classifier on pre-trained language models such as BERT. In this section we will first discuss
the different models we considered, how do they differ from each other, what are the pros and cons of each
model, and then we will implement each model, train them and evaluate them. Here are the models we considered:

1. BERT
2. RoBERTa
3. DistilBERT
4. XLM-RoBERTa
5. GPT2
6. RoBERTa-Large
7. DistilBERT-Large
8. GPT2-Medium
9. GPT2-Large

We have 9 different models, let's go over each model and explain how they differ from each other.

### BERT

BERT is a transformer-based model introduced by Google in 2018 in the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). 
It was a significant milestone in the field of NLP, as it achieved state-of-the-art results on several tasks. 
BERT is pre-trained on large amounts of text data and can be fine-tuned for various NLP tasks, such as sentiment 
analysis, question-answering, and more. BERT uses bidirectional context, which means that it considers both left 
and right contexts in text when learning representations. This characteristic allows the model to have a better 
understanding of the textual context. Even though it is still used by many researchers, it is not the most recent 
model, so it usually is outperformed by newer models.

**Pros:**

- Achieves high performance on many NLP tasks.
- Can be fine-tuned for specific tasks.
- Bidirectional context improves the understanding of textual information.

**Cons:**

- Large model size.

### RoBERTa

RoBERTa is an optimized version of BERT, introduced by Facebook AI in 2019 in the paper [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692). 
It builds upon BERT's architecture but implements several modifications that improve its performance. Some key 
changes include a **larger training dataset**, **longer training time**, and **the removal of the next-sentence 
prediction task** during pre-training. RoBERTa also uses dynamic masking, which allows the model to see multiple masks for 
the same token during pre-training, resulting in better performance on downstream tasks.

**Pros:**

- Improved performance compared to BERT on several NLP tasks.
- Retains the benefits of BERT, such as fine-tuning capabilities and bidirectional context.

**Cons:**

- Still has large model size and high computational requirements, similar to BERT.

### DistilBERT

DistilBERT is a smaller version of BERT, introduced by Hugging Face in 2019 in the paper [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108). 
It aims to maintain most of BERT's performance while reducing its size and computational requirements. DistilBERT
has approximately half the number of parameters as BERT and is faster during training and inference. The 
distillation process involves training a smaller model (the student) to mimic the behavior of a larger model 
(the teacher), in this case, BERT.

**Pros:**

- Reduced model size and faster training and inference.
- Retains a substantial portion of BERT's performance.
- Can be fine-tuned for specific tasks.

**Cons:**

- Slightly lower performance compared to BERT and RoBERTa.

### XLM-RoBERTa

XLM-RoBERTa is a multilingual version of RoBERTa, introduced by Facebook AI in 2019 in the paper [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116). 
It is pre-trained on a large dataset comprising 100 languages. XLM-RoBERTa builds upon the RoBERTa architecture 
and aims to offer improved performance on cross-lingual tasks, such as machine translation and multilingual sentiment 
analysis.

**Pros:**

- Multilingual model that can be used for cross-lingual tasks.
- Retains the benefits of RoBERTa, such as improved performance compared to BERT and DistilBERT.
- Can be fine-tuned for specific tasks.

**Cons:**

- Still has large model size and high computational requirements.

We will now implement each of these models and train them on our data. 

### GPT2

GPT2 is a transformer-based language model introduced by OpenAI in 2019 in the paper [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).
It is a large model that is pre-trained on a large dataset of text data. GPT2 is a generative model, which means that it
generates text one token at a time. It uses a left-to-right autoregressive language modeling (LM) objective, which means
that it tries to predict the next token in the sequence given the previous tokens. GPT is generally better at generating
text than BERT, meaning it imagines more creative text. Since we are trying to imitate a GPT generated output, we will
give it a shot as well.

**Pros:**

- Generates creative text.
- Can be fine-tuned for specific tasks.

**Cons:**

- Generally worse at classification tasks compared to BERT and RoBERTa.

## Training

We will train each of the models on the training set, and evaluate them on the validation set. We use the
`transformers` library to train our models. Switching between models is very easy, as the `transformers` library
provides a unified API for all the models.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
```

We will use the `Trainer` class to train our models. The `Trainer` class is a high-level API that handles
the training loop, evaluation loop, and prediction loop. It also handles the data loading, model saving,
and model loading. We will use the `TrainingArguments` class to specify the training arguments, such as
the number of epochs, the batch size, and the learning rate.

```python
model_name = 'bert-base-uncased' # we are changing this for each model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

train_dataset = SentimentDataset('train.csv', tokenizer)
val_dataset = SentimentDataset('val.csv', tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=64,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    evaluation_strategy='epoch',     # evaluation strategy to adopt during training
    save_strategy='epoch',           # model saving strategy
    load_best_model_at_end=True,     # load the best model found during training
    metric_for_best_model='accuracy',# metric to use for selecting the best model
    seeed=42,                        # random seed for initialization
    learning_rate=5e-5,              # learning rate
    warmup_steps=100,                # number of warmup steps for learning rate scheduler
    weight_decay=0.05,               # strength of weight decay
    learning_rate_scheduler='linear',# learning rate scheduler
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)
```

After setting the arguments, we also set up the optimizer and the learning rate scheduler. We use the AdamW optimizer
with a linear learning rate scheduler. We also set the random seed to 42 for reproducibility.

```python
from transformers import AdamW, get_linear_schedule_with_warmup

optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=training_args.warmup_steps,
    num_training_steps=len(train_dataset) * training_args.per_device_train_batch_size * training_args.num_train_epochs
)
```

We are now ready to train our model. We instantiate the `Trainer` class and call the `train` method to start training.

```python
# Trainer
trainer = Trainer(
    model=model,                        # the instantiated 🤗 Transformers model to be trained
    args=training_args,                 # training arguments, defined above
    train_dataset=train_dataset,        # training dataset
    eval_dataset=val_dataset,           # evaluation dataset
    tokenizer=tokenizer,                # the instantiated 🤗 Transformers tokenizer
    compute_metrics=compute_metrics,    # the callback that computes metrics of interest
    optimizers=(optimizer, scheduler)   # the optimizer and the learning rate scheduler
)

# Depending on the model there might be a need to setup the padding token manually
# tokenizer.pad_token = tokenizer.eos_token
# model.config.pad_token_id = model.config.eos_token_id
trainer.train()
```

We can then evaluate the model on the validation set one more time. I am keeping the test set for when we are fully
done with training and evaluation, so we don't make biased decisions.

```python
# Evaluate the model on the validation set
eval_results = trainer.evaluate()
```

We can also save the model and the tokenizer.

```python
trainer.save_model('./results')
tokenizer.save_pretrained('./results')
```

## Evaluation

So we have finalized the training of our models. After running each model and tweaking the parameters to be able to
get the best performance, we can see the results. Here are the results for different metrics:

| Model | Accuracy | F1 Score | Precision | Recall |
| --- | --- | --- | --- | --- |
| BERT | x | x | x | x |
| RoBERTa | x | x | x | x |
| DistilBERT | x | x | x | x |
| XLM-RoBERTa | x | x | x | x |
| GPT2 | x | x | x | x |
| --- | --- | --- | --- | --- |
| RoBERTa Large | x | x | x | x |
| DistilBERT Large | x | x | x | x |
| GPT2 Medium | x | x | x | x |
| GPT2 Large | x | x | x | x |

...
