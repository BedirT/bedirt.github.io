---
author: "Bedir Tapkan"
title: "Best Methods for Sentiment Analysis for your weird data"
date: 2023-04-06
description: "Talking about my research and details on Sentiment Analysis"
tags: ["NLP", "Machine Learning", "Sentiment Analysis", "Transfer Learning", "HuggingFace", "Transformers"]
ShowToc: true
draft: true
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

