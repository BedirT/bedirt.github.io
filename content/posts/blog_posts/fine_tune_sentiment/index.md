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
talk about my research, thought process, and the methods I used to get the results I wanted.

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

With that being said, I decided to go with the most common sentiment analysis task which is classifying the text
into positive, negative, or neutral. This gives us the most flexibility in terms of the types of data we can use
for training, or models we can test that are already available.

So now, we have a sentiment analysis task that will classify the text into positive, negative, or neutral. The
next step is to find a pre-trained sentiment analysis model to have a baseline to compare our results to. But
we have a problem prior to that. Our data is not in the format that the models are trained on, or any publicly
available data. So we need to have some labelled data to test these models on to compare their results (to know,
which model performs the best for our data). 

# Data Labeling

The first step is to label our data. We have a lot of data, but we don't have the time to label all of it. So
we decided to label a very small subset of our data. We used [Doccano](https://github.com/doccano/doccano) to
label our data. It is a very simple tool that allows you to label your data. You can read more about it on their
github page.

After labeling our data, we had a small dataset that we can use to test our models. We had 200 samples that were
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
and zero-shot models using GPT. I used the GPT-3.5 and GPT-4 models using [OpenAI](https://openai.com/) framework.

So let's list out all the models I used:
1. VADER
2. ...
3. ...
4. ...
5. GPT-3.5 (zero-shot, few-shot)
6. GPT-4 (zero-shot, few-shot)

# Evaluation Metrics

