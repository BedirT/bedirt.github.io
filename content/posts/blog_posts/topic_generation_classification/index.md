---
author: "Bedir Tapkan"
title: "Using GPT for Topic Generation and Classification"
date: 2023-04-25
description: "Talking about my research and experiments on Topic Generation and Classification using GPT-3.5/4."
tags: ["NLP", "Machine Learning", "Topic Classification", "Topic Modelling", "GPT-3.5", "GPT-4", "Few-shot Learning", "Zero-shot Learning", "LLMs"]
ShowToc: true
---

A new tool we are investigating with [my company](https://www.avalancheinsights.com/) is Topic Generation and Classification. This is an extensive experimentation process on Topic Modelling and GPT-3.5/4. I first start with topic modelling, how well can we actually describe topics using a simple BERTopic model, which seems to be doing the best job. We then investigate how strong this model actually is in matching our human experts. After discussing the weaknesses and strengths of this approach, we go ahead and investigate how GPT can help us. 

For this purpose I came up with an experimentation road map. I tried asking every question I could think of and tried to answer them in a systematic way. In this post we will go over this journey and discuss the results.

# What is Topic Modelling?

Topic modelling is a technique that allows us to extract topics from a corpus of text. It is an unsupervised technique that allows us to discover hidden semantic structures in a text. When we talk about text analysis, Topic Modelling is the main tool that comes to mind in classifying text without any labels. So if we have a bunch of documents, the method we use to assign meaning to bulk of them is Topic Modelling. 

This is a probabilistic model, that does not provide much accuracy. It is a very simple method in essence. Different methods have differnt approaches to this problem, but one of the most popular ones is BERTopic which uses BERT embeddings to cluster documents. That pretty much is it, even though the library is amazingly implemented and well maintained, the method is very simple. It uses sentence similarity to cluster documents, and analyze word frequency to assign topics and extract keywords.

We can mention why you would want to use topic modelling, and why not. 

**Pros**
- It is an unsupervised method, so you do not need labelled data
- It is very fast, and can be used on large datasets
- It is very easy to use, and does not require much tuning (maybe except for the number of topics)

**Cons**
- It is not very accurate, and can be very subjective (I cannot stress this enough)
- It is not very robust, and can be very sensitive to the number of topics
- Overall for a quality analysis, it does not provide much value
- For smaller datasets, it is not very useful

So if you were interested in a quality analysis, you would not want to use topic modelling. But if you were interested in a quick analysis, and you did not have any labelled data, then topic modelling is a great tool. If you want to read more about topic modelling I strongly suggest that you checkout [BERTopic](https://maartengr.github.io/BERTopic/algorithm/algorithm.html). It is a great library, and the documentation is very well written.

What we are aiming however is seeing the potential of GPT-3.5/4 in topic classification and generation, in a high quality analysis. We hypothesize that GPT models could speed up the process of qualitative analysis, and provide a more robust and accurate analysis for experts to start with.

Along the way, we use topic modelling as a baseline, since we don't have a better choice. One thing to mention here is that we are not making use of existing topic classification models. This is due to the fact that topic classification assumes the new results to be in already classified (labelled) topics. This is not the case for us, since we are trying to discover new topics, and then classify them with no prior knowledge. This begs for few-shot or zero-shot learning, which is what we test with GPT models.

One thing we did not mention, and it is crucial in any part of this process is that topic classification is a multi-label multi-class classification task. Which makes it much harder than any other classification method. We will discuss this in more detail later on when we talk about the evaluation metrics.

# Experiment 1: Creating a baseline, how does BERTopic perform in classifying existing topics?

In this experiment we assume that BERTopic has all the correct labels to given dataset, and should classify them into these classes. We will then compare the results with the actual labels, and see how well it performs. This is a very simple experiment, but it will give us a good idea of how well BERTopic performs in classifying topics.

BERTopic is not designed to perform a classification task with no training. What we do instead is, perform topic modelling on the dataset, and then map the topic labels generated to the closest class we have by looking at their cosine similarity. This gives us a proper class for each cluster and document. One good thing is that we also have the exact number of topics, so we can use that as a hyperparameter.

We use our internal data for the experiments, but you can use any dataset you want. We have a relatively large survey data, with small and big surveys (ranges from 50-20k responses per survey). We want to make sure the method we end up with can handle both ends of the spectrum. We also have a lot of different topics, which is another thing we want to make sure we can handle.


```python
import pandas as pd
from tqdm import tqdm

from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
from transformers.pipelines import pipeline


responses = pd.read_csv("responses.csv")
true_labels = pd.read_csv("true_labels.csv")
```

We first load the data, and then we will use BERTopic to perform topic modelling. We will use the default settings, and then we will map the topics to the closest class. We will then compare the results with the true labels, and see how well it performs.


```python
topic_model = BERTopic(nr_topics=len(class_names), embedding_model=embedding_model)
topics, _ = topic_model.fit_transform(responses["response"].tolist())

# Since we don't care about False Positives at this point (just looking for a raw accuracy) we get rid of the outliers
new_topics = topic_model.reduce_outliers(responses, topics, strategy="embeddings")
topic_model.update_topics(responses, topics=new_topics)

# Generate unique topic labels
class_names = true_labels["label"].unique().tolist()
```

Now that we trained and reduced the outliers, we can map the topics to the closest class.


```python
class_embeddings = {}
for class_name in class_names:
    class_embeddings[class_name] = embedding_model(class_name)[0][0]

topic_mapping = {}
for topic_id in tqdm(topic_model.get_topic_freq().index):
    topic_keywords = topic_model.get_topic(topic_id)[:10]
    topic_keywords = ' '.join([keyword for keyword, _ in topic_keywords])
    emb_topic = embedding_model(topic_keywords)[0][0]
    best_class, best_score = None, 0
    for class_name in class_names:
        emb_class = class_embeddings[class_name]
        score = cosine_similarity([emb_topic], [emb_class])[0][0]
        if score > best_score:
            best_score, best_class = score, class_name
    topic_mapping[topic_id] = best_class
```

Now we can check the accuracy of the topic modelling.


```python
acc = (responses['topic'] == responses['topic_mapping']).mean()
```

With our data, we get an average of `0.09` accuracy. Which I think is not much of a surprise. We have a lot of topics, and the topics are very similar to each other. This is a very hard task for topic modelling, and we cannot expect it to perform well. But we needed this experiment to see what we are dealing with, and what we can expect from topic modelling.

After this experiment we speculated that skipping topic modelling and testing just using cosine similarity and BERT embeddings might be a better approach. Due to the similarity of the approach, we include this experiment under the same section. The change is only in the main loop, so let's just see that.


```python
acc = 0.
for response, true_class in tqdm(zip(responses, true_classes), total=len(responses)):
    emb_response = embedding_model(response)[0][0]
    best_class, best_score = None, 0
    for class_name in survey_data[survey_id]['themes']:
        emb_class = class_embeddings[class_name]
        score = cosine_similarity([emb_response], [emb_class])[0][0]
        if score > best_score:
            best_score, best_class = score, class_name
    
    if best_class == true_class:
        acc += 1

acc /= len(responses)
```

This method yielded a `0.21` accuracy. As we can see, it is better than topic modelling, but still not good enough. Still, works better as the baseline, so we will use this method for the comparison when it comes to the final results. 

Since we mentioned accuracy couple times here, let's talk about what metrics should we be using to properly evaluate our model (hint: it is not accuracy).

# Evaluation Metrics
We cannot really calculate simple accuracy for multi-label classification. We need to use a different metrics. For our case we care the most about not labelling a response with a wrong class. We can tolerate not labelling a response with the correct class, but we cannot tolerate labelling a response with a wrong class. This is why we will be using precision as our main metric. We will also use recall and f1-score to get a better idea of how well our model performs.

Besides these, we will use another common metric for multi-label classification, that replaces the accuracy. It is called Jaccard similarity, and it is the intersection over union of the predicted and true labels. It is a good metric to use when we have a lot of classes, and we want to see how well our model performs in general. We will use this metric to compare our model with the baseline.

Before talking about each metric, we introduce two other friends of ours, price and time. Since we are actually hoping to productionize this method, it is important to talk about these two metrics as well. We will be using the same dataset for all the experiments, so we can compare the time it takes to train and predict for each method. We will also talk about the price of each method, and how much it would cost to run it in production.

## Precision
Precision is the number of true positives divided by the sum of true positives and false positives. In other words, it measures how well the model predicts the positive instances of each class. A high precision means that the model is good at avoiding false positives, which is important in our case since we want to avoid labeling a response with the wrong class. 

Precision can be calculated as:

```
Precision = TP / (TP + FP)
```

where TP is the number of true positives and FP is the number of false positives.

## Recall

Recall is the number of true positives divided by the sum of true positives and false negatives. It measures how well the model identifies the positive instances of each class. A high recall means that the model is good at finding the relevant instances, but it might also produce more false positives.

Recall can be calculated as:

```
Recall = TP / (TP + FN)
```

where TP is the number of true positives and FN is the number of false negatives.

## F1-Score

The F1-score is the harmonic mean of precision and recall, which provides a balance between these two metrics. It ranges from 0 to 1, with 1 being the best possible score. A high F1-score indicates that the model is good at both avoiding false positives and finding relevant instances.

F1-score can be calculated as:

```
F1-score = 2 * (Precision * Recall) / (Precision + Recall)
```

## Jaccard Similarity

Jaccard similarity, also known as the Jaccard index, is a measure of similarity between two sets. In our case, it is used to measure the similarity between the predicted and true labels. The Jaccard similarity ranges from 0 to 1, with 1 being a perfect match between the two sets.

Jaccard similarity can be calculated as:

```
Jaccard Similarity = (Intersection of Predicted and True Labels) / (Union of Predicted and True Labels)
```

## Time and Cost

In addition to the above-mentioned evaluation metrics, time and cost are also important factors when considering a model for production use. The time required for training and predicting with each method should be compared, as well as the cost associated with using a particular method, such as the price of using GPT-3.5/4 API, which could be significant depending on the size of the dataset.

With the metrics and the baseline ready, we can start talking about the implementation of our second experiment, how well GPT performs on classification.

# GPT-3.5/4 for Classification


# References
- https://www.clearpeaks.com/using-chatgpt-for-topic-modelling-and-analysis-of-customer-feedback/
- https://medium.com/@stephensonebinezer/transform-your-topic-modeling-with-chatgpt-cutting-edge-nlp-f4654b4eac99
- https://www.width.ai/post/gpt3-topic-extraction
- https://arxiv.org/abs/1908.10084
- https://maartengr.github.io/BERTopic/changelog.html
- https://monkeylearn.com/blog/introduction-to-topic-modeling/#:~:text=Topic%20modeling%20is%20an%20unsupervised,characterize%20a%20set%20of%20documents