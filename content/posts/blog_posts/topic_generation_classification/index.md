---
author: "Bedir Tapkan"
title: "Using GPT for Qualitative Analysis"
date: 2023-04-25
description: "Talking about my research and experiments on Performing Qualitative Analysis using GPT-3.5/4."
tags: ["NLP", "Machine Learning", "Topic Classification", "Topic Modelling", "GPT-3.5", "GPT-4", "Few-shot Learning", "Zero-shot Learning", "LLMs"]
ShowToc: true
---

A new topic we are investigating with [my company](https://www.fathomthat.ai/) is Topic Generation and Classification. This is an extensive experimentation process on Topic Modelling and GPT-3.5/4 for qualitative analysis. I first start with topic modelling, how well can we actually describe topics using a simple BERTopic model, which currently is the state of the art model. We then investigate how strong this model actually is compared to our human experts. After discussing the weaknesses and strengths of this approach, we go ahead and investigate how GPT can help us improve this performence. 

For this purpose I came up with an experimentation road map. I tried asking every question I could think of and tried to answer them in a systematic way. In this post we will go over this journey and discuss the results.

{{< collapse title="Click to Expand This Section!" openByDefault=true >}}
- We explore **Topic Generation and Classification**, focusing on BERTopic and GPT-3.5/4 for qualitative analysis.
- **Qualitative Analysis** is about understanding non-numerical data, and its challenges include the time-consuming "qualitative coding" process.
- **Topic Modelling** is introduced, with BERTopic as a primary tool.
    - **Pros:** Fast, unsupervised, easy to use.
    - **Cons:** Not very accurate, subjective, not ideal for small datasets.
- Our **experimentation roadmap**:
    - Establish a baseline using BERTopic for classification.
    - Test GPT models on the same classification task.
    - Understand the potential human errors in data labeling.
    - Divide complex tasks into sub-tasks for better GPT performance.
    - Separate tasks of classification and generation.
- **Evaluation Metrics** are crucial:
    - Precision, Recall, F1-score, and Jaccard Similarity are discussed.
    - Time and cost are also considered for real-world deployment.
- We try a **combined approach** using a single prompt for both generation and classification, aided by the "Chain of Thought" prompting technique.
- The **final system**:
    - Generates themes.
    - Merges redundant themes.
    - Classifies responses into these themes.
{{< /collapse >}}


# What is Qualitative Analysis?

Qualitative analysis is a method of analyzing data that is not numerical. It is a method of analysis that is used to understand the meaning of data. Qualitative analysis is used in many different fields, such as psychology, sociology, and anthropology. It is also used in business to understand the meaning of data. We perform qualitative analysis via different methods, such as interviews, focus groups, and surveys. After the collection of data, we need to analyze it to understand the meaning of the data since it is not numerical and extracting meaning is non-trivial. 

This is where we start "qualitative coding" process. Qualitative coding is the process of assigning labels to data. These labels are called "codes" or "themes", and they are used to describe the meaning of the data. The process of qualitative coding is very time consuming and requires a lot of effort. It is also very subjective, since it is done by humans. This is why we want to automate this process as much as we can, and make it more robust, accurate and fast. 

As much research showed recently, LLMs are still not at the point where they can outperform a quality coding done by a human expert. However, we are speculating that, they can be used to speed up the process, and provide a more robust and accurate analysis for experts to start with. This is what we are aiming to do in this research, and we will discuss the results in detail.

# What is Topic Modelling?

Topic modelling is a technique that allows us to extract topics from a corpus of text. It is an unsupervised technique that allows us to discover hidden semantic structures in a text. This is a probabilistic model, that does not provide much accuracy. It is a very simple method in essence. Different methods have differnt approaches to this problem, but one of the most popular ones is BERTopic which uses BERT embeddings to cluster documents. That pretty much is it, even though the library is amazingly implemented and well maintained, the method is very simple. It uses sentence similarity to cluster documents, and analyze word frequency to assign topics and extract keywords.

We can mention why you would want to use topic modelling, and why not. 

**Pros**
- It is an unsupervised method, so you do not need labelled data
- It is very fast, and can be used on large datasets
- It is very easy to use, and does not require much tuning (maybe except for the number of topics)

**Cons**
- It is not very accurate, and can be very subjective (I cannot stress this enough)
- It is not very robust, and can be very sensitive to the number of topics
- Overall for a quality analysis, it does not provide much value
- For smaller datasets, it is not very useful (which is largely the case for surveys etc. in real world)

So if you were interested in a quality analysis, you would not want to use topic modelling. But if you were interested in a quick analysis, and you did not have any labelled data, then topic modelling is a great tool. If you want to read more about topic modelling I strongly suggest that you checkout [BERTopic](https://maartengr.github.io/BERTopic/algorithm/algorithm.html). It is a great library, and the documentation is very well written.

What we are aiming however is seeing the potential of GPT-3.5/4 in topic classification and generation, in a high quality analysis. We hypothesize that GPT models could speed up the process of qualitative analysis, and provide a more robust and accurate analysis for experts to start with.

Along the way, we use topic modelling as a baseline, since we don't have a better choice. One thing to mention here is that we are not making use of existing topic classification models. This is due to the fact that topic classification assumes the new results to be in already classified (labelled) topics. This is not the case for us, since we are trying to discover new topics, and then classify them with no prior knowledge. This begs for few-shot or zero-shot learning, which is what we test with GPT models.

One thing we did not mention, and it is crucial in any part of this process is that topic classification is a multi-label multi-class classification task. Which makes it much harder than any other classification method. We will discuss this in further detail later on when we talk about the evaluation metrics.


# Seperation of Tasks: Classification and Generation

It is a clear statement that GPT (and all the other LLMs) performs better with [divided sub-tasks when it comes to handling complex tasks](https://github.com/openai/openai-cookbook/blob/main/techniques_to_improve_reliability.md). This means that we are expected to get better results if we can divide our end-goal into smaller pieces. For our case, this seemed like we could actually divide our task into classification and generation. This will help us evaluate existing methods, so that we actually can have two seperate baselines to compare with. 

One thing to consider in this seperation is that, these pieces must work well together. So this begs for the question of cohesion. How well do two models do together rather than alone. So in the end of testing the models on their seperate tasks, we will also test them together and see how well they perform for the end goal. 

Another consideration we have is that, these tasks might actually be harmful for the task at hand (at least cost wise), since we are repeating a lot of the information to divide the task. This is why we will also try a combined approach (one prompt) and try to tackle the complexity issues with prompting techniques.

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

# Experiment 2: GPT-3.5/4 for Classification

Second experiment is to see how well GPT-3.5/4 performs on the same classification task, multi-label multi-class classification. We use the same dataset and the same metrics to compare the results. We also compare the time and cost of each method, to see how well they perform in production.

## Human Error

When an Analyst handles the data, there are couple human error that are expected to happen time to time:

1. Analyst might forget to label a response with a class. Especially when there are many classes and it is hard to keep track of them.
2. There might be a coverage expectation from the client, which means that the analyst is going for covering some amount of responses and not all of them. This is usually the case when there are a lot of responses, and the client wants to get a speed up in the process.
3. The naming analyst used might not explicitly indicate their purpose on creating the theme. This leads to misunderstanding of the theme, and might lead to wrong labeling. This is highly avoidable if the analyst notes down a description or a purpose for the theme.

I am mentioning these here, since we are bout to use GPT for the classification task, and these errors in general will lead to wrong labeling. We will see how well GPT performs regardless of these errors here because we are using human generated labels to begin with. 

Later on when we are checking the results for cohesion, we will actually be using GPT generated themes and a human will manually evaluate the results. This will help us see how well GPT performs in a real world scenario. There are some issues with this method but we will discuss them later on.

## Prompting

Prompting is the single most important component when it comes to zero/few-shot learning. If you are not familiar with prompting I highly suggest you go through [Lilian Weng's Blog Post](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/). It is a great resource to understand the techniques and the importance of prompting.

Let's talk about what prompting techniques we will be using for this experiment. I won't be able to provide the exact prompts we have used, since they are company knowledge, but I will mention the general idea.


## Let's Classify, Shall We?

For this step, we feed the existing themes to GPT, and ask it to classify into these bins, and then we compare the results with the true labels. We use the same metrics as before, and we also compare the time and cost of each method.

The parameters that change during the experiment is only the GPT model used (3.5 or 4) and if we do few-shot or zero-shot learning.

Here are the results:

| Model   | Batch Size | Prompt ID | Zero-shot/ Few-shot | Precision | Recall | Jaccard Similarity (Acc) | Price | Time  |
|---------|------------|-----------|---------------------|-----------|--------|--------------------------|-------|-------|
| GPT 3.5 | 1          | 2         | zero-shot           | 0.412     | 0.636  | 0.412                    | 0.664 | 8 min |
| GPT 3.5 | 10         | 1         | zero-shot           | 0.425     | 0.678  | 0.487                    |       |       |
| GPT 3.5 | 25         | 1         | zero-shot           | 0.394     | 0.665  | 0.459                    | 0.096 | 5 min |
| GPT 3.5 | 25         | 3         | few-shot            | 0.425     | 0.574  | 0.459                    | 0.128 | 12 min|
| GPT 3.5 | 1          | 4         | few-shot            | 0.411     | 0.663  | 0.411                    | 0.661 | 21 min|
| GPT 4   | 1          | 2         | zero-shot           | 0.46      | 0.74   | 0.46                     | 6.46  | 24 min|
| GPT 4   | 25         | 1         | zero-shot           | 0.475     | 0.770  | 0.551                    | 0.823 | 11 min|
| GPT 4   | 25         | 3         | few-shot            | 0.506     | 0.663  | 0.561                    | 1.166 | 8.5 min|
| GPT 4   | 1          | 4         | few-shot            | 0.463     | 0.738  | 0.463                    | 6.43  | 18 min|

The prompts here are simply just explaining the task. We do not use any chain of thought or any other prompting technique. We simply ask GPT to classify the response into one of the themes. This is due to the time constraints we were aiming for (since we already have two layers here, we tested out prompts to just assign) and since this is not really a complex task.

We can see that the results clearly indicates that GPT models are outperforming the baseline by a large margin. Though we also see that the results are not near what we wanted. We are thinking some of this is due to the human error we mentioned before. Once we have the complete pipeline (with generation) we will be able to see how well GPT performs in a real world scenario, eliminating the human error.

# Experiment 3: GPT-3.5/4 for Theme Generation

We have a baseline for classification, but we do not have a baseline for generation. This is due to the fact that there is no existing method for theme generation. We can use topic modelling, but that is not really a generation method. When we mention generation, we mean that we want to generate a theme from scratch after reading the responses. We kind of come close to doing this in Topic Modelling, since we group the themes together (clustering) and then assign a name to the cluster. But this is not really generation, since we are not generating a theme from scratch and just assigning a name to a cluster.

Anyhow baseline is irrelevant here (since we can only perform zero/one-shot anyways.) We will just go ahead and test GPT-3.5/4 for generation and see how well it performs. Since we use human evaluator to evaluate the results, we will be able to see how acceptable GPT performs in a real world scenario (this is pretty much the final workflow we will be going through anyways.)

Again for this task we used simple prompting. After going through multiple iterations of prompting we picked the best performing one (from a quick evaluation), and used that for the final results. This is the step we test the "cohesion" between our models, and get a human evaluator to evaluate the results.

# Experiment 4: Cohesion, What did we Achieve?

We now came to the end of the first phase where we can evaluate the results. We run the generation and classification one after the other, report the results and ask a human expert to analyze these results. 

We have evaluated the results of 130 responses, and got to `F-Beta Score` of `0.81`. This is a very good result, and we are very happy with it. We also got a lot of feedback from the evaluator, and we used these feedbacks to improve the prompting. For `Beta` value we used `0.65` as we give more importance to precision.

This evaluation happens in two steps: Analyst first looks through the generated themes, and evaluates how good they are (and how descriptive). Then they look at the classification results in the context of the generated themes, and evaluate how well the classification results fit into the generated themes.

Overall we are happy with the current state of the model. But this process gave us the idea that the seperation might not have been a good idea. 

# Experiment 5: One Prompt to Rule Them All

Next we test out a combined approach, where we use a single prompt for both generation and classification. This will help us see if the seperation is actually helping us or not.

To handle some of the complications and give a clearer direction to GPT, we use a prompting technique called "Chain of Thought". This is a very powerful technique, and it is very easy to implement. We will be using this technique for both generation and classification.

We also gave a quite descriptive expert analyst personality to GPT that directs the model to think like an analyst we would approve of. This is a very important step, since we want to make sure that GPT is not generating themes that are not useful for us.

# Final System: What we found to be the best approach?

After all the experiments, we finally have a system in production. I might have missed some of the details while experimenting, but this took a long time to get to this point and I am a little lazy to fill in so much detail that don't really matter at this point. Especially since I am working on something new now.

I will just go ahead and explain the final system, and what we found to be the best approach. If you had any further questions, feel free to reach out to me. 

We have implemented a three stage system, where we first generate themes, and since we are doing this in parellel compute we then merge the redundant themes. We then classify the responses into these themes. While doing this we are using GPT function calling to reduce the parsing errors in the end. As much as it sounds simple, this whole process is a quite complex system to implement into production. We are using a lot of different techniques to make sure the system is robust and accurate. 

Overall we found this to be the best resulting approach using GPT. We are now focused on iterating and reducing the errors we found in production. As a final goal, we are hoping to train our own proprietary fine-tuned model using our own data. This will help us reduce the cost and increase the accuracy of the system. Stay tuned for the results.

# References
- https://www.clearpeaks.com/using-chatgpt-for-topic-modelling-and-analysis-of-customer-feedback/
- https://medium.com/@stephensonebinezer/transform-your-topic-modeling-with-chatgpt-cutting-edge-nlp-f4654b4eac99
- https://www.width.ai/post/gpt3-topic-extraction
- https://arxiv.org/abs/1908.10084
- https://maartengr.github.io/BERTopic/changelog.html
- https://monkeylearn.com/blog/introduction-to-topic-modeling/#:~:text=Topic%20modeling%20is%20an%20unsupervised,characterize%20a%20set%20of%20documents
- https://arxiv.org/pdf/2203.11171.pdf 
- https://arxiv.org/pdf/2303.07142.pdf 
- https://arxiv.org/pdf/2210.03629.pdf
- https://arxiv.org/abs/2211.01910
- https://arxiv.org/abs/2201.11903
