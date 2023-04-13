---
author: "Bedir Tapkan"
title: "Tackling Unconventional Data: A Guide to Optimizing Sentiment Analysis Models for Atypical Text"
date: 2023-04-13
description: "Talking about my research and details on Sentiment Analysis"
tags: ["NLP", "Machine Learning", "Sentiment Analysis", "Transfer Learning", "HuggingFace", "Transformers"]
ShowToc: true
---

I recently had the opportunity to develop a sentiment analysis tool for [my company](https://www.avalancheinsights.com/). Although I had some prior experience in this area, I quickly realized that I had more to learn. After extensive research and experimentation, we achieved the desired results. In this post, I'll share my journey, thought process, and the techniques I employed to meet our objectives.

# Identifying the Issue & Setting the Stage

Our startup specializes in delivering top-notch qualitative coding services to businesses, presenting the results on a user-friendly dashboard for our clients. In an effort to better serve their needs, we decided to incorporate sentiment analysis as a key feature.

Sentiment analysis is a popular NLP task that classifies text based on its sentiment. This can be accomplished in various ways, such as categorizing text as positive, negative, or neutral. Alternatively, more nuanced classifications like very positive, positive, neutral, negative, and very negative can be used. Other sentiment analysis tasks, like emotion classification or aspect-based sentiment analysis, focus on different aspects of the text. You can learn more about these tasks [here](https://www.surveymonkey.co.uk/mp/what-customers-really-think-how-sentiment-analysis-can-help/).

Ultimately, we chose the most common sentiment analysis task, which classifies text as positive, negative, or neutral. This approach offers the greatest flexibility in terms of data use and compatibility with existing models.

Having settled on our sentiment analysis task, the next step was to find a pre-trained model to serve as a baseline for comparison. However, we first encountered a challenge: our data was not in the same format as the models or publicly available data. Consequently, we needed labeled data to test the models and determine which one performed best for our specific needs.

# Data Labeling

Our first task was to label our data. Given the sheer volume of data and time constraints, we opted to label a small subset. We employed [Doccano](https://github.com/doccano/doccano), a user-friendly tool designed for effortless data labeling. You can find more details about Doccano on its GitHub page.

With the labeling complete, we had a modest dataset of 200 samples, chosen via stratified sampling, to test our models. While our initial plan was to label 1,000 samples, we reduced it to 200 to save time.

# Pre-trained Models

Armed with our labeled data, we set out to test various models. Our first port of call was [HuggingFace's Transformers](https://huggingface.co/models?pipeline_tag=text-classification), which offers a range of attention-based Transformer models known for their exceptional performance in NLP tasks, including sentiment analysis. Later in this post, I'll discuss some specific base models I used, their distinctions, and my rationale for selecting them.

For our initial testing, I chose several top-ranked models from HuggingFace's Transformers and a base model, 'VADER,' a rule-based sentiment analysis tool. I compared the Transformer models' results with those of the base model. In light of GPT-3.5 and GPT-4's success, I also incorporated a few zero-shot and few-shot models from GPT using the [OpenAI](https://openai.com/) framework.

Here's a list of the models I utilized:
1. [VADER](https://github.com/cjhutto/vaderSentiment)
2. [Huggingface "sbcBI/sentiment_analysis_model"](https://huggingface.co/sbcBI/sentiment_analysis_model)
3. [Huggingface "cardiffnlp/Twitter-xlm-roberta-base-sentiment"](https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base-sentiment)
4. [Huggingface "Seethal/sentiment_analysis_generic_dataset"](https://huggingface.co/Seethal/sentiment_analysis_generic_dataset)
5. [Huggingface "LiYuan/amazon-review-sentiment-analysis"](https://huggingface.co/LiYuan/amazon-review-sentiment-analysis)
6. [Huggingface "ahmedrachid/FinancialBERT-Sentiment-Analysis"](https://huggingface.co/ahmedrachid/FinancialBERT-Sentiment-Analysis)
7. [Huggingface "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"](https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis)
8. [PySentimento](https://github.com/pysentimiento/pysentimiento")
5. GPT-3.5 (zero-shot, few-shot)
6. GPT-4 (zero-shot, few-shot)

Now, let's delve into basic usage examples for each model type and our initial results.

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

To effectively assess our models' performance, we need to employ appropriate evaluation metrics. Common metrics for sentiment analysis include:
1. Accuracy
2. Precision
3. Recall
4. F1 Score

Using a combination of these metrics allows for a more comprehensive understanding of the model's performance, especially when dealing with an imbalanced dataset. For instance, if we have 1,000 samples with 900 positive and 100 negative, we could achieve a high accuracy score by consistently predicting positive outcomes. However, this doesn't necessarily indicate a good model. Therefore, we need to utilize additional metrics to evaluate our model's performance.

The F1 score combines precision and recall, making it an ideal choice for our evaluation. Consequently, we opted to use both F1 score and accuracy as our evaluation metrics.

Below is the function we'll use to calculate accuracy and F1 score.

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

With our models and evaluation metrics in place, we can now test the pre-trained models using the 200 labeled samples. Since no training is involved, we'll use all the data for testing.

These results serve as a sanity check and a general evaluation of how closely our data aligns with the training data used for the models. If our data happens to be highly similar to the training data, we can expect favorable results and stop there. However, if the results are unsatisfactory, we'll need to put in more effort to obtain better results or find a more suitable model.

Below are the accuracy and F1 score plots for all the models:

![Accuracy Plot](img/accuracy_plot.png)

![F1 Score Plot](img/f1_score_plot.png)

As evident from the plots, the VADER model performs the worst, while the GPT-4 model emerges as the best-performing one. GPT-3.5 also delivers relatively strong results. The Hugging Face models, on the other hand, don't perform quite as well. The best open-source model is PySentimento, but its performance still falls short of our desired level.

It's worth noting that our data labeling is complex, making it difficult even for humans. This could introduce some bias in the data, but we won't delve into that in this post since the data itself won't be disclosed.

The GPT-3.5 and GPT-4 models, both zero-shot, show promising performance. We could potentially achieve better results with few-shot training.

Considering the potential of GPT models and the underwhelming performance of the pre-trained sentiment analysis models, we decided to first explore GPT-3.5 and GPT-4 models and then attempt to train our own sentiment analysis model using GPT as the labeler. This approach will provide us with a smaller open-source model for our system, offering performance comparable to GPT models without any associated costs.

# Evaluating GPT-3.5 and GPT-4

We began by testing different prompting methods on the same small dataset to determine the best approach for labeling our sentiment analysis model.

Aside from the prompts, we also tested the general prompting technique. We introduced a parameter called `sample batch size` for this individually dependent task. This parameter controls the number of samples sent to the model at once. It is crucial since sending all samples simultaneously makes it more challenging for the model to generate all labels. However, a benefit of this approach is cost efficiency since the same pre-prompt (or instructions) doesn't need to be repeated for each sample.

While we won't delve into the specifics of the prompts used, we ensured that our instructions to the model were clear. GPT models allow us to explain what we want from the model, so we provided detailed definitions of positive, negative, and neutral sentiments.

The results for different prompting methods are shown below:

![GPT Prompting Results](img/gpt_prompting_results.png)

We included four metrics in the plot:

1. Accuracy: This primary measure of our model's prediction capabilities shows that both GPT-3.5 and GPT-4 perform well with a `sample batch size` of 1. The performance drops significantly with a `sample batch size` of 10.
2. F1 Score: This combination of precision and recall follows the same pattern as accuracy.
3. Price: This cost metric is essential if we plan to use this model in production. For example, the `sample batch size` of 1 is more expensive than the `sample batch size` of 10.
4. Time: This measures the time it takes to generate labels, which is important if we use this model in production.
Both GPT-3.5 and GPT-4 perform well, with the `sample batch size` of 1 outperforming the `sample batch size` of 10. Though GPT-4 performs slightly better, we chose GPT-3.5 due to its lower cost and faster processing time.

To train an open-source model, we'll use GPT-3.5 to generate the majority of the labels (120,000 data points) and GPT-4 for an additional 10,000 data points. This approach will help us assess how closely we can achieve GPT-4 performance with a smaller model.


# Training a Sentiment Analysis Model

Now that we have the labels for our data, we can start training our sentiment analysis model. We will discuss
in order of the following steps:
1. Data Preprocessing and Preparation
2. Model Selection
3. Model Training
4. Model Evaluation

## Data Preprocessing and Preparation

Before training our sentiment analysis model, we need to preprocess and prepare the data. The process involves the following steps:

1. Split the data into train and test sets. We will allocate 80% of the data for training and 20% for validation. The extra 10,000 data points generated with GPT-4 will serve as our test set.
2. Convert the labels to numbers using the following mapping:
    - Positive: 2
    - Neutral: 1
    - Negative: 0
3. Address any incorrect labels generated by GPT-3.5. We can either drop these data points or manually correct them. In our case, we chose to manually fix most of them and leave some outliers untouched. These examples include dots, different phrasings, etc. A few examples are provided below.
4. Check the distribution of labels in the train and test sets. This will give us an idea of how balanced our data is. It's essential to ensure that our dataset is balanced to avoid biases during training and evaluation.

Here's the code for the above steps:

```python
# Correct labels
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

# Check the distribution of labels
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

The distribution of labels in our dataset is somewhat imbalanced. We've observed that negative labels are the most common, while positive labels are the least common. We took this into account when deciding how to evaluate our model. Still, it's a good idea to be aware of the label distribution in the dataset.

Next, we'll create a data loader for Pytorch and prepare the data for training by creating a `Dataset` class.

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

Now it's time to choose a model to train. We'll use the `transformers` library to train our models, building a classifier on top of pre-trained language models such as BERT. In this section, we'll discuss different models we considered, their differences, pros and cons, and then we'll implement, train, and evaluate each model. The models we considered include:

1. BERT
2. RoBERTa
3. DistilBERT
4. XLM-RoBERTa
5. GPT2
6. RoBERTa-Large
7. DistilBERT-Large
8. GPT2-Medium

We have 8 different models. Let's go over each model and explain how they differ.

### BERT

[BERT](https://arxiv.org/abs/1810.04805), introduced by Google in 2018, is a transformer-based model that marked a significant milestone in the field of NLP. It achieved state-of-the-art results on various tasks and can be fine-tuned for specific tasks like sentiment analysis and question-answering. BERT uses bidirectional context, allowing the model to better understand the textual context. However, it's not the most recent model, so it's usually outperformed by newer models.

**Pros:**

- High performance on many NLP tasks.
- Fine-tuning capabilities.
- Bidirectional context for better understanding.

**Cons:**

- Large model size.

### RoBERTa

[RoBERTa](https://arxiv.org/abs/1907.11692), an optimized version of BERT, was introduced by Facebook AI in 2019. It builds upon BERT's architecture but includes several modifications that improve its performance. RoBERTa uses **a larger training dataset**, **longer training time**, and **removes the next-sentence prediction task** during pre-training. It also employs **dynamic masking**, resulting in better performance on downstream tasks.

**Pros:**

- Improved performance compared to BERT.
- Retains benefits of BERT.
- Fine-tuning capabilities.

**Cons:**

- Large model size and high computational requirements.

### DistilBERT

[DistilBERT](https://arxiv.org/abs/1910.01108), a smaller version of BERT, was introduced by Hugging Face in 2019. It aims to maintain most of BERT's performance while reducing its size and computational requirements. DistilBERT has about half the parameters of BERT and is faster during training and inference.

**Pros:**

- Reduced model size and faster training and inference.
- Retains a substantial portion of BERT's performance.
- Fine-tuning capabilities.

**Cons:**

- Slightly lower performance compared to BERT and RoBERTa.

### XLM-RoBERTa

[XLM-RoBERTa](https://arxiv.org/abs/1911.02116) is a multilingual version of RoBERTa, introduced by Facebook AI in 2019. It's pre-trained on a dataset comprising 100 languages and aims to offer improved performance on cross-lingual tasks, such as machine translation and multilingual sentiment analysis.

**Pros:**

- Multilingual model for cross-lingual tasks.
- Retains benefits of RoBERTa.
- Fine-tuning capabilities.

**Cons:**

- Large model size and high computational requirements.

### GPT2

[GPT2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), a transformer-based language model, was introduced by OpenAI in 2019. It is a large, generative model that generates text one token at a time, using a left-to-right autoregressive language modeling (LM) objective. GPT2 is generally better at generating creative text compared to BERT. Since our goal is to imitate GPT-generated output, we'll give it a try.

**Pros:**

- Generates creative text.
- Closer to GPT's which we used to generate our dataset.
- Fine-tuning capabilities.

**Cons:**

- Generally worse at classification tasks compared to BERT and RoBERTa.

## Training

We'll train each model on the training set and evaluate them on the validation set, using the `transformers` library. The library provides a unified API for all the models, making it easy to switch between them.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
```

To train our models, we'll use the `Trainer` class, a high-level API that handles the training loop, evaluation loop, and prediction loop. It also manages data loading, model saving, and model loading. We'll use the `TrainingArguments` class to specify the training arguments, such as the number of epochs, the batch size, and the learning rate.

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
trainer.save_model('./results)
tokenizer.save_pretrained('./results)
```

## Evaluation

So we have finalized the training of our models. After running each model and tweaking the parameters to
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

...
