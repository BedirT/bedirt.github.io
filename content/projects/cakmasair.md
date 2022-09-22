---
title: "Cakma Sair"
date: 2018-03-02
description: "A turkish poem generator using NLP (twitter bot)."
---

[Github Link](https://github.com/AhmetHamzaEmra/CakmaSair)

**Tools Used:** Python, Tflearn, Numpy, Pandas, BeautifulSoup, Tweepy

**Topics:** Machine Learning, Deep Learning, Natural Language Processing, Twitter Bot, Web Scraping, Recurrent Neural Networks

## Introduction

Cakma Sair is a twitter bot that generates turkish poems using NLP. It uses a dataset of 1000+ turkish poems we scraped from multiple turkish poetry websites. We think that poetry is a very important part of turkish culture, and we wanted to create a bot that can generate turkish poems.

We are using Recurrent Neural Networks (RNN) with word2vec to train the model. We then use tweepy to tweet the generated poems based on people's inputs. 

## Tools

- **CakmaSair**: The main tool that generates the poems. It is a python script that uses a trained model to generate poems. It uses tweepy to tweet the generated poems.
- **ScrapePoems**: A tool that scrapes turkish poems from multiple turkish poetry websites. It uses BeautifulSoup to scrape the poems and save them to a txt file.

Here is a sample of the generated poems:

![poem1](https://raw.githubusercontent.com/AhmetHamzaEmra/CakmaSairOrganization/master/Screen%20Shot%202018-03-01%20at%201.40.08%20PM.png?token=ATpOheGbaJ1HdoXIO3LVCmiC2BREXJOgks5aoZLkwA%3D%3D)

Go ahead and tweet at [@CakmaSair](https://twitter.com/Cakma_Sair) to generate a poem!
