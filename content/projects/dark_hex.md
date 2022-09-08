---
title: "Dark Hex"
date: 2022-08-20
description: "An open-source library for training and evaluating agents in Dark Hex, a large-scale imperfect information game."
---

[Github Link](https://github.com/BedirT/darkhex)

**Tools Used:** Python, C++, Tensorflow, Open-Spiel, Pandas, Numpy, Matplotlib, PyGame, PyDot, Tkinter

**Topics:** Reinforcement Learning, Game Theory, Imperfect Information Games, Deep Learning, Machine Learning

## Introduction

Dark Hex is an imperfect information version of the [Hex](https://en.wikipedia.org/wiki/Hex_(board_game)). Dark Hex is a phantom game, where a player have a chance to play concecutively. Due to this property Dark Hex is an extremely huge game. Which makes it really hard to train agents, and develop algorithms for. 

In this project I implemented certain tools to help generate new strategies, and come up with better players eventually. The project includes my thesis work, along side all the results and experiments I have done. We have the best known players implemented (that we developed) as well as the methods we used to train them.

We base the tools on DeepMind's Open-Spiel library, most of the game specific functions are used from there. I am yet to finish documentation for the project, but I will be adding it soon.

We heavily rely on [MCCFR](http://mlanctot.info/files/papers/nips09mccfr.pdf) and [NFSP](https://www.davidsilver.uk/wp-content/uploads/2020/03/nfsp-1.pdf) algorithms for the training phase.

## Tools

- **MCCFR** implementation for training agents in Dark Hex.
- **PolGen**: A UI-based tool to generate complete policies for the agents. When we are creating new strategies/policies, we need to make sure we covered all the cases possible. PolGen helps us do that, and makes it even easier with extra features we added.
- **Tree Generator**: A tool to generate the game tree for two players. This is a crucial part of evaluating and understanding the agent behaviour. We can use this tool to generate the game tree for any given state, and see the probabilities of the agent choosing a certain action.
- **SimPly/SimPly+**: The algorithms we used to train our agents. The details of the algorithm can be found in the thesis.
- More tools are on the previous version. I am still cleaning up and organizing the code, and will be adding them soon.