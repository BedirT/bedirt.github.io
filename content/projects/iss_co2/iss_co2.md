---
title: "CO2 clustring detection on ISS"
date: 2018-04-20
description: "Detecting CO2 clusters on ISS using a custom built neural network model with time-series data. Data processing and visualization tools for further evaluation."
---

**Tools Used:** Python, PyTorch, Numpy, Pandas, JavaScript, Scikit-learn, Flask

**Topics:** Machine Learning, Deep Learning, Time-Series, Data Visualization, Data Processing

## Problem

The International Space Station (ISS) is a space station, or a habitable artificial satellite, in low Earth orbit. Given that it is in space, with no gravity, air circulation is very hard to manage. Due to lack of circulation any CO2 that is produced by the astronauts is not dispersed, and accumulates in the same area. This can be a problem, as the CO2 levels can reach a dangerous level, and can cause health issues for the astronauts. Especially when they are sleeping, or doing other activities that require them to be in a specific area for a long time. Lack of oxygen first causes dizziness, and headaches, and if the levels are high enough, it can cause unconsciousness, and even death.

## Solution

Astranouts have a very strict schedule, and they are always doing something based on that schedule. Some of the activities requires heavy physical activity, and some of them are more relaxed. The CO2 levels are higher during the physical activities, and lower during the relaxed activities. We can use the schedule information to see which activities are being performed at any time that the CO2 levels are high. This way we can detect the CO2 clusters, and see which activities are causing them. We can then use this information to make the schedule more efficient, and reduce the CO2 levels.

## Data

The data is collected from the ISS, and it is a time-series data. The data is collected every 5 minutes, and it contains the CO2 levels in localized areas from the sensors placed on the ISS, and for certain times on the astronauts themselves. The sensors that were placed in the room are less effective since the CO2's are packed in small areas, and the sensors are not able to detect them. The sensors that are placed on the astronauts are more effective, since they are able to detect the CO2's that are being produced by the astronauts. 

The data collected proved not to be quite enough and that further collection plans with new systems was needed. (This is a part of this project).

Unfortunately, the data is not public, even I have not seen the full data due to restrictions by NASA.

## Solution

The solution has multiple stages as we have multiple problems to solve. The first stage is to detect the CO2 clusters, and see which activities are causing them. The second stage is to collect more data by providing better tools, and improve the model. The third stage is to use the detected clusters to make the schedule more efficient, and reduce the CO2 levels. 

### Stage 1

The first stage was to detect the CO2 clusters, and see which activities are causing them. We first adjusted the datasets to match each other and be usable in sync, since we had multiple sensors, from different rooms, and multiple astranouts with different schedules it was quite a challange to arrange the dataset properly.

After arranging the dataset we looked for a suitable model to predict the CO2 levels based on activities performed. We tried a few different models, including linear ML models such as **ARIMA**, but ended up using a custom **LSTM** model. After long testing sessions we have concluded that the data was not enough to train a better model, and we needed to collect more data. In the end we achieved near **75%** accuracy on the test set, which is not bad, but not good enough for our purposes. So we decided to move on to the next stage, and leave the further model improvements for after we collect more data.

### Stage 2

The second stage was to collect more data. We decided to build an easy to use offline tool (since internet is not an option) that is easily installable to ISS computers, and can be used to collect data. So we decided to build a web application that can be used to visualize the data, and make it easier to understand for the astranouts. The tool is written in **Python**, and uses the **Flask** framework. For the visualization we used **plotly** and **javascript** and made it easy to use for the astranouts. The tool collects data from the sensors, and the astranouts annotate where needed. The data is then saved in a database, and can be used for further analysis.

We sent the tool to ISS, and they started using it. Since the wait times are long to send anything to ISS, we delivered the data collection tools along with the models for the further studies. Another team will continue the work for the third stage once they have collected enough data.

Here is the [poster we have presented](pdf/nasa_poster_co2_iss.pdf) at the **NASA Wearable Technologies**