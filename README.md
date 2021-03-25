# LegalSearcher
## Problem Statement:
A client requires the construction of an application that works as a search engine on the political constitution of Colombia (available online) in which, through this, the ordinary citizen can consult articles related to a compound word or by tags, that is, if the client wanted to search for "human rights" or "rights" the filter could be precise and show information from the text that is related to its tags.
For now, he would like a team of developers to advise him on the most appropriate form of medium for use, be it a responsive web application or a mobile app, this decision will be at the discretion of the proposal that will be presented in the first deliverable to the central team.

## Basic Text Pre-Processing
* Pandas
* Numpy
* re

Before jumping to the exploration stage, we need to perform basic data pre-processing steps like null value imputation and removal of unwanted data. So, let’s start by importing libraries and reading our dataset. The dataset contains 439 rows and 3 columns. But these columns are in dictionary format and we need title, chapter and article information. Therefore we are transforming our Dataset in order to extract relevant data.

![alt text](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/3f5fefb3-35f8-4b2d-9735-8b690fbea323/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210325%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210325T015235Z&X-Amz-Expires=86400&X-Amz-Signature=74810cf97ec1d3481ed1cc8eaf046a78f4f1e735db959eaf698093584718712d&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D"Untitled.png") 

## Exploratory Data Analysis
We will start by looking at the common words present in the articles for each title. For this, we will use the document term matrix created earlier with word clouds for plotting these words. Word clouds are the visual representations of the frequency of different words present in a document. It gives importance to the more frequent words which are bigger in size compared to other less frequent words.

![alt_text](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/5c3c8069-48af-4e08-9cfa-d1836c536c79/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20210325%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20210325T031608Z&X-Amz-Expires=86400&X-Amz-Signature=eca056eb2fbaa6e764e98891747fceb5edccc223141de15edd28e0e43fdd111d&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D"Untitled.png")

### Preparing Text Data for Exploratory Data Analysis (EDA)
A Document Term Matrix provides the frequency of a word in a corpus (collection of documents), which in this case are articles from the Political Constitution. It helps in analyzing the occurrence of words in different documents in a corpus. The following figure is an example of a document term matrix

