---
title: "Introduction to Word Embeddings"
thumbnailImagePosition: left
thumbnailImage: /images/Thumbnail/Word2Vec1.jpeg
metaAlignment: center
coverMeta: out
date: 2024-07-13
categories:
- NLP
tags:
- Embedding
---
**What is Word Embeddings?**
<!--more-->

Word Embedding is one of the key concept in Natural Language Processing. It is the way to transform your textual information into a bunch of numbers that could still represent the context but using bunch of vectors.

In any Machine learning/AI problem, you deal with bunch of vectors and matrices. While numeric data could be used directly as vectors and matrices, the non numeric data need to be converted into a numbers while still retaining the original context to build effective language models. and this is where word embeddings plays major role in processing textual information to train language models

To make things more interesting, Word embedding forms as the basis for the GPT models. Word embedding transforms the input the user keys in the chat and in turn converting the textual information to a numerical vectors and hence getting the responses from the underlying GPT models.

Official Wikipedia definition for the Word Embedding:

"In natural language processing (NLP), a word embedding is a representation of a word. The embedding is used in text analysis. Typically, the representation is a real-valued vector that encodes the meaning of the word in such a way that the words that are closer in the vector space are expected to be similar in meaning.[1] Word embeddings can be obtained using language modeling and feature learning techniques, where words or phrases from the vocabulary are mapped to vectors of real numbers."



I am going to explain some of the key word embeddings techniques in this blog, let's get started! Some of the major word embeddings as follows


# 1. Word2Vec

This simple map depicts the number of fire stations and its location in Dublin.

![Fire.png](/images/Markdown_Images/Kepler/Fire_Station_Final.png)
