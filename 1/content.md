---
layout: post
title: "attention mechanisms: an informal approach"
---

# attention mechanisms: an informal approach

![what the hell does even mean?](../assets/1/image.png)


if you are even vaguely familiar with the term "ChatGPT" chances are that you might have come across a term called **attention**. 

attention is what powers "transformers" - the seemingly complex architecture behind large language models (LLMs) like ChatGPT.

this blog attempts to take you through an informal approach of answering the question, "what the hell does attention even mean?"


# some background first

before going deeper into the concept of attention, let me quickly tell you what the transformer architecture does in short. Don't worry, this will (maybe) feel like a breeze.

so...

a transformer has two main parts: 

- an encoder and 
- a decoder


now, given some input *words* making a sentence/prompt, the encoder is responsible for converting the plain-text input words into **tokens** where each token has a unique id associated with it AND is "represented" by a *high-dimensional vector*.


> wait, high-dimensional vector? why?

this is because, neural networks and hence machines do not understand text as us humans do, so we need to convert our text into something which neural networks understand very well that is... YES! a vector!

these vector "representations" capture a lot of information about the input words such as:

1. the word's semantic information
2. the word's positional information in the sentence
3. the word's "attention" with respect to other words in the sentence (this is what we will discuss in this blog btw)

condensing all this information into a matrix composed of high-dimensional token vectors is what the encoder does.

for example, a token representing the word "cat" will be encoded as a vector in some *n-dimensional space*.

$$
cat \space => [v_1,\space v_2,\space v_3,\space ...,\space v_{n}]
$$


> semantic information? positional information? ahhh, i don't understand

hey don't worry...

here i'm shamelessly skipping the fine-details of how the plain-text words are converted into these vector "embeddings" that capture the semantic (1st point) and positional (2nd point) information, since our focus is mainly on attention today. **Word embeddings** can be a whole topic in itself. But, for now, imagine using magic we convert words into some vector "embeddings".