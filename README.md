# RNN-Attention
Implementation of  RNN with Attention for NLP

# Requirements
tensorflow == 2.11.0

numpy = 1.23.5

# Description
The RNN-Attention is a reccurent neural network architecture that utilizes the strengths autoencoder, teacher forcing and the attention mechanism which is partially used compared to regular transformers. This architecture works with time series analyzes such as in this project where we use it for NLP task such as seq2seq translation. The architecture starts with an embedding layer in the encoder model in order to find token relations between words,  follow by a bidirectional-LSTM which is a recurrent model that takes into consideration both the start and end of a sequence. We then take the hidden state of the encoders bidirectional LSTM to be fed into the input of the hidden state of the LSTM within the decoder, the attention mechanism is used as a teacher forcing technique so that the LSTM is force to see the true previous word token so that it can predict the next word token.

# Dataset
spa.txt (spanish to english translations)

# Tokenizer
Custom

# Architecture
RNN-Attention for Seq2seq

# optimizer
Adam

# loss function
Sparse Categorical Crossentropy

# Text Results:

![RNN_attention1](https://github.com/Santiagor2230/RNN-Attention/assets/52907423/a38886bb-d821-4be4-be0a-bfd6b64a658e)
![attention-rnn3](https://github.com/Santiagor2230/RNN-Attention/assets/52907423/1e1ff1eb-11e7-446b-bee3-c29fb7ec0454)
