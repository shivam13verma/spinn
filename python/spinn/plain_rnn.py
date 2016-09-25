"""Theano-based RNN implementations."""

import numpy as np
# import theano

# from theano import tensor as T
from spinn import util

# Chainer imports
# import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

class RNN_Sentence(Chain):
    def __init__(self, model_dim, word_embedding_dim, vocab_size, compose_network,
                 seq_length,
                 num_classes,
                 initial_embeddings,
                 ):
        super(RNN_Sentence, self).__init__(
            embed=L.EmbedID(vocab_size, word_embedding_dim, initialW=initial_embeddings),  # word embedding
            mid=L.LSTM(word_embedding_dim, model_dim),  # the first LSTM layer
            out=L.Linear(model_dim, num_classes),  # the feed-forward output layer
        )
        self.seq_length = seq_length

    def reset_state(self):
        self.mid.reset_state()

    def __call__(self, sent):
        x = self.embed(sent)
        for i in range(self.seq_length):
            self.mid(x[:, i:i+1])
        h = self.mid.h
        y = self.out(h)
        return y

class RNN_Sentence_Pair(Chain):
    def __init__(self, model_dim, word_embedding_dim, vocab_size, compose_network,
                 seq_length,
                 num_classes,
                 initial_embeddings,
                 ):
        super(RNN_Sentence_Pair, self).__init__(
            embed=L.EmbedID(vocab_size, word_embedding_dim, initialW=initial_embeddings),  # word embedding
            prem=compose_network(word_embedding_dim, model_dim),  # the first LSTM layer
            hyp=compose_network(word_embedding_dim, model_dim),  # the first LSTM layer
            out=L.Linear(model_dim * 2, num_classes),  # the feed-forward output layer
        )
        self.seq_length = seq_length

    def reset_state(self):
        self.prem.reset_state()
        self.hyp.reset_state()

    def __call__(self, X):
        prem_sent = X[:, :, 0]
        hyp_sent = X[:, :, 1]

        x_prem = self.embed(prem_sent)
        for i in range(self.seq_length):
            self.prem(x_prem[:, i:i+1])

        x_hyp = self.embed(hyp_sent)
        for i in range(self.seq_length):
            self.hyp(x_hyp[:, i:i+1])

        h = F.concat((self.prem.h, self.hyp.h), axis = 1)
        y = self.out(h)
        return y

class RNN(object):
    """Plain RNN encoder implementation. Can use any activation function.
    """

    def __init__(self, model_dim, word_embedding_dim, vocab_size, compose_network,
                 seq_length,
                 num_classes,
                 initial_embeddings=None,
                 training_mode=True,
                 use_sentence_pair=False,
                 **kwargs):
        """Construct an RNN.

        Args:
            model_dim: Dimensionality of hidden state.
            vocab_size: Number of unique tokens in vocabulary.
            compose_network: Blocks-like function which accepts arguments
              `prev_hidden_state, inp, inp_dim, hidden_dim, vs, name` (see e.g. `util.LSTMLayer`).
            X: Theano batch describing input matrix, or `None` (in which case
              this instance will make its own batch variable).
        """

        self.model_dim = model_dim
        self.word_embedding_dim = word_embedding_dim
        self.vocab_size = vocab_size
        self._compose_network = compose_network
        self.initial_embeddings = initial_embeddings
        self.training_mode = training_mode
        self.seq_length = seq_length

        if use_sentence_pair:
            self.rnn = RNN_Sentence_Pair(model_dim, word_embedding_dim, vocab_size, compose_network,
                seq_length=seq_length,
                num_classes=num_classes,
                initial_embeddings=initial_embeddings,
                )
        else:
            self.rnn = RNN_Sentence(model_dim, word_embedding_dim, vocab_size, compose_network,
                seq_length=seq_length,
                num_classes=num_classes,
                initial_embeddings=initial_embeddings,
                )
        self.model = L.Classifier(self.rnn)
