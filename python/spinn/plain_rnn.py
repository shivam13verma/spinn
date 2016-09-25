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

DEBUG = True

if DEBUG:
    import ipdb

class RNN_Chainer(Chain):
    def __init__(self, model_dim, word_embedding_dim, vocab_size, compose_network,
                 initial_embeddings, num_classes=3):
        super(RNN_Chainer, self).__init__(
            embed=L.EmbedID(vocab_size, word_embedding_dim, initialW=initial_embeddings),  # word embedding
            mid=compose_network(word_embedding_dim, model_dim),  # the first LSTM layer
            out=L.Linear(model_dim, num_classes),  # the feed-forward output layer
        )

    def reset_state(self):
        self.mid.reset_state()

    def __call__(self, sent):
        seq_length = sent.shape[1]
        x = self.embed(sent)
        for i in range(seq_length):
            self.mid(x[:, i:i+1])
        h = self.mid.h
        y = self.out(h)
        return y

class RNN(object):
    """Plain RNN encoder implementation. Can use any activation function.
    """

    def __init__(self, model_dim, word_embedding_dim, vocab_size, compose_network,
                 X=None,
                 initial_embeddings=None,
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

        self.rnn = RNN_Chainer(model_dim, word_embedding_dim, vocab_size, compose_network, initial_embeddings)
        self.model = L.Classifier(self.rnn)
        self.optimizer = optimizers.SGD()
        self.optimizer.setup(self.model)

    def _compute_loss(self, X, Y):
        loss = self.model(X, Y)
        return loss

    def step(self, X, Y):
        self.rnn.reset_state()
        self.model.cleargrads()
        self.loss = self._compute_loss(X, Y)
        self.loss.backward()
        self.optimizer.update()
