# Recurrent computation library

The recurrent computation library contains code to perform recurrent
computations.

Its chief application is to implement recurrent neural networks (RNNs, LSTMs,
etc), which is implemented in `functional_rnn.py`. Similar techniques may be
used to implement deep networks.

The computation saves the activations in the forward pass, and computes the
gradients in the backward pass using a single accumulator.

The `functional_rnn` interface is compatible with the `dynamic_rnn` API.
