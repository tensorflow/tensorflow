"""Various high level TF models."""
#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import division, print_function, absolute_import

import tensorflow as tf

from skflow.ops import mean_squared_error_regressor, softmax_classifier, dnn


def linear_regression(X, y):
    """Creates linear regression TensorFlow subgraph.

    Args:
        X: tensor or placeholder for input features.
        y: tensor or placeholder for target.

    Returns:
        Predictions and loss tensors.
    """
    with tf.variable_scope('linear_regression'):
        tf.histogram_summary('linear_regression.X', X)
        tf.histogram_summary('linear_regression.y', y)
        y_shape = y.get_shape()
        if len(y_shape) == 1:
            output_shape = 1
        else:
            output_shape = y_shape[1]
        weights = tf.get_variable('weights', [X.get_shape()[1], output_shape])
        bias = tf.get_variable('bias', [output_shape])
        tf.histogram_summary('linear_regression.weights', weights)
        tf.histogram_summary('linear_regression.bias', bias)
        return mean_squared_error_regressor(X, y, weights, bias)


def logistic_regression(X, y, class_weight=None):
    """Creates logistic regression TensorFlow subgraph.

    Args:
        X: tensor or placeholder for input features,
           shape should be [batch_size, n_features].
        y: tensor or placeholder for target,
           shape should be [batch_size, n_classes].
        class_weight: tensor, [n_classes], where for each class
                      it has weight of the class. If not provided
                      will check if graph contains tensor `class_weight:0`.
                      If that is not provided either all ones are used.

    Returns:
        Predictions and loss tensors.
    """
    with tf.variable_scope('logistic_regression'):
        tf.histogram_summary('logistic_regression.X', X)
        tf.histogram_summary('logistic_regression.y', y)
        weights = tf.get_variable('weights', [X.get_shape()[1],
                                              y.get_shape()[-1]])
        bias = tf.get_variable('bias', [y.get_shape()[-1]])
        tf.histogram_summary('logistic_regression.weights', weights)
        tf.histogram_summary('logistic_regression.bias', bias)
        # If no class weight provided, try to retrieve one from pre-defined
        # tensor name in the graph.
        if not class_weight:
            try:
                class_weight = tf.get_default_graph().get_tensor_by_name('class_weight:0')
            except KeyError:
                pass
        return softmax_classifier(X, y, weights, bias,
                                  class_weight=class_weight)


def get_dnn_model(hidden_units, target_predictor_fn):
    """Returns a function that creates a DNN TensorFlow subgraph with given
    params.

    Args:
        hidden_units: List of values of hidden units for layers.
        target_predictor_fn: Function that will predict target from input
                             features. This can be logistic regression,
                             linear regression or any other model,
                             that takes X, y and returns predictions and loss tensors.

    Returns:
        A function that creates the subgraph.
    """
    def dnn_estimator(X, y):
        """DNN estimator with target predictor function on top."""
        layers = dnn(X, hidden_units)
        return target_predictor_fn(layers, y)
    return dnn_estimator

## This will be in Tensorflow 0.7.
## TODO(ilblackdragon): Clean this up when it's released


def _reverse_seq(input_seq, lengths):
    """Reverse a list of Tensors up to specified lengths.
    Args:
        input_seq: Sequence of seq_len tensors of dimension (batch_size, depth)
        lengths:   A tensor of dimension batch_size, containing lengths for each
                   sequence in the batch. If "None" is specified, simply reverses
                   the list.
    Returns:
        time-reversed sequence
    """
    if lengths is None:
        return list(reversed(input_seq))

    for input_ in input_seq:
        input_.set_shape(input_.get_shape().with_rank(2))

    # Join into (time, batch_size, depth)
    s_joined = tf.pack(input_seq)

    # Reverse along dimension 0
    s_reversed = tf.reverse_sequence(s_joined, lengths, 0, 1)
    # Split again into list
    result = tf.unpack(s_reversed)
    return result


def bidirectional_rnn(cell_fw, cell_bw, inputs,
                      initial_state_fw=None, initial_state_bw=None,
                      dtype=None, sequence_length=None, scope=None):
    """Creates a bidirectional recurrent neural network.
    Similar to the unidirectional case (rnn) but takes input and builds
    independent forward and backward RNNs with the final forward and backward
    outputs depth-concatenated, such that the output will have the format
    [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
    forward and backward cell must match. The initial state for both directions
    is zero by default (but can be set optionally) and no intermediate states are
    ever returned -- the network is fully unrolled for the given (passed in)
    length(s) of the sequence(s) or completely unrolled if length(s) is not given.
    Args:
        cell_fw: An instance of RNNCell, to be used for forward direction.
        cell_bw: An instance of RNNCell, to be used for backward direction.
        inputs: A length T list of inputs, each a tensor of shape
          [batch_size, cell.input_size].
        initial_state_fw: (optional) An initial state for the forward RNN.
          This must be a tensor of appropriate type and shape
          [batch_size x cell.state_size].
        initial_state_bw: (optional) Same as for initial_state_fw.
        dtype: (optional) The data type for the initial state.  Required if either
          of the initial states are not provided.
        sequence_length: (optional) An int64 vector (tensor) of size [batch_size],
          containing the actual lengths for each of the sequences.
        scope: VariableScope for the created subgraph; defaults to "BiRNN"
    Returns:
        A pair (outputs, state) where:
          outputs is a length T list of outputs (one for each input), which
            are depth-concatenated forward and backward outputs
          state is the concatenated final state of the forward and backward RNN
    Raises:
        TypeError: If "cell_fw" or "cell_bw" is not an instance of RNNCell.
        ValueError: If inputs is None or an empty list.
    """

    if not isinstance(cell_fw, tf.nn.rnn_cell.RNNCell):
        raise TypeError("cell_fw must be an instance of RNNCell")
    if not isinstance(cell_bw, tf.nn.rnn_cell.RNNCell):
        raise TypeError("cell_bw must be an instance of RNNCell")
    if not isinstance(inputs, list):
        raise TypeError("inputs must be a list")
    if not inputs:
        raise ValueError("inputs must not be empty")

    name = scope or "BiRNN"
    # Forward direction
    with tf.variable_scope(name + "_FW"):
        output_fw, state_fw = tf.nn.rnn(cell_fw, inputs, initial_state_fw, dtype,
                                        sequence_length)

    # Backward direction
    with tf.variable_scope(name + "_BW"):
        tmp, state_bw = tf.nn.rnn(cell_bw, _reverse_seq(inputs, sequence_length),
                                  initial_state_bw, dtype, sequence_length)
    output_bw = _reverse_seq(tmp, sequence_length)
    # Concat each of the forward/backward outputs
    outputs = [tf.concat(1, [fw, bw])
               for fw, bw in zip(output_fw, output_bw)]

    return outputs, tf.concat(1, [state_fw, state_bw])

# End of Tensorflow 0.7


def get_rnn_model(rnn_size, cell_type, num_layers, input_op_fn,
                  bidirectional, target_predictor_fn,
                  sequence_length, initial_state):
    """Returns a function that creates a RNN TensorFlow subgraph with given
    params.

    Args:
        rnn_size: The size for rnn cell, e.g. size of your word embeddings.
        cell_type: The type of rnn cell, including rnn, gru, and lstm.
        num_layers: The number of layers of the rnn model.
        input_op_fn: Function that will transform the input tensor, such as
                     creating word embeddings, byte list, etc. This takes
                     an argument X for input and returns transformed X.
        bidirectional: boolean, Whether this is a bidirectional rnn.
        target_predictor_fn: Function that will predict target from input
                             features. This can be logistic regression,
                             linear regression or any other model,
                             that takes X, y and returns predictions and loss tensors.
        sequence_length: If sequence_length is provided, dynamic calculation is performed.
                         This saves computational time when unrolling past max sequence length.
                         Required for bidirectional RNNs.
        initial_state: An initial state for the RNN. This must be a tensor of appropriate type
                       and shape [batch_size x cell.state_size].

    Returns:
        A function that creates the subgraph.
    """
    def rnn_estimator(X, y):
        """RNN estimator with target predictor function on top."""
        X = input_op_fn(X)
        if cell_type == 'rnn':
            cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif cell_type == 'gru':
            cell_fn = tf.nn.rnn_cell.GRUCell
        elif cell_type == 'lstm':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        else:
            raise ValueError("cell_type {} is not supported. ".format(cell_type))
        if bidirectional:
            # forward direction cell
            rnn_fw_cell = tf.nn.rnn_cell.MultiRNNCell([cell_fn(rnn_size)] * num_layers)
            # backward direction cell
            rnn_bw_cell = tf.nn.rnn_cell.MultiRNNCell([cell_fn(rnn_size)] * num_layers)
            # pylint: disable=unexpected-keyword-arg, no-value-for-parameter
            _, encoding = bidirectional_rnn(rnn_fw_cell, rnn_bw_cell, X,
                                            dtype=tf.float32,
                                            sequence_length=sequence_length,
                                            initial_state_fw=initial_state,
                                            initial_state_bw=initial_state)
        else:
            cell = tf.nn.rnn_cell.MultiRNNCell([cell_fn(rnn_size)] * num_layers)
            _, encoding = tf.nn.rnn(cell, X, dtype=tf.float32,
                                    sequence_length=sequence_length,
                                    initial_state=initial_state)
        return target_predictor_fn(encoding, y)
    return rnn_estimator
