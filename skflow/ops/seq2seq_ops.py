"""TensorFlow Ops for Sequence to Sequence models."""
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

from skflow.ops import array_ops


def sequence_classifier(decoding, labels, name=None):
    """Returns predictions and loss for sequence of predictions.

    Args:
        decoding: List of Tensors with predictions.
        labels: List of Tensors with labels.

    Returns:
        Predictions and losses tensors.
    """
    with tf.op_scope([decoding, labels], name, "sequence_classifier"):
        predictions, xent_list = [], []
        for i, pred in enumerate(decoding):
            xent_list.append(
                tf.nn.softmax_cross_entropy_with_logits(
                    pred, labels[i], name="sequence_loss/xent_raw{0}".format(i)))
            predictions.append(tf.nn.softmax(pred))
        xent = tf.add_n(xent_list, name="sequence_loss/xent")
        loss = tf.reduce_mean(xent, name="sequence_loss")
        return predictions, loss


def seq2seq_inputs(X, y, input_length, output_length, sentinel=None, name=None):
    """Processes inputs for Sequence to Sequence models.

    Args:
        X: Input Tensor [batch_size, input_length, embed_dim].
        y: Output Tensor [batch_size, output_length, embed_dim].
        input_length: length of input X.
        output_length: length of output y.
        sentinel: optional first input to decoder and final output expected.
                  if sentinel is not provided, zeros are used.

    Returns:
        Encoder input from X, and decoder inputs and outputs from y.
    """
    with tf.op_scope([X, y], name, "seq2seq_inputs"):
        in_X = array_ops.split_squeeze(1, input_length, X)
        y = array_ops.split_squeeze(1, output_length, y)
        if not sentinel:
            # Set to zeros of shape of y[0]
            sentinel = tf.zeros(tf.shape(y[0]))
            sentinel.set_shape(y[0].get_shape())
        in_y = [sentinel] + y
        out_y = y + [sentinel]
        return in_X, in_y, out_y
     
