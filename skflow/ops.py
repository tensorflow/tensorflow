"""General TensorFlow ops."""

import tensorflow as tf


def mean_squared_error_regressor(tensor_in, labels, weights, biases, name=None):
    """Returns prediction and loss for mean squared error regression."""
    with tf.op_scope([tensor_in, labels], name, "mean_squared_error_regressor"):
        predictions = tf.nn.xw_plus_b(tensor_in, weights, biases)
        diff = predictions - labels
        loss = tf.reduce_mean(tf.mul(diff, diff))
        return predictions, loss


def softmax_classifier(tensor_in, labels, weights, biases, name=None):
    """Returns prediction and loss for softmax classifier."""
    with tf.op_scope([tensor_in, labels], name, "softmax_classifier"):
        logits = tf.nn.xw_plus_b(tensor_in, weights, biases)
        xent = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                       labels,
                                                       name="xent_raw")
        loss = tf.reduce_mean(xent, name="xent")
        predictions = tf.nn.softmax(logits, name=name)
        return predictions, loss


