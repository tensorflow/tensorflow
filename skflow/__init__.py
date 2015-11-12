import random

import numpy as np
import tensorflow as tf

OPTIMIZER_CLS_NAMES = {
    "SGD": tf.train.GradientDescentOptimizer,
    "Adagrad": tf.train.AdagradOptimizer,
    "Adam": tf.train.AdamOptimizer,
}


def mean_squared_error_regressor(tensor_in, labels, weights, biases, name=None):
  with tf.op_scope([tensor_in, labels], name, "mean_squared_error_regressor"):
    predictions = tf.nn.xw_plus_b(tensor_in, weights, biases)
    diff = predictions - labels
    loss = tf.reduce_mean(tf.mul(diff, diff))
    return predictions, loss


def softmax_classifier(tensor_in, labels, weights, biases, name=None):
  with tf.op_scope([tensor_in, labels], name, "softmax_classifier"):
    logits = tf.nn.xw_plus_b(tensor_in, weights, biases)
    xent = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                   labels,
                                                   name="xent_raw")
    loss = tf.reduce_mean(xent, name="xent")
    predictions = tf.nn.softmax(logits, name=name)
    return predictions, loss


class TFModel(object):

  def __init__(self, n_classes, graph, input_shape):
    with graph.as_default():
      self.inp = tf.placeholder(tf.float32, [None, input_shape], name="input")
      if n_classes > 1:
        self.out = tf.placeholder(tf.float32, [None, n_classes], name="output")
      else:
        self.out = tf.placeholder(tf.float32, [None], name="output")
      self.global_step = tf.Variable(0, name="global_step", trainable=False)
      out_dim = n_classes if n_classes > 1 else 1
      self.weights = tf.get_variable("weights", [input_shape, out_dim])
      self.bias = tf.get_variable("bias", [out_dim])
      if n_classes > 1:
        self.predictions, self.loss = softmax_classifier(
            self.inp, self.out, self.weights, self.bias)
      else:
        self.predictions, self.loss = mean_squared_error_regressor(
            self.inp, self.out, self.weights, self.bias)


class TFTrainer(object):

  def __init__(self, model, trainer, learning_rate, clip_gradients=5.0):
    """Build a trainer part of graph."""
    self.model = model
    self.learning_rate = tf.get_variable(
        "learning_rate",
        [],
        initializer=tf.constant_initializer(learning_rate))
    params = tf.trainable_variables()
    self.gradients = tf.gradients(model.loss, params)
    if clip_gradients > 0.0:
      self.gradients, self.gradients_norm = tf.clip_by_global_norm(
          self.gradients, clip_gradients)
    grads_and_vars = zip(self.gradients, params)
    if isinstance(trainer, str):
      optimizer = OPTIMIZER_CLS_NAMES[trainer](self.learning_rate)
    else:
      optimizer = trainer(self.learning_rate)
    self.trainer = optimizer.apply_gradients(grads_and_vars,
                                             global_step=model.global_step,
                                             name="train")
    # Get all initializers for all trainable variables.
    self.initializers = tf.initialize_all_variables()

  def initialize(self, sess):
    """Initalizes all variables."""
    return sess.run(self.initializers)

  def train(self, sess, feed_dict_fn, steps, print_steps=50):
    """Trains a model for given number of steps, given feed_dict function."""
    for step in xrange(steps):
      feed_dict = feed_dict_fn()
      global_step, loss, _, p = sess.run(
          [self.model.global_step, self.model.loss, self.trainer, self.model.predictions],
          feed_dict=feed_dict)
      if step % print_steps == 0:
        print "Step #%d, loss: %.5f" % (global_step, loss)


class TensorFlowBase(object):

  def __init__(self,
               n_classes,
               tf_master="",
               batch_size=32,
               steps=50,
               trainer="SGD",
               learning_rate=0.1):
    self.n_classes = n_classes
    self.tf_master = tf_master
    self.batch_size = batch_size
    self.steps = steps
    self.trainer = trainer
    self.learning_rate = learning_rate

  def _get_feed_dict_fn(self, model, X, y):

    def _feed_dict():
      inp = np.zeros([self.batch_size, X.shape[1]])
      if self.n_classes > 1:
        out = np.zeros([self.batch_size, self.n_classes])
      else:
        out = np.zeros([self.batch_size])
      for i in xrange(self.batch_size):
        sample = random.randint(0, X.shape[0] - 1)
        inp[i, :] = X[sample, :]
        if self.n_classes > 1:
          out[i, y[sample]] = 1.0
        else:
          out[i] = y[sample]
      return {model.inp.name: inp, model.out.name: out,}

    return _feed_dict

  def fit(self, X, y):
    with tf.Graph().as_default() as graph:
      self._model = TFModel(self.n_classes, graph, X.shape[1])
      self._trainer = TFTrainer(self._model, self.trainer, self.learning_rate)
      self._session = tf.Session(self.tf_master)
      self._trainer.initialize(self._session)
      self._trainer.train(self._session,
                          self._get_feed_dict_fn(self._model, X, y), self.steps)

  def predict(self, X):
    pred = self._session.run(self._model.predictions,
                             feed_dict={
                                 self._model.inp.name: X
                             })
    if self.n_classes < 2:
      return pred
    return pred.argmax(axis=1)


class TensorFlowRegressor(TensorFlowBase):
  pass


class TensorFlowClassifier(TensorFlowBase):
  pass

