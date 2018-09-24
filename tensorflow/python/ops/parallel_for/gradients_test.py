# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for jacobian and batch_jacobian ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.keras.engine import training as keras_training
from tensorflow.python.layers import layers as tf_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops as tf_control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients as gradient_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.ops.parallel_for import gradients
from tensorflow.python.platform import test
from tensorflow.python.util import nest


class FullyConnectedModel(object):

  def __init__(self, activation_size, num_layers):
    self._layers = [
        tf_layers.Dense(activation_size, activation=nn.relu)
        for _ in range(num_layers)
    ]

  def __call__(self, inp):
    activation = inp
    for layer in self._layers:
      activation = layer(activation)
    return activation


def fully_connected_model_fn(batch_size, activation_size, num_layers):
  model = FullyConnectedModel(activation_size, num_layers)
  inp = random_ops.random_normal([batch_size, activation_size])
  return inp, model(inp)


def lstm_model_fn(batch_size, state_size, steps):
  inputs = [
      random_ops.random_normal([batch_size, state_size]) for _ in range(steps)
  ]
  cell = rnn_cell.BasicLSTMCell(state_size)
  init_state = cell.zero_state(batch_size, dtypes.float32)
  state = init_state
  for inp in inputs:
    _, state = cell(inp, state)
  return init_state.c, state.c


def dynamic_lstm_model_fn(batch_size, state_size, max_steps):
  # We make inputs and sequence_length constant so that multiple session.run
  # calls produce the same result.
  inputs = constant_op.constant(
      np.random.rand(batch_size, max_steps, state_size), dtype=dtypes.float32)
  sequence_length = constant_op.constant(
      np.random.randint(0, size=[batch_size], high=max_steps + 1),
      dtype=dtypes.int32)

  cell = rnn_cell.BasicLSTMCell(state_size)
  initial_state = cell.zero_state(batch_size, dtypes.float32)
  return inputs, rnn.dynamic_rnn(
      cell,
      inputs,
      sequence_length=sequence_length,
      initial_state=initial_state)


def create_fc_batch_jacobian(batch_size, activation_size, num_layers):
  inp, output = fully_connected_model_fn(batch_size, activation_size,
                                         num_layers)
  pfor_jacobian = gradients.batch_jacobian(output, inp, use_pfor=True)
  while_jacobian = gradients.batch_jacobian(output, inp, use_pfor=False)
  return pfor_jacobian, while_jacobian


def create_lstm_batch_jacobian(batch_size, state_size, steps):
  inp, output = lstm_model_fn(batch_size, state_size, steps)
  pfor_jacobian = gradients.batch_jacobian(output, inp, use_pfor=True)
  while_jacobian = gradients.batch_jacobian(output, inp, use_pfor=False)
  return pfor_jacobian, while_jacobian


def create_dynamic_lstm_batch_jacobian(batch_size, state_size, max_steps):
  inp, (_, final_state) = dynamic_lstm_model_fn(batch_size, state_size,
                                                max_steps)
  pfor_jacobian = gradients.batch_jacobian(final_state.c, inp, use_pfor=True)
  # Note that use_pfor=False does not work above given the current limitations
  # on implementation of while_loop. So we statically unroll the looping in the
  # jacobian computation.
  while_gradients = [
      gradient_ops.gradients(array_ops.gather(final_state.c, i, axis=1), inp)[0]
      for i in range(state_size)
  ]
  return pfor_jacobian, while_gradients


def create_lstm_batch_hessian(batch_size, state_size, steps):
  inp, output = lstm_model_fn(batch_size, state_size, steps)
  pfor_jacobian = gradients.batch_jacobian(output, inp, use_pfor=True)
  pfor_jacobian = array_ops.reshape(pfor_jacobian, [batch_size, -1])
  pfor_hessian = gradients.batch_jacobian(pfor_jacobian, inp, use_pfor=True)
  # TODO(agarwal): using two nested while_loop doesn't seem to work here.
  # Hence we use pfor_jacobian for computing while_hessian.
  while_jacobian = pfor_jacobian
  while_hessian = gradients.batch_jacobian(while_jacobian, inp, use_pfor=False)
  return pfor_hessian, while_hessian


def create_lstm_hessian(batch_size, state_size, steps):
  _, output = lstm_model_fn(batch_size, state_size, steps)
  weights = variables.trainable_variables()
  pfor_jacobians = gradients.jacobian(output, weights, use_pfor=True)
  pfor_hessians = [
      gradients.jacobian(x, weights, use_pfor=True) for x in pfor_jacobians
  ]
  # TODO(agarwal): using two nested while_loop doesn't seem to work here.
  # Hence we use pfor_jacobians for computing while_hessians.
  while_jacobians = pfor_jacobians
  while_hessians = [
      gradients.jacobian(x, weights, use_pfor=False) for x in while_jacobians
  ]
  return pfor_hessians, while_hessians


def create_fc_per_eg_grad(batch_size, activation_size, num_layers):
  inp = random_ops.random_normal([batch_size, activation_size])
  layers = [
      tf_layers.Dense(activation_size, activation=nn.relu)
      for _ in range(num_layers)
  ]
  projection = tf_layers.Dense(1)

  def model_fn(activation):
    for layer in layers:
      activation = layer(activation)
    activation = projection(activation)
    activation = nn.l2_loss(activation)
    return gradient_ops.gradients(activation, variables.trainable_variables())

  def loop_fn(i):
    return model_fn(array_ops.expand_dims(array_ops.gather(inp, i), 0))

  pfor_outputs = control_flow_ops.pfor(loop_fn, batch_size)
  loop_fn_dtypes = [x.dtype for x in variables.trainable_variables()]
  while_outputs = control_flow_ops.for_loop(loop_fn, loop_fn_dtypes, batch_size)
  return pfor_outputs, while_outputs


def create_lstm_per_eg_grad(batch_size, state_size, steps):
  inputs = [
      random_ops.random_normal([batch_size, state_size]) for _ in range(steps)
  ]
  cell = rnn_cell.BasicLSTMCell(state_size)
  init_state = cell.zero_state(batch_size, dtypes.float32)

  def model_fn(inps, init_state):
    state = init_state
    for inp in inps:
      _, state = cell(inp, state)
    output = nn.l2_loss(state.c)
    return gradient_ops.gradients(output, variables.trainable_variables())

  def loop_fn(i):
    loop_inputs = [
        array_ops.expand_dims(array_ops.gather(x, i), 0) for x in inputs
    ]
    loop_init_state = rnn_cell.LSTMStateTuple(
        *[array_ops.expand_dims(array_ops.gather(x, i), 0) for x in init_state])
    return model_fn(loop_inputs, loop_init_state)

  pfor_outputs = control_flow_ops.pfor(loop_fn, batch_size)
  loop_fn_dtypes = [x.dtype for x in variables.trainable_variables()]
  while_outputs = control_flow_ops.for_loop(loop_fn, loop_fn_dtypes, batch_size)
  return pfor_outputs, while_outputs


# Importing the code from tensorflow_models seems to cause errors. Hence we
# duplicate the model definition here.
# TODO(agarwal): Use the version in tensorflow_models/official instead.
class Mnist(keras_training.Model):

  def __init__(self, data_format):
    """Creates a model for classifying a hand-written digit.

    Args:
      data_format: Either 'channels_first' or 'channels_last'.
    """
    super(Mnist, self).__init__()
    if data_format == "channels_first":
      self._input_shape = [-1, 1, 28, 28]
    else:
      assert data_format == "channels_last"
      self._input_shape = [-1, 28, 28, 1]

    self.conv1 = tf_layers.Conv2D(
        32, 5, padding="same", data_format=data_format, activation=nn.relu)
    self.conv2 = tf_layers.Conv2D(
        64, 5, padding="same", data_format=data_format, activation=nn.relu)
    self.fc1 = tf_layers.Dense(1024, activation=nn.relu)
    self.fc2 = tf_layers.Dense(10)
    self.dropout = tf_layers.Dropout(0.4)
    self.max_pool2d = tf_layers.MaxPooling2D(
        (2, 2), (2, 2), padding="same", data_format=data_format)

  def __call__(self, inputs, training):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, 10].
    """
    y = array_ops.reshape(inputs, self._input_shape)
    y = self.conv1(y)
    y = self.max_pool2d(y)
    y = self.conv2(y)
    y = self.max_pool2d(y)
    y = tf_layers.flatten(y)
    y = self.fc1(y)
    y = self.dropout(y, training=training)
    return self.fc2(y)


def create_mnist_per_eg_grad(batch_size, data_format, training):
  images = random_ops.random_uniform([batch_size, 28, 28])
  sparse_labels = np.random.randint(
      low=0, high=10, size=[batch_size]).astype(np.int32)
  labels = np.zeros((batch_size, 10)).astype(np.float32)
  labels[np.arange(batch_size), sparse_labels] = 1.
  model = Mnist(data_format)

  def loop_fn(i):
    image = array_ops.gather(images, i)
    label = array_ops.gather(labels, i)
    logits = array_ops.reshape(model(image, training=training), [-1])
    loss = losses.softmax_cross_entropy(
        logits=logits, onehot_labels=label, reduction=losses.Reduction.NONE)
    return gradient_ops.gradients(loss, variables.trainable_variables())

  pfor_outputs = control_flow_ops.pfor(loop_fn, batch_size)
  while_outputs = control_flow_ops.for_loop(
      loop_fn, [dtypes.float32] * len(variables.trainable_variables()),
      batch_size)
  return pfor_outputs, while_outputs


def create_mnist_per_eg_jacobian(batch_size, data_format, training):
  images = random_ops.random_uniform([batch_size, 28, 28])
  model = Mnist(data_format)

  def loop_fn(i, use_pfor):
    image = array_ops.gather(images, i)
    logits = array_ops.reshape(model(image, training=training), [-1])
    return gradients.jacobian(
        logits, variables.trainable_variables(), use_pfor=use_pfor)

  pfor_outputs = control_flow_ops.pfor(
      functools.partial(loop_fn, use_pfor=True),
      batch_size)
  while_outputs = control_flow_ops.for_loop(
      functools.partial(loop_fn, use_pfor=False),
      [dtypes.float32] * len(variables.trainable_variables()), batch_size)
  return pfor_outputs, while_outputs


def create_fc_per_eg_jacobians(batch_size, activation_size, num_layers):
  model = FullyConnectedModel(activation_size=activation_size,
                              num_layers=num_layers)
  inp = random_ops.random_normal([batch_size, activation_size])
  output = model(inp)
  jacobians = gradients.jacobian(output, variables.trainable_variables())

  def loop_fn(i, use_pfor):
    inp_i = array_ops.expand_dims(array_ops.gather(inp, i), 0)
    output = array_ops.reshape(model(inp_i), [-1])
    return gradients.jacobian(
        output, variables.trainable_variables(), use_pfor=use_pfor)

  per_eg_jacobians_pfor = control_flow_ops.pfor(
      functools.partial(loop_fn, use_pfor=True),
      batch_size)
  per_eg_jacobians_while = control_flow_ops.for_loop(
      functools.partial(loop_fn, use_pfor=False),
      [dtypes.float32] * len(variables.trainable_variables()), batch_size)
  return jacobians, per_eg_jacobians_pfor, per_eg_jacobians_while


class GradientsTest(test.TestCase):

  def run_and_assert_equal(self, targets1, targets2, atol=1e-4, rtol=1e-4):
    targets1 = nest.flatten(targets1)
    targets2 = nest.flatten(targets2)
    assert len(targets1) == len(targets2)
    init = variables.global_variables_initializer()
    self.evaluate(init)
    outputs = self.evaluate(targets1 + targets2)
    n = len(outputs) // 2
    for i in range(n):
      self.assertAllClose(outputs[i], outputs[i + n], rtol=rtol, atol=atol)

  def test_no_path(self):
    for grad_func in [gradients.jacobian, gradients.batch_jacobian]:
      for use_pfor in [True, False]:
        x = constant_op.constant([[1.0]])
        y = constant_op.constant([[2.0]])
        self.assertIsNone(grad_func(y, x, use_pfor=use_pfor))

  def test_jacobian_fixed_shape(self):
    x = random_ops.random_uniform([2, 2])
    y = math_ops.matmul(x, x, transpose_a=True)
    jacobian_pfor = gradients.jacobian(y, x, use_pfor=True)
    jacobian_while = gradients.jacobian(y, x, use_pfor=False)
    answer = ops.convert_to_tensor([[
        gradient_ops.gradients(y[0][0], x)[0],
        gradient_ops.gradients(y[0][1], x)[0]
    ], [
        gradient_ops.gradients(y[1][0], x)[0],
        gradient_ops.gradients(y[1][1], x)[0]
    ]])
    self.run_and_assert_equal(answer, jacobian_pfor)
    self.run_and_assert_equal(answer, jacobian_while)

  def test_jacobian_scan_shape(self):
    # Shape x: [3, 4]
    x = random_ops.random_uniform([3, 4])
    elems = random_ops.random_uniform([6])
    # Shape y: [6, 3, 4]
    y = functional_ops.scan(lambda a, e: a + e, elems, initializer=x)
    jacobian = gradients.jacobian(y, x)

    expected_shape = [6, 3, 4, 3, 4]
    self.assertAllEqual(expected_shape, jacobian.shape.as_list())

  def test_jacobian_while_loop_shape(self):
    # Shape x: [3, 4]
    x = random_ops.random_uniform([3, 4])
    _, y = tf_control_flow_ops.while_loop(lambda i, a: i > 5.,
                                          lambda i, a: (i + 1, a + i),
                                          (constant_op.constant(0.), x))
    # Shape y: [2, 3]
    y = y[:2, :3]
    jacobian = gradients.jacobian(y, x)

    expected_shape = [2, 3, 3, 4]
    self.assertAllEqual(expected_shape, jacobian.shape.as_list())

  def test_jacobian_unknown_shape(self):
    with self.cached_session() as sess:
      x = array_ops.placeholder(dtypes.float32, shape=[None, None])
      y = math_ops.matmul(x, x, transpose_a=True)
      jacobian_pfor = gradients.jacobian(y, x, use_pfor=True)
      jacobian_while = gradients.jacobian(y, x, use_pfor=False)
      answer = ops.convert_to_tensor([[
          gradient_ops.gradients(y[0][0], x)[0],
          gradient_ops.gradients(y[0][1], x)[0]
      ], [
          gradient_ops.gradients(y[1][0], x)[0],
          gradient_ops.gradients(y[1][1], x)[0]
      ]])
      ans, pfor_value, while_value = sess.run(
          [answer, jacobian_pfor, jacobian_while],
          feed_dict={x: [[1, 2], [3, 4]]})
      self.assertAllClose(ans, pfor_value)
      self.assertAllClose(ans, while_value)

  def test_batch_jacobian_bad_shapes(self):
    x = random_ops.random_uniform([2, 2])
    y = random_ops.random_uniform([3, 2])
    with self.assertRaisesRegexp(ValueError, "Need first dimension of output"):
      gradients.batch_jacobian(y, x, use_pfor=True)

  def test_batch_jacobian_bad_unknown_shapes(self):
    with self.cached_session() as sess:
      x = array_ops.placeholder(dtypes.float32)
      y = array_ops.concat([x, x], axis=0)
      jacobian = gradients.batch_jacobian(y, x)
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "assertion failed"):
        sess.run(jacobian, feed_dict={x: [[1, 2], [3, 4]]})

  def test_batch_jacobian_fixed_shape(self):
    x = random_ops.random_uniform([2, 3, 5])
    y = x * x
    batch_jacobian_pfor = gradients.batch_jacobian(y, x, use_pfor=True)
    batch_jacobian_while = gradients.batch_jacobian(y, x, use_pfor=False)
    two_x = 2 * x
    answer = array_ops.stack(
        [array_ops.diag(two_x[0]),
         array_ops.diag(two_x[1])])
    self.run_and_assert_equal(answer, batch_jacobian_pfor)
    self.run_and_assert_equal(answer, batch_jacobian_while)

  def test_batch_jacobian_unknown_shape(self):
    with self.cached_session() as sess:
      x = array_ops.placeholder(dtypes.float32)
      y = x * x
      batch_jacobian_pfor = gradients.batch_jacobian(y, x, use_pfor=True)
      batch_jacobian_while = gradients.batch_jacobian(y, x, use_pfor=False)
      two_x = 2 * x
      answer = array_ops.stack(
          [array_ops.diag(two_x[0]),
           array_ops.diag(two_x[1])])
      ans, pfor_value, while_value = sess.run(
          [answer, batch_jacobian_pfor, batch_jacobian_while],
          feed_dict={x: [[1, 2], [3, 4]]})
      self.assertAllClose(ans, pfor_value)
      self.assertAllClose(ans, while_value)

  def test_fc_batch_jacobian(self):
    pfor_jacobian, while_jacobian = create_fc_batch_jacobian(8, 4, 2)
    self.run_and_assert_equal(pfor_jacobian, while_jacobian)

  def test_lstm_batch_jacobian(self):
    pfor_jacobian, while_jacobian = create_lstm_batch_jacobian(8, 4, 2)
    self.run_and_assert_equal(pfor_jacobian, while_jacobian)

  def test_dynamic_lstm_batch_jacobian(self):
    pfor_jacobian, while_gradients = create_dynamic_lstm_batch_jacobian(8, 4, 3)
    with session.Session() as sess:
      init = variables.global_variables_initializer()
      sess.run(init)
      pfor = sess.run(pfor_jacobian)
      for i in range(4):
        while_i = sess.run(while_gradients[i])
        self.assertAllClose(while_i, pfor[:, i, ...])

  def test_lstm_hessian(self):
    pfor_hessian, while_hessian = create_lstm_hessian(2, 2, 2)
    self.run_and_assert_equal(pfor_hessian, while_hessian)

  def test_lstm_batch_hessian(self):
    pfor_hessian, while_hessian = create_lstm_batch_hessian(2, 2, 2)
    self.run_and_assert_equal(pfor_hessian, while_hessian)

  def test_fc_per_eg_grad(self):
    pfor_outputs, while_outputs = create_fc_per_eg_grad(8, 4, 2)
    self.run_and_assert_equal(pfor_outputs, while_outputs)

  def test_lstm_per_eg_grad(self):
    pfor_outputs, while_outputs = create_lstm_per_eg_grad(8, 4, 2)
    self.run_and_assert_equal(pfor_outputs, while_outputs)

  def test_mnist_per_eg_grad(self):
    # It looks like CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
    # configuration of Winograd can cause low precision output resulting in
    # tests failing. So we disable that here.
    os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "0"
    data_format = ("channels_first"
                   if test.is_gpu_available() else "channels_last")
    # Note that we we are setting training=False here so that dropout produces
    # the same result with pfor and with while_loop.
    pfor_outputs, while_outputs = create_mnist_per_eg_grad(
        4, data_format, training=False)
    self.run_and_assert_equal(pfor_outputs, while_outputs, rtol=1e-3)
    os.environ.pop("TF_ENABLE_WINOGRAD_NONFUSED", None)

  def test_mnist_per_eg_jacobian(self):
    # It looks like CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
    # configuration of Winograd can cause low precision output resulting in
    # tests failing. So we disable that here.
    os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "0"
    data_format = ("channels_first"
                   if test.is_gpu_available() else "channels_last")
    # Note that we we are setting training=False here so that dropout produces
    # the same result with pfor and with while_loop.
    pfor_outputs, while_outputs = create_mnist_per_eg_jacobian(
        2, data_format, training=False)
    self.run_and_assert_equal(pfor_outputs, while_outputs, rtol=1e-3)
    os.environ.pop("TF_ENABLE_WINOGRAD_NONFUSED", None)

  def test_fc_jacobian(self):
    jacobians, per_eg_jacobians_pfor, per_eg_jacobians_while = (
        create_fc_per_eg_jacobians(batch_size=8,
                                   activation_size=4,
                                   num_layers=2))
    self.run_and_assert_equal(jacobians, per_eg_jacobians_pfor,
                              rtol=2e-3, atol=1e-3)
    self.run_and_assert_equal(jacobians, per_eg_jacobians_while,
                              rtol=2e-3, atol=1e-3)


class GradientsBenchmarks(test.Benchmark):

  def _run(self, targets, iters, name=None):

    def _done(t):
      # Note that we don't use tf.control_dependencies since that will not make
      # sure that the computation on GPU has actually finished. So we fetch the
      # first element of the output, and assume that this will not be called on
      # empty tensors.
      return array_ops.gather(array_ops.reshape(t, [-1]), 0)

    targets = [_done(x) for x in nest.flatten(targets)]
    sess = session.Session()
    with sess:
      init = variables.global_variables_initializer()
      sess.run(init)
      sess.run(targets)
      begin = time.time()
      for _ in range(iters):
        sess.run(targets)
      end = time.time()
    avg_time_ms = 1000 * (end - begin) / iters
    self.report_benchmark(iters=iters, wall_time=avg_time_ms, name=name)
    return avg_time_ms

  def benchmark_fc_batch_jacobian(self):
    with ops.Graph().as_default():
      pfor_jacobian, while_jacobian = create_fc_batch_jacobian(100, 32, 20)
      self._run(pfor_jacobian, 100, name="fc_batch_jacobian_pfor")
      self._run(while_jacobian, 20, name="fc_batch_jacobian_while")

  def benchmark_lstm_batch_jacobian(self):
    with ops.Graph().as_default():
      pfor_jacobian, while_jacobian = create_lstm_batch_jacobian(100, 32, 8)
      self._run(pfor_jacobian, 100, name="lstm_batch_jacobian_pfor")
      self._run(while_jacobian, 20, name="lstm_batch_jacobian_while")

  def benchmark_lstm_hessian(self):
    with ops.Graph().as_default():
      pfor_hessian, while_hessian = create_lstm_hessian(2, 2, 10)
      self._run(pfor_hessian, 20, name="lstm_hessian_pfor")
      self._run(while_hessian, 3, name="lstm_hessian_while_pfor")

  def benchmark_lstm_batch_hessian(self):
    with ops.Graph().as_default():
      pfor_hessian, while_hessian = create_lstm_batch_hessian(4, 4, 10)
      self._run(pfor_hessian, 100, name="lstm_batch_hessian_pfor")
      self._run(while_hessian, 20, name="lstm_batch_hessian_while_pfor")

  def benchmark_fc_per_eg_grad(self):
    with ops.Graph().as_default():
      pfor_outputs, while_outputs = create_fc_per_eg_grad(100, 32, 3)
      self._run(pfor_outputs, 100, name="fc_per_eg_grad_pfor")
      self._run(while_outputs, 20, name="fc_per_eg_grad_while")

  def benchmark_lstm_per_eg_grad(self):
    with ops.Graph().as_default():
      pfor_outputs, while_outputs = create_lstm_per_eg_grad(100, 32, 8)
      self._run(pfor_outputs, 100, name="lstm_per_eg_grad_pfor")
      self._run(while_outputs, 20, name="lstm_per_eg_grad_while")

  def benchmark_mnist_per_eg_grad(self):
    with ops.Graph().as_default():
      data_format = ("channels_first"
                     if test.is_gpu_available() else "channels_last")
      pfor_outputs, while_outputs = create_mnist_per_eg_grad(
          128, data_format, training=True)
      self._run(pfor_outputs, 20, name="mnist_per_eg_grad_pfor")
      self._run(while_outputs, 20, name="mnist_per_eg_grad_while")

  def benchmark_mnist_per_eg_jacobian(self):
    with ops.Graph().as_default():
      data_format = ("channels_first"
                     if test.is_gpu_available() else "channels_last")
      pfor_outputs, while_outputs = create_mnist_per_eg_jacobian(
          16, data_format, training=True)
      self._run(pfor_outputs, 20, name="mnist_per_eg_jacobian_pfor")
      self._run(while_outputs, 20, name="mnist_per_eg_jacobian_while")

  def benchmark_fc_per_eg_jacobian(self):
    with ops.Graph().as_default():
      jacobians, per_eg_jacobians_pfor, per_eg_jacobians_while = (
          create_fc_per_eg_jacobians(batch_size=128,
                                     activation_size=32,
                                     num_layers=3))
      self._run(jacobians, 30, name="fc_jacobians_pfor")
      self._run(per_eg_jacobians_pfor, 100,
                name="fc_per_eg_jacobians_pfor")
      self._run(per_eg_jacobians_while, 10,
                name="fc_per_eg_jacobians_while")


if __name__ == "__main__":
  test.main()
