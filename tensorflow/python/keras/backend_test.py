# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras backend."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import scipy.sparse

from tensorflow.core.protobuf import config_pb2
from tensorflow.python import keras
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


def compare_single_input_op_to_numpy(keras_op,
                                     np_op,
                                     input_shape,
                                     dtype='float32',
                                     negative_values=True,
                                     keras_args=None,
                                     keras_kwargs=None,
                                     np_args=None,
                                     np_kwargs=None):
  keras_args = keras_args or []
  keras_kwargs = keras_kwargs or {}
  np_args = np_args or []
  np_kwargs = np_kwargs or {}
  inputs = 2. * np.random.random(input_shape)
  if negative_values:
    inputs -= 1.
  keras_output = keras_op(keras.backend.variable(inputs, dtype=dtype),
                          *keras_args, **keras_kwargs)
  keras_output = keras.backend.eval(keras_output)
  np_output = np_op(inputs.astype(dtype), *np_args, **np_kwargs)
  try:
    np.testing.assert_allclose(keras_output, np_output, atol=1e-4)
  except AssertionError:
    raise AssertionError('Test for op `' + str(keras_op.__name__) + '` failed; '
                         'Expected ' + str(np_output) + ' but got ' +
                         str(keras_output))


def compare_two_inputs_op_to_numpy(keras_op,
                                   np_op,
                                   input_shape_a,
                                   input_shape_b,
                                   dtype='float32',
                                   keras_args=None,
                                   keras_kwargs=None,
                                   np_args=None,
                                   np_kwargs=None):
  keras_args = keras_args or []
  keras_kwargs = keras_kwargs or {}
  np_args = np_args or []
  np_kwargs = np_kwargs or {}
  input_a = np.random.random(input_shape_a)
  input_b = np.random.random(input_shape_b)
  keras_output = keras_op(keras.backend.variable(input_a, dtype=dtype),
                          keras.backend.variable(input_b, dtype=dtype),
                          *keras_args, **keras_kwargs)
  keras_output = keras.backend.eval(keras_output)
  np_output = np_op(input_a.astype(dtype), input_b.astype(dtype),
                    *np_args, **np_kwargs)
  try:
    np.testing.assert_allclose(keras_output, np_output, atol=1e-4)
  except AssertionError:
    raise AssertionError('Test for op `' + str(keras_op.__name__) + '` failed; '
                         'Expected ' + str(np_output) + ' but got ' +
                         str(keras_output))


class BackendResetTest(test.TestCase, parameterized.TestCase):

  @test_util.run_all_in_graph_and_eager_modes
  def test_new_config(self):
    # User defined jit setting
    config.set_optimizer_jit(False)
    sess = keras.backend.get_session()
    default_config = context.context().config
    self.assertEqual(
        sess._config.graph_options.optimizer_options.global_jit_level,
        default_config.graph_options.optimizer_options.global_jit_level)
    keras.backend.clear_session()

    # New session has the same jit setting
    sess = keras.backend.get_session()
    default_config = context.context().config
    self.assertEqual(
        sess._config.graph_options.optimizer_options.global_jit_level,
        default_config.graph_options.optimizer_options.global_jit_level)
    keras.backend.clear_session()

    # Change respected
    config.set_optimizer_jit(True)
    sess = keras.backend.get_session()
    default_config = context.context().config
    self.assertEqual(
        sess._config.graph_options.optimizer_options.global_jit_level,
        default_config.graph_options.optimizer_options.global_jit_level)
    keras.backend.clear_session()

  # We can't use the normal parameterized decorator because the test session
  # will block graph clearing.
  @parameterized.named_parameters(('_v1', context.graph_mode),
                                  ('_v2', context.eager_mode))
  def test_new_graph(self, test_context):
    with test_context():
      g_old = keras.backend.get_graph()
      keras.backend.clear_session()
      g = keras.backend.get_graph()

      assert g_old is not g


@test_util.run_all_in_graph_and_eager_modes
class BackendUtilsTest(test.TestCase):

  def test_backend(self):
    self.assertEqual(keras.backend.backend(), 'tensorflow')

  def test_get_reset_uids(self):
    self.assertEqual(keras.backend.get_uid('foo'), 1)
    self.assertEqual(keras.backend.get_uid('foo'), 2)

    keras.backend.reset_uids()
    self.assertEqual(keras.backend.get_uid('foo'), 1)

  def test_learning_phase(self):
    with self.cached_session() as sess:
      with self.assertRaises(ValueError):
        keras.backend.set_learning_phase(2)

      # Test running with a learning-phase-consuming layer
      with keras.backend.learning_phase_scope(0):
        x = keras.Input((3,))
        y = keras.layers.BatchNormalization()(x)
        if not context.executing_eagerly():
          self.evaluate(variables.global_variables_initializer())
          sess.run(y, feed_dict={x: np.random.random((2, 3))})

  def test_learning_phase_name(self):
    with ops.name_scope('test_scope'):
      # Test that outer name scopes do not affect the learning phase's name.
      lp = keras.backend.symbolic_learning_phase()
    self.assertEqual(lp.name, 'keras_learning_phase:0')

  def test_learning_phase_scope(self):
    initial_learning_phase = keras.backend.learning_phase()
    with keras.backend.learning_phase_scope(1):
      self.assertEqual(keras.backend.learning_phase(), 1)
    self.assertEqual(keras.backend.learning_phase(), initial_learning_phase)
    with keras.backend.learning_phase_scope(0):
      self.assertEqual(keras.backend.learning_phase(), 0)
    self.assertEqual(keras.backend.learning_phase(), initial_learning_phase)
    with self.assertRaises(ValueError):
      with keras.backend.learning_phase_scope(None):
        pass
    self.assertEqual(keras.backend.learning_phase(), initial_learning_phase)

    new_learning_phase = 0
    keras.backend.set_learning_phase(new_learning_phase)
    self.assertEqual(keras.backend.learning_phase(), new_learning_phase)
    with keras.backend.learning_phase_scope(1):
      self.assertEqual(keras.backend.learning_phase(), 1)
    self.assertEqual(keras.backend.learning_phase(), new_learning_phase)

  def test_learning_phase_scope_in_graph(self):
    initial_learning_phase_outside_graph = keras.backend.learning_phase()
    with keras.backend.get_graph().as_default():
      initial_learning_phase_in_graph = keras.backend.learning_phase()

    self.assertEqual(keras.backend.learning_phase(),
                     initial_learning_phase_outside_graph)
    with keras.backend.learning_phase_scope(1):
      self.assertEqual(keras.backend.learning_phase(), 1)
    self.assertEqual(keras.backend.learning_phase(),
                     initial_learning_phase_outside_graph)

    with keras.backend.get_graph().as_default():
      self.assertIs(keras.backend.learning_phase(),
                    initial_learning_phase_in_graph)

    self.assertEqual(keras.backend.learning_phase(),
                     initial_learning_phase_outside_graph)

  def test_int_shape(self):
    x = keras.backend.ones(shape=(3, 4))
    self.assertEqual(keras.backend.int_shape(x), (3, 4))

    if not context.executing_eagerly():
      x = keras.backend.placeholder(shape=(None, 4))
      self.assertEqual(keras.backend.int_shape(x), (None, 4))

  def test_in_train_phase(self):
    y1 = keras.backend.variable(1)
    y2 = keras.backend.variable(2)
    if context.executing_eagerly():
      with keras.backend.learning_phase_scope(0):
        y_val_test = keras.backend.in_train_phase(y1, y2).numpy()
      with keras.backend.learning_phase_scope(1):
        y_val_train = keras.backend.in_train_phase(y1, y2).numpy()
    else:
      y = keras.backend.in_train_phase(y1, y2)
      f = keras.backend.function([keras.backend.learning_phase()], [y])
      y_val_test = f([0])[0]
      y_val_train = f([1])[0]
    self.assertAllClose(y_val_test, 2)
    self.assertAllClose(y_val_train, 1)

  def test_is_keras_tensor(self):
    x = keras.backend.variable(1)
    self.assertEqual(keras.backend.is_keras_tensor(x), False)
    x = keras.Input(shape=(1,))
    self.assertEqual(keras.backend.is_keras_tensor(x), True)
    with self.assertRaises(ValueError):
      keras.backend.is_keras_tensor(0)

  def test_stop_gradient(self):
    x = keras.backend.variable(1)
    y = keras.backend.stop_gradient(x)
    if not context.executing_eagerly():
      self.assertEqual(y.op.name[:12], 'StopGradient')

    xs = [keras.backend.variable(1) for _ in range(3)]
    ys = keras.backend.stop_gradient(xs)
    if not context.executing_eagerly():
      for y in ys:
        self.assertEqual(y.op.name[:12], 'StopGradient')

  def test_placeholder(self):
    x = keras.backend.placeholder(shape=(3, 4))
    self.assertEqual(x.shape.as_list(), [3, 4])
    x = keras.backend.placeholder(shape=(3, 4), sparse=True)
    self.assertEqual(x.shape.as_list(), [3, 4])

  def test_is_placeholder(self):
    x = keras.backend.placeholder(shape=(1,))
    self.assertEqual(keras.backend.is_placeholder(x), True)
    x = keras.backend.variable(1)
    self.assertEqual(keras.backend.is_placeholder(x), False)

  def test_print_tensor(self):
    # Unfortunately it seems impossible to use `mock` (or any other method)
    # to capture stdout when used inside a graph or graph function, thus
    # we cannot test correctness.
    # The message gets correctly printed in practice.
    x = keras.backend.placeholder(shape=())
    y = keras.backend.print_tensor(x, 'eager=%s' % context.executing_eagerly())
    f = keras.backend.function(x, y)
    f(0)

  def test_cast_to_floatx(self):
    x = keras.backend.variable(1, dtype='float64')
    x = keras.backend.cast_to_floatx(x)
    self.assertEqual(x.dtype.name, 'float32')
    x = keras.backend.cast_to_floatx(2)
    self.assertEqual(x.dtype.name, 'float32')


@test_util.run_all_in_graph_and_eager_modes
class BackendVariableTest(test.TestCase):

  def test_zeros(self):
    x = keras.backend.zeros((3, 4))
    val = keras.backend.eval(x)
    self.assertAllClose(val, np.zeros((3, 4)))

  def test_ones(self):
    x = keras.backend.ones((3, 4))
    val = keras.backend.eval(x)
    self.assertAllClose(val, np.ones((3, 4)))

  def test_eye(self):
    x = keras.backend.eye(4)
    val = keras.backend.eval(x)
    self.assertAllClose(val, np.eye(4))

  def test_zeros_like(self):
    x = keras.backend.zeros((3, 4))
    y = keras.backend.zeros_like(x)
    val = keras.backend.eval(y)
    self.assertAllClose(val, np.zeros((3, 4)))

  def test_ones_like(self):
    x = keras.backend.zeros((3, 4))
    y = keras.backend.ones_like(x)
    val = keras.backend.eval(y)
    self.assertAllClose(val, np.ones((3, 4)))

  def test_random_uniform_variable(self):
    x = keras.backend.random_uniform_variable((30, 20), low=1, high=2, seed=0)
    val = keras.backend.eval(x)
    self.assertAllClose(val.mean(), 1.5, atol=1e-1)
    self.assertAllClose(val.max(), 2., atol=1e-1)
    self.assertAllClose(val.min(), 1., atol=1e-1)

  def test_random_normal_variable(self):
    x = keras.backend.random_normal_variable((30, 20), 1., 0.5, seed=0)
    val = keras.backend.eval(x)
    self.assertAllClose(val.mean(), 1., atol=1e-1)
    self.assertAllClose(val.std(), 0.5, atol=1e-1)

  def test_count_params(self):
    x = keras.backend.zeros((4, 5))
    val = keras.backend.count_params(x)
    self.assertAllClose(val, 20)

  def test_constant(self):
    ref_val = np.random.random((3, 4)).astype('float32')
    x = keras.backend.constant(ref_val)
    val = keras.backend.eval(x)
    self.assertAllClose(val, ref_val)

  def test_sparse_variable(self):
    val = scipy.sparse.eye(10)
    x = keras.backend.variable(val)
    self.assertTrue(isinstance(x, sparse_tensor.SparseTensor))

    y = keras.backend.to_dense(x)
    self.assertFalse(keras.backend.is_sparse(y))


@test_util.run_all_in_graph_and_eager_modes
class BackendLinearAlgebraTest(test.TestCase, parameterized.TestCase):

  def test_dot(self):
    x = keras.backend.ones(shape=(2, 3))
    y = keras.backend.ones(shape=(3, 4))
    xy = keras.backend.dot(x, y)
    self.assertEqual(xy.shape.as_list(), [2, 4])

    x = keras.backend.ones(shape=(32, 28, 3))
    y = keras.backend.ones(shape=(3, 4))
    xy = keras.backend.dot(x, y)
    self.assertEqual(xy.shape.as_list(), [32, 28, 4])

  @parameterized.parameters(
      [(2, 3, 4, 5), (2, 5, 6, 7), (2, 3, 4, 6, 7), (3, 1)],
      [(2, 20, 1), (2, 30, 20), (2, 1, 30), (1, 2)],
      [(4, 2, 3), (4, 5, 3), (4, 2, 5), (2, 2)],
      [(4, 2), (4, 2, 3), (4, 3), (1, 1)],
      [(4, 2), (4, 2, 3), (4, 3), 1],
      [(4, 2, 3), (4, 3), (4, 2), (2, 1)],
  )
  def test_batch_dot(self, x_shape, y_shape, output_shape, axes):
    x_val = np.random.random(x_shape)
    y_val = np.random.random(y_shape)
    x = keras.backend.variable(x_val)
    y = keras.backend.variable(y_val)
    xy = keras.backend.batch_dot(x, y, axes=axes)
    self.assertEqual(tuple(xy.shape.as_list()), output_shape)
    xy_val = keras.backend.eval(xy)
    ref_val = self._reference_batch_dot(x_val, y_val, axes)
    self.assertAllClose(xy_val, ref_val, atol=1e-5)

  def _reference_batch_dot(self, x, y, axes):
    if isinstance(axes, int):
      axes = [axes, axes]
    elif isinstance(axes, tuple):
      axes = list(axes)
    if axes is None:
      if y.ndim == 2:
        axes = [x.ndim - 1, y.ndim - 1]
      else:
        axes = [x.ndim - 1, y.ndim - 2]
    if axes[0] < 0:
      axes[0] += x.ndim
    if axes[1] < 0:
      axes[1] += y.ndim
    result = []
    axes = [axes[0] - 1, axes[1] - 1]
    for xi, yi in zip(x, y):
      result.append(np.tensordot(xi, yi, axes))
    result = np.array(result)
    if result.ndim == 1:
      result = np.expand_dims(result, -1)
    return result

  def test_reduction_ops(self):
    ops_to_test = [
        (keras.backend.max, np.max),
        (keras.backend.min, np.min),
        (keras.backend.sum, np.sum),
        (keras.backend.prod, np.prod),
        (keras.backend.var, np.var),
        (keras.backend.std, np.std),
        (keras.backend.mean, np.mean),
        (keras.backend.argmin, np.argmin),
        (keras.backend.argmax, np.argmax),
    ]
    for keras_op, np_op in ops_to_test:
      compare_single_input_op_to_numpy(keras_op, np_op, input_shape=(4, 7, 5),
                                       keras_kwargs={'axis': 1},
                                       np_kwargs={'axis': 1})
      compare_single_input_op_to_numpy(keras_op, np_op, input_shape=(4, 7, 5),
                                       keras_kwargs={'axis': -1},
                                       np_kwargs={'axis': -1})
      if 'keepdims' in tf_inspect.getargspec(keras_op).args:
        compare_single_input_op_to_numpy(keras_op, np_op,
                                         input_shape=(4, 7, 5),
                                         keras_kwargs={'axis': 1,
                                                       'keepdims': True},
                                         np_kwargs={'axis': 1,
                                                    'keepdims': True})

  def test_elementwise_ops(self):
    ops_to_test = [
        (keras.backend.square, np.square),
        (keras.backend.abs, np.abs),
        (keras.backend.round, np.round),
        (keras.backend.sign, np.sign),
        (keras.backend.sin, np.sin),
        (keras.backend.cos, np.cos),
        (keras.backend.exp, np.exp),
    ]
    for keras_op, np_op in ops_to_test:
      compare_single_input_op_to_numpy(keras_op, np_op, input_shape=(4, 7))

    ops_to_test = [
        (keras.backend.sqrt, np.sqrt),
        (keras.backend.log, np.log),
    ]
    for keras_op, np_op in ops_to_test:
      compare_single_input_op_to_numpy(keras_op, np_op,
                                       input_shape=(4, 7),
                                       negative_values=False)

    compare_single_input_op_to_numpy(
        keras.backend.clip, np.clip,
        input_shape=(6, 4),
        keras_kwargs={'min_value': 0.1, 'max_value': 2.4},
        np_kwargs={'a_min': 0.1, 'a_max': 1.4})

    compare_single_input_op_to_numpy(
        keras.backend.pow, np.power,
        input_shape=(6, 4),
        keras_args=[3],
        np_args=[3])

  def test_two_tensor_ops(self):
    ops_to_test = [
        (keras.backend.equal, np.equal),
        (keras.backend.not_equal, np.not_equal),
        (keras.backend.greater, np.greater),
        (keras.backend.greater_equal, np.greater_equal),
        (keras.backend.less, np.less),
        (keras.backend.less_equal, np.less_equal),
        (keras.backend.maximum, np.maximum),
        (keras.backend.minimum, np.minimum),
    ]
    for keras_op, np_op in ops_to_test:
      compare_two_inputs_op_to_numpy(keras_op, np_op,
                                     input_shape_a=(4, 7),
                                     input_shape_b=(4, 7))

  def test_relu(self):
    x = ops.convert_to_tensor([[-4, 0], [2, 7]], 'float32')

    # standard relu
    relu_op = keras.backend.relu(x)
    self.assertAllClose(keras.backend.eval(relu_op), [[0, 0], [2, 7]])

    # alpha (leaky relu used)
    relu_op = keras.backend.relu(x, alpha=0.5)
    if not context.executing_eagerly():
      self.assertTrue('LeakyRelu' in relu_op.name)
    self.assertAllClose(keras.backend.eval(relu_op), [[-2, 0], [2, 7]])

    # max_value < some elements
    relu_op = keras.backend.relu(x, max_value=5)
    self.assertAllClose(keras.backend.eval(relu_op), [[0, 0], [2, 5]])

    # nn.relu6 used
    relu_op = keras.backend.relu(x, max_value=6)
    if not context.executing_eagerly():
      self.assertTrue('Relu6' in relu_op.name)  # uses tf.nn.relu6
    self.assertAllClose(keras.backend.eval(relu_op), [[0, 0], [2, 6]])

    # max value > 6
    relu_op = keras.backend.relu(x, max_value=10)
    self.assertAllClose(keras.backend.eval(relu_op), [[0, 0], [2, 7]])

    # max value is float
    relu_op = keras.backend.relu(x, max_value=4.3)
    self.assertAllClose(keras.backend.eval(relu_op), [[0, 0], [2, 4.3]])

    # max value == 0
    relu_op = keras.backend.relu(x, max_value=0)
    self.assertAllClose(keras.backend.eval(relu_op), [[0, 0], [0, 0]])

    # alpha and max_value
    relu_op = keras.backend.relu(x, alpha=0.25, max_value=3)
    self.assertAllClose(keras.backend.eval(relu_op), [[-1, 0], [2, 3]])

    # threshold
    relu_op = keras.backend.relu(x, threshold=3)
    self.assertAllClose(keras.backend.eval(relu_op), [[0, 0], [0, 7]])

    # threshold is float
    relu_op = keras.backend.relu(x, threshold=1.5)
    self.assertAllClose(keras.backend.eval(relu_op), [[0, 0], [2, 7]])

    # threshold is negative
    relu_op = keras.backend.relu(x, threshold=-5)
    self.assertAllClose(keras.backend.eval(relu_op), [[-4, 0], [2, 7]])

    # threshold and max_value
    relu_op = keras.backend.relu(x, threshold=3, max_value=5)
    self.assertAllClose(keras.backend.eval(relu_op), [[0, 0], [0, 5]])

    # threshold and alpha
    relu_op = keras.backend.relu(x, alpha=0.25, threshold=4)
    self.assertAllClose(keras.backend.eval(relu_op), [[-2, -1], [-0.5, 7]])

    # threshold, alpha, and max_value
    relu_op = keras.backend.relu(x, alpha=0.25, threshold=4, max_value=5)
    self.assertAllClose(keras.backend.eval(relu_op), [[-2, -1], [-0.5, 5]])


@test_util.run_all_in_graph_and_eager_modes
class BackendShapeOpsTest(test.TestCase):

  def test_reshape(self):
    compare_single_input_op_to_numpy(keras.backend.reshape, np.reshape,
                                     input_shape=(4, 7),
                                     keras_args=[(2, 14)],
                                     np_args=[(2, 14)])

  def test_concatenate(self):
    a = keras.backend.variable(np.ones((1, 2, 3)))
    b = keras.backend.variable(np.ones((1, 2, 2)))
    y = keras.backend.concatenate([a, b], axis=-1)
    self.assertEqual(y.shape.as_list(), [1, 2, 5])

  def test_permute_dimensions(self):
    compare_single_input_op_to_numpy(keras.backend.permute_dimensions,
                                     np.transpose,
                                     input_shape=(4, 7),
                                     keras_args=[(1, 0)],
                                     np_args=[(1, 0)])

  def test_resize_images(self):
    height_factor = 2
    width_factor = 2
    data_format = 'channels_last'
    x = keras.backend.variable(np.ones((1, 2, 2, 3)))
    y = keras.backend.resize_images(x,
                                    height_factor,
                                    width_factor,
                                    data_format)
    self.assertEqual(y.shape.as_list(), [1, 4, 4, 3])

    data_format = 'channels_first'
    x = keras.backend.variable(np.ones((1, 3, 2, 2)))
    y = keras.backend.resize_images(x,
                                    height_factor,
                                    width_factor,
                                    data_format)
    self.assertEqual(y.shape.as_list(), [1, 3, 4, 4])

    # Invalid use:
    with self.assertRaises(ValueError):
      keras.backend.resize_images(x,
                                  height_factor,
                                  width_factor,
                                  data_format='unknown')

  def test_resize_volumes(self):
    height_factor = 2
    width_factor = 2
    depth_factor = 2
    data_format = 'channels_last'
    x = keras.backend.variable(np.ones((1, 2, 2, 2, 3)))
    y = keras.backend.resize_volumes(x,
                                     depth_factor,
                                     height_factor,
                                     width_factor,
                                     data_format)
    self.assertEqual(y.shape.as_list(), [1, 4, 4, 4, 3])

    data_format = 'channels_first'
    x = keras.backend.variable(np.ones((1, 3, 2, 2, 2)))
    y = keras.backend.resize_volumes(x,
                                     depth_factor,
                                     height_factor,
                                     width_factor,
                                     data_format)
    self.assertEqual(y.shape.as_list(), [1, 3, 4, 4, 4])

    # Invalid use:
    with self.assertRaises(ValueError):
      keras.backend.resize_volumes(x,
                                   depth_factor,
                                   height_factor,
                                   width_factor,
                                   data_format='unknown')

  def test_repeat_elements(self):
    x = keras.backend.variable(np.ones((1, 3, 2)))
    y = keras.backend.repeat_elements(x, 3, axis=1)
    self.assertEqual(y.shape.as_list(), [1, 9, 2])

    # Use with a dynamic axis:
    if not context.executing_eagerly():
      x = keras.backend.placeholder(shape=(2, None, 2))
      y = keras.backend.repeat_elements(x, 3, axis=1)
      self.assertEqual(y.shape.as_list(), [2, None, 2])

  def test_repeat(self):
    x = keras.backend.variable(np.ones((1, 3)))
    y = keras.backend.repeat(x, 2)
    self.assertEqual(y.shape.as_list(), [1, 2, 3])

  def test_flatten(self):
    compare_single_input_op_to_numpy(keras.backend.flatten,
                                     np.reshape,
                                     input_shape=(4, 7, 6),
                                     np_args=[(4 * 7 * 6,)])

  def test_batch_flatten(self):
    compare_single_input_op_to_numpy(keras.backend.batch_flatten,
                                     np.reshape,
                                     input_shape=(4, 7, 6),
                                     np_args=[(4, 7 * 6)])

  def test_temporal_padding(self):

    def ref_op(x, padding):
      shape = list(x.shape)
      shape[1] += padding[0] + padding[1]
      y = np.zeros(tuple(shape))
      y[:, padding[0]:-padding[1], :] = x
      return y

    compare_single_input_op_to_numpy(keras.backend.temporal_padding,
                                     ref_op,
                                     input_shape=(4, 7, 6),
                                     keras_args=[(2, 3)],
                                     np_args=[(2, 3)])

  def test_spatial_2d_padding(self):

    def ref_op(x, padding, data_format='channels_last'):
      shape = list(x.shape)
      if data_format == 'channels_last':
        shape[1] += padding[0][0] + padding[0][1]
        shape[2] += padding[1][0] + padding[1][1]
        y = np.zeros(tuple(shape))
        y[:, padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1], :] = x
      else:
        shape[2] += padding[0][0] + padding[0][1]
        shape[3] += padding[1][0] + padding[1][1]
        y = np.zeros(tuple(shape))
        y[:, :, padding[0][0]:-padding[0][1], padding[1][0]:-padding[1][1]] = x
      return y

    compare_single_input_op_to_numpy(
        keras.backend.spatial_2d_padding,
        ref_op,
        input_shape=(2, 3, 2, 3),
        keras_args=[((2, 3), (1, 2))],
        keras_kwargs={'data_format': 'channels_last'},
        np_args=[((2, 3), (1, 2))],
        np_kwargs={'data_format': 'channels_last'})
    compare_single_input_op_to_numpy(
        keras.backend.spatial_2d_padding,
        ref_op,
        input_shape=(2, 3, 2, 3),
        keras_args=[((2, 3), (1, 2))],
        keras_kwargs={'data_format': 'channels_first'},
        np_args=[((2, 3), (1, 2))],
        np_kwargs={'data_format': 'channels_first'})

  def test_spatial_3d_padding(self):

    def ref_op(x, padding, data_format='channels_last'):
      shape = list(x.shape)
      if data_format == 'channels_last':
        shape[1] += padding[0][0] + padding[0][1]
        shape[2] += padding[1][0] + padding[1][1]
        shape[3] += padding[2][0] + padding[2][1]
        y = np.zeros(tuple(shape))
        y[:,
          padding[0][0]:-padding[0][1],
          padding[1][0]:-padding[1][1],
          padding[2][0]:-padding[2][1],
          :] = x
      else:
        shape[2] += padding[0][0] + padding[0][1]
        shape[3] += padding[1][0] + padding[1][1]
        shape[4] += padding[2][0] + padding[2][1]
        y = np.zeros(tuple(shape))
        y[:, :,
          padding[0][0]:-padding[0][1],
          padding[1][0]:-padding[1][1],
          padding[2][0]:-padding[2][1]] = x
      return y

    compare_single_input_op_to_numpy(
        keras.backend.spatial_3d_padding,
        ref_op,
        input_shape=(2, 3, 2, 3, 2),
        keras_args=[((2, 3), (1, 2), (2, 3))],
        keras_kwargs={'data_format': 'channels_last'},
        np_args=[((2, 3), (1, 2), (2, 3))],
        np_kwargs={'data_format': 'channels_last'})
    compare_single_input_op_to_numpy(
        keras.backend.spatial_3d_padding,
        ref_op,
        input_shape=(2, 3, 2, 3, 2),
        keras_args=[((2, 3), (1, 2), (2, 3))],
        keras_kwargs={'data_format': 'channels_first'},
        np_args=[((2, 3), (1, 2), (2, 3))],
        np_kwargs={'data_format': 'channels_first'})


@test_util.run_all_in_graph_and_eager_modes
class BackendNNOpsTest(test.TestCase, parameterized.TestCase):

  def test_bias_add(self):
    keras_op = keras.backend.bias_add
    np_op = np.add
    compare_two_inputs_op_to_numpy(keras_op, np_op,
                                   input_shape_a=(4, 7),
                                   input_shape_b=(7,))
    compare_two_inputs_op_to_numpy(keras_op, np_op,
                                   input_shape_a=(4, 3, 7),
                                   input_shape_b=(7,))
    compare_two_inputs_op_to_numpy(keras_op, np_op,
                                   input_shape_a=(4, 3, 5, 7),
                                   input_shape_b=(7,))
    compare_two_inputs_op_to_numpy(keras_op, np_op,
                                   input_shape_a=(4, 3, 5, 2, 7),
                                   input_shape_b=(7,))

    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      x = keras.backend.variable((3, 4))
      b = keras.backend.variable((3, 4))
      keras.backend.bias_add(x, b)
    with self.assertRaises(ValueError):
      x = keras.backend.variable((3, 4))
      b = keras.backend.variable((4,))
      keras.backend.bias_add(x, b, data_format='unknown')

  def test_bias_add_channels_first(self):

    def keras_op(x, b):
      return keras.backend.bias_add(x, b, data_format='channels_first')

    def np_op(x, b):
      if x.ndim == 3:
        b = b.reshape((1, b.shape[0], 1))
      if x.ndim == 4:
        b = b.reshape((1, b.shape[0], 1, 1))
      return x + b

    compare_two_inputs_op_to_numpy(keras_op, np_op,
                                   input_shape_a=(4, 3, 7),
                                   input_shape_b=(3,))
    compare_two_inputs_op_to_numpy(keras_op, np_op,
                                   input_shape_a=(4, 3, 5, 7),
                                   input_shape_b=(3,))

  def test_pool2d(self):
    val = np.random.random((10, 3, 10, 10))
    x = keras.backend.variable(val)
    y = keras.backend.pool2d(x, (2, 2), strides=(1, 1),
                             padding='valid', data_format='channels_first',
                             pool_mode='max')
    self.assertEqual(y.shape.as_list(), [10, 3, 9, 9])

    y = keras.backend.pool2d(x, (2, 2), strides=(1, 1),
                             padding='valid', data_format='channels_first',
                             pool_mode='avg')
    self.assertEqual(y.shape.as_list(), [10, 3, 9, 9])

    val = np.random.random((10, 10, 10, 3))
    x = keras.backend.variable(val)
    y = keras.backend.pool2d(x, (2, 2), strides=(1, 1),
                             padding='valid', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 9, 9, 3])

    val = np.random.random((10, 10, 10, 3))
    x = keras.backend.variable(val)
    y = keras.backend.pool2d(x, (2, 2), strides=(1, 1),
                             padding='same', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 10, 10, 3])

    val = np.random.random((10, 10, 10, 3))
    x = keras.backend.variable(val)
    y = keras.backend.pool2d(x, (2, 2), strides=(2, 2),
                             padding='same', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 5, 5, 3])

    with self.assertRaises(ValueError):
      y = keras.backend.pool2d(x, (2, 2), strides=(2, 2),
                               padding='other', data_format='channels_last')
    with self.assertRaises(ValueError):
      y = keras.backend.pool2d(x, (2, 2), strides=(2, 2),
                               data_format='other')
    with self.assertRaises(ValueError):
      y = keras.backend.pool2d(x, (2, 2, 2), strides=(2, 2))
    with self.assertRaises(ValueError):
      y = keras.backend.pool2d(x, (2, 2), strides=(2, 2, 2))
    with self.assertRaises(ValueError):
      y = keras.backend.pool2d(x, (2, 2), strides=(2, 2), pool_mode='other')

  def test_pool3d(self):
    if test.is_built_with_rocm():
      self.skipTest('Pooling with 3D tensors is not supported in ROCm')
    val = np.random.random((10, 3, 10, 10, 10))
    x = keras.backend.variable(val)
    y = keras.backend.pool3d(x, (2, 2, 2), strides=(1, 1, 1),
                             padding='valid', data_format='channels_first',
                             pool_mode='max')
    self.assertEqual(y.shape.as_list(), [10, 3, 9, 9, 9])

    y = keras.backend.pool3d(x, (2, 2, 2), strides=(1, 1, 1),
                             padding='valid', data_format='channels_first',
                             pool_mode='avg')
    self.assertEqual(y.shape.as_list(), [10, 3, 9, 9, 9])

    val = np.random.random((10, 10, 10, 10, 3))
    x = keras.backend.variable(val)
    y = keras.backend.pool3d(x, (2, 2, 2), strides=(1, 1, 1),
                             padding='valid', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 9, 9, 9, 3])

    val = np.random.random((10, 10, 10, 10, 3))
    x = keras.backend.variable(val)
    y = keras.backend.pool3d(x, (2, 2, 2), strides=(1, 1, 1),
                             padding='same', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 10, 10, 10, 3])

    val = np.random.random((10, 10, 10, 10, 3))
    x = keras.backend.variable(val)
    y = keras.backend.pool3d(x, (2, 2, 2), strides=(2, 2, 2),
                             padding='same', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 5, 5, 5, 3])

  def test_conv1d(self):
    val = np.random.random((10, 4, 10))
    x = keras.backend.variable(val)
    kernel_val = np.random.random((3, 4, 5))
    k = keras.backend.variable(kernel_val)
    y = keras.backend.conv1d(x, k, strides=(1,),
                             padding='valid', data_format='channels_first')
    self.assertEqual(y.shape.as_list(), [10, 5, 8])

    val = np.random.random((10, 10, 4))
    x = keras.backend.variable(val)
    y = keras.backend.conv1d(x, k, strides=(1,),
                             padding='valid', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 8, 5])

    val = np.random.random((10, 10, 4))
    x = keras.backend.variable(val)
    y = keras.backend.conv1d(x, k, strides=(1,),
                             padding='same', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 10, 5])

    val = np.random.random((10, 10, 4))
    x = keras.backend.variable(val)
    y = keras.backend.conv1d(x, k, strides=(2,),
                             padding='same', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 5, 5])

  def test_local_conv_channels_dim(self):
    filters = 3
    batch_size = 2

    for input_shape in [(3, 5), (2, 3, 5), (2, 5, 3, 4)]:
      channels_in = input_shape[0]
      input_spatial_shape = input_shape[1:]
      dim = len(input_spatial_shape)

      inputs = np.random.normal(0, 1, (batch_size,) + input_shape)
      inputs_cf = keras.backend.variable(inputs)

      for kernel_size in [1, 2]:
        for stride in [1, 2]:
          kernel_sizes = (kernel_size,) * dim
          strides = (stride,) * dim

          output_shape = tuple([(i - kernel_size + stride) // stride
                                for i in input_spatial_shape])

          kernel_shape = (np.prod(output_shape),
                          np.prod(kernel_sizes) * channels_in,
                          filters)

          kernel = np.random.normal(
              0,
              1,
              output_shape + (channels_in, np.prod(kernel_sizes), filters)
          )

          kernel_cf = np.reshape(kernel, kernel_shape)
          kernel_cf = keras.backend.variable(kernel_cf)

          conv_cf = keras.backend.local_conv(inputs_cf,
                                             kernel_cf,
                                             kernel_sizes,
                                             strides,
                                             output_shape,
                                             'channels_first')

          inputs_cl = np.transpose(inputs, [0, 2] + list(range(3, dim + 2)) +
                                   [1])
          inputs_cl = keras.backend.variable(inputs_cl)

          kernel_cl = np.reshape(
              np.transpose(kernel, list(range(dim)) + [dim + 1, dim, dim + 2]),
              kernel_shape
          )
          kernel_cl = keras.backend.variable(kernel_cl)

          conv_cl = keras.backend.local_conv(inputs_cl,
                                             kernel_cl,
                                             kernel_sizes,
                                             strides,
                                             output_shape,
                                             'channels_last')

          conv_cf = keras.backend.eval(conv_cf)
          conv_cl = keras.backend.eval(conv_cl)

          self.assertAllCloseAccordingToType(
              conv_cf,
              np.transpose(conv_cl,
                           [0, dim + 1] + list(range(1, dim + 1))),
              atol=1e-5
          )

  @parameterized.named_parameters(
      ('local_conv1d', (5, 6), (3,), (1,), (3,)),
      ('local_conv2d', (4, 5, 6), (3, 3), (1, 1), (2, 3)))
  def test_local_conv_1d_and_2d(self,
                                input_shape,
                                kernel_sizes,
                                strides,
                                output_shape):
    filters = 3
    batch_size = 2

    inputs = np.random.normal(0, 1, (batch_size,) + input_shape)
    inputs = keras.backend.variable(inputs)

    kernel = np.random.normal(0, 1, (np.prod(output_shape),
                                     np.prod(kernel_sizes) * input_shape[-1],
                                     filters))
    kernel = keras.backend.variable(kernel)

    local_conv = keras.backend.local_conv(inputs,
                                          kernel,
                                          kernel_sizes,
                                          strides,
                                          output_shape,
                                          'channels_last')
    if len(output_shape) == 1:
      local_conv_dim = keras.backend.local_conv1d(inputs,
                                                  kernel,
                                                  kernel_sizes,
                                                  strides,
                                                  'channels_last')
    else:
      local_conv_dim = keras.backend.local_conv2d(inputs,
                                                  kernel,
                                                  kernel_sizes,
                                                  strides,
                                                  output_shape,
                                                  'channels_last')

    local_conv = keras.backend.eval(local_conv)
    local_conv_dim = keras.backend.eval(local_conv_dim)

    self.assertAllCloseAccordingToType(local_conv, local_conv_dim)

  def test_conv2d(self):
    kernel_val = np.random.random((3, 3, 4, 5))
    k = keras.backend.variable(kernel_val)

    # Test channels_first
    val = np.random.random((10, 4, 10, 10))
    x = keras.backend.variable(val)
    y = keras.backend.conv2d(x, k,
                             padding='valid', data_format='channels_first')
    self.assertEqual(y.shape.as_list(), [10, 5, 8, 8])

    # Test channels_last
    val = np.random.random((10, 10, 10, 4))
    x = keras.backend.variable(val)
    y = keras.backend.conv2d(x, k, strides=(1, 1),
                             padding='valid', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 8, 8, 5])

    # Test same padding
    val = np.random.random((10, 10, 10, 4))
    x = keras.backend.variable(val)
    y = keras.backend.conv2d(x, k,
                             padding='same', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 10, 10, 5])

    # Test dilation_rate
    val = np.random.random((10, 10, 10, 4))
    x = keras.backend.variable(val)
    y = keras.backend.conv2d(x, k, dilation_rate=(2, 2),
                             padding='same', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 10, 10, 5])

    # Test strides
    val = np.random.random((10, 10, 10, 4))
    x = keras.backend.variable(val)
    y = keras.backend.conv2d(x, k, strides=(2, 2),
                             padding='same', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 5, 5, 5])

    # Test invalid arguments
    with self.assertRaises(ValueError):
      y = keras.backend.conv2d(x, k, (2, 2),
                               padding='other', data_format='channels_last')
    with self.assertRaises(ValueError):
      y = keras.backend.conv2d(x, k, (2, 2),
                               data_format='other')
    with self.assertRaises(ValueError):
      y = keras.backend.conv2d(x, k, (2, 2, 2))

  def test_conv2d_transpose(self):
    input_size = (7, 8)
    kernel_size = (3, 3)
    input_depth = 6
    filters = 6
    batch_size = 2

    kernel_val = np.random.random(kernel_size + (input_depth, filters))
    k = keras.backend.variable(kernel_val)

    # Test channels_first
    input_val = np.random.random((batch_size, input_depth) + input_size)
    x = keras.backend.variable(input_val)
    y = keras.backend.conv2d_transpose(x, k, (batch_size, filters) + input_size,
                                       padding='same',
                                       data_format='channels_first')
    self.assertEqual(
        tuple(y.shape.as_list()), (batch_size, filters) + input_size)

    # Test channels_last
    input_val = np.random.random((batch_size,) + input_size + (input_depth,))
    x = keras.backend.variable(input_val)
    y = keras.backend.conv2d_transpose(
        x, k, (batch_size,) + input_size + (filters,),
        padding='same',
        data_format='channels_last')
    self.assertEqual(
        tuple(y.shape.as_list()), (batch_size,) + input_size + (filters,))

    # Test dilation_rate
    y = keras.backend.conv2d_transpose(
        x, k, (batch_size,) + input_size + (filters,),
        padding='same',
        data_format='channels_last',
        dilation_rate=(2, 2))
    self.assertEqual(
        tuple(y.shape.as_list()), (batch_size,) + input_size + (filters,))

    # Test batch size of None in output_shape
    y = keras.backend.conv2d_transpose(x, k, (None,) + input_size + (filters,),
                                       padding='same',
                                       data_format='channels_last')
    self.assertEqual(
        tuple(y.shape.as_list()), (batch_size,) + input_size + (filters,))

    # Test invalid values
    with self.assertRaises(ValueError):
      y = keras.backend.conv2d_transpose(x, k, (2, 2, 8, 9),
                                         padding='other',
                                         data_format='channels_last')
    with self.assertRaises(ValueError):
      y = keras.backend.conv2d_transpose(x, k, (2, 2, 8, 9),
                                         data_format='other')

  def test_separable_conv2d(self):
    val = np.random.random((10, 4, 10, 10))
    x = keras.backend.variable(val)
    depthwise_kernel_val = np.random.random((3, 3, 4, 1))
    pointwise_kernel_val = np.random.random((1, 1, 4, 5))
    dk = keras.backend.variable(depthwise_kernel_val)
    pk = keras.backend.variable(pointwise_kernel_val)
    y = keras.backend.separable_conv2d(
        x, dk, pk, padding='valid', data_format='channels_first')
    self.assertEqual(y.shape.as_list(), [10, 5, 8, 8])

    val = np.random.random((10, 10, 10, 4))
    x = keras.backend.variable(val)
    y = keras.backend.separable_conv2d(
        x, dk, pk, strides=(1, 1), padding='valid', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 8, 8, 5])

    val = np.random.random((10, 10, 10, 4))
    x = keras.backend.variable(val)
    y = keras.backend.separable_conv2d(
        x, dk, pk, strides=(1, 1), padding='same', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 10, 10, 5])

    val = np.random.random((10, 10, 10, 4))
    x = keras.backend.variable(val)
    y = keras.backend.separable_conv2d(
        x, dk, pk, strides=(2, 2), padding='same', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 5, 5, 5])
    with self.assertRaises(ValueError):
      y = keras.backend.separable_conv2d(
          x, dk, pk, (2, 2), padding='other', data_format='channels_last')
    with self.assertRaises(ValueError):
      y = keras.backend.separable_conv2d(
          x, dk, pk, (2, 2), data_format='other')
    with self.assertRaises(ValueError):
      y = keras.backend.separable_conv2d(x, dk, pk, (2, 2, 2))

  def test_conv3d(self):
    val = np.random.random((10, 4, 10, 10, 10))
    x = keras.backend.variable(val)
    kernel_val = np.random.random((3, 3, 3, 4, 5))
    k = keras.backend.variable(kernel_val)
    y = keras.backend.conv3d(x, k,
                             padding='valid', data_format='channels_first')
    self.assertEqual(y.shape.as_list(), [10, 5, 8, 8, 8])

    val = np.random.random((10, 10, 10, 10, 4))
    x = keras.backend.variable(val)
    y = keras.backend.conv3d(x, k, strides=(1, 1, 1),
                             padding='valid', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 8, 8, 8, 5])

    val = np.random.random((10, 10, 10, 10, 4))
    x = keras.backend.variable(val)
    y = keras.backend.conv3d(x, k, strides=(1, 1, 1),
                             padding='same', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 10, 10, 10, 5])

    val = np.random.random((10, 10, 10, 10, 4))
    x = keras.backend.variable(val)
    y = keras.backend.conv3d(x, k, strides=(2, 2, 2),
                             padding='same', data_format='channels_last')
    self.assertEqual(y.shape.as_list(), [10, 5, 5, 5, 5])
    with self.assertRaises(ValueError):
      y = keras.backend.conv3d(x, k, (2, 2, 2),
                               padding='other', data_format='channels_last')
    with self.assertRaises(ValueError):
      y = keras.backend.conv3d(x, k, (2, 2, 2),
                               data_format='other')
    with self.assertRaises(ValueError):
      y = keras.backend.conv3d(x, k, (2, 2))

  def test_rnn(self):
    # implement a simple RNN
    num_samples = 4
    input_dim = 5
    output_dim = 3
    timesteps = 6

    input_val = np.random.random(
        (num_samples, timesteps, input_dim)).astype(np.float32)
    init_state_val = np.random.random(
        (num_samples, output_dim)).astype(np.float32)
    w_i_val = np.random.random((input_dim, output_dim)).astype(np.float32)
    w_o_val = np.random.random((output_dim, output_dim)).astype(np.float32)
    np_mask = np.random.randint(2, size=(num_samples, timesteps))

    def rnn_step_fn():
      w_i = keras.backend.variable(w_i_val)
      w_o = keras.backend.variable(w_o_val)

      def step_function(x, states):
        assert len(states) == 1
        prev_output = states[0]
        output = keras.backend.dot(x, w_i) + keras.backend.dot(prev_output, w_o)
        return output, [output]

      return step_function

    # test default setup
    last_output_list = [[], [], [], [], [], []]
    outputs_list = [[], [], [], [], [], []]
    state_list = [[], [], [], [], [], []]

    rnn_fn = rnn_step_fn()
    inputs = keras.backend.variable(input_val)
    initial_states = [keras.backend.variable(init_state_val)]
    mask = keras.backend.variable(np_mask)

    kwargs_list = [
        {'go_backwards': False, 'mask': None},
        {'go_backwards': False, 'mask': None, 'unroll': True},
        {'go_backwards': True, 'mask': None},
        {'go_backwards': True, 'mask': None, 'unroll': True},
        {'go_backwards': False, 'mask': mask},
        {'go_backwards': False, 'mask': mask, 'unroll': True},
    ]
    for i, kwargs in enumerate(kwargs_list):
      last_output, outputs, new_states = keras.backend.rnn(rnn_fn, inputs,
                                                           initial_states,
                                                           **kwargs)
      # check static shape inference
      self.assertEqual(last_output.shape.as_list(), [num_samples, output_dim])
      self.assertEqual(outputs.shape.as_list(),
                       [num_samples, timesteps, output_dim])
      for state in new_states:
        self.assertEqual(state.shape.as_list(), [num_samples, output_dim])

      last_output_list[i].append(keras.backend.eval(last_output))
      outputs_list[i].append(keras.backend.eval(outputs))
      self.assertLen(new_states, 1)
      state_list[i].append(keras.backend.eval(new_states[0]))

      def assert_list_pairwise(z_list, atol=1e-05):
        for (z1, z2) in zip(z_list[1:], z_list[:-1]):
          self.assertAllClose(z1, z2, atol=atol)

      assert_list_pairwise(last_output_list[0], atol=1e-04)
      assert_list_pairwise(outputs_list[0], atol=1e-04)
      assert_list_pairwise(state_list[0], atol=1e-04)
      assert_list_pairwise(last_output_list[2], atol=1e-04)
      assert_list_pairwise(outputs_list[2], atol=1e-04)
      assert_list_pairwise(state_list[2], atol=1e-04)

      for l, u_l in zip(last_output_list[0], last_output_list[1]):
        self.assertAllClose(l, u_l, atol=1e-04)

      for o, u_o in zip(outputs_list[0], outputs_list[1]):
        self.assertAllClose(o, u_o, atol=1e-04)

      for s, u_s in zip(state_list[0], state_list[1]):
        self.assertAllClose(s, u_s, atol=1e-04)

      for b_l, b_u_l in zip(last_output_list[2], last_output_list[3]):
        self.assertAllClose(b_l, b_u_l, atol=1e-04)

      for b_o, b_u_o in zip(outputs_list[2], outputs_list[3]):
        self.assertAllClose(b_o, b_u_o, atol=1e-04)

      for b_s, b_u_s in zip(state_list[2], state_list[3]):
        self.assertAllClose(b_s, b_u_s, atol=1e-04)

  def test_rnn_additional_states(self):
    # implement a simple RNN
    num_samples = 4
    input_dim = 5
    output_dim = 3
    timesteps = 6

    input_val = np.random.random(
        (num_samples, timesteps, input_dim)).astype(np.float32)
    init_state_val = np.random.random(
        (num_samples, output_dim)).astype(np.float32)
    w_i_val = np.random.random((input_dim, output_dim)).astype(np.float32)
    w_o_val = np.random.random((output_dim, output_dim)).astype(np.float32)
    np_mask = np.random.randint(2, size=(num_samples, timesteps))

    def rnn_step_fn():
      w_i = keras.backend.variable(w_i_val)
      w_o = keras.backend.variable(w_o_val)

      def step_function(x, states):
        assert len(states) == 2
        prev_output = states[0]
        output = keras.backend.dot(x, w_i) + keras.backend.dot(prev_output, w_o)
        return output, [output,
                        keras.backend.concatenate([output, output], axis=-1)]

      return step_function

    # test default setup
    last_output_list = [[], [], [], [], [], []]
    outputs_list = [[], [], [], [], [], []]
    state_list = [[], [], [], [], [], []]
    additional_state_list = [[], [], [], [], [], []]

    rnn_fn = rnn_step_fn()
    inputs = keras.backend.variable(input_val)
    initial_states = [
        keras.backend.variable(init_state_val),
        ops.convert_to_tensor(
            np.concatenate([init_state_val, init_state_val], axis=-1))
    ]
    mask = keras.backend.variable(np_mask)

    kwargs_list = [
        {'go_backwards': False, 'mask': None},
        {'go_backwards': False, 'mask': None, 'unroll': True},
        {'go_backwards': True, 'mask': None},
        {'go_backwards': True, 'mask': None, 'unroll': True},
        {'go_backwards': False, 'mask': mask},
        {'go_backwards': False, 'mask': mask, 'unroll': True},
    ]
    for i, kwargs in enumerate(kwargs_list):
      last_output, outputs, new_states = keras.backend.rnn(rnn_fn, inputs,
                                                           initial_states,
                                                           **kwargs)
      # check static shape inference
      self.assertEqual(last_output.shape.as_list(), [num_samples, output_dim])
      self.assertEqual(outputs.shape.as_list(),
                       [num_samples, timesteps, output_dim])
      # for state in new_states:
      #   self.assertEqual(state.shape.as_list(),
      #                     [num_samples, output_dim])
      self.assertEqual(new_states[0].shape.as_list(), [num_samples, output_dim])
      self.assertEqual(new_states[1].shape.as_list(),
                       [num_samples, 2 * output_dim])

      last_output_list[i].append(keras.backend.eval(last_output))
      outputs_list[i].append(keras.backend.eval(outputs))
      self.assertLen(new_states, 2)
      state_list[i].append(keras.backend.eval(new_states[0]))
      additional_state_list[i].append(keras.backend.eval(new_states[1]))

      def assert_list_pairwise(z_list, atol=1e-05):
        for (z1, z2) in zip(z_list[1:], z_list[:-1]):
          self.assertAllClose(z1, z2, atol=atol)

      assert_list_pairwise(last_output_list[0], atol=1e-04)
      assert_list_pairwise(outputs_list[0], atol=1e-04)
      assert_list_pairwise(state_list[0], atol=1e-04)
      assert_list_pairwise(additional_state_list[0], atol=1e-04)
      assert_list_pairwise(last_output_list[2], atol=1e-04)
      assert_list_pairwise(outputs_list[2], atol=1e-04)
      assert_list_pairwise(state_list[2], atol=1e-04)
      assert_list_pairwise(additional_state_list[2], atol=1e-04)

      for l, u_l in zip(last_output_list[0], last_output_list[1]):
        self.assertAllClose(l, u_l, atol=1e-04)

      for o, u_o in zip(outputs_list[0], outputs_list[1]):
        self.assertAllClose(o, u_o, atol=1e-04)

      for s, u_s in zip(state_list[0], state_list[1]):
        self.assertAllClose(s, u_s, atol=1e-04)

      for s, u_s in zip(additional_state_list[0], additional_state_list[1]):
        self.assertAllClose(s, u_s, atol=1e-04)

      for b_l, b_u_l in zip(last_output_list[2], last_output_list[3]):
        self.assertAllClose(b_l, b_u_l, atol=1e-04)

      for b_o, b_u_o in zip(outputs_list[2], outputs_list[3]):
        self.assertAllClose(b_o, b_u_o, atol=1e-04)

      for b_s, b_u_s in zip(state_list[2], state_list[3]):
        self.assertAllClose(b_s, b_u_s, atol=1e-04)

      for s, u_s in zip(additional_state_list[2], additional_state_list[3]):
        self.assertAllClose(s, u_s, atol=1e-04)

  def test_rnn_output_and_state_masking_independent(self):
    num_samples = 2
    num_timesteps = 4
    state_and_io_size = 2
    mask_last_num_timesteps = 2  # for second sample only

    # a step function that just outputs inputs,
    # but increments states +1 per timestep
    def step_function(inputs, states):
      return inputs, [s + 1 for s in states]

    inputs_vals = np.random.random((num_samples, num_timesteps,
                                    state_and_io_size))
    initial_state_vals = np.random.random((num_samples, state_and_io_size))
    # masking of two last timesteps for second sample only
    mask_vals = np.ones((num_samples, num_timesteps))
    mask_vals[1, -mask_last_num_timesteps:] = 0

    # outputs expected to be same as inputs for the first sample
    expected_outputs = inputs_vals.copy()
    # but for the second sample all outputs in masked region should be the same
    # as last output before masked region
    expected_outputs[1, -mask_last_num_timesteps:] = \
        expected_outputs[1, -(mask_last_num_timesteps + 1)]

    expected_last_state = initial_state_vals.copy()
    # first state should be incremented for every timestep (no masking)
    expected_last_state[0] += num_timesteps
    # second state should not be incremented for last two timesteps
    expected_last_state[1] += (num_timesteps - mask_last_num_timesteps)

    # verify same expected output for `unroll=true/false`
    inputs = keras.backend.variable(inputs_vals)
    initial_states = [keras.backend.variable(initial_state_vals)]
    mask = keras.backend.variable(mask_vals)
    for unroll in [True, False]:
      _, outputs, last_states = keras.backend.rnn(
          step_function,
          inputs,
          initial_states,
          mask=mask,
          unroll=unroll,
          input_length=num_timesteps if unroll else None)

      self.assertAllClose(keras.backend.eval(outputs), expected_outputs)
      self.assertAllClose(
          keras.backend.eval(last_states[0]), expected_last_state)

  def test_rnn_output_num_dim_larger_than_2_masking(self):
    num_samples = 3
    num_timesteps = 4
    num_features = 5

    def step_function(inputs, states):
      outputs = keras.backend.tile(keras.backend.expand_dims(inputs), [1, 1, 2])
      return outputs, [keras.backend.identity(s) for s in states]
      # Note: cannot just return states (which can be a problem) ->
      # tensorflow/python/ops/resource_variable_ops.py", line 824, in set_shape
      # NotImplementedError: ResourceVariable does not implement set_shape()

    inputs_vals = np.random.random((num_samples, num_timesteps, num_features))
    initial_state_vals = np.random.random((num_samples, 6))
    mask_vals = np.ones((num_samples, num_timesteps))
    mask_vals[-1, -1] = 0  # final timestep masked for last sample

    expected_outputs = np.repeat(inputs_vals[..., None], repeats=2, axis=-1)
    # for the last sample, the final timestep (in masked region) should be the
    # same as the second to final output (before masked region)
    expected_outputs[-1, -1] = expected_outputs[-1, -2]

    inputs = keras.backend.variable(inputs_vals)
    initial_states = [keras.backend.variable(initial_state_vals)]
    mask = keras.backend.variable(mask_vals)
    for unroll in [True, False]:
      _, outputs, _ = keras.backend.rnn(
          step_function,
          inputs,
          initial_states,
          mask=mask,
          unroll=unroll,
          input_length=num_timesteps if unroll else None)

      self.assertAllClose(keras.backend.eval(outputs), expected_outputs)

  def test_rnn_state_num_dim_larger_than_2_masking(self):
    num_samples = 3
    num_timesteps = 4

    def step_function(inputs, states):
      return inputs, [s + 1 for s in states]

    inputs_vals = np.random.random((num_samples, num_timesteps, 5))
    initial_state_vals = np.random.random((num_samples, 6, 7))
    mask_vals = np.ones((num_samples, num_timesteps))
    mask_vals[0, -2:] = 0  # final two timesteps masked for first sample

    expected_last_state = initial_state_vals.copy()
    expected_last_state[0] += (num_timesteps - 2)
    expected_last_state[1:] += num_timesteps

    inputs = keras.backend.variable(inputs_vals)
    initial_states = [keras.backend.variable(initial_state_vals)]
    mask = keras.backend.variable(mask_vals)
    for unroll in [True, False]:
      _, _, last_states = keras.backend.rnn(
          step_function,
          inputs,
          initial_states,
          mask=mask,
          unroll=unroll,
          input_length=num_timesteps if unroll else None)

      self.assertAllClose(
          keras.backend.eval(last_states[0]), expected_last_state)

  def test_batch_normalization(self):
    g_val = np.random.random((3,))
    b_val = np.random.random((3,))
    gamma = keras.backend.variable(g_val)
    beta = keras.backend.variable(b_val)

    # 3D NHC case
    val = np.random.random((10, 5, 3))
    x = keras.backend.variable(val)
    mean, var = nn.moments(x, (0, 1), None, None, False)
    normed = keras.backend.batch_normalization(
        x, mean, var, beta, gamma, axis=-1, epsilon=1e-3)
    self.assertEqual(normed.shape.as_list(), [10, 5, 3])

    # 4D NHWC case
    val = np.random.random((10, 5, 5, 3))
    x = keras.backend.variable(val)
    mean, var = nn.moments(x, (0, 1, 2), None, None, False)
    normed = keras.backend.batch_normalization(
        x, mean, var, beta, gamma, axis=-1, epsilon=1e-3)
    self.assertEqual(normed.shape.as_list(), [10, 5, 5, 3])

    # 4D NCHW case
    if not context.executing_eagerly():
      # Eager CPU kernel for NCHW does not exist.
      val = np.random.random((10, 3, 5, 5))
      x = keras.backend.variable(val)
      mean, var = nn.moments(x, (0, 2, 3), None, None, False)
      normed = keras.backend.batch_normalization(
          x, mean, var, beta, gamma, axis=1, epsilon=1e-3)
      self.assertEqual(normed.shape.as_list(), [10, 3, 5, 5])

  def test_normalize_batch_in_training(self):
    val = np.random.random((10, 3, 10, 10))
    x = keras.backend.variable(val)
    reduction_axes = (0, 2, 3)

    g_val = np.random.random((3,))
    b_val = np.random.random((3,))
    gamma = keras.backend.variable(g_val)
    beta = keras.backend.variable(b_val)
    normed, mean, var = keras.backend.normalize_batch_in_training(
        x, gamma, beta, reduction_axes, epsilon=1e-3)
    self.assertEqual(normed.shape.as_list(), [10, 3, 10, 10])
    self.assertEqual(mean.shape.as_list(), [
        3,
    ])
    self.assertEqual(var.shape.as_list(), [
        3,
    ])

    # case: gamma=None
    gamma = None
    normed, mean, var = keras.backend.normalize_batch_in_training(
        x, gamma, beta, reduction_axes, epsilon=1e-3)
    self.assertEqual(normed.shape.as_list(), [10, 3, 10, 10])
    self.assertEqual(mean.shape.as_list(), [
        3,
    ])
    self.assertEqual(var.shape.as_list(), [
        3,
    ])

    # case: beta=None
    beta = None
    normed, mean, var = keras.backend.normalize_batch_in_training(
        x, gamma, beta, reduction_axes, epsilon=1e-3)
    self.assertEqual(normed.shape.as_list(), [10, 3, 10, 10])
    self.assertEqual(mean.shape.as_list(), [
        3,
    ])
    self.assertEqual(var.shape.as_list(), [
        3,
    ])

  def test_dropout(self):
    inputs = array_ops.ones((200, 200))
    outputs = keras.backend.dropout(inputs, 0.2)
    outputs_val = keras.backend.eval(outputs)
    self.assertEqual(np.min(outputs_val), 0)
    self.assertAllClose(np.count_nonzero(outputs_val), 32000, atol=1000)
    # Test noise shape
    outputs = keras.backend.dropout(inputs, 0.2, noise_shape=(200, 1))
    outputs_val = keras.backend.eval(outputs)
    self.assertAllClose(outputs_val[2, :], outputs_val[3, :], atol=1e-5)


class BackendCrossEntropyLossesTest(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_categorical_crossentropy_loss(self):
    t = keras.backend.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    p = keras.backend.constant([[.9, .05, .05], [.05, .89, .06],
                                [.05, .01, .94]])
    result = keras.backend.categorical_crossentropy(t, p)
    self.assertArrayNear(self.evaluate(result), [.105, .116, .062], 1e-3)

    p = keras.backend.constant([[.9, .05, .05], [.05, .89, .01],
                                [.05, .06, .94]])
    result = keras.backend.categorical_crossentropy(t, p, axis=0)
    self.assertArrayNear(self.evaluate(result), [.105, .116, .062], 1e-3)

    p = keras.backend.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    result = keras.backend.categorical_crossentropy(t, p, from_logits=True),
    self.assertArrayNear(self.evaluate(result)[0], [.002, 0, .17], 1e-3)

    p = keras.backend.constant([[8., 0., 2.], [1., 9., 3.], [1., 1., 5.]])
    result = keras.backend.categorical_crossentropy(
        t, p, from_logits=True, axis=0),
    self.assertArrayNear(self.evaluate(result)[0], [.002, 0, .17], 1e-3)

  @test_util.run_in_graph_and_eager_modes
  def test_categorical_crossentropy_loss_with_unknown_rank_tensor(self):
    t = keras.backend.placeholder()
    p = keras.backend.placeholder()
    o = keras.backend.categorical_crossentropy(t, p)

    t_val = ops.convert_to_tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    p_val = ops.convert_to_tensor([[.9, .05, .05], [.05, .89, .06],
                                   [.05, .01, .94]])
    f = keras.backend.function([t, p], o)

    result = f([t_val, p_val])
    self.assertArrayNear(result, [.105, .116, .062], 1e-3)

    # With axis set
    o = keras.backend.categorical_crossentropy(t, p, axis=0)
    f = keras.backend.function([t, p], o)

    result = f([t_val, p_val])
    self.assertArrayNear(result, [.105, .065, .111], 1e-3)

    # from logits
    p_val = ops.convert_to_tensor([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    o = keras.backend.categorical_crossentropy(t, p, from_logits=True)
    f = keras.backend.function([t, p], o)

    result = f([t_val, p_val])
    self.assertArrayNear(result, [.002, 0, .17], 1e-3)

    # from logits and axis set
    o = keras.backend.categorical_crossentropy(t, p, from_logits=True, axis=0)
    f = keras.backend.function([t, p], o)

    result = f([t_val, p_val])
    self.assertArrayNear(result, [.002, .003, .036], 1e-3)

  @test_util.run_in_graph_and_eager_modes
  def test_sparse_categorical_crossentropy_loss(self):
    t = keras.backend.constant([0, 1, 2])

    p = keras.backend.constant([[.9, .05, .05], [.05, .89, .06],
                                [.05, .01, .94]])
    result = keras.backend.sparse_categorical_crossentropy(t, p)
    self.assertArrayNear(self.evaluate(result), [.105, .116, .062], 1e-3)

    p = keras.backend.constant([[.9, .05, .05], [.05, .89, .01],
                                [.05, .06, .94]])
    result = keras.backend.sparse_categorical_crossentropy(t, p, axis=0)
    self.assertArrayNear(self.evaluate(result), [.105, .116, .062], 1e-3)

    p = keras.backend.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    result = keras.backend.sparse_categorical_crossentropy(
        t, p, from_logits=True),
    self.assertArrayNear(self.evaluate(result)[0], [.002, 0, .17], 1e-3)

    p = keras.backend.constant([[8., 0., 2.], [1., 9., 3.], [1., 1., 5.]])
    result = keras.backend.sparse_categorical_crossentropy(
        t, p, from_logits=True, axis=0),
    self.assertArrayNear(self.evaluate(result)[0], [.002, 0, .17], 1e-3)

  @test_util.run_in_graph_and_eager_modes
  def test_sparse_categorical_crossentropy_loss_with_unknown_rank_tensor(self):
    t = keras.backend.placeholder()
    p = keras.backend.placeholder()
    o = keras.backend.sparse_categorical_crossentropy(t, p)

    t_val = ops.convert_to_tensor([0, 1, 2])
    p_val = ops.convert_to_tensor([[.9, .05, .05], [.05, .89, .06],
                                   [.05, .01, .94]])
    f = keras.backend.function([t, p], o)

    result = f([t_val, p_val])
    self.assertArrayNear(result, [.105, .116, .062], 1e-3)

    # With axis set
    with self.assertRaisesRegex(
        ValueError,
        'Cannot compute sparse categorical crossentropy with `axis=0`'):
      o = keras.backend.sparse_categorical_crossentropy(t, p, axis=0)
      f = keras.backend.function([t, p], o)

      _ = f([t_val, p_val])

    # from logits
    p_val = ops.convert_to_tensor([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
    o = keras.backend.sparse_categorical_crossentropy(t, p, from_logits=True)
    f = keras.backend.function([t, p], o)

    result = f([t_val, p_val])
    self.assertArrayNear(result, [.002, 0, .17], 1e-3)

    # from logits and axis set
    with self.assertRaisesRegex(
        ValueError,
        'Cannot compute sparse categorical crossentropy with `axis=0`'):
      o = keras.backend.sparse_categorical_crossentropy(
          t, p, from_logits=True, axis=0)
      f = keras.backend.function([t, p], o)

      _ = f([t_val, p_val])


@test_util.run_all_in_graph_and_eager_modes
@test_util.with_control_flow_v2
class TestCTC(test.TestCase):

  def test_ctc_decode(self):
    depth = 6
    seq_len_0 = 5
    input_prob_matrix_0 = np.asarray(
        [[0.30999, 0.309938, 0.0679938, 0.0673362, 0.0708352, 0.173908],
         [0.215136, 0.439699, 0.0370931, 0.0393967, 0.0381581, 0.230517],
         [0.199959, 0.489485, 0.0233221, 0.0251417, 0.0233289, 0.238763],
         [0.279611, 0.452966, 0.0204795, 0.0209126, 0.0194803, 0.20655],
         [0.51286, 0.288951, 0.0243026, 0.0220788, 0.0219297, 0.129878],
         # Random entry added in at time=5
         [0.155251, 0.164444, 0.173517, 0.176138, 0.169979, 0.160671]],
        dtype=np.float32)

    # len max_time_steps array of batch_size x depth matrices
    inputs = ([input_prob_matrix_0[t, :][np.newaxis, :]
               for t in range(seq_len_0)] +  # Pad to max_time_steps = 8
              2 * [np.zeros((1, depth), dtype=np.float32)])

    inputs = keras.backend.variable(np.asarray(inputs).transpose((1, 0, 2)))

    # batch_size length vector of sequence_lengths
    input_length = keras.backend.variable(
        np.array([seq_len_0], dtype=np.int32))
    # batch_size length vector of negative log probabilities
    log_prob_truth = np.array([
        -3.5821197,  # output beam 0
        -3.777835    # output beam 1
    ], np.float32)[np.newaxis, :]

    decode_truth = [np.array([1, 0]), np.array([0, 1, 0])]
    beam_width = 2
    top_paths = 2

    decode_pred_tf, log_prob_pred_tf = keras.backend.ctc_decode(
        inputs,
        input_length,
        greedy=False,
        beam_width=beam_width,
        top_paths=top_paths)

    self.assertEqual(len(decode_pred_tf), top_paths)
    log_prob_pred = keras.backend.eval(log_prob_pred_tf)
    for i in range(top_paths):
      self.assertTrue(
          np.alltrue(
              decode_truth[i] == keras.backend.eval(decode_pred_tf[i])))
    self.assertAllClose(log_prob_truth, log_prob_pred)

  @test_util.run_v1_only('b/120545219')
  def test_ctc_batch_cost(self):
    with self.cached_session():
      label_lens = np.expand_dims(np.asarray([5, 4]), 1)
      input_lens = np.expand_dims(np.asarray([5, 5]), 1)  # number of timesteps
      loss_log_probs = [3.34211, 5.42262]

      # dimensions are batch x time x categories
      labels = np.asarray([[0, 1, 2, 1, 0], [0, 1, 1, 0, -1]])
      inputs = np.asarray(
          [[[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
            [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
            [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
            [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
            [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],
           [[0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
            [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549],
            [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456],
            [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345],
            [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]]],
          dtype=np.float32)

      labels = keras.backend.variable(labels, dtype='int32')
      inputs = keras.backend.variable(inputs, dtype='float32')
      input_lens = keras.backend.variable(input_lens, dtype='int32')
      label_lens = keras.backend.variable(label_lens, dtype='int32')
      res = keras.backend.eval(
          keras.backend.ctc_batch_cost(labels, inputs, input_lens, label_lens))
      self.assertAllClose(res[:, 0], loss_log_probs, atol=1e-05)

      # test when batch_size = 1, that is, one sample only
      ref = [3.34211]
      input_lens = np.expand_dims(np.asarray([5]), 1)
      label_lens = np.expand_dims(np.asarray([5]), 1)

      labels = np.asarray([[0, 1, 2, 1, 0]])
      inputs = np.asarray(
          [[[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553], [
              0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436
          ], [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
            [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
            [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]]
          ],
          dtype=np.float32)

      k_labels = keras.backend.variable(labels, dtype='int32')
      k_inputs = keras.backend.variable(inputs, dtype='float32')
      k_input_lens = keras.backend.variable(input_lens, dtype='int32')
      k_label_lens = keras.backend.variable(label_lens, dtype='int32')
      res = keras.backend.eval(
          keras.backend.ctc_batch_cost(k_labels, k_inputs, k_input_lens,
                                       k_label_lens))
      self.assertAllClose(res[:, 0], ref, atol=1e-05)


@test_util.run_all_in_graph_and_eager_modes
class TestRandomOps(test.TestCase):

  def test_random_normal(self):
    np.random.seed(123)
    x = keras.backend.random_normal((500, 500))
    val = keras.backend.eval(x)
    self.assertAllClose(np.mean(val), 0., atol=0.01)
    self.assertAllClose(np.std(val), 1., atol=0.01)

  def test_random_uniform(self):
    np.random.seed(123)
    x = keras.backend.random_uniform((500, 500))
    val = keras.backend.eval(x)
    self.assertAllClose(np.mean(val), 0.5, atol=0.01)
    self.assertAllClose(np.max(val), 1., atol=0.01)
    self.assertAllClose(np.min(val), 0., atol=0.01)

  def test_random_binomial(self):
    np.random.seed(123)
    x = keras.backend.random_binomial((500, 500), p=0.5)
    self.assertAllClose(np.mean(keras.backend.eval(x)), 0.5, atol=0.01)

  def test_truncated_normal(self):
    np.random.seed(123)
    x = keras.backend.truncated_normal((500, 500), mean=0.0, stddev=1.0)
    x = keras.backend.truncated_normal((1000, 1000), mean=0.0, stddev=1.0)
    y = keras.backend.eval(x)
    self.assertAllClose(np.mean(y), 0., atol=0.01)
    self.assertAllClose(np.std(y), 0.88, atol=0.01)
    self.assertAllClose(np.max(y), 2., atol=0.01)
    self.assertAllClose(np.min(y), -2., atol=0.01)


@test_util.run_all_in_graph_and_eager_modes
class FunctionTest(test.TestCase):

  def test_function_basics(self):
    x1 = keras.backend.placeholder(shape=(), dtype='float32')
    x2 = keras.backend.placeholder(shape=(), dtype='int32')
    v = keras.backend.variable(10.)

    y1 = x1 + keras.backend.cast(x2, 'float32') + v
    y2 = x1 * keras.backend.cast(x2, 'float32')

    with ops.control_dependencies([y1]):
      u = keras.backend.update(v, x1)

    f = keras.backend.function([x1, x2], [y1, y2], updates=[u])
    output_values = f([2, 3])
    self.assertEqual(output_values, [15., 6.])
    self.assertEqual(keras.backend.eval(v), 2.)

  def test_function_dict_outputs(self):
    x_ph = keras.backend.placeholder(shape=(), name='x')
    y_ph = keras.backend.placeholder(shape=(), name='y')
    outputs = {'x*y': y_ph * x_ph, 'x*x': x_ph * x_ph}

    f = keras.backend.function(inputs=[x_ph, y_ph], outputs=outputs)
    x, y = 2., 5.
    results = f([x, y])

    self.assertEqual(results['x*y'], 10.)
    self.assertEqual(results['x*x'], 4)

  def test_function_dict_inputs(self):
    placeholders = {
        'x': keras.backend.placeholder(shape=()),
        'y': keras.backend.placeholder(shape=())
    }
    outputs = [placeholders['x'] * placeholders['y']]

    f = keras.backend.function(inputs=placeholders, outputs=outputs)
    results = f({'x': 2., 'y': 3.})
    self.assertEqual(results[0], 6.)

  def test_function_single_input_output(self):
    x_ph = keras.backend.placeholder(shape=(), name='x')
    output = x_ph * x_ph
    f = keras.backend.function(x_ph, output)
    result = f(2.)
    self.assertEqual(result, 4.)

  def test_tuple_updates(self):
    x_ph = keras.backend.placeholder(ndim=2)
    v = keras.backend.variable(np.ones((4, 2)))
    output = x_ph ** 2 + v
    new_v = v + x_ph
    f = keras.backend.function(x_ph, output, updates=[(v, new_v)])
    input_val = np.random.random((4, 2))
    result = f(input_val)
    self.assertAllClose(result, input_val ** 2 + 1)
    self.assertAllClose(keras.backend.get_value(v), np.ones((4, 2)) + input_val)


class BackendGraphTests(test.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_function_placeholder_with_default(self):
    with keras.backend.get_graph().as_default():
      x1 = array_ops.placeholder_with_default(
          np.array(2., dtype='float32'), shape=())
      x2 = array_ops.placeholder_with_default(
          np.array(3, dtype='int32'), shape=())
    y1 = x1 + keras.backend.cast(x2, 'float32')
    y2 = x1 * keras.backend.cast(x2, 'float32')
    f = keras.backend.function([x1, x2], [y1, y2])
    output_values = f([4, 5])
    self.assertEqual(output_values, [9., 20.])
    output_values = f([None, None])
    self.assertEqual(output_values, [5., 6.])

  @test_util.run_deprecated_v1
  def test_function_tf_feed_symbols(self):
    # Test Keras backend functions with TF tensor inputs.
    with self.cached_session():
      # Test feeding a resource variable to `function`.
      x1 = keras.backend.placeholder(shape=())
      x2 = keras.backend.placeholder(shape=())
      lr = keras.backend.learning_phase()  # Include a placeholder_with_default.

      y1 = keras.backend.variable(10.)
      y2 = 3

      f = keras.backend.function(
          inputs=[x1, x2, lr],
          outputs=[x1 + 1, keras.backend.in_train_phase(x2 + 2, x2 - 1)])
      outs = f([y1, y2, None])  # Use default learning_phase value.
      self.assertEqual(outs, [11., 2.])
      outs = f([y1, y2, 1])  # Set learning phase value.
      self.assertEqual(outs, [11., 5.])

      # Test triggering a callable refresh by changing the input.
      y3 = keras.backend.constant(20.)  # Test with tensor
      outs = f([y3, y2, None])
      self.assertEqual(outs, [21., 2.])

      y4 = 4  # Test with non-symbol
      outs = f([y4, y2, None])
      self.assertEqual(outs, [5., 2.])

      # Test with a different dtype
      y5 = keras.backend.constant(10., dtype='float64')
      outs = f([y5, y2, None])
      self.assertEqual(outs, [11., 2.])

  @test_util.run_deprecated_v1
  def test_function_tf_fetches(self):
    # Additional operations can be passed to tf.compat.v1.Session().run() via
    # its `fetches` arguments. In contrast to `updates` argument of
    # keras.backend.function() these do not have control dependency on `outputs`
    # so they can run in parallel. Also they should not contribute to output of
    # keras.backend.function().
    with self.cached_session():
      x = keras.backend.variable(0.)
      y = keras.backend.variable(0.)
      x_placeholder = keras.backend.placeholder(shape=())
      y_placeholder = keras.backend.placeholder(shape=())

      f = keras.backend.function(
          inputs=[x_placeholder, y_placeholder],
          outputs=[x_placeholder + y_placeholder],
          updates=[(x, x_placeholder + 1.)],
          fetches=[keras.backend.update(y, 5.)])
      output = f([10., 20.])
      self.assertEqual(output, [30.])
      self.assertEqual(keras.backend.get_session().run(fetches=[x, y]),
                       [11., 5.])

  @test_util.run_deprecated_v1
  def test_function_tf_feed_dict(self):
    # Additional substitutions can be passed to `tf.compat.v1.Session().run()`
    # via its `feed_dict` arguments. Note that the feed_dict is passed once in
    # the constructor but we can modify the values in the dictionary. Through
    # this feed_dict we can provide additional substitutions besides Keras
    # inputs.
    with self.cached_session():
      x = keras.backend.variable(0.)
      y = keras.backend.variable(0.)
      x_placeholder = keras.backend.placeholder(shape=())
      y_placeholder = keras.backend.placeholder(shape=())

      feed_dict = {y_placeholder: 3.}
      fetches = [keras.backend.update(y, y_placeholder * 10.)]
      f = keras.backend.function(
          inputs=[x_placeholder],
          outputs=[x_placeholder + 1.],
          updates=[(x, x_placeholder + 10.)],
          feed_dict=feed_dict,
          fetches=fetches)
      output = f([10.])
      self.assertEqual(output, [11.])
      self.assertEqual(keras.backend.get_session().run(fetches=[x, y]),
                       [20., 30.])

      # updated value in feed_dict will be modified within the K.function()
      feed_dict[y_placeholder] = 4.
      output = f([20.])
      self.assertEqual(output, [21.])
      self.assertEqual(keras.backend.get_session().run(fetches=[x, y]),
                       [30., 40.])

  @test_util.run_deprecated_v1
  def test_function_tf_run_options_with_run_metadata(self):
    with self.cached_session():
      x_placeholder = keras.backend.placeholder(shape=())
      y_placeholder = keras.backend.placeholder(shape=())

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      run_metadata = config_pb2.RunMetadata()
      # enable run_options.
      f = keras.backend.function(
          inputs=[x_placeholder, y_placeholder],
          outputs=[x_placeholder + y_placeholder],
          options=run_options,
          run_metadata=run_metadata)
      output = f([10., 20.])
      self.assertEqual(output, [30.])
      self.assertGreater(len(run_metadata.partition_graphs), 0)
      # disable run_options.
      f1 = keras.backend.function(
          inputs=[x_placeholder, y_placeholder],
          outputs=[x_placeholder + y_placeholder],
          run_metadata=run_metadata)
      output1 = f1([10., 20.])
      self.assertEqual(output1, [30.])
      self.assertEqual(len(run_metadata.partition_graphs), 0)

  @test_util.run_deprecated_v1
  def test_function_fetch_callbacks(self):

    class CallbackStub(object):

      def __init__(self):
        self.times_called = 0
        self.callback_result = 0

      def _fetch_callback(self, result):
        self.times_called += 1
        self.callback_result = result

    with self.cached_session():
      callback = CallbackStub()
      x_placeholder = keras.backend.placeholder(shape=())
      y_placeholder = keras.backend.placeholder(shape=())

      callback_op = x_placeholder * y_placeholder

      f = keras.backend.function(
          inputs=[x_placeholder, y_placeholder],
          outputs=[x_placeholder + y_placeholder])
      f.fetches.append(callback_op)
      f.fetch_callbacks[callback_op] = callback._fetch_callback

      _ = f([10., 20.])

      self.assertEqual(callback.times_called, 1)
      self.assertEqual(callback.callback_result, 200)

  def test_get_session_different_graphs(self):
    with ops.Graph().as_default():
      x = keras.backend.constant(1)
      session = keras.backend.get_session()
      self.assertIs(session, keras.backend.get_session((x,)))
      self.assertIs(session, keras.backend.get_session())
    with ops.Graph().as_default():
      self.assertIs(session, keras.backend.get_session((x,)))
      self.assertIsNot(session, keras.backend.get_session())


@test_util.run_all_in_graph_and_eager_modes
class ControlOpsTests(test.TestCase):

  def test_function_switch_basics(self):
    x = array_ops.constant(2.0)
    y = array_ops.constant(3.0)

    def xpowy():
      return keras.backend.pow(x, y)

    def ypowx():
      return keras.backend.pow(y, x)

    tensor = keras.backend.switch(keras.backend.less(x, y), xpowy, ypowx)
    self.assertEqual(keras.backend.eval(tensor), [8.0])

    tensor = keras.backend.switch(keras.backend.greater(x, y), xpowy, ypowx)
    self.assertEqual(keras.backend.eval(tensor), [9.0])

  def test_unequal_rank(self):
    x = ops.convert_to_tensor(np.array([[1, 2, 3], [4, 5, 6]]), dtype='float32')
    y = ops.convert_to_tensor(np.array([1, 2, 3]), dtype='float32')

    def true_func():
      return x

    def false_func():
      return y

    with self.assertRaisesRegexp(ValueError,
                                 'Rank of `condition` should be less than'):
      keras.backend.switch(keras.backend.equal(x, x), false_func, true_func)


if __name__ == '__main__':
  test.main()
