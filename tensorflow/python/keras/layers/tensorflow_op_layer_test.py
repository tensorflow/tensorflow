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
"""Test for allowing TF ops to work with Keras Functional API."""

import time

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.saving import model_config
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest


def _single_op_at_end():
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(10)(inputs)
  outputs = gen_nn_ops.relu(x)
  return keras.Model(inputs, outputs)


def _single_identity_op_at_end():
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(10)(inputs)
  outputs = array_ops.identity(x)
  return keras.Model(inputs, outputs)


def _multiple_ops_at_end():
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(10)(inputs)
  x = gen_nn_ops.relu(x)
  outputs = gen_nn_ops.relu(x)
  return keras.Model(inputs, outputs)


def _single_op_in_middle():
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(10)(inputs)
  x = gen_nn_ops.relu(x)
  outputs = keras.layers.Dense(10)(x)
  return keras.Model(inputs, outputs)


def _multiple_ops_in_middle():
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(10)(inputs)
  x = gen_nn_ops.relu(x)
  x = gen_nn_ops.relu(x)
  outputs = keras.layers.Dense(10)(x)
  return keras.Model(inputs, outputs)


def _shape_op_inference():
  inputs = keras.Input(shape=(10,))
  x = array_ops.shape(inputs)
  x = array_ops.ones(x)
  assert x.shape.as_list() == [None, 10]
  outputs = keras.layers.Dense(10)(x)
  return keras.Model(inputs, outputs)


def _shape_op_known_batch_size():
  inputs = keras.Input(batch_size=2, shape=(10,))
  x = array_ops.shape(inputs)
  x = array_ops.ones(x)
  assert x.shape.as_list() == [2, 10]
  outputs = keras.layers.Dense(10)(x)
  if context.executing_eagerly():
    return keras.Model(inputs, outputs)
  else:
    # In V1 the op layer fails for some reason,
    # but we don't have access to the test case to call
    # self.skip_test in this util method
    return keras.Model(inputs, inputs)


def _shape_op_slice_and_range():
  inputs = keras.Input(shape=(10,))
  batch_size = array_ops.shape(inputs)[0]
  x = math_ops.range(batch_size * 2)
  assert x.shape.as_list() == [None]
  x = array_ops.reshape(x, (batch_size, 2))
  x = math_ops.cast(x, dtype='float32')
  outputs = keras.layers.Dense(10)(x)
  return keras.Model(inputs, outputs)


def _shape_op_slice_and_range_known_dim():
  inputs = keras.Input(batch_size=2, shape=(10,))
  batch_size = array_ops.shape(inputs)[0]
  x = math_ops.range(batch_size * 3)
  assert x.shape.as_list() == [6]
  x = array_ops.reshape(x, (batch_size, 3))
  x = math_ops.cast(x, dtype='float32')
  outputs = keras.layers.Dense(10)(x)
  if context.executing_eagerly():
    return keras.Model(inputs, outputs)
  else:
    # In V1 the op layer fails for some reason,
    # but we don't have access to the test case to call
    # self.skip_test in this util method
    return keras.Model(inputs, inputs)


def _int32_manipulation_too_big_for_shape():
  # This test verifies that the Keras Functional API
  # won't crash when manipulating int32 tensors that are too large
  # to represent shapes.
  inputs = keras.Input(batch_size=2, shape=(10,))
  batch_size = array_ops.shape(inputs)[0]
  num_features = 3 * 1024 * 16
  x = math_ops.range(batch_size * num_features, dtype='int32')
  assert x.shape.as_list() == [inputs.shape[0] * num_features]
  x = array_ops.reshape(x, (batch_size, num_features))
  x = math_ops.cast(x, dtype='float32')
  outputs = keras.layers.Dense(10)(x)
  if context.executing_eagerly():
    return keras.Model(inputs, outputs)
  else:
    # In V1 the op layer fails for some reason,
    # but we don't have access to the test case to call
    # self.skip_test in this util method
    return keras.Model(inputs, inputs)


def _int32_manipulation_at_max_shape_dims_limit():
  # This test verifies that the Keras Functional API
  # won't crash when manipulating int32 tensors that are at the limit
  # of the max tensor size Keras can try inferring values for.
  inputs = keras.Input(batch_size=2, shape=(10,))
  batch_size = array_ops.shape(inputs)[0]
  num_features = int(keras_tensor._MAX_TENSOR_RANK / int(inputs.shape[0]))
  x = math_ops.range(batch_size * num_features, dtype='int32')
  assert x.shape.as_list() == [keras_tensor._MAX_TENSOR_RANK]

  # Verify that a value was actually inferred for a tensor that *might*
  # represent the shape, bying checking that a value in
  # the range appears in the printed inferred value
  if ops.executing_eagerly_outside_functions():
    assert str(keras_tensor._MAX_TENSOR_RANK - 1) in str(x)

  x = array_ops.reshape(x, (batch_size, num_features))
  x = math_ops.cast(x, dtype='float32')
  outputs = keras.layers.Dense(10)(x)
  if context.executing_eagerly():
    return keras.Model(inputs, outputs)
  else:
    # In V1 the op layer fails for some reason,
    # but we don't have access to the test case to call
    # self.skip_test in this util method
    return keras.Model(inputs, inputs)


def _single_standalone_branch():
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(10)(inputs)
  outputs = x * 2
  return keras.Model(inputs, outputs)


def _single_op_with_attrs():
  inputs = keras.Input(shape=(10,))
  x = math_ops.reduce_mean(inputs, axis=1, keepdims=True)
  outputs = keras.layers.Dense(10)(x)
  return keras.Model(inputs, outputs)


def _multiple_uses():
  inputs = keras.Input(shape=(10,))
  x = math_ops.reduce_mean(inputs, axis=1, keepdims=True)
  x1 = keras.layers.Dense(10)(x)
  x2 = keras.layers.Dense(10)(x)
  outputs = x1 + x2
  return keras.Model(inputs, outputs)


def _op_with_tensor_list():
  inputs = keras.Input(shape=(10,))
  x = array_ops.concat([inputs, inputs], axis=1)
  outputs = keras.layers.Dense(10)(x)
  return keras.Model(inputs, outputs)


def _add_n():
  inputs = keras.Input(shape=(10,))
  outputs = math_ops.add_n([inputs, inputs, inputs])
  return keras.Model(inputs, outputs)


def _reuse_op():
  inputs = keras.Input(shape=(10,))
  # This op needs to be checked multiple times.
  x = gen_nn_ops.relu(inputs)
  y = keras.layers.Dense(10)(x)
  x2 = x * 2
  y2 = keras.layers.Dense(10)(x2)
  outputs = y + y2
  return keras.Model(inputs, outputs)


def _float64_op():
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(10, dtype='float64')(inputs)
  x = gen_nn_ops.relu(x)
  assert x.dtype == 'float64', 'x has dtype: %s' % x.dtype
  outputs = keras.layers.Dense(10)(x)
  return keras.Model(inputs, outputs)


class MyAdd(keras.layers.Layer):

  def call(self, x, y):
    return x + y


def _layer_with_tensor_arg():
  inputs = keras.Input(shape=(10,))
  x = inputs * 2
  outputs = MyAdd()(inputs, x)
  return keras.Model(inputs, outputs)


class LayerWithLayer(keras.layers.Layer):

  def build(self, input_shape):
    self.bias = self.add_weight(name='bias', dtype='float32')
    self.layer = keras.layers.Dense(10)

  def call(self, inputs):
    inputs = inputs * self.bias
    # Would throw an error if Keras History was created here.
    return self.layer(inputs)


def _inner_layer():
  inputs = keras.Input(shape=(10,))
  outputs = LayerWithLayer()(inputs)
  return keras.Model(inputs, outputs)


def _reuse_ancillary_layer():
  inputs = (keras.Input(shape=(5,)), keras.Input(shape=(5,)))
  base_model = keras.Sequential([
      keras.layers.Dense(3, input_shape=(5,)),
  ])
  outputs = base_model(inputs[0])
  model = keras.Model(inputs, outputs)
  # The second input is only involved in ancillary layers.
  outputs_delta = outputs - base_model(0.5 * inputs[1])
  l2_loss = math_ops.reduce_mean(
      math_ops.reduce_sum(math_ops.square(outputs_delta), -1))
  model.add_loss(l2_loss)
  model.add_metric(l2_loss, aggregation='mean', name='l2_loss')
  l1_loss = 0.01 * math_ops.reduce_mean(
      math_ops.reduce_sum(math_ops.abs(outputs_delta), -1))
  model.add_loss(l1_loss)
  model.add_metric(l1_loss, aggregation='mean', name='l1_loss')
  return model


@keras_parameterized.run_all_keras_modes()
class AutoLambdaTest(keras_parameterized.TestCase):

  @parameterized.named_parameters(
      ('single_op_at_end', _single_op_at_end),
      ('single_identity_op_at_end', _single_identity_op_at_end),
      ('multiple_ops_at_end', _multiple_ops_at_end),
      ('single_op_in_middle', _single_op_in_middle),
      ('multiple_ops_in_middle', _multiple_ops_in_middle),
      ('shape_op_inference', _shape_op_inference),
      ('shape_op_known_batch_size', _shape_op_known_batch_size),
      ('shape_op_slice_and_range', _shape_op_slice_and_range),
      ('shape_op_slice_and_range_known_dim',
       _shape_op_slice_and_range_known_dim),
      ('int32_manipulation_too_big_for_shape',
       _int32_manipulation_too_big_for_shape),
      ('int32_manipulation_at_max_shape_dims_limit',
       _int32_manipulation_at_max_shape_dims_limit),
      ('single_standalone_branch', _single_standalone_branch),
      ('single_op_with_attrs', _single_op_with_attrs),
      ('multiple_uses', _multiple_uses),
      ('op_with_tensor_list', _op_with_tensor_list),
      ('add_n', _add_n),
      ('_reuse_op', _reuse_op),
      ('_float64_op', _float64_op),
      ('_inner_layer', _inner_layer),
      ('_reuse_ancillary_layer', _reuse_ancillary_layer),
      ('_layer_with_tensor_arg', _layer_with_tensor_arg),
  )
  def test_autolambda(self, model_fn):
    model = model_fn()
    model.compile(
        adam.Adam(0.001),
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())

    np_inputs = nest.map_structure(
        lambda x: np.ones((2,) + tuple(x.shape[1:]), 'float32'), model.inputs)
    np_outputs = nest.map_structure(
        lambda x: np.ones((2,) + tuple(x.shape[1:]), 'float32'), model.outputs)
    model.fit(np_inputs, np_outputs, batch_size=2)
    model(np_inputs)  # Test calling the model directly on inputs.

    new_model = keras.Model.from_config(
        model.get_config(),
        custom_objects={
            'LayerWithLayer': LayerWithLayer,
            'MyAdd': MyAdd
        })
    new_model.compile(
        adam.Adam(0.001),
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    new_model.fit(np_inputs, np_outputs, batch_size=2)
    new_model(np_inputs)  # Test calling the new model directly on inputs.
    # Assert that metrics are preserved and in the right order.
    self.assertAllEqual(model.metrics_names, new_model.metrics_names)
    # Assert that layer names don't change.
    self.assertAllEqual([layer.name for layer in model.layers],
                        [layer.name for layer in new_model.layers])

  def test_stack_preserves_correct_shape(self):
    ## Test stack([x])
    inp = keras.Input(shape=(), dtype='float32')

    out = array_ops.stack([inp])
    model = keras.Model(
        inputs=inp,
        outputs=out)
    model.compile(
        adam.Adam(0.001),
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())

    x = array_ops.ones(shape=(4, 4))
    expected = array_ops.stack([x])
    self.assertAllEqual(expected.shape, (1, 4, 4))

    self.assertAllEqual(model(x).shape, (1, 4, 4))
    self.assertAllEqual(model(x), expected)

    config = model.get_config()
    model = keras.Model.from_config(config)

    self.assertAllEqual(model(x).shape, (1, 4, 4))
    self.assertAllEqual(model(x), expected)

    ## Test stack(x)
    inp = keras.Input(shape=(), dtype='float32')

    out = array_ops.stack(inp)
    model = keras.Model(
        inputs=inp,
        outputs=out)
    model.compile(
        adam.Adam(0.001),
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())

    x = array_ops.ones(shape=(4, 4))
    expected = array_ops.stack(x)
    self.assertAllEqual(expected.shape, (4, 4))

    self.assertAllEqual(model(x).shape, (4, 4))
    self.assertAllEqual(model(x), expected)

    config = model.get_config()
    model = keras.Model.from_config(config)

    self.assertAllEqual(model(x).shape, (4, 4))
    self.assertAllEqual(model(x), expected)

  def test_getitem_slice_with_step_only(self):
    if not context.executing_eagerly():
      self.skipTest('Complex slicing like this fails in v1')
    inp = keras.Input(shape=(8,))
    slice_step = keras.Input(shape=(), dtype='int32')

    out = inp[..., ::slice_step[0]]
    model = keras.Model(
        inputs=[inp, slice_step],
        outputs=out)
    model.compile(
        adam.Adam(0.001),
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    batch_size = 7
    step = 3
    x = array_ops.stack([
        math_ops.range(8) for _ in range(batch_size)])
    args = [x, constant_op.constant(step, shape=(batch_size,))]
    expected = array_ops.stack([
        math_ops.range(8)[::step] for _ in range(batch_size)])

    if ops.executing_eagerly_outside_functions():
      self.assertIn('tf.__operators__.getitem', (
          x.name for x in model.layers))
      self.assertNotIn('tf.strided_slice', (
          x.name for x in model.layers))
    self.assertAllEqual(model(args), expected)
    self.assertAllEqual(model.predict(args, batch_size=batch_size), expected)

    # Make sure it can be successfully saved and loaded
    config = model.get_config()
    model = keras.Model.from_config(config)

    self.assertAllEqual(model(args), expected)
    self.assertAllEqual(model.predict(args, batch_size=batch_size), expected)

  def test_getitem_slice_real_tensor(self):
    if not context.executing_eagerly():
      self.skipTest('Complex slicing like this fails in v1')
    x = math_ops.range(10.0)
    slice_stop = keras.Input(shape=(), dtype='int32')

    out = x[:slice_stop[0]]
    model = keras.Model(
        inputs=slice_stop,
        outputs=out)
    model.compile(
        adam.Adam(0.001),
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    batch_size = 7
    stop = 6
    args = constant_op.constant(stop, shape=(batch_size,))
    expected = x[:stop]

    if ops.executing_eagerly_outside_functions():
      self.assertIn('tf.__operators__.getitem', (
          x.name for x in model.layers))
      # TODO(b/161925288): Fix the dispatch triggering then uncomment:
      # self.assertNotIn('tf.strided_slice', (
      #     x.name for x in model.layers))
    self.assertAllEqual(model(args), expected)
    self.assertAllEqual(model.predict(args, batch_size=batch_size), expected)

    config = model.get_config()
    model = keras.Model.from_config(config)

    self.assertAllEqual(model(args), expected)
    self.assertAllEqual(model.predict(args, batch_size=batch_size), expected)

  def test_getitem_index_real_tensor(self):
    if not context.executing_eagerly():
      self.skipTest('Complex slicing like this fails in v1')
    x = math_ops.range(10.0)
    slice_stop = keras.Input(shape=(), dtype='int32')

    out = x[slice_stop[0]]
    model = keras.Model(
        inputs=slice_stop,
        outputs=out)
    model.compile(
        adam.Adam(0.001),
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    batch_size = 7
    index = 6
    args = constant_op.constant(index, shape=(batch_size,))
    expected = x[index]

    if ops.executing_eagerly_outside_functions():
      self.assertIn('tf.__operators__.getitem', (
          x.name for x in model.layers))
      # TODO(b/161925288): Fix the bug then uncomment:
      # self.assertNotIn('tf.strided_slice', (
      #     x.name for x in model.layers))
    self.assertAllEqual(model(args), expected)
    self.assertAllEqual(model.predict(args, batch_size=batch_size), expected)

    # Make sure it can be successfully saved and loaded
    config = model.get_config()
    model = keras.Model.from_config(config)

    self.assertAllEqual(model(args), expected)
    self.assertAllEqual(model.predict(args, batch_size=batch_size), expected)

  def test_getitem_slice_with_stop_only(self):
    if not context.executing_eagerly():
      self.skipTest('Complex slicing like this fails in v1')
    inp = keras.Input(shape=(8,))
    slice_stop = keras.Input(shape=(), dtype='int32')

    out = inp[:slice_stop[0]]
    model = keras.Model(
        inputs=[inp, slice_stop],
        outputs=out)
    model.compile(
        adam.Adam(0.001),
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    batch_size = 7
    stop = 6
    x = array_ops.stack([
        math_ops.range(8) for _ in range(batch_size)])
    args = [x, constant_op.constant(stop, shape=(batch_size,))]
    expected = x[:stop]

    if ops.executing_eagerly_outside_functions():
      self.assertIn('tf.__operators__.getitem', (
          x.name for x in model.layers))
      self.assertNotIn('tf.strided_slice', (
          x.name for x in model.layers))
    self.assertAllEqual(model(args), expected)
    self.assertAllEqual(model.predict(args, batch_size=batch_size), expected)

    # Make sure it can be successfully saved and loaded
    config = model.get_config()
    model = keras.Model.from_config(config)

    self.assertAllEqual(model(args), expected)
    self.assertAllEqual(model.predict(args, batch_size=batch_size), expected)

  def test_getitem_slice_with_stop_and_ellipsis_only(self):
    if not context.executing_eagerly():
      self.skipTest('Complex slicing like this fails in v1')
    inp = keras.Input(shape=(8,))
    slice_stop = keras.Input(shape=(), dtype='int32')

    out = inp[..., :slice_stop[0]]
    model = keras.Model(
        inputs=[inp, slice_stop],
        outputs=out)
    model.compile(
        adam.Adam(0.001),
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    batch_size = 7
    stop = 6
    x = array_ops.stack([
        math_ops.range(8) for _ in range(batch_size)])
    args = [x, constant_op.constant(stop, shape=(batch_size,))]
    expected = array_ops.stack([
        math_ops.range(8)[:stop] for _ in range(batch_size)])

    if ops.executing_eagerly_outside_functions():
      self.assertIn('tf.__operators__.getitem', (
          x.name for x in model.layers))
      self.assertNotIn('tf.strided_slice', (
          x.name for x in model.layers))
    self.assertAllEqual(model(args), expected)
    self.assertAllEqual(model.predict(args, batch_size=batch_size), expected)

    # Make sure it can be successfully saved and loaded
    config = model.get_config()
    model = keras.Model.from_config(config)

    self.assertAllEqual(model(args), expected)
    self.assertAllEqual(model.predict(args, batch_size=batch_size), expected)

  def test_getitem_complex_slicing(self):
    if not context.executing_eagerly():
      self.skipTest('Complex slicing like this fails in v1')
    inp = keras.Input(shape=(4, 3, 8))
    first_dim = keras.Input(shape=(), dtype='int32')
    slice_start = keras.Input(shape=(), dtype='int32')
    slice_stop = keras.Input(shape=(), dtype='int32')
    slice_stride = keras.Input(shape=(), dtype='int32')

    out = inp[..., first_dim[0], slice_start[0]:slice_stop[0]:slice_stride[0]]
    model = keras.Model(
        inputs=[inp, first_dim, slice_start, slice_stop, slice_stride],
        outputs=out)
    model.compile(
        adam.Adam(0.001),
        'mse',
        run_eagerly=testing_utils.should_run_eagerly())
    batch_size = 7
    start = 1
    stop = 6
    step = 2
    x = array_ops.stack([array_ops.stack([array_ops.stack([
        math_ops.range(8)
        for _ in range(3)]) for _ in range(4)]) for _ in range(batch_size)])
    args = [x,
            constant_op.constant(0, shape=(batch_size,)),
            constant_op.constant(start, shape=(batch_size,)),
            constant_op.constant(stop, shape=(batch_size,)),
            constant_op.constant(step, shape=(batch_size,))]
    # Slice the innermost dim. only grab one index from the second-to-innermost
    # dim, removing that dim from the shape.
    expected = array_ops.stack([array_ops.stack([
        math_ops.range(8)[start:stop:step]
        for _ in range(4)]) for _ in range(batch_size)])

    if ops.executing_eagerly_outside_functions():
      self.assertIn('tf.__operators__.getitem', (
          x.name for x in model.layers))
      self.assertNotIn('tf.strided_slice', (
          x.name for x in model.layers))
    self.assertAllEqual(model(args), expected)
    self.assertAllEqual(model.predict(args, batch_size=batch_size), expected)

    # Make sure it can be successfully saved and loaded
    config = model.get_config()
    model = keras.Model.from_config(config)

    self.assertAllEqual(model(args), expected)
    self.assertAllEqual(model.predict(args, batch_size=batch_size), expected)

  def test_left_hand_numpy_multiplication(self):
    x = np.asarray([3.0])
    inputs = keras.Input(shape=(4,))
    outputs = x * inputs
    model = keras.Model(inputs, outputs)
    ones = array_ops.ones((5, 4), dtype='float32')
    self.assertAllEqual(model(ones), 3.0 * ones)

  def test_numerical_correctness_simple(self):
    x = ops.convert_to_tensor_v2_with_dispatch([[-1., 0., -2., 1.]])
    inputs = keras.Input(shape=(4,))
    outputs = gen_nn_ops.relu(inputs)
    model = keras.Model(inputs, outputs)
    y = self.evaluate(model(x))
    self.assertAllClose(y, [[0., 0., 0., 1.]])

  def test_numerical_correctness_with_attrs(self):
    x = ops.convert_to_tensor_v2_with_dispatch([[1.5, 1.5], [2.5, 3.5]])
    inputs = keras.Input(shape=(2,))
    outputs = math_ops.reduce_mean(inputs, axis=1)
    model = keras.Model(inputs, outputs)
    y = self.evaluate(model(x))
    self.assertAllClose(y, [1.5, 3.])

  def test_numerical_correctness_serialization(self):
    x = ops.convert_to_tensor_v2_with_dispatch([[-1., 0., -2., 1.]])
    inputs = keras.Input(shape=(4,))
    outputs = gen_nn_ops.relu(inputs)
    model1 = keras.Model(inputs, outputs)
    y1 = self.evaluate(model1(x))
    model2 = keras.Model.from_config(model1.get_config())
    y2 = self.evaluate(model2(x))
    self.assertAllClose(y1, y2)

  def test_gradient_tape_in_function(self):
    z = keras.Input((1,))
    x = math_ops.matmul(z, constant_op.constant(2.0, shape=(1, 1)))
    x = math_ops.reduce_mean(x, axis=0, keepdims=True)
    h = gen_nn_ops.relu(x)
    m = keras.Model(z, h)

    @def_function.function()
    def f(x):
      with backprop.GradientTape() as t:
        t.watch(x)
        z = m(x ** 2)
      grads = t.gradient(z, x)
      return grads

    self.assertAllEqual(f(constant_op.constant(10.0, shape=(1, 1))),
                        constant_op.constant(40.0, shape=(1, 1)))

    f = def_function.function(f)

    self.assertAllEqual(f(constant_op.constant(10.0, shape=(1, 1))),
                        constant_op.constant(40.0, shape=(1, 1)))

  def test_no_tracking(self):
    if not context.executing_eagerly():
      x = constant_op.constant(1.0, shape=(10, 10))
      keras.layers.Dense(1)(x)
      self.assertTrue(x._keras_history_checked)

  def test_timing_scales_linearly(self):

    def _construct_graph_of_size(size):
      start = time.time()
      x = keras.backend.placeholder(shape=(10, 4))

      for _ in range(size):
        x = keras.layers.Dense(4)(x)
        x = gen_nn_ops.relu(x)

      end = time.time()
      return end - start

    size_50 = _construct_graph_of_size(50)
    size_500 = _construct_graph_of_size(500)

    # Check construction time grows approx. linearly with size.
    e = 3  # Fudge factor to prevent flakiness.
    self.assertLess(size_500, (10 * e) * size_50)

  def test_built(self):
    inputs = keras.Input(shape=(10,))
    outputs = gen_nn_ops.relu(inputs)
    model = keras.Model(inputs, outputs)
    model.compile('sgd', 'mse')
    for layer in model.layers:
      self.assertTrue(layer.built)
    # Test something that requires Layers to be built.
    model.summary()

  def test_json_serialization(self):
    inputs = keras.Input(shape=(4,), dtype='uint8')
    outputs = math_ops.cast(inputs, 'float32') / 4.
    model = model_config.model_from_json(keras.Model(inputs, outputs).to_json())
    self.assertAllEqual(
        self.evaluate(model(np.array([0, 64, 128, 192], np.uint8))),
        [0., 16., 32., 48.])
    model.summary()


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class InputInEagerTest(keras_parameterized.TestCase):
  """Tests ops on keras inputs in Eager runtime.

  Input returns graph/symbolic tensors in the Eager runtime (this
  happens, for example, with tensors returned from Keras layers). These
  should be routed to the graph-style branch of these ops (b/134715641)
  """

  def test_identity(self):
    x = keras.Input(shape=(1,))
    ident = array_ops.identity(x)

    # This is now a graph tensor, and should be able to continue in graphland
    self.assertIn('Identity', ident.name)

  def test_size(self):
    x = keras.Input(shape=(3,))
    self.assertAllEqual(x.get_shape().as_list(), [None, 3])
    sz = array_ops.size(x)

    # This is now a graph tensor, and should be able to continue in graphland
    self.assertIn('Size', sz.name)


if __name__ == '__main__':
  test.main()
