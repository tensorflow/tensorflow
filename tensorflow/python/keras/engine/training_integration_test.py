# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""End-to-end tests for a variety of small models."""

import collections
import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def _conv2d_filter(**kwargs):
  """Convolution with non-default strides and dilation rate is not supported."""
  return kwargs['strides'] <= 1 or kwargs['dilation_rate'] <= 1


# Scheme: (layer_class, data_shape, fuzz_dims, constructor_args, filter_fn)
#   layer_class:
#     A keras Layer class to be tested.
#   data_shape:
#     The shape of the input data. (not including batch dim)
#   fuzz_dims:
#     Dimensions which can be unspecified during model construction. For
#     instance, if data_shape is (2, 5) and fuzz_dims is (False, True), a pass
#     with model input shape of (2, None) will also be performed.
#   constructor_args:
#     An OrderedDict (to ensure consistent test names) with a key and a list
#     of values to test. Test cases will be generated for the Cartesian product
#     of all constructor args, so adding more fields can cause the drastically
#     increase the testing load.
#   filter_fn:
#     If not None, this function will be called on each set of generated
#     constructor args, and prevents generation of contradictory combinations.
#     A True return value indicates a valid test.
_LAYERS_TO_TEST = [
    (keras.layers.Dense, (1,), (False,), collections.OrderedDict([
        ('units', [1])]), None),
    (keras.layers.Activation, (2, 2), (True, True), collections.OrderedDict([
        ('activation', ['relu'])]), None),
    (keras.layers.Dropout, (16,), (False,), collections.OrderedDict([
        ('rate', [0.25])]), None),
    (keras.layers.BatchNormalization, (8, 8, 3), (True, True, False),
     collections.OrderedDict([
         ('axis', [3]),
         ('center', [True, False]),
         ('scale', [True, False])
     ]), None),
    (keras.layers.Conv1D, (8, 8), (False, False), collections.OrderedDict([
        ('filters', [1]),
        ('kernel_size', [1, 3]),
        ('strides', [1, 2]),
        ('padding', ['valid', 'same']),
        ('use_bias', [True]),
        ('kernel_regularizer', ['l2']),
        ('data_format', ['channels_last'])
    ]), None),
    (keras.layers.Conv2D, (8, 8, 3), (True, True, False),
     collections.OrderedDict([
         ('filters', [1]),
         ('kernel_size', [1, 3]),
         ('strides', [1, 2]),
         ('padding', ['valid', 'same']),
         ('use_bias', [True, False]),
         ('kernel_regularizer', ['l2']),
         ('dilation_rate', [1, 2]),
         ('data_format', ['channels_last'])
     ]), _conv2d_filter),
    (keras.layers.LSTM, (4, 4), (False, False), collections.OrderedDict([
        ('units', [1]),
        ('kernel_regularizer', ['l2']),
        ('dropout', [0, 0.5]),
        ('stateful', [True, False]),
        ('unroll', [True, False]),
        ('return_sequences', [True, False])
    ]), None),
]


def _gather_test_cases():
  cases = []
  for layer_type, inp_shape, fuzz_dims, arg_dict, filter_fn in _LAYERS_TO_TEST:
    arg_combinations = [[(k, i) for i in v] for k, v in arg_dict.items()]  # pylint: disable=g-complex-comprehension
    for arguments in itertools.product(*arg_combinations):
      layer_kwargs = {k: v for k, v in arguments}
      if filter_fn is not None and not filter_fn(**layer_kwargs):
        continue

      name = '_{}_{}'.format(layer_type.__name__,
                             '_'.join('{}_{}'.format(*i) for i in arguments))
      cases.append((name, layer_type, inp_shape, fuzz_dims, layer_kwargs))
  return cases


OUTPUT_TEST_CASES = _gather_test_cases()


class CoreLayerIntegrationTest(keras_parameterized.TestCase):
  """Test that layers and models produce the correct tensor types."""

  # In v1 graph there are only symbolic tensors.
  @keras_parameterized.run_all_keras_modes(always_skip_v1=True)
  @parameterized.named_parameters(*OUTPUT_TEST_CASES)
  def test_layer_output_type(self, layer_to_test, input_shape, _, layer_kwargs):
    layer = layer_to_test(**layer_kwargs)

    input_data = np.ones(shape=(2,) + input_shape, dtype=np.float32)
    layer_result = layer(input_data)

    inp = keras.layers.Input(shape=input_shape, batch_size=2)
    model = keras.models.Model(inp, layer_to_test(**layer_kwargs)(inp))
    model_result = model(input_data)

    for x in [layer_result, model_result]:
      if not isinstance(x, ops.Tensor):
        raise ValueError('Tensor or EagerTensor expected, got type {}'
                         .format(type(x)))

      if isinstance(x, ops.EagerTensor) != context.executing_eagerly():
        expected_type = (ops.EagerTensor if context.executing_eagerly()
                         else ops.Tensor)
        raise ValueError('Expected type {}, got type {}'
                         .format(expected_type, type(x)))

  def _run_fit_eval_predict(self, layer_to_test, input_shape, data_shape,
                            layer_kwargs):
    batch_size = 2
    run_eagerly = testing_utils.should_run_eagerly()

    def map_fn(_):
      x = keras.backend.random_uniform(shape=data_shape)
      y = keras.backend.random_uniform(shape=(1,))
      return x, y

    dataset = dataset_ops.DatasetV2.range(4).map(map_fn).batch(batch_size)

    inp = keras.layers.Input(shape=input_shape, batch_size=batch_size)
    layer = layer_to_test(**layer_kwargs)(inp)

    # Condense the output down to a single scalar.
    layer = keras.layers.Flatten()(layer)
    layer = keras.layers.Lambda(
        lambda x: math_ops.reduce_mean(x, keepdims=True))(layer)
    layer = keras.layers.Dense(1, activation=None)(layer)
    model = keras.models.Model(inp, layer)

    model.compile(loss='mse', optimizer='sgd', run_eagerly=run_eagerly)
    model.fit(dataset, verbose=2, epochs=2)

    model.compile(loss='mse', optimizer='sgd', run_eagerly=run_eagerly)
    model.fit(dataset.repeat(2), verbose=2, epochs=2, steps_per_epoch=2)

    eval_dataset = dataset_ops.DatasetV2.range(4).map(map_fn).batch(batch_size)
    model.evaluate(eval_dataset, verbose=2)

    def pred_map_fn(_):
      return keras.backend.random_uniform(shape=data_shape)

    pred_dataset = dataset_ops.DatasetV2.range(4)
    pred_dataset = pred_dataset.map(pred_map_fn).batch(batch_size)
    model.predict(pred_dataset, verbose=2)

  @keras_parameterized.run_all_keras_modes(always_skip_v1=False)
  @parameterized.named_parameters(*OUTPUT_TEST_CASES)
  def test_model_loops(self, layer_to_test, input_shape, fuzz_dims,
                       layer_kwargs):
    self._run_fit_eval_predict(layer_to_test, input_shape,
                               input_shape, layer_kwargs)

    if any(fuzz_dims):
      fuzzed_shape = []
      for dim, should_fuzz in zip(input_shape, fuzz_dims):
        fuzzed_shape.append(None if should_fuzz else dim)

      self._run_fit_eval_predict(layer_to_test, fuzzed_shape,
                                 input_shape, layer_kwargs)


if __name__ == '__main__':
  test.main()
