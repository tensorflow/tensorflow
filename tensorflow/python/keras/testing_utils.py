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
"""Utilities for unit-testing Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import tensor_shape
from tensorflow.python.training.rmsprop import RMSPropOptimizer
from tensorflow.python.util import tf_inspect


def get_test_data(train_samples,
                  test_samples,
                  input_shape,
                  num_classes):
  """Generates test data to train a model on.

  Arguments:
    train_samples: Integer, how many training samples to generate.
    test_samples: Integer, how many test samples to generate.
    input_shape: Tuple of integers, shape of the inputs.
    num_classes: Integer, number of classes for the data and targets.

  Returns:
    A tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  num_sample = train_samples + test_samples
  templates = 2 * num_classes * np.random.random((num_classes,) + input_shape)
  y = np.random.randint(0, num_classes, size=(num_sample,))
  x = np.zeros((num_sample,) + input_shape, dtype=np.float32)
  for i in range(num_sample):
    x[i] = templates[y[i]] + np.random.normal(loc=0, scale=1., size=input_shape)
  return ((x[:train_samples], y[:train_samples]),
          (x[train_samples:], y[train_samples:]))


def layer_test(layer_cls, kwargs=None, input_shape=None, input_dtype=None,
               input_data=None, expected_output=None,
               expected_output_dtype=None):
  """Test routine for a layer with a single input and single output.

  Arguments:
    layer_cls: Layer class object.
    kwargs: Optional dictionary of keyword arguments for instantiating the
      layer.
    input_shape: Input shape tuple.
    input_dtype: Data type of the input data.
    input_data: Numpy array of input data.
    expected_output: Shape tuple for the expected shape of the output.
    expected_output_dtype: Data type expected for the output.

  Returns:
    The output data (Numpy array) returned by the layer, for additional
    checks to be done by the calling code.
  """
  if input_data is None:
    assert input_shape
    if not input_dtype:
      input_dtype = 'float32'
    input_data_shape = list(input_shape)
    for i, e in enumerate(input_data_shape):
      if e is None:
        input_data_shape[i] = np.random.randint(1, 4)
    input_data = 10 * np.random.random(input_data_shape)
    if input_dtype[:5] == 'float':
      input_data -= 0.5
    input_data = input_data.astype(input_dtype)
  elif input_shape is None:
    input_shape = input_data.shape
  if input_dtype is None:
    input_dtype = input_data.dtype
  if expected_output_dtype is None:
    expected_output_dtype = input_dtype

  # instantiation
  kwargs = kwargs or {}
  layer = layer_cls(**kwargs)

  # test get_weights , set_weights at layer level
  weights = layer.get_weights()
  layer.set_weights(weights)

  # test and instantiation from weights
  if 'weights' in tf_inspect.getargspec(layer_cls.__init__):
    kwargs['weights'] = weights
    layer = layer_cls(**kwargs)

  # test in functional API
  x = keras.layers.Input(shape=input_shape[1:], dtype=input_dtype)
  y = layer(x)
  if keras.backend.dtype(y) != expected_output_dtype:
    raise AssertionError('When testing layer %s, for input %s, found output '
                         'dtype=%s but expected to find %s.\nFull kwargs: %s' %
                         (layer_cls.__name__,
                          x,
                          keras.backend.dtype(y),
                          expected_output_dtype,
                          kwargs))
  # check shape inference
  model = keras.models.Model(x, y)
  expected_output_shape = tuple(
      layer.compute_output_shape(
          tensor_shape.TensorShape(input_shape)).as_list())
  actual_output = model.predict(input_data)
  actual_output_shape = actual_output.shape
  for expected_dim, actual_dim in zip(expected_output_shape,
                                      actual_output_shape):
    if expected_dim is not None:
      if expected_dim != actual_dim:
        raise AssertionError(
            'When testing layer %s, for input %s, found output_shape='
            '%s but expected to find %s.\nFull kwargs: %s' %
            (layer_cls.__name__,
             x,
             actual_output_shape,
             expected_output_shape,
             kwargs))
  if expected_output is not None:
    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3)

  # test serialization, weight setting at model level
  model_config = model.get_config()
  recovered_model = keras.models.Model.from_config(model_config)
  if model.weights:
    weights = model.get_weights()
    recovered_model.set_weights(weights)
    output = recovered_model.predict(input_data)
    np.testing.assert_allclose(output, actual_output, rtol=1e-3)

  # test training mode (e.g. useful for dropout tests)
  model.compile(RMSPropOptimizer(0.01), 'mse')
  model.train_on_batch(input_data, actual_output)

  # test as first layer in Sequential API
  layer_config = layer.get_config()
  layer_config['batch_input_shape'] = input_shape
  layer = layer.__class__.from_config(layer_config)

  model = keras.models.Sequential()
  model.add(layer)
  actual_output = model.predict(input_data)
  actual_output_shape = actual_output.shape
  for expected_dim, actual_dim in zip(expected_output_shape,
                                      actual_output_shape):
    if expected_dim is not None:
      if expected_dim != actual_dim:
        raise AssertionError(
            'When testing layer %s, for input %s, found output_shape='
            '%s but expected to find %s.\nFull kwargs: %s' %
            (layer_cls.__name__,
             x,
             actual_output_shape,
             expected_output_shape,
             kwargs))
  if expected_output is not None:
    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3)

  # test serialization, weight setting at model level
  model_config = model.get_config()
  recovered_model = keras.models.Sequential.from_config(model_config)
  if model.weights:
    weights = model.get_weights()
    recovered_model.set_weights(weights)
    output = recovered_model.predict(input_data)
    np.testing.assert_allclose(output, actual_output, rtol=1e-3)

  # for further checks in the caller function
  return actual_output

