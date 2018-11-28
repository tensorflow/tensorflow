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
"""Tests for locally-connected layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test
from tensorflow.python.training.rmsprop import RMSPropOptimizer


class LocallyConnected1DLayersTest(test.TestCase):
  # TODO(fchollet): investigate why LocallyConnected1D
  # fails inside a graph function in an eager context (fails with error
  # "Incompatible shapes between op input and calculated input gradient").

  def test_locallyconnected_1d(self):
    with self.cached_session():
      num_samples = 2
      num_steps = 8
      input_dim = 5
      filter_length = 3
      filters = 4

      for padding in ['valid', 'same']:
        for strides in [1]:
          if padding == 'same' and strides != 1:
            continue
          for data_format in ['channels_first', 'channels_last']:
            for implementation in [1, 2]:
              kwargs = {
                  'filters': filters,
                  'kernel_size': filter_length,
                  'padding': padding,
                  'strides': strides,
                  'data_format': data_format,
                  'implementation': implementation
              }

              if padding == 'same' and implementation == 1:
                self.assertRaises(ValueError,
                                  keras.layers.LocallyConnected1D,
                                  **kwargs)
              else:
                testing_utils.layer_test(
                    keras.layers.LocallyConnected1D,
                    kwargs=kwargs,
                    input_shape=(num_samples, num_steps, input_dim))

  def test_locallyconnected_1d_regularization(self):
    num_samples = 2
    num_steps = 8
    input_dim = 5
    filter_length = 3
    filters = 4
    for data_format in ['channels_first', 'channels_last']:
      for padding in ['valid', 'same']:
        for implementation in [1, 2]:
          kwargs = {
              'filters': filters,
              'kernel_size': filter_length,
              'kernel_regularizer': 'l2',
              'bias_regularizer': 'l2',
              'activity_regularizer': 'l2',
              'data_format': data_format,
              'implementation': implementation,
              'padding': padding
          }

          if padding == 'same' and implementation == 1:
            self.assertRaises(ValueError,
                              keras.layers.LocallyConnected1D,
                              **kwargs)
          else:
            with self.cached_session():
              layer = keras.layers.LocallyConnected1D(**kwargs)
              layer.build((num_samples, num_steps, input_dim))
              self.assertEqual(len(layer.losses), 2)
              layer(
                  keras.backend.variable(np.ones((num_samples,
                                                  num_steps,
                                                  input_dim))))
              self.assertEqual(len(layer.losses), 3)

            k_constraint = keras.constraints.max_norm(0.01)
            b_constraint = keras.constraints.max_norm(0.01)
            kwargs = {
                'filters': filters,
                'kernel_size': filter_length,
                'kernel_constraint': k_constraint,
                'bias_constraint': b_constraint,
            }
            with self.cached_session():
              layer = keras.layers.LocallyConnected1D(**kwargs)
              layer.build((num_samples, num_steps, input_dim))
              self.assertEqual(layer.kernel.constraint, k_constraint)
              self.assertEqual(layer.bias.constraint, b_constraint)


class LocallyConnected2DLayersTest(test.TestCase):
  # TODO(fchollet): investigate why LocallyConnected2D
  # fails inside a graph function in an eager context (fails with error
  # "Incompatible shapes between op input and calculated input gradient").

  def test_locallyconnected_2d(self):
    with self.cached_session():
      num_samples = 8
      filters = 3
      stack_size = 4
      num_row = 6
      num_col = 10

      for padding in ['valid', 'same']:
        for strides in [(1, 1), (2, 2)]:
          for implementation in [1, 2]:
            if padding == 'same' and strides != (1, 1):
              continue

            kwargs = {
                'filters': filters,
                'kernel_size': 3,
                'padding': padding,
                'kernel_regularizer': 'l2',
                'bias_regularizer': 'l2',
                'strides': strides,
                'data_format': 'channels_last',
                'implementation': implementation
            }

            if padding == 'same' and implementation == 1:
              self.assertRaises(ValueError,
                                keras.layers.LocallyConnected2D,
                                **kwargs)
            else:
              testing_utils.layer_test(
                  keras.layers.LocallyConnected2D,
                  kwargs=kwargs,
                  input_shape=(num_samples, num_row, num_col, stack_size))

  def test_locallyconnected_2d_channels_first(self):
    with self.cached_session():
      num_samples = 8
      filters = 3
      stack_size = 4
      num_row = 6
      num_col = 10

      for implementation in [1, 2]:
        for padding in ['valid', 'same']:
          kwargs = {
              'filters': filters,
              'kernel_size': 3,
              'data_format': 'channels_first',
              'implementation': implementation,
              'padding': padding
          }

          if padding == 'same' and implementation == 1:
            self.assertRaises(ValueError,
                              keras.layers.LocallyConnected2D,
                              **kwargs)
          else:
            testing_utils.layer_test(
                keras.layers.LocallyConnected2D,
                kwargs=kwargs,
                input_shape=(num_samples, num_row, num_col, stack_size))

  def test_locallyconnected_2d_regularization(self):
    num_samples = 2
    filters = 3
    stack_size = 4
    num_row = 6
    num_col = 7
    for implementation in [1, 2]:
      for padding in ['valid', 'same']:
        kwargs = {
            'filters': filters,
            'kernel_size': 3,
            'kernel_regularizer': 'l2',
            'bias_regularizer': 'l2',
            'activity_regularizer': 'l2',
            'implementation': implementation,
            'padding': padding
        }

        if padding == 'same' and implementation == 1:
          self.assertRaises(ValueError,
                            keras.layers.LocallyConnected2D,
                            **kwargs)
        else:
          with self.cached_session():
            layer = keras.layers.LocallyConnected2D(**kwargs)
            layer.build((num_samples, num_row, num_col, stack_size))
            self.assertEqual(len(layer.losses), 2)
            layer(
                keras.backend.variable(
                    np.ones((num_samples, num_row, num_col, stack_size))))
            self.assertEqual(len(layer.losses), 3)

          k_constraint = keras.constraints.max_norm(0.01)
          b_constraint = keras.constraints.max_norm(0.01)
          kwargs = {
              'filters': filters,
              'kernel_size': 3,
              'kernel_constraint': k_constraint,
              'bias_constraint': b_constraint,
          }
          with self.cached_session():
            layer = keras.layers.LocallyConnected2D(**kwargs)
            layer.build((num_samples, num_row, num_col, stack_size))
            self.assertEqual(layer.kernel.constraint, k_constraint)
            self.assertEqual(layer.bias.constraint, b_constraint)


class LocallyConnectedImplementationModeTest(test.TestCase):

  def test_locallyconnected_implementation(self):
    with self.cached_session():
      num_samples = 4
      num_classes = 3
      num_epochs = 2

      np.random.seed(1)
      targets = np.random.randint(0, num_classes, (num_samples,))

      for width in [1, 6]:
        for height in [7]:
          for filters in [2]:
            for data_format in ['channels_first', 'channels_last']:
              inputs = get_inputs(
                  data_format, filters, height, num_samples, width)

              for kernel_x in [(3,)]:
                for kernel_y in [()] if width == 1 else [(2,)]:
                  for stride_x in [(1,)]:
                    for stride_y in [()] if width == 1 else [(3,)]:
                      for layers in [2]:
                        kwargs = {
                            'layers': layers,
                            'filters': filters,
                            'kernel_size': kernel_x + kernel_y,
                            'strides': stride_x + stride_y,
                            'data_format': data_format,
                            'num_classes': num_classes
                        }
                        model_1 = get_model(implementation=1, **kwargs)
                        model_2 = get_model(implementation=2, **kwargs)

                        # Build models.
                        model_1.train_on_batch(inputs, targets)
                        model_2.train_on_batch(inputs, targets)

                        # Copy weights.
                        copy_model_weights(model_2, model_1)

                        # Compare outputs at initialization.
                        out_1 = model_1.call(inputs)
                        out_2 = model_2.call(inputs)
                        self.assertAllCloseAccordingToType(out_1, out_2,
                                                           rtol=1e-5, atol=1e-5)

                        # Train.
                        model_1.fit(x=inputs,
                                    y=targets,
                                    epochs=num_epochs,
                                    batch_size=num_samples)
                        model_2.fit(x=inputs,
                                    y=targets,
                                    epochs=num_epochs,
                                    batch_size=num_samples)

                        # Compare outputs after a few training steps.
                        out_1 = model_1.call(inputs)
                        out_2 = model_2.call(inputs)
                        self.assertAllCloseAccordingToType(out_1, out_2,
                                                           atol=1e-4)

  @tf_test_util.run_in_graph_and_eager_modes
  def test_make_2d(self):
    input_shapes = [
        (0,),
        (0, 0),
        (1,),
        (2,),
        (3,),
        (1, 0),
        (0, 3),
        (1, 1),
        (1, 2),
        (3, 1),
        (2, 2),
        (3, 3),
        (1, 0, 1),
        (5, 2, 3),
        (3, 5, 6, 7, 0),
        (3, 2, 2, 4, 4),
        (1, 2, 3, 4, 7, 2),
    ]
    np.random.seed(1)

    for input_shape in input_shapes:
      inputs = np.random.normal(0, 1, input_shape)
      inputs_tf = keras.backend.variable(inputs)

      split_dim = np.random.randint(0, inputs.ndim + 1)
      shape_2d = (int(np.prod(inputs.shape[:split_dim])),
                  int(np.prod(inputs.shape[split_dim:])))
      inputs_2d = np.reshape(inputs, shape_2d)

      inputs_2d_tf = keras.layers.local.make_2d(inputs_tf, split_dim)
      inputs_2d_tf = keras.backend.get_value(inputs_2d_tf)

      self.assertAllCloseAccordingToType(inputs_2d, inputs_2d_tf)


def get_inputs(data_format, filters, height, num_samples, width):
  if data_format == 'channels_first':
    if width == 1:
      input_shape = (filters, height)
    else:
      input_shape = (filters, height, width)

  elif data_format == 'channels_last':
    if width == 1:
      input_shape = (height, filters)
    else:
      input_shape = (height, width, filters)

  else:
    raise NotImplementedError(data_format)

  inputs = np.random.normal(0, 1,
                            (num_samples,) + input_shape).astype(np.float32)
  return inputs


def xent(y_true, y_pred):
  y_true = keras.backend.cast(
      keras.backend.reshape(y_true, (-1,)),
      keras.backend.dtypes_module.int32)

  return keras.backend.nn.sparse_softmax_cross_entropy_with_logits(
      labels=y_true,
      logits=y_pred)


def get_model(implementation,
              filters,
              kernel_size,
              strides,
              layers,
              num_classes,
              data_format):
  model = keras.Sequential()

  if len(kernel_size) == 1:
    lc_layer = keras.layers.LocallyConnected1D
  elif len(kernel_size) == 2:
    lc_layer = keras.layers.LocallyConnected2D
  else:
    raise NotImplementedError(kernel_size)

  for _ in range(layers):
    model.add(lc_layer(
        padding='valid',
        kernel_initializer=keras.initializers.random_normal(),
        bias_initializer=keras.initializers.random_normal(),
        filters=filters,
        strides=strides,
        kernel_size=kernel_size,
        activation=keras.activations.relu,
        data_format=data_format,
        implementation=implementation))

  model.add(keras.layers.Flatten())
  model.add(keras.layers.Dense(num_classes))
  model.compile(
      optimizer=RMSPropOptimizer(0.01),
      metrics=[keras.metrics.categorical_accuracy],
      loss=xent
  )
  return model


def copy_lc_weights(lc_layer_2_from, lc_layer_1_to):
  lc_2_kernel, lc_2_bias = lc_layer_2_from.weights
  lc_2_kernel_masked = lc_2_kernel * lc_layer_2_from.kernel_mask

  data_format = lc_layer_2_from.data_format

  if data_format == 'channels_first':
    if isinstance(lc_layer_2_from, keras.layers.LocallyConnected1D):
      permutation = (3, 0, 1, 2)
    elif isinstance(lc_layer_2_from, keras.layers.LocallyConnected2D):
      permutation = (4, 5, 0, 1, 2, 3)
    else:
      raise NotImplementedError(lc_layer_2_from)

  elif data_format == 'channels_last':
    if isinstance(lc_layer_2_from, keras.layers.LocallyConnected1D):
      permutation = (2, 0, 1, 3)
    elif isinstance(lc_layer_2_from, keras.layers.LocallyConnected2D):
      permutation = (3, 4, 0, 1, 2, 5)
    else:
      raise NotImplementedError(lc_layer_2_from)

  else:
    raise NotImplementedError(data_format)

  lc_2_kernel_masked = keras.backend.permute_dimensions(
      lc_2_kernel_masked, permutation)

  lc_2_kernel_mask = keras.backend.math_ops.not_equal(
      lc_2_kernel_masked, 0)
  lc_2_kernel_flat = keras.backend.array_ops.boolean_mask(
      lc_2_kernel_masked, lc_2_kernel_mask)
  lc_2_kernel_reshaped = keras.backend.reshape(lc_2_kernel_flat,
                                               lc_layer_1_to.kernel.shape)

  lc_2_kernel_reshaped = keras.backend.get_value(lc_2_kernel_reshaped)
  lc_2_bias = keras.backend.get_value(lc_2_bias)

  lc_layer_1_to.set_weights([lc_2_kernel_reshaped, lc_2_bias])


def copy_model_weights(model_2_from, model_1_to):
  for l in range(len(model_2_from.layers)):
    layer_2_from = model_2_from.layers[l]
    layer_1_to = model_1_to.layers[l]

    if isinstance(layer_2_from, (keras.layers.LocallyConnected2D,
                                 keras.layers.LocallyConnected1D)):
      copy_lc_weights(layer_2_from, layer_1_to)

    elif isinstance(layer_2_from, keras.layers.Dense):
      weights_2, bias_2 = layer_2_from.weights
      weights_2 = keras.backend.get_value(weights_2)
      bias_2 = keras.backend.get_value(bias_2)
      layer_1_to.set_weights([weights_2, bias_2])

    else:
      continue


if __name__ == '__main__':
  test.main()
