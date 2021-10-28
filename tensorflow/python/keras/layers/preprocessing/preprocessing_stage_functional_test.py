# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Functional preprocessing stage tests."""
# pylint: disable=g-classes-have-attributes

import time
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers import convolutional
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import merge
from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from tensorflow.python.keras.layers.preprocessing import normalization
from tensorflow.python.keras.layers.preprocessing import preprocessing_stage
from tensorflow.python.keras.layers.preprocessing import preprocessing_test_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class PL(base_preprocessing_layer.PreprocessingLayer):

  def __init__(self, **kwargs):
    self.adapt_time = None
    self.adapt_count = 0
    super(PL, self).__init__(**kwargs)

  def adapt(self, data, reset_state=True):
    self.adapt_time = time.time()
    self.adapt_count += 1

  def call(self, inputs):
    return inputs + 1


class PLMerge(PL):

  def call(self, inputs):
    return inputs[0] + inputs[1]


class PLSplit(PL):

  def call(self, inputs):
    return inputs + 1, inputs - 1


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class PreprocessingStageTest(keras_parameterized.TestCase,
                             preprocessing_test_utils.PreprocessingLayerTest):

  def test_adapt_preprocessing_stage_with_single_input_output(self):

    x = Input(shape=(3,))

    l0 = PL()
    y = l0(x)

    l1 = PL()
    z = l1(y)

    stage = preprocessing_stage.FunctionalPreprocessingStage(x, z)
    stage.compile()

    # Test with NumPy array
    one_array = np.ones((4, 3), dtype='float32')
    stage.adapt(one_array)
    self.assertEqual(l0.adapt_count, 1)
    self.assertEqual(l1.adapt_count, 1)
    self.assertLessEqual(l0.adapt_time, l1.adapt_time)

    # Check call
    z = stage(array_ops.ones((4, 3), dtype='float32'))
    self.assertAllClose(z, np.ones((4, 3), dtype='float32') + 2.)

    # Test with dataset
    adapt_data = dataset_ops.Dataset.from_tensor_slices(one_array)
    adapt_data = adapt_data.batch(2)  # 5 batches of 2 samples

    stage.adapt(adapt_data)
    self.assertEqual(l0.adapt_count, 2)
    self.assertEqual(l1.adapt_count, 2)
    self.assertLessEqual(l0.adapt_time, l1.adapt_time)

    # Test error with bad data
    with self.assertRaisesRegex(ValueError, 'requires a '):
      stage.adapt(None)

    # Disallow calling fit
    with self.assertRaisesRegex(ValueError, 'Preprocessing stage'):
      stage.fit(None)

  def test_adapt_preprocessing_stage_with_list_input(self):

    x0 = Input(shape=(3,))
    x1 = Input(shape=(3,))
    x2 = Input(shape=(3,))

    l0 = PLMerge()
    y = l0([x0, x1])

    l1 = PLMerge()
    y = l1([y, x2])

    l2 = PLSplit()
    z, y = l2(y)

    stage = preprocessing_stage.FunctionalPreprocessingStage([x0, x1, x2],
                                                             [y, z])
    stage.compile()

    # Test with NumPy array
    one_array = np.ones((4, 3), dtype='float32')
    stage.adapt([one_array, one_array, one_array])
    self.assertEqual(l0.adapt_count, 1)
    self.assertEqual(l1.adapt_count, 1)
    self.assertEqual(l2.adapt_count, 1)
    self.assertLessEqual(l0.adapt_time, l1.adapt_time)
    self.assertLessEqual(l1.adapt_time, l2.adapt_time)

    # Check call
    y, z = stage([
        array_ops.ones((4, 3), dtype='float32'),
        array_ops.ones((4, 3), dtype='float32'),
        array_ops.ones((4, 3), dtype='float32')
    ])
    self.assertAllClose(y, np.ones((4, 3), dtype='float32') + 1.)
    self.assertAllClose(z, np.ones((4, 3), dtype='float32') + 3.)

    # Test with dataset
    adapt_data = dataset_ops.Dataset.from_tensor_slices(
        (one_array, one_array, one_array))
    adapt_data = adapt_data.batch(2)  # 5 batches of 2 samples

    stage.adapt(adapt_data)
    self.assertEqual(l0.adapt_count, 2)
    self.assertEqual(l1.adapt_count, 2)
    self.assertEqual(l2.adapt_count, 2)
    self.assertLessEqual(l0.adapt_time, l1.adapt_time)
    self.assertLessEqual(l1.adapt_time, l2.adapt_time)

    # Test error with bad data
    with self.assertRaisesRegex(ValueError, 'requires a '):
      stage.adapt(None)

  def test_adapt_preprocessing_stage_with_dict_input(self):
    x0 = Input(shape=(3,), name='x0')
    x1 = Input(shape=(4,), name='x1')
    x2 = Input(shape=(3, 5), name='x2')

    # dimension will mismatch if x1 incorrectly placed.
    x1_sum = core.Lambda(
        lambda x: math_ops.reduce_sum(x, axis=-1, keepdims=True))(
            x1)
    x2_sum = core.Lambda(lambda x: math_ops.reduce_sum(x, axis=-1))(x2)

    l0 = PLMerge()
    y = l0([x0, x1_sum])

    l1 = PLMerge()
    y = l1([y, x2_sum])

    l2 = PLSplit()
    z, y = l2(y)
    stage = preprocessing_stage.FunctionalPreprocessingStage(
        {
            'x2': x2,
            'x0': x0,
            'x1': x1
        }, [y, z])
    stage.compile()

    # Test with dict of NumPy array
    one_array0 = np.ones((4, 3), dtype='float32')
    one_array1 = np.ones((4, 4), dtype='float32')
    one_array2 = np.ones((4, 3, 5), dtype='float32')
    adapt_data = {'x1': one_array1, 'x0': one_array0, 'x2': one_array2}
    stage.adapt(adapt_data)
    self.assertEqual(l0.adapt_count, 1)
    self.assertEqual(l1.adapt_count, 1)
    self.assertEqual(l2.adapt_count, 1)
    self.assertLessEqual(l0.adapt_time, l1.adapt_time)
    self.assertLessEqual(l1.adapt_time, l2.adapt_time)

    # Check call
    y, z = stage({
        'x1': array_ops.constant(one_array1),
        'x2': array_ops.constant(one_array2),
        'x0': array_ops.constant(one_array0)
    })
    self.assertAllClose(y, np.zeros((4, 3), dtype='float32') + 9.)
    self.assertAllClose(z, np.zeros((4, 3), dtype='float32') + 11.)

    # Test with list of NumPy array
    adapt_data = [one_array0, one_array1, one_array2]
    stage.adapt(adapt_data)
    self.assertEqual(l0.adapt_count, 2)
    self.assertEqual(l1.adapt_count, 2)
    self.assertEqual(l2.adapt_count, 2)
    self.assertLessEqual(l0.adapt_time, l1.adapt_time)
    self.assertLessEqual(l1.adapt_time, l2.adapt_time)

    # Test with flattened dataset
    adapt_data = dataset_ops.Dataset.from_tensor_slices(
        (one_array0, one_array1, one_array2))
    adapt_data = adapt_data.batch(2)  # 5 batches of 2 samples

    stage.adapt(adapt_data)
    self.assertEqual(l0.adapt_count, 3)
    self.assertEqual(l1.adapt_count, 3)
    self.assertEqual(l2.adapt_count, 3)
    self.assertLessEqual(l0.adapt_time, l1.adapt_time)
    self.assertLessEqual(l1.adapt_time, l2.adapt_time)

    # Test with dataset in dict shape
    adapt_data = dataset_ops.Dataset.from_tensor_slices({
        'x0': one_array0,
        'x2': one_array2,
        'x1': one_array1
    })
    adapt_data = adapt_data.batch(2)  # 5 batches of 2 samples
    stage.adapt(adapt_data)
    self.assertEqual(l0.adapt_count, 4)
    self.assertEqual(l1.adapt_count, 4)
    self.assertEqual(l2.adapt_count, 4)
    self.assertLessEqual(l0.adapt_time, l1.adapt_time)
    self.assertLessEqual(l1.adapt_time, l2.adapt_time)

    # Test error with bad data
    with self.assertRaisesRegex(ValueError, 'requires a '):
      stage.adapt(None)

  def test_adapt_preprocessing_stage_with_dict_output(self):
    x = Input(shape=(3,), name='x')

    l0 = PLSplit()
    y0, y1 = l0(x)

    l1 = PLSplit()
    z0, z1 = l1(y0)
    stage = preprocessing_stage.FunctionalPreprocessingStage({'x': x}, {
        'y1': y1,
        'z1': z1,
        'y0': y0,
        'z0': z0
    })
    stage.compile()

    # Test with NumPy array
    one_array = np.ones((4, 3), dtype='float32')
    adapt_data = {'x': one_array}
    stage.adapt(adapt_data)
    self.assertEqual(l0.adapt_count, 1)
    self.assertEqual(l1.adapt_count, 1)
    self.assertLessEqual(l0.adapt_time, l1.adapt_time)

    # Check call
    outputs = stage({'x': array_ops.constant(one_array)})
    self.assertEqual(set(outputs.keys()), {'y0', 'y1', 'z0', 'z1'})
    self.assertAllClose(outputs['y0'], np.ones((4, 3), dtype='float32') + 1.)
    self.assertAllClose(outputs['y1'], np.ones((4, 3), dtype='float32') - 1.)
    self.assertAllClose(outputs['z0'], np.ones((4, 3), dtype='float32') + 2.)
    self.assertAllClose(outputs['z1'], np.ones((4, 3), dtype='float32'))

  def test_preprocessing_stage_with_nested_input(self):
    # Test with NumPy array
    x0 = Input(shape=(3,))
    x1 = Input(shape=(3,))
    x2 = Input(shape=(3,))

    l0 = PLMerge()
    y = l0([x0, x1])

    l1 = PLMerge()
    y = l1([y, x2])

    l2 = PLSplit()
    z, y = l2(y)

    stage = preprocessing_stage.FunctionalPreprocessingStage([x0, [x1, x2]],
                                                             [y, z])
    stage.compile()
    one_array = np.ones((4, 3), dtype='float32')
    stage.adapt([one_array, [one_array, one_array]])
    self.assertEqual(l0.adapt_count, 1)
    self.assertEqual(l1.adapt_count, 1)
    self.assertEqual(l2.adapt_count, 1)
    self.assertLessEqual(l0.adapt_time, l1.adapt_time)
    self.assertLessEqual(l1.adapt_time, l2.adapt_time)

    # Check call
    y, z = stage([
        array_ops.ones((4, 3), dtype='float32'),
        [
            array_ops.ones((4, 3), dtype='float32'),
            array_ops.ones((4, 3), dtype='float32')
        ]
    ])
    self.assertAllClose(y, np.ones((4, 3), dtype='float32') + 1.)
    self.assertAllClose(z, np.ones((4, 3), dtype='float32') + 3.)

    # Test with dataset
    adapt_data = dataset_ops.Dataset.from_tensor_slices(
        (one_array, (one_array, one_array)))
    adapt_data = adapt_data.batch(2)  # 5 batches of 2 samples

    stage.adapt(adapt_data)
    self.assertEqual(l0.adapt_count, 2)
    self.assertEqual(l1.adapt_count, 2)
    self.assertEqual(l2.adapt_count, 2)
    self.assertLessEqual(l0.adapt_time, l1.adapt_time)
    self.assertLessEqual(l1.adapt_time, l2.adapt_time)

    # Test error with bad data
    with self.assertRaisesRegex(ValueError, 'requires a '):
      stage.adapt(None)

  def test_include_layers_with_dict_input(self):

    class PLMergeDict(PLMerge):

      def call(self, inputs):
        return inputs['a'] + inputs['b']

    x0 = Input(shape=(3,))
    x1 = Input(shape=(3,))

    l0 = PLMergeDict()
    y = l0({'a': x0, 'b': x1})

    l1 = PLSplit()
    z, y = l1(y)

    stage = preprocessing_stage.FunctionalPreprocessingStage([x0, x1], [y, z])
    stage.compile()

    one_array = np.ones((4, 3), dtype='float32')
    adapt_data = dataset_ops.Dataset.from_tensor_slices((one_array, one_array))
    stage.adapt(adapt_data)
    self.assertEqual(l0.adapt_count, 1)
    self.assertEqual(l1.adapt_count, 1)
    self.assertLessEqual(l0.adapt_time, l1.adapt_time)

    # Check call
    y, z = stage([
        array_ops.ones((4, 3), dtype='float32'),
        array_ops.ones((4, 3), dtype='float32')
    ])
    self.assertAllClose(y, np.ones((4, 3), dtype='float32'))
    self.assertAllClose(z, np.ones((4, 3), dtype='float32') + 2.)

  def test_include_layers_with_nested_input(self):

    class PLMergeNest(PLMerge):

      def call(self, inputs):
        a = inputs[0]
        b = inputs[1][0]
        c = inputs[1][1]
        return a + b + c

    x0 = Input(shape=(3,))
    x1 = Input(shape=(3,))
    x2 = Input(shape=(3,))

    l0 = PLMergeNest()
    y = l0([x0, [x1, x2]])

    stage = preprocessing_stage.FunctionalPreprocessingStage([x0, x1, x2], y)
    stage.compile()

    one_array = np.ones((4, 3), dtype='float32')
    adapt_data = dataset_ops.Dataset.from_tensor_slices((one_array,) * 3)
    stage.adapt(adapt_data)
    self.assertEqual(l0.adapt_count, 1)

    # Check call
    y = stage([
        array_ops.ones((4, 3), dtype='float32'),
        array_ops.ones((4, 3), dtype='float32'),
        array_ops.ones((4, 3), dtype='float32')
    ])
    self.assertAllClose(y, np.ones((4, 3), dtype='float32') + 2.)

  def test_mixing_preprocessing_and_regular_layers(self):
    x0 = Input(shape=(10, 10, 3))
    x1 = Input(shape=(10, 10, 3))
    x2 = Input(shape=(10, 10, 3))

    y0 = merge.Add()([x0, x1])
    y1 = image_preprocessing.CenterCrop(8, 8)(x2)
    y1 = convolutional.ZeroPadding2D(padding=1)(y1)

    z = merge.Add()([y0, y1])
    z = normalization.Normalization()(z)
    z = convolutional.Conv2D(4, 3)(z)

    stage = preprocessing_stage.FunctionalPreprocessingStage([x0, x1, x2], z)

    data = [
        np.ones((12, 10, 10, 3), dtype='float32'),
        np.ones((12, 10, 10, 3), dtype='float32'),
        np.ones((12, 10, 10, 3), dtype='float32')
    ]

    stage.adapt(data)
    _ = stage(data)
    stage.compile('rmsprop', 'mse')
    with self.assertRaisesRegex(ValueError, 'Preprocessing stage'):
      stage.fit(data, np.ones((12, 8, 8, 4)))

    ds_x0 = dataset_ops.Dataset.from_tensor_slices(np.ones((12, 10, 10, 3)))
    ds_x1 = dataset_ops.Dataset.from_tensor_slices(np.ones((12, 10, 10, 3)))
    ds_x2 = dataset_ops.Dataset.from_tensor_slices(np.ones((12, 10, 10, 3)))
    ds_x = dataset_ops.Dataset.zip((ds_x0, ds_x1, ds_x2))
    ds_y = dataset_ops.Dataset.from_tensor_slices(np.ones((12, 8, 8, 4)))
    dataset = dataset_ops.Dataset.zip((ds_x, ds_y)).batch(4)

    with self.assertRaisesRegex(ValueError, 'Preprocessing stage'):
      stage.fit(dataset)
    _ = stage.evaluate(data, np.ones((12, 8, 8, 4)))
    _ = stage.predict(data)


if __name__ == '__main__':
  test.main()
