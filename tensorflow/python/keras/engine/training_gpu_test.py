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
"""Tests for training routines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import combinations
from tensorflow.python.keras.engine import input_layer
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.platform import test


class TrainingGPUTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(combinations.combine(mode=['graph', 'eager']))
  def test_model_with_crossentropy_losses_channels_first(self):
    """Tests use of all crossentropy losses with `channels_first`.

    Tests `sparse_categorical_crossentropy`, `categorical_crossentropy`,
    and `binary_crossentropy`.
    Verifies that evaluate gives the same result with either `channels_first`
    or `channels_last` image_data_format.
    """
    def prepare_simple_model(input_tensor, loss_name, target):
      axis = 1 if K.image_data_format() == 'channels_first' else -1
      loss = None
      num_channels = None
      activation = None
      if loss_name == 'sparse_categorical_crossentropy':
        loss = lambda y_true, y_pred: K.sparse_categorical_crossentropy(  # pylint: disable=g-long-lambda
            y_true, y_pred, axis=axis)
        num_channels = int(np.amax(target) + 1)
        activation = 'softmax'
      elif loss_name == 'categorical_crossentropy':
        loss = lambda y_true, y_pred: K.categorical_crossentropy(  # pylint: disable=g-long-lambda
            y_true, y_pred, axis=axis)
        num_channels = target.shape[axis]
        activation = 'softmax'
      elif loss_name == 'binary_crossentropy':
        loss = lambda y_true, y_pred: K.binary_crossentropy(y_true, y_pred)  # pylint: disable=unnecessary-lambda
        num_channels = target.shape[axis]
        activation = 'sigmoid'

      predictions = Conv2D(num_channels,
                           1,
                           activation=activation,
                           kernel_initializer='ones',
                           bias_initializer='ones')(input_tensor)
      simple_model = training.Model(inputs=input_tensor, outputs=predictions)
      simple_model.compile(optimizer='rmsprop', loss=loss)
      return simple_model

    if test.is_gpu_available(cuda_only=True):
      with test_util.use_gpu():
        losses_to_test = ['sparse_categorical_crossentropy',
                          'categorical_crossentropy', 'binary_crossentropy']

        data_channels_first = np.array([[[[8., 7.1, 0.], [4.5, 2.6, 0.55],
                                          [0.9, 4.2, 11.2]]]], dtype=np.float32)
        # Labels for testing 4-class sparse_categorical_crossentropy, 4-class
        # categorical_crossentropy, and 2-class binary_crossentropy:
        labels_channels_first = [np.array([[[[0, 1, 3], [2, 1, 0], [2, 2, 1]]]], dtype=np.float32),  # pylint: disable=line-too-long
                                 np.array([[[[0, 1, 0], [0, 1, 0], [0, 0, 0]],
                                            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                                            [[0, 0, 0], [1, 0, 0], [0, 0, 1]],
                                            [[0, 0, 1], [0, 0, 0], [1, 0, 0]]]], dtype=np.float32),  # pylint: disable=line-too-long
                                 np.array([[[[0, 1, 0], [0, 1, 0], [0, 0, 1]],
                                            [[1, 0, 1], [1, 0, 1], [1, 1, 0]]]], dtype=np.float32)]  # pylint: disable=line-too-long
        # Compute one loss for each loss function in the list `losses_to_test`:
        loss_channels_last = [0., 0., 0.]
        loss_channels_first = [0., 0., 0.]

        old_data_format = K.image_data_format()

        # Evaluate a simple network with channels last, with all three loss
        # functions:
        K.set_image_data_format('channels_last')
        data = np.moveaxis(data_channels_first, 1, -1)
        for index, loss_function in enumerate(losses_to_test):
          labels = np.moveaxis(labels_channels_first[index], 1, -1)
          inputs = input_layer.Input(shape=(3, 3, 1))
          model = prepare_simple_model(inputs, loss_function, labels)
          loss_channels_last[index] = model.evaluate(x=data, y=labels,
                                                     batch_size=1, verbose=0)

        # Evaluate the same network with channels first, with all three loss
        # functions:
        K.set_image_data_format('channels_first')
        data = data_channels_first
        for index, loss_function in enumerate(losses_to_test):
          labels = labels_channels_first[index]
          inputs = input_layer.Input(shape=(1, 3, 3))
          model = prepare_simple_model(inputs, loss_function, labels)
          loss_channels_first[index] = model.evaluate(x=data, y=labels,
                                                      batch_size=1, verbose=0)

        K.set_image_data_format(old_data_format)

        np.testing.assert_allclose(
            loss_channels_first,
            loss_channels_last,
            rtol=1e-06,
            err_msg='{}{}'.format('Computed different losses for ',
                                  'channels_first and channels_last'))


if __name__ == '__main__':
  test.main()
