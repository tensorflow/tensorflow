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
"""Tests for distributed training utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import callbacks
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import adam as v1_adam


class DistributedTrainingUtilsTest(test.TestCase):

  @test.mock.patch.object(logging, 'warning', autospec=True)
  def test_validate_callbacks_predefined_callbacks(self, mock_warning):
    supported_predefined_callbacks = [
        callbacks.TensorBoard(),
        callbacks.CSVLogger(filename='./log.csv'),
        callbacks.EarlyStopping(),
        callbacks.ModelCheckpoint(filepath='./checkpoint'),
        callbacks.TerminateOnNaN(),
        callbacks.ProgbarLogger(),
        callbacks.History(),
        callbacks.RemoteMonitor()
    ]

    distributed_training_utils.validate_callbacks(
        supported_predefined_callbacks, adam.Adam())

    unsupported_predefined_callbacks = [
        callbacks.ReduceLROnPlateau(),
        callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001)
    ]

    for callback in unsupported_predefined_callbacks:
      with self.assertRaisesRegexp(
          ValueError, 'You must specify a Keras Optimizer V2'):
        distributed_training_utils.validate_callbacks([callback],
                                                      v1_adam.AdamOptimizer())

    self.assertEqual(0, mock_warning.call_count)

  @test.mock.patch.object(logging, 'warning', autospec=True)
  def test_validate_callbacks_custom_callback(self, mock_warning):

    class CustomCallback(callbacks.Callback):
      pass

    distributed_training_utils.validate_callbacks([CustomCallback()],
                                                  adam.Adam())

    self.assertEqual(1, mock_warning.call_count)

    call_args, call_kwargs = mock_warning.call_args

    self.assertEqual(('Your input callback is not one of the predefined '
                      'Callbacks that supports DistributionStrategy. You '
                      'might encounter an error if you access one of the '
                      'model\'s attributes as part of the callback since '
                      'these attributes are not set. You can access each of '
                      'the individual distributed models using the '
                      '`_grouped_model` attribute of your original model.',),
                     call_args)

    self.assertEqual(0, len(call_kwargs))


if __name__ == '__main__':
  test.main()
