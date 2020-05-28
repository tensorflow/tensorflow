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
"""Tests of `worker_training_state.py` utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys

from absl.testing import parameterized
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import multi_worker_test_base as test_base
from tensorflow.python.framework.errors_impl import NotFoundError
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.distribute import multi_worker_testing_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import test


class ModelCheckpointTest(test_base.IndependentWorkerTestBase,
                          parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          mode=['graph'],
          required_gpus=[0, 1],
          file_format=['h5', 'tf'],
          save_weights_only=[True, False]))
  def testCheckpointExists(self, file_format, save_weights_only):
    train_ds, _ = multi_worker_testing_utils.mnist_synthetic_dataset(64, 2)
    model = multi_worker_testing_utils.get_mnist_model((28, 28, 1))
    saving_dir = self.get_temp_dir()
    saving_filepath = os.path.join(saving_dir, 'checkpoint.' + file_format)
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=saving_filepath, save_weights_only=save_weights_only)
    ]
    self.assertFalse(file_io.file_exists(saving_filepath))

    try:
      model.fit(
          x=train_ds, epochs=2, steps_per_epoch=2, callbacks=callbacks_list)
    except NotFoundError as e:
      if 'Failed to create a NewWriteableFile' in e.message:
        self.skipTest('b/138941852, path not found error in Windows py35.')
    tf_saved_model_exists = file_io.file_exists(saving_filepath)
    tf_weights_only_checkpoint_exists = file_io.file_exists(saving_filepath +
                                                            '.index')
    self.assertTrue(tf_saved_model_exists or tf_weights_only_checkpoint_exists)


if __name__ == '__main__':
  with test.mock.patch.object(sys, 'exit', os._exit):
    test.main()
