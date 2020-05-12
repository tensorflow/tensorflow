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
"""Test for multi-worker training tutorial."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import os
import re
from absl.testing import parameterized
import numpy as np
from tensorflow.python import keras
from tensorflow.python.data.experimental.ops import distribute_options
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow.python.platform import test
from tensorflow.python.util import nest


class MultiWorkerTutorialTest(parameterized.TestCase, test.TestCase):
  """Test multi-worker training flow demo'ed in go/multi-worker-with-keras."""

  @contextlib.contextmanager
  def skip_fetch_failure_exception(self):
    try:
      yield
    except Exception as e:  # pylint: disable=broad-except
      if 'URL fetch failure' in str(e):
        self.skipTest('URL fetch error not considered failure of the test.')
      else:
        raise

  @combinations.generate(
      combinations.combine(
          mode=['eager'],
          shard_policy=[None] + list(distribute_options.AutoShardPolicy)))
  def testMultiWorkerTutorial(self, mode, shard_policy):
    """Test multi-worker training flow demo'ed in go/multi-worker-with-keras.

    This test should be kept in sync with the code samples in
    go/multi-worker-with-keras.

    Args:
      mode: Runtime mode.
      shard_policy: None or any of tf.data.experimental.AutoShardPolicy for
        testing.
    """
    if shard_policy is distribute_options.AutoShardPolicy.FILE:
      self.skipTest('TensorSliceDataset is not shardable with FILE policy.')

    def mnist_dataset(batch_size):
      with self.skip_fetch_failure_exception():
        (x_train, y_train), _ = mnist.load_data()
      # The `x` arrays are in uint8 and have values in the range [0, 255].
      # We need to convert them to float32 with values in the range [0, 1]
      x_train = x_train / np.float32(255)
      y_train = y_train.astype(np.int64)
      train_dataset = dataset_ops.DatasetV2.from_tensor_slices(
          (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
      return train_dataset

    def build_and_compile_cnn_model():
      model = keras.Sequential([
          keras.layers.Input(shape=(28, 28)),
          keras.layers.Reshape(target_shape=(28, 28, 1)),
          keras.layers.Conv2D(32, 3, activation='relu'),
          keras.layers.Flatten(),
          keras.layers.Dense(128, activation='relu'),
          keras.layers.Dense(10)
      ])
      model.compile(
          loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          optimizer=gradient_descent.SGD(learning_rate=0.001),
          metrics=['accuracy'])
      return model

    per_worker_batch_size = 64

    single_worker_dataset = mnist_dataset(per_worker_batch_size)
    single_worker_model = build_and_compile_cnn_model()
    single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)

    num_workers = 4

    def proc_func():
      global_batch_size = per_worker_batch_size * num_workers
      strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()
      with strategy.scope():
        multi_worker_model = build_and_compile_cnn_model()

      callbacks = [
          keras.callbacks.ModelCheckpoint(
              filepath=os.path.join(self.get_temp_dir(), 'checkpoint'))
      ]

      multi_worker_dataset = mnist_dataset(global_batch_size)
      if shard_policy:
        options = dataset_ops.Options()
        options.experimental_distribute.auto_shard_policy = shard_policy
        multi_worker_dataset = multi_worker_dataset.with_options(options)

      multi_worker_model.fit(
          multi_worker_dataset,
          epochs=3,
          steps_per_epoch=70,
          callbacks=callbacks)

    with test_util.skip_if_error(self, errors_impl.UnavailableError):
      mpr_result = multi_process_runner.run(
          proc_func,
          multi_worker_test_base.create_cluster_spec(num_workers=num_workers),
          list_stdout=True)

    def extract_accuracy(worker_id, input_string):
      match = re.match(
          r'\[worker\-{}\].*accuracy: (\d+\.\d+).*'.format(worker_id),
          input_string)
      return None if match is None else float(match.group(1))

    for worker_id in range(num_workers):
      accu_result = nest.map_structure(
          lambda x: extract_accuracy(worker_id, x),  # pylint: disable=cell-var-from-loop
          mpr_result.stdout)
      self.assertTrue(
          any(accu_result), 'Every worker is supposed to have accuracy result.')


if __name__ == '__main__':
  multi_process_runner.test_main()
