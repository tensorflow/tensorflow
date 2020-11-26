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
"""Tests for CollectiveAllReduceStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations as ds_combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_combinations as combinations
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_keras
from tensorflow.python.ops import array_ops


@ds_combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
        ],
        mode=['eager']))
class MultiWorkerMirroredStrategyTest(test.TestCase, parameterized.TestCase):

  def testFitWithoutStepsPerEpochPartialBatch(self, strategy):

    def _model_fn():
      x = layers.Input(shape=(1,), name='input')
      y = layers.Dense(1, name='dense')(x)
      model = training.Model(x, y)
      return model

    def _get_dataset():
      inputs = array_ops.expand_dims_v2(
          constant_op.constant(range(10)), axis=1)
      targets = array_ops.expand_dims_v2(
          constant_op.constant(range(10)), axis=1)
      # Make global batch size 12 for 2 replicas and a non-repeated dataset
      # with 10 elements so that we have partial batch
      dataset = dataset_ops.Dataset.from_tensor_slices(
          (inputs, targets)).batch(
              12, drop_remainder=False)
      return dataset

    with strategy.scope():
      optimizer_fn = gradient_descent_keras.SGD
      optimizer = optimizer_fn(0.001)
      model = _model_fn()
      loss = 'mse'
      metrics = ['mae']
      model.compile(optimizer, loss, metrics=metrics)
    dataset = _get_dataset()
    kernel_before = model.get_weights()[0][0]
    model.fit(dataset, epochs=10)
    kernel_after = model.get_weights()[0][0]
    self.assertNotEqual(kernel_before, kernel_after)
    self.assertGreater(abs(kernel_before - 1), abs(kernel_after - 1))


if __name__ == '__main__':
  v2_compat.enable_v2_behavior()
  multi_process_runner.test_main()
