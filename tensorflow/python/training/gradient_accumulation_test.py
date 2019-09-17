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
"""Functional test for GradientAccumulation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distribute.python import mirrored_strategy
from tensorflow.contrib.distribute.python import one_device_strategy
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator import run_config
from tensorflow.python.framework import constant_op
from tensorflow.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import training_util

class GradientAccumulationTest(test.TestCase):

  def _build_model_fn_optimizer(self, iter_size=None):
    """Simple model_fn with optimizer."""
    optimizer = gradient_descent.GradientDescentOptimizer(0.2)

    def model_fn(features, labels, mode):  # pylint: disable=unused-argument
      """model_fn which uses a single unit Dense layer."""
      layer = layers.Dense(1, use_bias=True)
      logits = layer(features)

      def loss_fn():
        y = array_ops.reshape(logits, []) - constant_op.constant(1.)
        return y * y

      assert mode == model_fn_lib.ModeKeys.TRAIN

      global_step = training_util.get_global_step()
      train_op = optimizer.minimize(loss_fn(), global_step=global_step,
          iter_size=iter_size)
      return model_fn_lib.EstimatorSpec(mode, loss=loss_fn(), train_op=train_op)

    return model_fn

  def testMirroredStrategyWithGA(self):
    distribution = mirrored_strategy.MirroredStrategy(["/device:GPU:0"])
    config = run_config.RunConfig(train_distribute=distribution)

    def train_input_fn():
      features = dataset_ops.Dataset.from_tensors([[1.]]).repeat(10)
      labels = dataset_ops.Dataset.from_tensors([1.]).repeat(10)
      return dataset_ops.Dataset.zip((features, labels))

    est = estimator.Estimator(
        model_fn=self._build_model_fn_optimizer(iter_size=2), config=config)
    est.train(input_fn=train_input_fn)

    # Since there are 10 samples and 1 device and each global_step will
    # consume iter_size=2 samples, so the final global_step until `input_fn`
    # generates the `tf.errors.OutOfRange` error should be 10 / 2 = 5.
    global_step = est.get_variable_value("global_step")
    self.assertAllCloseAccordingToType(5, global_step)

  def testOneDeviceStrategyWithGA(self):
    distribution = one_device_strategy.OneDeviceStrategy(
        device="/device:GPU:0")
    config = run_config.RunConfig(train_distribute=distribution)

    def train_input_fn():
      features = dataset_ops.Dataset.from_tensors([[1.]]).repeat(10)
      labels = dataset_ops.Dataset.from_tensors([1.]).repeat(10)
      return dataset_ops.Dataset.zip((features, labels))

    est = estimator.Estimator(
        model_fn=self._build_model_fn_optimizer(iter_size=2), config=config)
    est.train(input_fn=train_input_fn)

    # Since there are 10 samples and 1 device and each global_step will
    # consume iter_size=2 samples, so the final global_step until `input_fn`
    # generates the `tf.errors.OutOfRange` error should be 10 / 2 = 5.
    global_step = est.get_variable_value("global_step")
    self.assertAllCloseAccordingToType(5, global_step)

  def testMirroredStrategyWithGAMultiDevices(self):
    distribution = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:GPU:1", "device:GPU:2"])
    config = run_config.RunConfig(train_distribute=distribution)

    def train_input_fn():
      features = dataset_ops.Dataset.from_tensors([[1.]]).repeat(10)
      labels = dataset_ops.Dataset.from_tensors([1.]).repeat(10)
      return dataset_ops.Dataset.zip((features, labels))

    est = estimator.Estimator(
        model_fn=self._build_model_fn_optimizer(iter_size=2), config=config)
    est.train(input_fn=train_input_fn)

    # Since there are 10 samples and 3 device and each global_step will
    # consume iter_size*num_gpus=6 samples, so the final global_step until
    # `input_fn` generates the `tf.errors.OutOfRange` error should be
    # 10 / 6 = 1. Note that in this case each device will remain some
    # local accumulated grads which are not communicated with each other
    # as the next apply_op has not been triggered before `input_fn` runs
    # OutOfRange.
    global_step = est.get_variable_value("global_step")
    self.assertAllCloseAccordingToType(1, global_step)

  def testMirroredStrategyWithGAWithIllegalIterSize(self):
    distribution = mirrored_strategy.MirroredStrategy(
        ["/device:GPU:0", "/device:GPU:1"])
    config = run_config.RunConfig(train_distribute=distribution)

    def train_input_fn():
      features = dataset_ops.Dataset.from_tensors([[1.]]).repeat(10)
      labels = dataset_ops.Dataset.from_tensors([1.]).repeat(10)
      return dataset_ops.Dataset.zip((features, labels))

    est = estimator.Estimator(
        model_fn=self._build_model_fn_optimizer(iter_size=-3), config=config)
    with self.assertRaises(ValueError):
      est.train(input_fn=train_input_fn)

if __name__ == "__main__":
  test.main()
