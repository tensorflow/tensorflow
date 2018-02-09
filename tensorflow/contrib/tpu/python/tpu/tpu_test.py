# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

"""Tests for tpu_function helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.contrib.tpu.python.tpu import tpu_feed
from tensorflow.contrib.tpu.python.tpu import training_loop

from tensorflow.python.framework import dtypes
from tensorflow.python.layers import convolutional
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops

from tensorflow.python.platform import test


class TPUContextTest(test.TestCase):

  def testIsInContext(self):
    """Test that control_flow_util can check that we're in a TPU context."""
    z1 = array_ops.identity(1)
    context = tpu.TPUReplicateContext(b"context")
    context.Enter()
    z2 = array_ops.identity(1)
    context.Exit()
    self.assertFalse(control_flow_util.IsInXLAContext(z1.op))
    self.assertTrue(control_flow_util.IsInXLAContext(z2.op))


class TPULayerRewriteTest(test.TestCase):

  def testUsingInfeedQueueWithRegularizer(self):
    """Test that Layer regularizers can reference data created in loops."""

    def make_regularizer(scale):
      return lambda inputs: scale * math_ops.reduce_sum(math_ops.square(inputs))

    def training_step(inputs, scale):
      outputs = convolutional.conv2d(
          inputs,
          filters=16,
          kernel_size=(3, 3),
          data_format="channels_first",
          kernel_regularizer=make_regularizer(scale))
      loss = math_ops.reduce_mean(math_ops.square(outputs))
      return loss.op

    inputs = array_ops.zeros(shape=(128, 32, 32, 16))
    scale = array_ops.ones(shape=())
    infeed = tpu_feed.InfeedQueue(
        tuple_types=[dtypes.float32, dtypes.float32],
        tuple_shapes=[inputs.shape, scale.shape])

    def loop():
      return training_loop.repeat(5, training_step, infeed_queue=infeed)

    # This should not throw an error.
    tpu.rewrite(loop)


if __name__ == "__main__":
  test.main()
