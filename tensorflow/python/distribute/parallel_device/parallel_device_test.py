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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading

from tensorflow.python.distribute.parallel_device import parallel_device
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training.tracking import util as tracking
from tensorflow.python.util import nest

# When running collectives asynchronously, we need to give each parallel device
# execution a unique ID so the collectives don't interfere. Since the op is
# replicated with group/instance key intact, the replicated nodes will
# communicate.
# TODO(allenl): Switch to using a collective manager.
_COUNTER_LOCK = threading.Lock()
_COUNTER = 0


def _collective_reduce(inputs, operation, num_replicas):

  def _reduce_tensor(tensor):
    with _COUNTER_LOCK:
      global _COUNTER
      keys = _COUNTER
      _COUNTER += 1
    return collective_ops.all_reduce(
        t=tensor,
        group_size=num_replicas,
        merge_op=operation,
        group_key=keys,
        instance_key=keys,
        final_op="Id")

  return nest.map_structure(_reduce_tensor, inputs)


def _collective_sum(inputs, num_replicas):
  return _collective_reduce(
      inputs=inputs, operation="Add", num_replicas=num_replicas)


class _Dense(module.Module):

  def __init__(self, output_size):
    self.output_size = output_size
    self.kernel = None
    self.bias = None

  def __call__(self, x):
    if self.kernel is None:
      self.kernel = variables.Variable(
          array_ops.ones(
              array_ops.stack([self.output_size,
                               array_ops.shape(x)[-1]])))
      self.bias = variables.Variable(array_ops.ones([self.output_size]))
    return math_ops.matmul(x, self.kernel, transpose_b=True) + self.bias


class _VirtualDeviceTestCase(test.TestCase):

  def setUp(self):
    super(_VirtualDeviceTestCase, self).setUp()
    cpus = context.context().list_physical_devices("CPU")
    # Set 4 virtual CPUs
    context.context().set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration()
    ])

    # TODO(allenl): Make CPU:0 and CPU:1 work (right now "CPU:1" soft-places
    # onto CPU:0, which seems wrong).
    components = [
        "/job:localhost/replica:0/task:0/device:CPU:0",
        "/job:localhost/replica:0/task:0/device:CPU:1"
    ]
    self.device = parallel_device.ParallelDevice(components)


class ParallelDeviceTests(_VirtualDeviceTestCase):

  def test_register_parallel_device(self):
    with ops.device(self.device.name):
      c = constant_op.constant(1.)
      d = constant_op.constant(2.)
      e = c + d
      outputs = self.device.unpack(e)
    self.assertAllClose([3., 3.], outputs)

    self.assertIn(self.device.components[0], outputs[0].backing_device)
    self.assertIn(self.device.components[1], outputs[1].backing_device)

  def test_device_id(self):
    device_ids = self.device.unpack(self.device.device_ids)
    self.assertAllClose([0, 1], device_ids)
    self.assertIn(self.device.components[0], device_ids[0].backing_device)
    self.assertIn(self.device.components[1], device_ids[1].backing_device)

  def test_collective_reduce(self):
    with ops.device(self.device.name):
      x = self.device.pack(
          [constant_op.constant(-1.5),
           constant_op.constant(3.5)])
      reduced = _collective_sum(x, num_replicas=2)
      outputs = self.device.unpack(reduced)
    self.assertAllClose([2., 2.], outputs)
    self.assertIn(self.device.components[0], outputs[0].backing_device)
    self.assertIn(self.device.components[1], outputs[1].backing_device)

  def test_checkpointing(self):
    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    with self.device.scope():
      different_values = self.device.pack(
          [constant_op.constant(-1.),
           constant_op.constant(3.)])
      v = variables.Variable(different_values)
      checkpoint = tracking.Checkpoint(v=v)
    save_path = checkpoint.save(prefix)
    with ops.device(self.device.name):
      v.assign(constant_op.constant(0.))
    # Make sure the checkpoint is actually written before we try to read it
    context.async_wait()
    checkpoint.restore(save_path).assert_consumed()
    with ops.device(self.device.name):
      outputs = self.device.unpack(v)
    self.assertAllClose([-1., 3.], outputs)


class LayerTests(_VirtualDeviceTestCase):

  def test_layer_forward(self):
    with ops.device(self.device.name):
      layer = _Dense(5)
      x = constant_op.constant([[2.]])
      y = layer(x)
      outputs = self.device.unpack(y)
    self.assertAllClose([[3.] * 5], outputs[0])
    self.assertAllClose([[3.] * 5], outputs[1])
    self.assertIn(self.device.components[0], outputs[0].backing_device)
    self.assertIn(self.device.components[1], outputs[1].backing_device)

    # With different Layer inputs we get different outputs
    with ops.device(self.device.name):
      x = self.device.pack(
          [constant_op.constant([[-0.5]]),
           constant_op.constant([[0.5]])])
      y = layer(x)
      outputs = self.device.unpack(y)
    self.assertGreater(
        math_ops.reduce_max(math_ops.abs(outputs[0] - outputs[1])), 1e-5)
    self.assertIn(self.device.components[0], outputs[0].backing_device)
    self.assertIn(self.device.components[1], outputs[1].backing_device)

  def test_layer_sync_training(self):
    with ops.device(self.device.name):
      layer = _Dense(5)

      with backprop.GradientTape() as tape:
        x = self.device.pack(
            [constant_op.constant([[-0.5]]),
             constant_op.constant([[0.5]])])
        y = layer(x)
        loss = (y - math_ops.range(5.))**2.
      parameters = layer.trainable_variables
      unreduced_gradients = tape.gradient(loss, parameters)
      reduced_gradients = _collective_sum(unreduced_gradients, num_replicas=2)
      for grad, param in zip(reduced_gradients, parameters):
        param.assign_sub(0.01 * grad)
    final_kernels = self.device.unpack(layer.kernel)
    self.assertAllClose(final_kernels[0], final_kernels[1])
    final_bias = self.device.unpack(layer.bias)
    expected_bias = (1. - 0.01 * 2. * (1. + .5 - math_ops.range(5.)) -
                     0.01 * 2. * (1. - .5 - math_ops.range(5.)))
    self.assertAllClose(expected_bias, final_bias[0])
    self.assertAllClose(expected_bias, final_bias[1])
    self.assertIn(self.device.components[0], final_kernels[0].backing_device)
    self.assertIn(self.device.components[1], final_kernels[1].backing_device)

  def test_layer_divergent_buffer_training(self):
    with ops.device(self.device.name):
      layer = _Dense(5)

      with backprop.GradientTape() as tape:
        x = self.device.pack(
            [constant_op.constant([[-0.5]]),
             constant_op.constant([[0.5]])])
        y = layer(x)
        loss = (y - math_ops.range(5.))**2.
      parameters = layer.trainable_variables
      unreduced_gradients = tape.gradient(loss, parameters)
      for grad, param in zip(unreduced_gradients, parameters):
        param.assign_sub(0.01 * grad)
    final_kernels = self.device.unpack(layer.kernel)
    self.assertNotAllClose(final_kernels[0], final_kernels[1])
    final_bias = self.device.unpack(layer.bias)
    self.assertAllClose(1. - 0.01 * 2. * (1. - .5 - math_ops.range(5.)),
                        final_bias[0])
    self.assertAllClose(1. - 0.01 * 2. * (1. + .5 - math_ops.range(5.)),
                        final_bias[1])
    self.assertIn(self.device.components[0], final_kernels[0].backing_device)
    self.assertIn(self.device.components[1], final_kernels[1].backing_device)

  def test_training_loop(self):
    for _ in range(5):
      layer = _Dense(5)
      checkpoint = tracking.Checkpoint(layer=layer)
      manager = checkpoint_management.CheckpointManager(
          checkpoint, directory=self.get_temp_dir(), max_to_keep=5)
      manager.restore_or_initialize()

      for _ in range(10):
        with self.device.scope():
          with backprop.GradientTape() as tape:
            x = self.device.pack(
                [constant_op.constant([[-0.5]]),
                 constant_op.constant([[0.5]])])
            y = layer(x)
            loss = (y - math_ops.range(5.))**2.
          parameters = layer.trainable_variables
          unreduced_gradients = tape.gradient(loss, parameters)
          reduced_gradients = _collective_sum(
              unreduced_gradients, num_replicas=len(self.device.components))
          for grad, param in zip(reduced_gradients, parameters):
            param.assign_sub(0.01 * grad)

        manager.save()


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
