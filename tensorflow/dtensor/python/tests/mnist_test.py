# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""DTensor MNIST test."""

from absl.testing import parameterized

import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import input_util
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


_BATCH_DIM = 'batch'
_DEVICE_IDS = test_util.create_device_ids_array((2,))
_ONE_D_MESH = layout_lib.Mesh(
    [_BATCH_DIM],
    _DEVICE_IDS,
    np.ravel(_DEVICE_IDS).tolist(),
    test_util.create_device_list((2,), 'CPU'),
)
_ONE_D_TPU_MESH = layout_lib.Mesh(
    [_BATCH_DIM],
    _DEVICE_IDS,
    np.ravel(_DEVICE_IDS).tolist(),
    test_util.create_device_list((2,), 'TPU'),
)
_BATCH_SIZE = 1024
_STEPS = 5
_LR = 1e-3
_ATOL = 1  # absolute error becomes large as gradients approach zero.
_RTOL = 1e-3
Layout = layout_lib.Layout


def mnist_fake_dataset():
  imgs = []
  labels = []
  for i in range(_STEPS * _BATCH_SIZE):
    img = stateless_random_ops.stateless_random_uniform(
        shape=(28, 28, 1),
        seed=[1, i],
        minval=0,
        maxval=256,
        dtype=dtypes.float32,
    )
    imgs.append(img)
    label = stateless_random_ops.stateless_random_uniform(
        shape=(1,), seed=[2, i], minval=0, maxval=10, dtype=dtypes.int64
    )
    labels.append(label)

  return dataset_ops.DatasetV2.from_tensor_slices(
      (array_ops_stack.stack(imgs), array_ops_stack.stack(labels))
  )


def _run_step(inputs, w, b, k):
  with backprop.GradientTape() as g:
    g.watch([w, b])
    logits = nn_ops.conv2d_v2(inputs, k, strides=[1, 1, 1, 1], padding='SAME')
    logits = array_ops.reshape(logits, [logits.shape[0], -1])
    logits = math_ops.matmul(logits, w)
    logits = logits + b
    loss = math_ops.reduce_sum(logits, axis=[0, 1])
  gw, gb = g.gradient(loss, [w, b])
  for v, v_grad in zip([w, b], [gw, gb]):
    v.assign_sub(_LR * v_grad)
  return gw, gb, loss


class DTensorMNISTTest(test_util.DTensorBaseTest):

  def setUp(self):
    super(DTensorMNISTTest, self).setUp()

    global_ids = test_util.create_device_ids_array((2,))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {
        device: layout_lib.Mesh(
            [_BATCH_DIM],
            global_ids,
            local_ids,
            test_util.create_device_list((2,), device),
        )
        for device in ['TPU', 'GPU', 'CPU']
    }
    self.mesh = self.configTestMesh(mesh_dict)

  def init_var(self, mesh):
    # Initialize TF randon normal variables(without using DTensor).
    w_initializer = stateless_random_ops.stateless_random_normal(
        shape=[28 * 28, 10], seed=[0, 1]
    )
    b_initializer = stateless_random_ops.stateless_random_normal(
        shape=[10], seed=[1, 2]
    )
    # A filter with 3x3 shape, 1 input channel and 1 output channel.
    k_initializer = stateless_random_ops.stateless_random_normal(
        [3, 3, 1, 1], seed=[2, 3]
    )

    n_w = variables.Variable(w_initializer)
    n_b = variables.Variable(b_initializer)
    n_k = variables.Variable(k_initializer)

    # Initialize DTensor variables.
    w_initializer_on_mesh = api.copy_to_mesh(
        w_initializer, Layout.replicated(mesh, 2)
    )
    b_initializer_on_mesh = api.copy_to_mesh(
        b_initializer, Layout.replicated(mesh, rank=1)
    )
    k_initializer_on_mesh = api.copy_to_mesh(
        k_initializer, Layout.replicated(mesh, rank=4)
    )

    w = d_variable.DVariable(w_initializer_on_mesh)
    b = d_variable.DVariable(b_initializer_on_mesh)
    k = d_variable.DVariable(k_initializer_on_mesh)

    return (n_w, n_b, n_k), (w, b, k)

  @parameterized.named_parameters(('Eager', False), ('Function', True))
  def testMnist(self, on_function):
    mnist_dataset = mnist_fake_dataset()

    (n_w, n_b, n_k), (w, b, k) = self.init_var(self.mesh)

    n_dataset = mnist_dataset.batch(_BATCH_SIZE, drop_remainder=True)
    n_iter = iter(n_dataset)

    input_layout = Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=4)
    label_layout = Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=2)
    dtensor_dataset = input_util.DTensorDataset(
        dataset=mnist_dataset,
        global_batch_size=_BATCH_SIZE,
        mesh=self.mesh,
        layouts=(input_layout, label_layout),
        batch_dim=_BATCH_DIM,
    )
    dtensor_iter = iter(dtensor_dataset)

    step_fn = (
        polymorphic_function.function(_run_step) if on_function else _run_step
    )

    # Training loop.
    for _ in range(_STEPS):
      # Normal run without DTensor.
      n_input, _ = next(n_iter)
      g_nw, g_nb, n_loss = step_fn(n_input, n_w, n_b, n_k)

      # DTensor Run
      dtensor_input, _ = next(dtensor_iter)
      with ops.device_v2(api.device_name()):
        gw, gb, loss = step_fn(dtensor_input, w, b, k)

      loss_unpack = api.unpack(loss)
      self.assertAllEqual(loss_unpack[0], loss_unpack[1])

      self.assertAllClose(n_loss, loss, atol=_ATOL, rtol=_RTOL)
      self.assertAllClose(g_nw, gw, atol=_ATOL, rtol=_RTOL)
      self.assertAllClose(g_nb, gb, atol=_ATOL, rtol=_RTOL)
      self.assertAllClose(n_w, w, atol=_ATOL, rtol=_RTOL)
      self.assertAllClose(n_b, b, atol=_ATOL, rtol=_RTOL)


if __name__ == '__main__':
  test.main()
