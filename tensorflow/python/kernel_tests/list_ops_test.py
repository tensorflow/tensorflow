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
"""Tests for ops which manipulate lists of tensors."""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np  # pylint: disable=unused-import

from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


def scalar_shape():
  return ops.convert_to_tensor([], dtype=dtypes.int32)


class ListOpsTest(test_util.TensorFlowTestCase):

  def testPushPop(self):
    l = list_ops.empty_tensor_list(element_dtype=dtypes.float32,
                                   element_shape=scalar_shape())
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    self.assertAllEqual(e, 1.0)

  def testPushPopGPU(self):
    if not context.num_gpus():
      return
    with context.device("gpu:0"):
      self.testPushPop()

  def testStack(self):
    l = list_ops.empty_tensor_list(element_dtype=dtypes.float32,
                                   element_shape=scalar_shape())
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    l = list_ops.tensor_list_push_back(l, constant_op.constant(2.0))
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    self.assertAllEqual(t, [1.0, 2.0])

  def testStackGPU(self):
    if not context.num_gpus():
      return
    with context.device("gpu:0"):
      self.testStack()

  def testTensorListFromTensor(self):
    t = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_from_tensor(t, element_shape=scalar_shape())
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    self.assertAllEqual(e, 2.0)
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    self.assertAllEqual(e, 1.0)
    self.assertAllEqual(list_ops.tensor_list_length(l), 0)

  def testFromTensorGPU(self):
    if not context.num_gpus():
      return
    with context.device("gpu:0"):
      self.testTensorListFromTensor()

  def testGetSetItem(self):
    t = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_from_tensor(t, element_shape=scalar_shape())
    e0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
    self.assertAllEqual(e0, 1.0)
    l = list_ops.tensor_list_set_item(l, 0, 3.0)
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    self.assertAllEqual(t, [3.0, 2.0])

  def testGetSetGPU(self):
    if not context.num_gpus():
      return
    with context.device("gpu:0"):
      self.testGetSetItem()

  def testUnknownShape(self):
    l = list_ops.empty_tensor_list(element_dtype=dtypes.float32,
                                   element_shape=-1)
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    l = list_ops.tensor_list_push_back(l, constant_op.constant([1.0, 2.0]))
    _, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    self.assertAllEqual(e, [1.0, 2.0])

  def testCPUGPUCopy(self):
    if not context.num_gpus():
      return
    t = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_from_tensor(t, element_shape=scalar_shape())
    with context.device("gpu:0"):
      l_gpu = array_ops.identity(l)
      self.assertAllEqual(
          list_ops.tensor_list_pop_back(
              l_gpu, element_dtype=dtypes.float32)[1],
          2.0)
    l_cpu = array_ops.identity(l_gpu)
    self.assertAllEqual(
        list_ops.tensor_list_pop_back(
            l_cpu, element_dtype=dtypes.float32)[1],
        2.0)

  def testGraphStack(self):
    with context.graph_mode(), self.test_session():
      tl = list_ops.empty_tensor_list(
          element_shape=constant_op.constant([1], dtype=dtypes.int32),
          element_dtype=dtypes.int32)
      tl = list_ops.tensor_list_push_back(tl, [1])
      self.assertAllEqual(
          list_ops.tensor_list_stack(tl, element_dtype=dtypes.int32).eval(),
          [[1]])

  def testSerialize(self):
    # pylint: disable=g-import-not-at-top
    try:
      import portpicker
    except ImportError:
      return
    with context.graph_mode():
      worker_port = portpicker.pick_unused_port()
      ps_port = portpicker.pick_unused_port()
      cluster_dict = {
          "worker": ["localhost:%s" % worker_port],
          "ps": ["localhost:%s" % ps_port]
      }
      cs = server_lib.ClusterSpec(cluster_dict)

      worker = server_lib.Server(
          cs, job_name="worker", protocol="grpc", task_index=0, start=True)
      unused_ps = server_lib.Server(
          cs, job_name="ps", protocol="grpc", task_index=0, start=True)
      with ops.Graph().as_default(), session.Session(target=worker.target):
        with ops.device("/job:worker"):
          t = constant_op.constant([[1.0], [2.0]])
          l = list_ops.tensor_list_from_tensor(t, element_shape=[1])
        with ops.device("/job:ps"):
          l_ps = array_ops.identity(l)
          l_ps, e = list_ops.tensor_list_pop_back(
              l_ps, element_dtype=dtypes.float32)
        with ops.device("/job:worker"):
          worker_e = array_ops.identity(e)
        self.assertAllEqual(worker_e.eval(), [2.0])

  def testPushPopGradients(self):
    with backprop.GradientTape() as tape:
      l = list_ops.empty_tensor_list(element_dtype=dtypes.float32,
                                     element_shape=scalar_shape())
      c = constant_op.constant(1.0)
      tape.watch(c)
      l = list_ops.tensor_list_push_back(l, c)
      l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      e = 2 * e
    self.assertAllEqual(tape.gradient(e, [c])[0], 2.0)

  def testStackFromTensorGradients(self):
    with backprop.GradientTape() as tape:
      c = constant_op.constant([1.0, 2.0])
      tape.watch(c)
      l = list_ops.tensor_list_from_tensor(c, element_shape=scalar_shape())
      c2 = list_ops.tensor_list_stack(
          l, element_dtype=dtypes.float32)
      result = c2 * 2.0
    self.assertAllEqual(tape.gradient(result, [c])[0], [2.0, 2.0])

  def testGetSetGradients(self):
    with backprop.GradientTape() as tape:
      c = constant_op.constant([1.0, 2.0])
      tape.watch(c)
      l = list_ops.tensor_list_from_tensor(c, element_shape=scalar_shape())
      c2 = constant_op.constant(3.0)
      tape.watch(c2)
      l = list_ops.tensor_list_set_item(l, 0, c2)
      e = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
      ee = list_ops.tensor_list_get_item(l, 1, element_dtype=dtypes.float32)
      y = e * e + ee * ee
    grad_c, grad_c2 = tape.gradient(y, [c, c2])
    self.assertAllEqual(grad_c, [0.0, 4.0])
    self.assertAllEqual(grad_c2, 6.0)

  def testSetOutOfBounds(self):
    c = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_from_tensor(c, element_shape=scalar_shape())
    with self.assertRaises(errors.InvalidArgumentError):
      list_ops.tensor_list_set_item(l, 20, 3.0)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
