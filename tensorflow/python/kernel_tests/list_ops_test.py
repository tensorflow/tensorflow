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
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


def scalar_shape():
  return ops.convert_to_tensor([], dtype=dtypes.int32)


@test_util.with_c_shapes
class ListOpsTest(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def testPushPop(self):
    l = list_ops.empty_tensor_list(element_dtype=dtypes.float32,
                                   element_shape=scalar_shape())
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(e), 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testPushPopGPU(self):
    if not context.num_gpus():
      return
    with context.device("gpu:0"):
      self.testPushPop()

  @test_util.run_in_graph_and_eager_modes
  def testStack(self):
    l = list_ops.empty_tensor_list(element_dtype=dtypes.float32,
                                   element_shape=scalar_shape())
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    l = list_ops.tensor_list_push_back(l, constant_op.constant(2.0))
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [1.0, 2.0])

  @test_util.run_in_graph_and_eager_modes
  def testStackGPU(self):
    if not context.num_gpus():
      return
    with context.device("gpu:0"):
      self.testStack()

  @test_util.run_in_graph_and_eager_modes
  def testTensorListFromTensor(self):
    t = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_from_tensor(t, element_shape=scalar_shape())
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(e), 2.0)
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(e), 1.0)
    self.assertAllEqual(self.evaluate(list_ops.tensor_list_length(l)), 0)

  @test_util.run_in_graph_and_eager_modes
  def testFromTensorGPU(self):
    if not context.num_gpus():
      return
    with context.device("gpu:0"):
      self.testTensorListFromTensor()

  @test_util.run_in_graph_and_eager_modes
  def testGetSetItem(self):
    t = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_from_tensor(t, element_shape=scalar_shape())
    e0 = list_ops.tensor_list_get_item(l, 0, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(e0), 1.0)
    l = list_ops.tensor_list_set_item(l, 0, 3.0)
    t = list_ops.tensor_list_stack(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(t), [3.0, 2.0])

  @test_util.run_in_graph_and_eager_modes
  def testGetSetGPU(self):
    if not context.num_gpus():
      return
    with context.device("gpu:0"):
      self.testGetSetItem()

  @test_util.run_in_graph_and_eager_modes
  def testUnknownShape(self):
    l = list_ops.empty_tensor_list(
        element_dtype=dtypes.float32, element_shape=-1)
    l = list_ops.tensor_list_push_back(l, constant_op.constant(1.0))
    l = list_ops.tensor_list_push_back(l, constant_op.constant([1.0, 2.0]))
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(e), [1.0, 2.0])
    l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
    self.assertAllEqual(self.evaluate(e), 1.0)

  @test_util.run_in_graph_and_eager_modes
  def testCPUGPUCopy(self):
    if not context.num_gpus():
      return
    t = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_from_tensor(t, element_shape=scalar_shape())
    with context.device("gpu:0"):
      l_gpu = array_ops.identity(l)
      self.assertAllEqual(
          self.evaluate(
              list_ops.tensor_list_pop_back(
                  l_gpu, element_dtype=dtypes.float32)[1]), 2.0)
    l_cpu = array_ops.identity(l_gpu)
    self.assertAllEqual(
        self.evaluate(
            list_ops.tensor_list_pop_back(
                l_cpu, element_dtype=dtypes.float32)[1]), 2.0)

  @test_util.run_in_graph_and_eager_modes
  def testGraphStack(self):
    with context.graph_mode(), self.test_session():
      tl = list_ops.empty_tensor_list(
          element_shape=constant_op.constant([1], dtype=dtypes.int32),
          element_dtype=dtypes.int32)
      tl = list_ops.tensor_list_push_back(tl, [1])
      self.assertAllEqual(
          self.evaluate(
              list_ops.tensor_list_stack(tl, element_dtype=dtypes.int32)),
          [[1]])

  @test_util.run_in_graph_and_eager_modes
  def testGraphStackInLoop(self):
    with context.graph_mode(), self.test_session():
      t1 = list_ops.empty_tensor_list(
          element_shape=constant_op.constant([], dtype=dtypes.int32),
          element_dtype=dtypes.int32)
      i = constant_op.constant(0, dtype=dtypes.int32)

      def body(i, t1):
        t1 = list_ops.tensor_list_push_back(t1, i)
        i += 1
        return i, t1

      i, t1 = control_flow_ops.while_loop(lambda i, t1: math_ops.less(i, 4),
                                          body, [i, t1])
      s1 = list_ops.tensor_list_stack(t1, element_dtype=dtypes.int32)
      self.assertAllEqual(self.evaluate(s1), [0, 1, 2, 3])

  @test_util.run_in_graph_and_eager_modes
  def testGraphStackSwitchDtype(self):
    with context.graph_mode(), self.test_session():
      list_ = list_ops.empty_tensor_list(
          element_shape=constant_op.constant([], dtype=dtypes.int32),
          element_dtype=dtypes.int32)
      m = constant_op.constant([1, 2, 3], dtype=dtypes.float32)

      def body(list_, m):
        list_ = control_flow_ops.cond(
            math_ops.equal(list_ops.tensor_list_length(list_), 0),
            lambda: list_ops.empty_tensor_list(m.shape, m.dtype), lambda: list_)
        list_ = list_ops.tensor_list_push_back(list_, m)
        return list_, m

      for _ in range(2):
        list_, m = body(list_, m)

      s1 = list_ops.tensor_list_stack(list_, element_dtype=dtypes.float32)
      np_s1 = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.float32)
      self.assertAllEqual(self.evaluate(s1), np_s1)

  @test_util.run_in_graph_and_eager_modes
  def testGraphStackInLoopSwitchDtype(self):
    with context.graph_mode(), self.test_session():
      t1 = list_ops.empty_tensor_list(
          element_shape=constant_op.constant([], dtype=dtypes.int32),
          element_dtype=dtypes.int32)
      i = constant_op.constant(0, dtype=dtypes.float32)
      m = constant_op.constant([1, 2, 3], dtype=dtypes.float32)

      def body(i, m, t1):
        t1 = control_flow_ops.cond(
            math_ops.equal(list_ops.tensor_list_length(t1), 0),
            lambda: list_ops.empty_tensor_list(m.shape, m.dtype), lambda: t1)

        t1 = list_ops.tensor_list_push_back(t1, m * i)
        i += 1.0
        return i, m, t1

      i, m, t1 = control_flow_ops.while_loop(
          lambda i, m, t1: math_ops.less(i, 4), body, [i, m, t1])
      s1 = list_ops.tensor_list_stack(t1, element_dtype=dtypes.float32)
      np_s1 = np.vstack([np.arange(1, 4) * i for i in range(4)])
      self.assertAllEqual(self.evaluate(s1), np_s1)

  @test_util.run_in_graph_and_eager_modes
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
        self.assertAllEqual(self.evaluate(worker_e), [2.0])

  @test_util.run_in_graph_and_eager_modes
  def testPushPopGradients(self):
    with backprop.GradientTape() as tape:
      l = list_ops.empty_tensor_list(element_dtype=dtypes.float32,
                                     element_shape=scalar_shape())
      c = constant_op.constant(1.0)
      tape.watch(c)
      l = list_ops.tensor_list_push_back(l, c)
      l, e = list_ops.tensor_list_pop_back(l, element_dtype=dtypes.float32)
      e = 2 * e
    self.assertAllEqual(self.evaluate(tape.gradient(e, [c])[0]), 2.0)

  @test_util.run_in_graph_and_eager_modes
  def testStackFromTensorGradients(self):
    with backprop.GradientTape() as tape:
      c = constant_op.constant([1.0, 2.0])
      tape.watch(c)
      l = list_ops.tensor_list_from_tensor(c, element_shape=scalar_shape())
      c2 = list_ops.tensor_list_stack(
          l, element_dtype=dtypes.float32, num_elements=2)
      result = c2 * 2.0
    grad = tape.gradient(result, [c])[0]
    self.assertAllEqual(self.evaluate(grad), [2.0, 2.0])

  @test_util.run_in_graph_and_eager_modes
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
    self.assertAllEqual(self.evaluate(grad_c), [0.0, 4.0])
    self.assertAllEqual(self.evaluate(grad_c2), 6.0)

  @test_util.run_in_graph_and_eager_modes
  def testSetOutOfBounds(self):
    c = constant_op.constant([1.0, 2.0])
    l = list_ops.tensor_list_from_tensor(c, element_shape=scalar_shape())
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(list_ops.tensor_list_set_item(l, 20, 3.0))

  @test_util.run_in_graph_and_eager_modes
  def testResourceVariableScatterGather(self):
    c = constant_op.constant([1.0, 2.0], dtype=dtypes.float32)
    l = list_ops.tensor_list_from_tensor(c, element_shape=scalar_shape())
    v = vs.get_variable("var", initializer=[l] * 10, use_resource=True)
    v_r_0_stacked = list_ops.tensor_list_stack(v[0], dtypes.float32)
    self.evaluate(v.initializer)
    self.assertAllEqual([1.0, 2.0], self.evaluate(v_r_0_stacked))
    v_r_sparse_stacked = list_ops.tensor_list_stack(
        v.sparse_read(0), dtypes.float32)
    self.assertAllEqual([1.0, 2.0], self.evaluate(v_r_sparse_stacked))
    l_new_0 = list_ops.tensor_list_from_tensor(
        [3.0, 4.0], element_shape=scalar_shape())
    l_new_1 = list_ops.tensor_list_from_tensor(
        [5.0, 6.0], element_shape=scalar_shape())
    updated_v = state_ops.scatter_update(v, [3, 5], [l_new_0, l_new_1])
    updated_v_elems = array_ops.unstack(updated_v)
    updated_v_stacked = [
        list_ops.tensor_list_stack(el, dtypes.float32) for el in updated_v_elems
    ]
    expected = ([[1.0, 2.0]] * 3 + [[3.0, 4.0], [1.0, 2.0], [5.0, 6.0]] +
                [[1.0, 2.0]] * 4)
    self.assertAllEqual(self.evaluate(updated_v_stacked), expected)

  @test_util.run_in_graph_and_eager_modes
  def testConcat(self):
    c = constant_op.constant([1.0, 2.0], dtype=dtypes.float32)
    l0 = list_ops.tensor_list_from_tensor(c, element_shape=scalar_shape())
    l1 = list_ops.tensor_list_from_tensor([-1.0], element_shape=scalar_shape())
    l_batch_0 = array_ops.stack([l0, l1])
    l_batch_1 = array_ops.stack([l1, l0])

    l_concat_01 = list_ops.tensor_list_concat_lists(
        l_batch_0, l_batch_1, element_dtype=dtypes.float32)
    l_concat_10 = list_ops.tensor_list_concat_lists(
        l_batch_1, l_batch_0, element_dtype=dtypes.float32)
    l_concat_00 = list_ops.tensor_list_concat_lists(
        l_batch_0, l_batch_0, element_dtype=dtypes.float32)
    l_concat_11 = list_ops.tensor_list_concat_lists(
        l_batch_1, l_batch_1, element_dtype=dtypes.float32)

    expected_00 = [[1.0, 2.0, 1.0, 2.0], [-1.0, -1.0]]
    expected_01 = [[1.0, 2.0, -1.0], [-1.0, 1.0, 2.0]]
    expected_10 = [[-1.0, 1.0, 2.0], [1.0, 2.0, -1.0]]
    expected_11 = [[-1.0, -1.0], [1.0, 2.0, 1.0, 2.0]]

    for i, (concat, expected) in enumerate(zip(
        [l_concat_00, l_concat_01, l_concat_10, l_concat_11],
        [expected_00, expected_01, expected_10, expected_11])):
      splitted = array_ops.unstack(concat)
      splitted_stacked_ret = self.evaluate(
          (list_ops.tensor_list_stack(splitted[0], dtypes.float32),
           list_ops.tensor_list_stack(splitted[1], dtypes.float32)))
      print("Test concat %d: %s, %s, %s, %s"
            % (i, expected[0], splitted_stacked_ret[0],
               expected[1], splitted_stacked_ret[1]))
      self.assertAllClose(expected[0], splitted_stacked_ret[0])
      self.assertAllClose(expected[1], splitted_stacked_ret[1])

    # Concatenating mismatched shapes fails.
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      self.evaluate(
          list_ops.tensor_list_concat_lists(
              l_batch_0,
              list_ops.empty_tensor_list(scalar_shape(), dtypes.float32),
              element_dtype=dtypes.float32))

    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "element shapes are not identical at index 0"):
      l_batch_of_vec_tls = array_ops.stack(
          [list_ops.tensor_list_from_tensor([[1.0]], element_shape=[1])] * 2)
      self.evaluate(
          list_ops.tensor_list_concat_lists(l_batch_0, l_batch_of_vec_tls,
                                            element_dtype=dtypes.float32))

    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 r"input_b\[0\].dtype != element_dtype."):
      l_batch_of_int_tls = array_ops.stack(
          [list_ops.tensor_list_from_tensor([1], element_shape=scalar_shape())]
          * 2)
      self.evaluate(
          list_ops.tensor_list_concat_lists(l_batch_0, l_batch_of_int_tls,
                                            element_dtype=dtypes.float32))

  @test_util.run_in_graph_and_eager_modes
  def testPushBackBatch(self):
    c = constant_op.constant([1.0, 2.0], dtype=dtypes.float32)
    l0 = list_ops.tensor_list_from_tensor(c, element_shape=scalar_shape())
    l1 = list_ops.tensor_list_from_tensor([-1.0], element_shape=scalar_shape())
    l_batch = array_ops.stack([l0, l1])
    l_push = list_ops.tensor_list_push_back_batch(l_batch, [3.0, 4.0])
    l_unstack = array_ops.unstack(l_push)
    l0_ret = list_ops.tensor_list_stack(l_unstack[0], dtypes.float32)
    l1_ret = list_ops.tensor_list_stack(l_unstack[1], dtypes.float32)
    self.assertAllClose([1.0, 2.0, 3.0], self.evaluate(l0_ret))
    self.assertAllClose([-1.0, 4.0], self.evaluate(l1_ret))

    with ops.control_dependencies([l_push]):
      l_unstack_orig = array_ops.unstack(l_batch)
      l0_orig_ret = list_ops.tensor_list_stack(l_unstack_orig[0],
                                               dtypes.float32)
      l1_orig_ret = list_ops.tensor_list_stack(l_unstack_orig[1],
                                               dtypes.float32)

    # Check that without aliasing, push_back_batch still works; and
    # that it doesn't modify the input.
    l0_r_v, l1_r_v, l0_orig_v, l1_orig_v = self.evaluate(
        (l0_ret, l1_ret, l0_orig_ret, l1_orig_ret))
    self.assertAllClose([1.0, 2.0, 3.0], l0_r_v)
    self.assertAllClose([-1.0, 4.0], l1_r_v)
    self.assertAllClose([1.0, 2.0], l0_orig_v)
    self.assertAllClose([-1.0], l1_orig_v)

    # Pushing back mismatched shapes fails.
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      self.evaluate(list_ops.tensor_list_push_back_batch(l_batch, []))

    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "incompatible shape to a list at index 0"):
      self.evaluate(
          list_ops.tensor_list_push_back_batch(l_batch, [[3.0], [4.0]]))

    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "Invalid data type at index 0"):
      self.evaluate(list_ops.tensor_list_push_back_batch(l_batch, [3, 4]))


if __name__ == "__main__":
  test.main()
