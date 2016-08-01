# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

# pylint: disable=g-long-lambda
"""Tests for tensorflow.ops.control_flow_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.framework import function
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.util import nest

def check_op_order(graph):
  """Sanity check on the ordering of op id."""

  for op in graph.get_operations():
    for v in op.inputs:
      assert v.op._id < op._id or op.type == "Merge", (
          "The id of %s must be less than the id of %s" % (v.op.name, op.name))
  return True


def check_consumers(graph):
  """Sanity check on the consumer list of the tensors."""

  consumer_count = {}
  for op in graph.get_operations():
    for v in op.inputs:
      cnt = consumer_count.get(v, 0)
      consumer_count[v] = cnt + 1
  for k, v in consumer_count.items():
    if len(k.consumers()) != v:
      return False
  return True


def isum(s):
  i = tf.constant(0, name="i")
  c = lambda i, s: tf.less(i, 10)
  b = lambda i, s: [tf.add(i, 1), tf.add(i, s)]
  _, r_s = tf.while_loop(c, b, [i, s])
  return r_s


class ControlFlowTest(tf.test.TestCase):

  def testRefIdentity(self):
    with self.test_session():
      v = tf.Variable(7)

      v = control_flow_ops._Identity(v)
      op = tf.assign(v, 9)
      v2 = control_flow_ops.with_dependencies([op], v)

      self.assertTrue(check_op_order(v.graph))
      self.assertTrue(isinstance(v2, tf.Tensor))
      tf.initialize_all_variables().run()
      self.assertEqual(9, v2.eval())

  def testRefEnter(self):
    with self.test_session():
      v = tf.Variable(7)

      enter_v = control_flow_ops._Enter(v, "foo_1", is_constant=True)
      nine = tf.constant(9)
      enter_nine = control_flow_ops.enter(nine, "foo_1")
      op = tf.assign(enter_v, enter_nine)
      v2 = control_flow_ops.with_dependencies([op], enter_v)
      v3 = control_flow_ops.exit(v2)
      tf.initialize_all_variables().run()
      self.assertEqual(9, v3.eval())

  def testRefSwitch(self):
    with self.test_session():
      v = tf.Variable(7)

      p = tf.constant(True)
      v1 = control_flow_ops._SwitchRefOrTensor(v.ref(), p)
      v2 = tf.assign(v1[1], 9)
      tf.initialize_all_variables().run()
      self.assertEqual(9, v2.eval())

  def testEnterMulExit(self):
    with self.test_session():
      data = tf.constant([1, 2, 3, 4, 5, 6], name="data")
      enter_data = control_flow_ops.enter(data, "foo_1", False)
      five = tf.constant(5)
      enter_five = control_flow_ops.enter(five, "foo_1", False)
      mul_op = tf.mul(enter_data, enter_five)
      exit_op = control_flow_ops.exit(mul_op)

      result = exit_op.eval()
    self.assertAllEqual(np.array([x * 5 for x in [1, 2, 3, 4, 5, 6]]), result)

  def testSwitchMergeIndexedSlices(self):
    with self.test_session():
      values = tf.constant([1, 2, 3, 4, 5, 6])
      indices = tf.constant([0, 2, 4, 6, 8, 10])
      data = tf.IndexedSlices(values, indices)
      pred = tf.convert_to_tensor(True)
      switch_op = control_flow_ops.switch(data, pred)
      merge_op = control_flow_ops.merge(switch_op)[0]

      val = merge_op.values.eval()
      ind = merge_op.indices.eval()
    self.assertAllEqual(np.arange(1, 7), val)
    self.assertAllEqual(np.arange(0, 12, 2), ind)

  def testSwitchDeadBranch(self):
    with self.test_session():
      data = tf.constant([1, 2, 3, 4, 5, 6], name="data")
      ports = tf.convert_to_tensor(True, name="ports")
      switch_op = control_flow_ops.switch(data, ports)
      dead_branch = tf.identity(switch_op[0])

      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          lambda e: "The tensor returned for" in str(e)):
        dead_branch.eval()

  def testSwitchMergeLess(self):
    with self.test_session():
      data = tf.constant([1, 2, 3, 4, 5, 6], name="data")
      zero = tf.convert_to_tensor(0)
      one = tf.convert_to_tensor(1)
      less_op = tf.less(zero, one)
      switch_op = control_flow_ops.switch(data, less_op)
      merge_op = control_flow_ops.merge(switch_op)[0]

      result = merge_op.eval()
    self.assertAllEqual(np.arange(1, 7), result)

  def testSwitchMergeAddIdentity(self):
    with self.test_session():
      data = tf.constant([1, 2, 3, 4, 5, 6], name="data")
      ports = tf.convert_to_tensor(False, name="ports")
      switch_op = control_flow_ops.switch(data, ports)
      one = tf.constant(1)
      add_op = tf.add(switch_op[0], one)
      id_op = tf.identity(switch_op[1])
      merge_op = control_flow_ops.merge([add_op, id_op])[0]

      result = merge_op.eval()
    self.assertAllEqual(np.array([x + 1 for x in [1, 2, 3, 4, 5, 6]]), result)

  def testSwitchMergeAddMul(self):
    with self.test_session():
      data = tf.constant([1, 2, 3, 4, 5, 6], name="data")
      ports = tf.convert_to_tensor(True, name="ports")
      switch_op = control_flow_ops.switch(data, ports)
      one = tf.constant(1)
      add_op = tf.add(switch_op[0], one)
      five = tf.constant(5)
      mul_op = tf.mul(switch_op[1], five)
      merge_op = control_flow_ops.merge([add_op, mul_op])[0]

      result = merge_op.eval()
    self.assertAllEqual(np.array([x * 5 for x in [1, 2, 3, 4, 5, 6]]), result)

  def testLoop_false(self):
    with self.test_session():
      false = tf.convert_to_tensor(False)
      n = tf.constant(10)

      enter_false = control_flow_ops.enter(false, "foo_1", False)
      enter_n = control_flow_ops.enter(n, "foo_1", False)

      merge_n = control_flow_ops.merge([enter_n, enter_n], name="merge_n")[0]
      switch_n = control_flow_ops.switch(merge_n, enter_false)
      exit_n = control_flow_ops.exit(switch_n[0])
      next_n = control_flow_ops.next_iteration(switch_n[0])
      merge_n.op._update_input(1, next_n)

      result = exit_n.eval()
    self.assertAllEqual(10, result)

  def testLoop_1(self):
    with self.test_session():
      zero = tf.constant(0)
      one = tf.constant(1)
      n = tf.constant(10)

      enter_i = control_flow_ops.enter(zero, "foo", False)
      enter_one = control_flow_ops.enter(one, "foo", True)
      enter_n = control_flow_ops.enter(n, "foo", True)

      with tf.device("/gpu:0"):
        merge_i = control_flow_ops.merge([enter_i, enter_i])[0]

      less_op = tf.less(merge_i, enter_n)
      cond_op = control_flow_ops.loop_cond(less_op)
      switch_i = control_flow_ops.switch(merge_i, cond_op)

      add_i = tf.add(switch_i[1], enter_one)

      next_i = control_flow_ops.next_iteration(add_i)
      merge_i.op._update_input(1, next_i)

      exit_i = control_flow_ops.exit(switch_i[0])
      result = exit_i.eval()
    self.assertAllEqual(10, result)

  def testLoop_2(self):
    with self.test_session():
      zero = tf.constant(0)
      one = tf.constant(1)
      n = tf.constant(10)

      enter_i = control_flow_ops.enter(zero, "foo", False)
      enter_one = control_flow_ops.enter(one, "foo", True)
      enter_n = control_flow_ops.enter(n, "foo", True)

      merge_i = control_flow_ops.merge([enter_i, enter_i])[0]

      less_op = tf.less(merge_i, enter_n)
      cond_op = control_flow_ops.loop_cond(less_op)
      switch_i = control_flow_ops.switch(merge_i, cond_op)

      add_i = tf.add(switch_i[1], enter_one)

      with tf.device("/gpu:0"):
        next_i = control_flow_ops.next_iteration(add_i)
      merge_i.op._update_input(1, next_i)

      exit_i = control_flow_ops.exit(switch_i[0])
      result = exit_i.eval()
    self.assertAllEqual(10, result)

  def testCondBool(self):
    values = tf.constant(10)
    fn1 = lambda: tf.add(values, 1)
    fn2 = lambda: tf.sub(values, 1)
    with self.assertRaisesRegexp(TypeError, "must not be a Python bool"):
      _ = tf.cond(False, fn1, fn2)

  def testCondIndexedSlices(self):
    with self.test_session():
      values = tf.constant(10)
      indices = tf.constant(0)
      x = tf.IndexedSlices(values, indices)
      pred = tf.less(1, 2)
      fn1 = lambda: tf.IndexedSlices(tf.add(x.values, 1), indices)
      fn2 = lambda: tf.IndexedSlices(tf.sub(x.values, 1), indices)
      r = tf.cond(pred, fn1, fn2)

      val = r.values.eval()
      ind = r.indices.eval()
    self.assertTrue(check_op_order(x.values.graph))
    self.assertAllEqual(11, val)
    self.assertAllEqual(0, ind)

  def testCondIndexedSlicesDifferentTypes(self):
    with self.test_session():
      values = tf.constant(10)
      i_32 = tf.convert_to_tensor(0, name="one", dtype=tf.int32)
      i_64 = tf.convert_to_tensor(0, name="one", dtype=tf.int64)
      x = tf.IndexedSlices(values, i_32)
      pred = tf.less(1, 2)
      fn1 = lambda: tf.IndexedSlices(tf.add(x.values, 1), i_32)
      fn2 = lambda: tf.IndexedSlices(tf.sub(x.values, 1), i_64)
      r = tf.cond(pred, fn1, fn2)

      val = r.values.eval()
      ind = r.indices.eval()
    self.assertTrue(check_op_order(x.values.graph))
    self.assertAllEqual(11, val)
    self.assertAllEqual(0, ind)
    self.assertTrue(ind.dtype == np.int64)

  def testCondColocation(self):
    with self.test_session(use_gpu=True):
      with tf.device("/cpu:0"):
        v = tf.Variable(7.0)

      x = tf.constant(10.0)
      pred = tf.less(1.0, 2.0)
      fn1 = lambda: tf.add(v, 1.0)
      fn2 = lambda: tf.sub(x, 1.0)
      r = tf.cond(pred, fn1, fn2)

      for op in x.graph.get_operations():
        if op.name == "cond/Add/Switch":
          self.assertDeviceEqual(op.device, "/cpu:0")

  def _testCond_1(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      x = tf.constant(10)
      pred = tf.less(1, 2)
      fn1 = lambda: tf.add(x, 1)
      fn2 = lambda: tf.sub(x, 1)
      r = tf.cond(pred, fn1, fn2)

      result = r.eval()
    self.assertTrue(check_op_order(x.graph))
    self.assertAllEqual(11, result)

  def testCond_1(self):
    self._testCond_1(use_gpu=False)
    self._testCond_1(use_gpu=True)

  def testCond_2(self):
    with self.test_session():
      x = tf.constant(10)
      r = tf.cond(tf.less(1, 0), lambda: tf.add(x, 1), lambda: tf.sub(x, 1))
      result = r.eval()
    self.assertTrue(check_op_order(x.graph))
    self.assertAllEqual(9, result)

  def testCond_3(self):
    with self.test_session():
      x = tf.constant(10)
      pred = tf.less(1, 2)
      fn1 = lambda: tf.add(x, 1)
      fn2 = lambda: tf.sub(x, 1)
      fn3 = lambda: tf.add(tf.cond(pred, fn1, fn2), 1)
      r = tf.cond(pred, fn3, fn2)

      result = r.eval()
    self.assertTrue(check_op_order(x.graph))
    self.assertAllEqual(12, result)

  def testCond_4(self):
    with self.test_session():
      v1 = tf.Variable(7)
      v2 = tf.Variable(7)
      v3 = tf.Variable(7)

      age = tf.constant(3)
      max_age = tf.constant(2)
      pred = tf.greater(age, max_age)
      fn1 = lambda: [tf.assign(v1, 1).op, tf.assign(v2, 2).op]
      fn2 = lambda: [tf.assign(v3, 3).op, tf.constant(10).op]
      r = tf.cond(pred, fn1, fn2)

      tf.initialize_all_variables().run()
      self.assertEqual(len(r), 2)
      result = r[1].eval()
      self.assertTrue(check_op_order(age.graph))
      self.assertAllEqual(True, result)
      self.assertAllEqual(7, v1.eval())
      self.assertAllEqual(2, v2.eval())
      self.assertAllEqual(7, v3.eval())

  def testCond_5(self):
    with self.test_session():
      alive = tf.constant(True, name="alive")
      count = tf.constant(0, name="count")

      def body(i):
        return tf.cond(
            alive, lambda: [tf.less(i, 3), tf.add(count, 1)],
            lambda: [alive, count])

      for i in range(10):
        alive, count = body(i)
      self.assertAllEqual(4, count.eval())

  def testCond_6(self):
    with self.test_session():
      v1 = tf.Variable([7])

      age = tf.constant(3)
      pred = tf.greater(age, 4)
      fn1 = lambda: age
      fn2 = lambda: v1
      r = tf.cond(pred, fn1, fn2)

      tf.initialize_all_variables().run()
      result = r.eval()
      self.assertAllEqual(np.array([7]), result)

  def testCond_7(self):
    with self.test_session() as sess:
      x = tf.constant(10)
      y = tf.constant(200)
      pred = tf.less(1, 2)
      fn1 = lambda: [tf.add(x, 1), tf.add(x, 2)]
      fn2 = lambda: [y, y]
      r = tf.cond(pred, fn1, fn2)
      self.assertAllEqual([11, 12], sess.run(r))

  def testCondRef(self):
    with self.test_session():
      x = state_ops.variable_op([1], tf.float32)
      true_fn = lambda: x
      false_fn = lambda: tf.constant([2.0])
      r = tf.cond(tf.constant(False), true_fn, false_fn)
      self.assertAllEqual([2.0], r.eval())

  def testUninitializedRefIdentity(self):
    with self.test_session() as sess:
      v = state_ops.variable_op([1], tf.float32)
      inited = state_ops.is_variable_initialized(v)
      v_f, v_t = control_flow_ops.ref_switch(v, inited)
      # Both v_f and v_t are uninitialized references. However, an actual use
      # of the reference in the 'true' branch in the 'tf.identity' op will
      # not 'fire' when v is uninitialized, so this is a valid construction.
      # This test tests that _ref_identity allows uninitialized ref as input
      # so that this construction is allowed.
      v_f_op = gen_array_ops._ref_identity(v_f)
      v_t_op = gen_array_ops._ref_identity(v_t)
      with tf.control_dependencies([v_f_op]):
        assign_v = tf.assign(v, [1.0])
      with tf.control_dependencies([v_t_op]):
        orig_v = tf.identity(v)
      merged_op = control_flow_ops.merge([assign_v, orig_v])
      self.assertAllEqual([1.0], sess.run(merged_op.output))

  def testCondGrad_1(self):
    with self.test_session():
      x = tf.constant(10.0, name="x")
      pred = tf.less(1, 2)
      fn1 = lambda: tf.identity(x)
      fn2 = lambda: tf.identity(x)
      r = tf.cond(pred, fn1, fn2)

      grad = tf.gradients(r, [x])[0]
      result = grad.eval()
    self.assertAllEqual(1.0, result)

  def testCondGrad_2(self):
    with self.test_session():
      c = tf.placeholder(tf.int32, shape=[])
      x = tf.constant(10.0)
      pred = tf.less(c, 2)
      fn1 = lambda: tf.mul(x, 42.0)
      fn2 = lambda: tf.mul(x, 3.0)
      r = tf.cond(pred, fn1, fn2)

      grad = tf.gradients(r, [x])[0]
      self.assertAllEqual(42.0, grad.eval(feed_dict={c: 1}))
      self.assertAllEqual(3.0, grad.eval(feed_dict={c: 3}))

  def testNestedCond_Simple(self):
    with self.test_session():
      x = tf.constant(0., name="X")
      y = tf.cond(tf.constant(True),
                  lambda: x,
                  lambda: tf.cond(x < 1., lambda: x, lambda: x))
      result = tf.gradients(y, x)[0]
      self.assertEqual(1.0, result.eval())

      z = tf.cond(tf.constant(False),
                  lambda: x,
                  lambda: tf.cond(x < 1., lambda: x, lambda: x))
      result = tf.gradients(z, x)[0]
      self.assertEqual(1.0, result.eval())

  def testCondGrad_Gather(self):
    with self.test_session() as sess:
      v1 = tf.Variable([1.0, 42.0])
      c = tf.placeholder(tf.int32, shape=[])
      pred = tf.less(c, 2)
      fn1 = lambda: tf.identity(v1)
      fn2 = lambda: tf.gather(v1, [1, 1])
      r = tf.cond(pred, fn1, fn2)
      grad = tf.gradients(r, [v1])[0]
      tf.initialize_all_variables().run()
      # Should just be [1, 1], but possibly a sparse representation
      gv, gi = sess.run([grad.values, grad.indices], feed_dict={c: 1})
      dense_gv = [sum([y for (x, y) in zip(gi, gv) if x == i]) for i in range(2)
                 ]
      self.assertAllEqual(dense_gv, [1.0, 1.0])
      # Should be [0, 2], as the else forwards v1[1] twice
      gv, gi = sess.run([grad.values, grad.indices], feed_dict={c: 3})
      dense_gv = [sum([y for (x, y) in zip(gi, gv) if x == i]) for i in range(2)
                 ]
      self.assertAllEqual(dense_gv, [0.0, 2.0])

  # Microbenchmark: 10,000 iterations took 0.21s.
  def testWhile_1(self):
    with self.test_session():
      n = tf.constant(0)
      c = lambda x: tf.less(x, 10000)
      b = lambda x: tf.add(x, 1)
      r = tf.while_loop(c, b, [n], parallel_iterations=20)
      self.assertEqual(10000, r.eval())

  def testWhileWithRefs_1(self):
    with self.test_session() as sess:
      x = tf.Variable(0).ref()
      i = tf.constant(0)
      c = lambda i, x: tf.less(i, 100)

      self.assertEqual(x.dtype, tf.int32_ref)

      def b(i, x):
        self.assertEqual(x.dtype, tf.int32_ref)
        return (i+1, gen_array_ops._ref_identity(x))

      r = tf.while_loop(c, b, [i, x], parallel_iterations=5)

      tf.initialize_all_variables().run()

      self.assertEqual(r[0].dtype, tf.int32)
      self.assertEqual(r[1].dtype, tf.int32_ref)

      value_i, value_x = sess.run(r)

    self.assertEqual(100, value_i)
    self.assertEqual(0, value_x)

  def testWhile_2(self):
    with self.test_session():
      s = tf.constant(0)
      r = isum(s)
      self.assertAllEqual(45, r.eval())

  # Have more than 10 parallel iterations and hence exercise k-bound
  # most of the time.
  def testWhile_3(self):
    with self.test_session():

      def compute(i, m, c, o):
        m, c = [tf.add(m, 1), tf.add(c, 1)]
        o = tf.add(o, m)
        o = tf.add(o, c)
        i = tf.add(i, 1)
        return [i, m, c, o]

      i = tf.convert_to_tensor(0)
      m = tf.convert_to_tensor(0)
      c = tf.convert_to_tensor(0)
      o = tf.convert_to_tensor(0)
      d = tf.convert_to_tensor(100)
      r = tf.while_loop(
          lambda i, m, c, o: tf.less(i, d), compute, [i, m, c, o])
      result = r[3].eval()
    self.assertTrue(check_op_order(i.graph))
    self.assertAllEqual(10100, result)

  def testWhile_4(self):
    with self.test_session():

      def compute(i, m, c, o):
        m, c = [tf.gather(x, i), tf.gather(x, i)]
        o = tf.add(o, m)
        o = tf.add(o, c)
        i = tf.add(i, 1)
        return [i, m, c, o]

      i = tf.convert_to_tensor(0)
      m = tf.convert_to_tensor(0)
      c = tf.convert_to_tensor(0)
      o = tf.convert_to_tensor(0)
      x = tf.convert_to_tensor([1, 2, 3, 4, 5, 6])
      s = tf.size(x)
      r = tf.while_loop(
          lambda i, m, c, o: tf.less(i, s), compute, [i, m, c, o])
      result = r[3].eval()
    self.assertTrue(check_op_order(i.graph))
    self.assertAllEqual(42, result)

  def testWhile_5(self):
    with self.test_session():

      def compute(i, c, o):
        c = tf.slice(x, tf.expand_dims(i, 0), [1])
        o = tf.concat(0, [o, c])
        i = tf.add(i, 1)
        return [i, c, o]

      i = tf.convert_to_tensor(0)
      c = tf.convert_to_tensor(0)
      o = tf.convert_to_tensor([0])
      x = tf.convert_to_tensor([1, 2, 3, 4, 5, 6])
      s = tf.size(x)
      r = tf.while_loop(
          lambda i, c, o: tf.less(i, s), compute, [i, c, o])
      result = r[2].eval()
    self.assertTrue(check_op_order(i.graph))
    self.assertAllEqual(np.array([0, 1, 2, 3, 4, 5, 6]), result)

  def _testWhile_Gpu_1(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      n = tf.constant(1.0)
      c = lambda x: tf.less(x, 10.0)
      b = lambda x: tf.add(x, 1.0)
      r = tf.while_loop(c, b, [n])
      self.assertAllClose(10.0, r.eval())

  def testWhile_Gpu_1(self):
    self._testWhile_Gpu_1(use_gpu=False)
    self._testWhile_Gpu_1(use_gpu=True)

  def _testWhile_Gpu_2(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      n = tf.constant(1.0)
      c = lambda x: tf.less(x, 10.0)
      def b(x):
        with tf.device("/cpu:0"):
          return tf.add(x, 1.0)
      r = tf.while_loop(c, b, [n])
      self.assertAllClose(10.0, r.eval())

  def testWhile_Gpu_2(self):
    self._testWhile_Gpu_1(use_gpu=False)
    self._testWhile_Gpu_1(use_gpu=True)

  def testWhileShape(self):
    with self.test_session():
      i = tf.constant(0)
      m = tf.ones([2, 2])
      c = lambda i, j: tf.less(i, 2)
      def _b(i, j):
        new_i = tf.add(i, 1)
        new_j = tf.tile(j, [2, 2])
        return [new_i, new_j]
      r = tf.while_loop(c, _b, [i, m])
      r = r[1] * tf.ones([8, 8])
      self.assertAllEqual(np.ones((8, 8)), r.eval())

  def testWhileWithNonTensorInput_Scalar(self):
    with self.test_session():
      n = 0
      c = lambda x: x < 10000
      b = lambda x: x + 1
      r = tf.while_loop(c, b, [n], parallel_iterations=20)
      self.assertEqual(10000, r.eval())

  def testWhileWithNonTensorInput_Vector(self):
    with self.test_session():
      n = np.array([0])  # Note, [0] would not work here; that is a list
      c = lambda x: x[0] < 10000
      b = lambda x: tf.pack([x[0] + 1])
      r = tf.while_loop(c, b, [n], parallel_iterations=20)
      self.assertEqual([10000], r.eval())

  def testWhileShapeInference(self):
    with self.test_session():
      i = tf.constant(0)
      m = tf.ones([2, 2])
      c = lambda i, j: tf.less(i, 2)
      def _b(i, j):
        new_i = tf.add(i, 1)
        new_j = tf.concat(0, [j, j])
        return [new_i, new_j]
      r = tf.while_loop(c, _b, [i, m])
      self.assertTrue(r[1].get_shape()[0].value is None)
      self.assertEqual(r[1].get_shape()[1], tf.Dimension(2))

  def _testNestedWhile_1(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      n = tf.constant(0)
      def cpu_sum(s):
        c = lambda i, s: tf.less(i, 10)
        def b(i, s):
          i1 = tf.add(i, 1)
          with tf.device("/cpu:0"):
            s1 = tf.add(i, s)
          return i1, s1
        _, r_s = tf.while_loop(c, b, [n, s])
        return r_s
      c = lambda x: tf.less(x, 200)
      b = lambda x: tf.add(x, cpu_sum(n))
      r = tf.while_loop(c, b, [n])
      self.assertEqual(225, r.eval())

  def testNestedWhile_1(self):
    self._testNestedWhile_1(use_gpu=False)
    self._testNestedWhile_1(use_gpu=True)

  def testWhileWithControl_1(self):
    with self.test_session():
      n = tf.constant(0)
      r = tf.constant(0)
      condition = lambda n_, r_: tf.less(n_, 10)

      def body(n_, r_):
        n_ = tf.add(n_, 1)
        with r_.graph.control_dependencies([r_]):
          r_ = tf.constant(12)
        return [n_, r_]

      res = tf.while_loop(condition, body, [n, r],
                          parallel_iterations=1)
      self.assertAllEqual(12, res[1].eval())

  def testWhileWithControl_2(self):
    with self.test_session():
      r = tf.constant(0)
      condition = lambda r_: tf.less(r_, 10)

      def body(r_):
        with r_.graph.control_dependencies([r_]):
          r_ = tf.constant(12)
        return [r_]

      res = tf.while_loop(condition, body, [r], parallel_iterations=1)
      self.assertAllEqual(12, res.eval())

  def testWhileWithControl_3(self):
    with self.test_session() as sess:
      b = tf.placeholder(tf.bool)
      c = tf.constant(1)
      x0 = tf.constant(0)
      with tf.control_dependencies([b]):
        r = tf.while_loop(lambda x: x < 10, lambda x: x + c, [x0])
      self.assertEqual(10, sess.run(r, {b: True}))

  def testWhileWithControl_4(self):
    with self.test_session() as sess:
      b = tf.placeholder(tf.bool)
      c = tf.constant(1)
      x0 = tf.constant(0)
      with tf.control_dependencies([b]):
        r = tf.while_loop(lambda x: x < 10, lambda x: x + tf.identity(c), [x0])
      self.assertEqual(10, sess.run(r, {b: True}))

  def testCondWhile_1(self):
    with self.test_session():
      n = tf.convert_to_tensor(0, name="n")
      c = lambda x: tf.less(x, 10)
      b = lambda x: tf.add(x, 1)
      r = tf.cond(tf.less(0, 1),
                  lambda: tf.while_loop(c, b, [n]),
                  lambda: n)
      self.assertAllEqual(10, r.eval())

  def testCondWhile_2(self):
    with self.test_session():
      n = tf.convert_to_tensor(0)
      c = lambda x: tf.less(x, 10)
      b = lambda x: tf.add(x, 1)
      r = tf.cond(tf.less(1, 0), lambda: tf.add(n, 1),
                  lambda: tf.while_loop(c, b, [n]))
      self.assertAllEqual(10, r.eval())

  def testWhileCond_1(self):
    with self.test_session():
      i = tf.convert_to_tensor(0, name="i")
      n = tf.convert_to_tensor(10, name="n")
      one = tf.convert_to_tensor(1, name="one")
      c = lambda x: tf.less(x, n)
      # pylint: disable=undefined-variable
      # for OSS build
      b = lambda x: tf.cond(
          tf.constant(True), lambda: tf.add(x, one), lambda: tf.sub(x, one))
      # pylint: enable=undefined-variable
      r = tf.while_loop(c, b, [i])
      self.assertAllEqual(10, r.eval())

  def testWhileCond_2(self):
    with self.test_session():
      n = tf.convert_to_tensor(0, name="n")
      c = lambda x: tf.less(x, 10)
      b = lambda x: tf.cond(tf.constant(True), lambda: tf.add(x, 1), lambda: n)
      r = tf.while_loop(c, b, [n])
      self.assertAllEqual(10, r.eval())

  def testWhileCond_3(self):
    with self.test_session():
      n = tf.convert_to_tensor(0)
      c = lambda x: tf.less(x, 10)
      # pylint: disable=undefined-variable
      # for OSS build
      b = lambda x: tf.cond(tf.less(0, 1), lambda: tf.add(x, 1),
                            lambda: tf.sub(x, 1))
      # pylint: enable=undefined-variable
      r = tf.while_loop(c, b, [n])
      self.assertAllEqual(10, r.eval())

  # NOTE: It is ok to have parallel_iterations > 1
  def testWhileUpdateVariable_1(self):
    with self.test_session():
      select = tf.Variable([3.0, 4.0, 5.0])
      n = tf.constant(0)

      def loop_iterator(j):
        return tf.less(j, 3)

      def loop_body(j):
        ns = tf.scatter_update(select, j, 10.0)
        nj = tf.add(j, 1)
        op = control_flow_ops.group(ns)
        nj = control_flow_ops.with_dependencies([op], nj)
        return [nj]

      r = tf.while_loop(loop_iterator, loop_body, [n],
                        parallel_iterations=1)
      self.assertTrue(check_op_order(n.graph))
      tf.initialize_all_variables().run()
      self.assertEqual(3, r.eval())
      result = select.eval()
      self.assertAllClose(np.array([10.0, 10.0, 10.0]), result)

  def testWhileUpdateVariable_2(self):
    with self.test_session():
      select1 = tf.Variable([3.0, 4.0, 5.0])
      select2 = tf.Variable([3.0, 4.0, 5.0])
      n = tf.constant(0)

      def loop_iterator(j):
        return tf.less(j, 3)

      def loop_body(j):
        ns1 = tf.scatter_update(select1, j, 10.0)
        ns2 = tf.scatter_update(select2, j, 10.0)
        nj = tf.add(j, 1)
        op = control_flow_ops.group(ns1, ns2)
        nj = control_flow_ops.with_dependencies([op], nj)
        return [nj]

      r = tf.while_loop(loop_iterator, loop_body, [n],
                        parallel_iterations=1)
      self.assertTrue(check_op_order(n.graph))
      tf.initialize_all_variables().run()
      self.assertEqual(3, r.eval())
      result1 = select1.eval()
      self.assertAllClose(np.array([10.0, 10.0, 10.0]), result1)
      result2 = select2.eval()
      self.assertAllClose(np.array([10.0, 10.0, 10.0]), result2)

  def testWhileUpdateVariable_3(self):
    with self.test_session():
      select = tf.Variable([3.0, 4.0, 5.0])
      n = tf.constant(0)

      def loop_iterator(j, _):
        return tf.less(j, 3)

      def loop_body(j, _):
        ns = tf.scatter_update(select, j, 10.0)
        nj = tf.add(j, 1)
        return [nj, ns]

      r = tf.while_loop(loop_iterator, loop_body,
                        [n, tf.identity(select)],
                        parallel_iterations=1)
      tf.initialize_all_variables().run()
      result = r[1].eval()
    self.assertTrue(check_op_order(n.graph))
    self.assertAllClose(np.array([10.0, 10.0, 10.0]), result)

  # b/24814703
  def testWhileUpdateVariable_4(self):
    with self.test_session():
      var_a = tf.Variable(0, name="a")
      var_b = tf.Variable(0, name="b")
      tf.initialize_all_variables().run()

      c = tf.constant(0, name="c")
      asn1 = tf.assign_add(var_a, 1, name="a_add")
      # Loop condition
      def pred(i):
        return tf.less(i, 10)
      # Loop body
      def loop_body(i):
        asn2 = tf.assign_add(var_b, asn1, name="b_add")
        with tf.control_dependencies([asn2]):
          ni = tf.add(i, 1, name="i_add")
        return ni

      lpa = tf.while_loop(pred, loop_body, [c],
                          parallel_iterations=1)

      self.assertEqual(0, var_b.eval())
      lpa.eval()  # Run the loop
      self.assertEqual(10, var_b.eval())

  # b/24736492
  def testWhileUpdateVariable_5(self):
    with self.test_session():
      # Create some variables.
      var_a = tf.Variable(0, name="a")
      var_b = tf.Variable(0, name="b")
      tf.initialize_all_variables().run()

      # Change condition to check var_b
      def pred(_):
        return tf.less(var_b, 10)

      # Change body to increment var_b
      def loop_body(i):
        asn1 = tf.assign_add(var_a, tf.constant(1), name="a_add")
        asn2 = tf.assign_add(var_b, tf.constant(1), name="b_add")
        with tf.control_dependencies([asn1, asn2]):
          inc_b = tf.identity(var_b)
        return inc_b

      lpa = tf.while_loop(pred, loop_body, [var_b], 1, name="loop")

      self.assertEqual(0, var_b.eval())
      lpa.eval()  # Run the loop
      self.assertEqual(10, var_a.eval())
      self.assertEqual(10, var_b.eval())

  # b/24814668
  def testWhileUpdateVariable_6(self):
    with self.test_session():
      # Create some variables.
      var_a = tf.Variable(0, name="a")
      var_b = tf.Variable(0, name="b")
      c = tf.constant(0)
      tf.initialize_all_variables().run()

      # Loop condition
      def pred(i):
        return tf.less(i, 10)

      # Loop body
      def loop_body(i):
        asn1 = tf.assign_add(var_a, 1, name="a_add")
        with tf.control_dependencies([asn1]):
          asn2 = tf.assign_add(var_b, var_a, name="b_add")
        with tf.control_dependencies([asn2]):
          ni = tf.add(i, 1, name="i_add")
          return ni

      lpa = tf.while_loop(pred, loop_body, [c], 1, name="loop")

      self.assertEqual(0, var_b.eval())
      lpa.eval()  # Run the loop
      self.assertEqual(55, var_b.eval())
      self.assertEqual(10, var_a.eval())

  def testWhileQueue_1(self):
    with self.test_session():
      q = tf.FIFOQueue(-1, tf.int32)
      i = tf.constant(0)

      def c(i):
        return tf.less(i, 10)

      def b(i):
        ni = tf.add(i, 1)
        ni = control_flow_ops.with_dependencies([q.enqueue((i,))], ni)
        return ni

      r = tf.while_loop(c, b, [i], parallel_iterations=1)
      self.assertEqual([10], r.eval())
      for i in xrange(10):
        self.assertEqual([i], q.dequeue().eval())

  def testWhileStack_1(self):
    with self.test_session():
      s = gen_data_flow_ops._stack(tf.int32, stack_name="foo")
      i = tf.constant(0)

      def c(i):
        return tf.less(i, 10)
      def b(i):
        ni = tf.add(i, 1)
        ni = control_flow_ops.with_dependencies(
            [gen_data_flow_ops._stack_push(s, i)], ni)
        return ni
      r = tf.while_loop(c, b, [i], parallel_iterations=1)

      x = tf.constant(0)
      def c1(i, _):
        return tf.greater(i, 0)
      def b1(i, x):
        ni = tf.sub(i, 1)
        nx = x + gen_data_flow_ops._stack_pop(s, tf.int32)
        return [ni, nx]
      _, rx = tf.while_loop(c1, b1, [r, x], parallel_iterations=1)
      self.assertEqual(45, rx.eval())

  def _testWhileGrad_ColocateGradients(self, colocate):
    with self.test_session(graph=tf.Graph()) as sess:
      v = tf.constant(2.0, name="v")
      c = lambda v: tf.less(v, 100.0)
      def b(x):
        with tf.device("/gpu:0"):
          return tf.square(x)
      loop = tf.while_loop(c, b, [v], parallel_iterations=1)
      r = tf.gradients(loop, v, colocate_gradients_with_ops=colocate)[0]
    r_ops = r.graph.get_operations()
    r_devices = [(op.name, op.device.lower()) for op in r_ops]

    self.assertTrue(any("Square" in op.name for op in r_ops))

    for (name, dev) in r_devices:
      if not colocate and name.endswith("Square"):
        # Only forward graph contain gpu in Square device
        self.assertTrue("gpu:0" in dev)
      elif colocate and "Square" in name:
        # Forward and backward graphs contain gpu in Square/Square_grad devices
        self.assertTrue("gpu:0" in dev)
      else:
        self.assertFalse("gpu:0" in dev)
    self.assertAllClose(1024.0, sess.run(r))

  def testWhileGrad_ColocateGradients(self):
    self._testWhileGrad_ColocateGradients(colocate=False)
    self._testWhileGrad_ColocateGradients(colocate=True)

  def testWhileGrad_Square(self):
    with self.test_session():
      v = tf.constant(2.0, name="v")
      c = lambda v: tf.less(v, 100.0)
      b = tf.square
      r = tf.while_loop(c, b, [v], parallel_iterations=1)
      r = control_flow_ops.cond(tf.less(1, 2), lambda: r, lambda: v)

      r = tf.gradients(r, v)[0]
      self.assertAllClose(1024.0, r.eval())

  def testWhileGrad_Shape(self):
    with self.test_session():
      x = tf.placeholder(tf.float32, shape=[None])
      v = tf.constant([2.0], name="v")
      n = tf.constant(0, name="n")
      c = lambda i, v: tf.less(i, 5)
      b = lambda i, v: [i + 1, tf.mul(x, v)]
      r = tf.while_loop(c, b, [n, v], parallel_iterations=1)

      r = tf.gradients(r[1], x)[0]
      self.assertEqual([None], r.get_shape().as_list())
      self.assertAllClose([810.0, 2560.0], r.eval(feed_dict={x: [3.0, 4.0]}))

  def testWhileGrad_BaseShape(self):
    with self.test_session() as sess:
      x = tf.placeholder(tf.float32, [None])
      v0 = tf.constant([2.0, 2.0], name="v")
      c = lambda v: tf.constant(False)
      b = lambda v: tf.mul(v, x)
      r = tf.while_loop(c, b, [v0])
      y = tf.square(x)

      r = tf.gradients([r, y], x)[0]
      self.assertAllClose([2.0, 4.0], sess.run(r, feed_dict={x: [1.0, 2.0]}))

  def testWhileGrad_MultipleUses(self):
    with self.test_session():
      v = tf.constant(2.0, name="v")
      c = lambda v: tf.less(v, 100.0)
      b = tf.square
      r = tf.while_loop(c, b, [v], parallel_iterations=1)
      r = tf.mul(r, r)

      r = tf.gradients(r, v)[0]
      self.assertEqual(524288.0, r.eval())

  def testWhileGrad_LoopAdd(self):
    with self.test_session():
      v = tf.constant(2.0, name="v")
      c = lambda v: tf.less(v, 100.0)
      b = tf.square
      r = tf.while_loop(c, b, [v], parallel_iterations=1)
      r = tf.add(r, r)

      r = tf.gradients(r, v)[0]
      self.assertAllClose(2048.0, r.eval())

  def _testWhileGrad_Mul(self, use_gpu, p_iters):
    with self.test_session(use_gpu=use_gpu) as sess:
      a = tf.constant(3.0, name="a")
      v = tf.constant(2.0, name="v")
      c = lambda v: tf.less(v, 100.0)
      b = lambda v: tf.mul(v, a)
      r = tf.while_loop(c, b, [v], parallel_iterations=p_iters)

      grad_a, grad_v = tf.gradients(r, [a, v])
      grad_a_val, grad_v_val = sess.run([grad_a, grad_v])
      self.assertAllClose(216.0, grad_a_val)
      self.assertAllClose(81.0, grad_v_val)

  def testWhileGrad_Mul(self):
    self._testWhileGrad_Mul(use_gpu=False, p_iters=1)
    self._testWhileGrad_Mul(use_gpu=False, p_iters=10)
    self._testWhileGrad_Mul(use_gpu=True, p_iters=1)
    self._testWhileGrad_Mul(use_gpu=True, p_iters=10)

  def testWhileGrad_Variable(self):
    with self.test_session():
      a = tf.Variable(3.0)
      v = tf.constant(2.0, name="v")
      c = lambda v: tf.less(v, 100.0)
      b = lambda v: tf.mul(v, a)
      r = tf.while_loop(c, b, [v], parallel_iterations=1)

      r = tf.gradients(r, a)
      tf.initialize_all_variables().run()
      self.assertAllClose(216.0, r[0].eval())

  def testWhile_NestedInput(self):
    with self.test_session() as sess:
      named = collections.namedtuple("named", ("a", "b"))
      loop_vars = [named(a=tf.constant(0.0), b=tf.constant(1.0)),
                   (tf.constant(2.0), tf.constant(3.0)),
                   tf.constant(4.0)]
      c = lambda lv0, _1, _2: lv0.a < 100.0
      def b(lv0, lv1, lv2):
        lv0 = named(a=lv0.a + 1, b=lv0.b)
        lv1 = (lv1[0] + 1, lv1[1])
        lv2 += 2
        return [lv0, lv1, lv2]
      r = tf.while_loop(c, b, loop_vars)

      self.assertTrue(isinstance(r, list))
      self.assertTrue(isinstance(r[0], named))
      self.assertTrue(isinstance(r[1], tuple))
      self.assertTrue(isinstance(r[2], tf.Tensor))

      r_flattened = nest.flatten(r)
      self.assertEqual(
          [100.0, 1.0, 102.0, 3.0, 4.0 + 100*2.0],
          sess.run(r_flattened))

  def testWhile_NestedBadArityFails(self):
    with self.test_session():
      named = collections.namedtuple("named", ("a", "b"))
      loop_vars = [named(a=tf.constant(0.0), b=tf.constant(1.0)),
                   (tf.constant(2.0), tf.constant(3.0)),
                   tf.constant(4.0)]
      c = lambda lv0, _1, _2: lv0.a < 100.0
      def b(lv0, lv1, _):
        return [lv0, lv1]

      with self.assertRaisesRegexp(ValueError, "the same number of elements"):
        tf.while_loop(c, b, loop_vars)

  def testWhileGrad_ys_xs(self):
    with self.test_session():
      x = tf.constant(3.0, name="x")
      y = tf.constant(2.0, name="y")

      c = lambda x, y: tf.less(x, 100.0)
      def b(x, y):
        y1 = tf.add(x, y)
        x1 = tf.mul(x, y1)
        return x1, y1
      rx, ry = tf.while_loop(c, b, [x, y], parallel_iterations=1)

      r = tf.gradients([rx, ry], x)
      self.assertAllClose(304.0, r[0].eval())
      r = tf.gradients([rx, ry], y)
      self.assertAllClose(124.0, r[0].eval())
      r = tf.gradients([rx], x)
      self.assertAllClose(295.0, r[0].eval())
      r = tf.gradients([rx], y)
      self.assertAllClose(120.0, r[0].eval())

  def testWhileGrad_Dependency(self):
    with self.test_session():
      i = tf.constant(0, name="i")
      x = tf.constant(2.0, name="x")

      c = lambda i, x: tf.less(i, 10)
      def b(i, x):
        x = tf.mul(x, 2.0)
        i = tf.add(i, 1)
        return i, x
      ri, rx = tf.while_loop(c, b, [i, x], parallel_iterations=1)

      r = tf.gradients([ri, rx], x)
      self.assertAllClose(1024.0, r[0].eval())
      r = tf.gradients([rx], x)
      self.assertAllClose(1024.0, r[0].eval())

  def testWhileGrad_NoGradient(self):
    with self.test_session():
      v = tf.constant(2.0, name="v")
      c = lambda v: tf.less(v, 100.0)
      b = tf.square
      r = tf.while_loop(c, b, [v], back_prop=False)
      r = tf.add(r, v)
      r = tf.gradients(r, v)
      self.assertAllClose(1.0, r[0].eval())

  def testWhileGrad_NoDependency(self):
    with self.test_session() as sess:
      variable = tf.Variable(tf.ones([2, 3]))
      time = tf.zeros([], dtype=tf.int32)
      def cond(time, tensor, _):
        return time < 10
      def body(time, tensor, _):
        return (time+1, tensor, tensor)
      loop_vars = [time, variable, variable]
      tensors = tf.while_loop(cond=cond, body=body, loop_vars=loop_vars)
      cost = tf.reduce_sum(tensors[2])
      grad = tf.gradients(cost, [variable])
      tf.initialize_all_variables().run()
      self.assertAllClose(np.ones([2, 3]), sess.run(grad[0]))

  def testWhileGrad_Const(self):
    with self.test_session() as sess:
      c0 = tf.constant(0.0, name="c0")
      c1 = tf.constant(1.0, name="c1")
      time = tf.constant(0, name="t")
      def cond(time, _):
        return time < 1
      def body(time, tensor):
        return time+1, c1
      loop_vars = [time, c0]
      tensors = tf.while_loop(cond=cond, body=body, loop_vars=loop_vars)
      cost = tf.reduce_sum(tensors[1])
      grad = tf.gradients(cost, [c0])
      self.assertAllClose(0.0, sess.run(grad[0]))

  def testWhileGrad_SerialTwoLoops(self):
    with self.test_session():
      i = tf.constant(0, name="i")
      x = tf.constant(2.0, name="x")

      c = lambda i, x: tf.less(i, 5)
      def b(i, x):
        x = tf.mul(x, 2.0)
        i = tf.add(i, 1)
        return i, x
      _, rx = tf.while_loop(c, b, [i, x], parallel_iterations=1)
      _, rx = tf.while_loop(c, b, [i, rx], parallel_iterations=1)

      r = tf.gradients([rx], x)
      self.assertAllClose(1024.0, r[0].eval())

  def testWhileGrad_ParallelTwoLoops(self):
    with self.test_session():
      i = tf.constant(0, name="i")
      x = tf.constant(2.0, name="x")

      c = lambda i, x: tf.less(i, 5)
      def b(i, x):
        x = tf.mul(x, 2.0)
        i = tf.add(i, 1)
        return i, x
      _, r1 = tf.while_loop(c, b, [i, x], parallel_iterations=1)
      _, r2 = tf.while_loop(c, b, [i, x], parallel_iterations=1)
      rx = tf.add(r1, r2)

      r = tf.gradients([rx], x)
      self.assertAllClose(64.0, r[0].eval())

  def testWhileGrad_OneOutputWithControlDependencyOnSecond(self):
    with self.test_session():
      i = tf.constant(0, name="i")
      x = tf.constant(1.0, name="x")
      y = tf.constant(1.0, name="y")
      c = lambda i, *_: tf.less(i, 1, name="cond_less")
      def b(i, xi, yi):
        # return (i + 1, xi, xi + yi)
        return (tf.add(i, 1, name="inc"),
                tf.identity(xi, name="xi"),
                tf.add(xi, yi, name="xi_plus_yi"))

      _, x_f, y_f = tf.while_loop(c, b, [i, x, y])
      with tf.control_dependencies([x_f]):
        y_f_d = tf.identity(y_f, name="y_f_d")

      self.assertAllClose(2.0, y_f_d.eval())  # y_f_d = 1.0 + 1.0
      g = tf.gradients([y_f_d], [x])[0]
      self.assertTrue(g is not None)
      self.assertAllClose(1.0, g.eval())  # y_f_d = x + 1.0, dy_f_d/dx = 1.0

  def _testNestedWhileGrad_Simple(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      v = tf.constant(1.0)
      def inner_loop(s):
        c = lambda x: tf.less(x, 4.0)
        b = lambda x: tf.mul(x, 2.0)
        return tf.while_loop(c, b, [s])
      c = lambda x: tf.less(x, 2.0)
      b = lambda x: tf.mul(inner_loop(x), 2.0)
      r = tf.while_loop(c, b, [v])

      r = tf.gradients(r, v)[0]
      self.assertAllClose(8.0, r.eval())

  def testNestedWhileGrad_Simple(self):
    self._testNestedWhileGrad_Simple(use_gpu=False)
    self._testNestedWhileGrad_Simple(use_gpu=True)

  def testNestedWhileGrad_SerialInner(self):
    with self.test_session():
      v = tf.constant(1.0)
      def inner_loop1(s):
        z = tf.constant(0)
        c = lambda i, x: tf.less(i, 4)
        b = lambda i, x: [tf.add(i, 1), tf.mul(x, 2.0)]
        return tf.while_loop(c, b, [z, s])
      def inner_loop2(s):
        z = tf.constant(0)
        c = lambda i, x: tf.less(i, 4)
        b = lambda i, x: [tf.add(i, 1), tf.mul(x, 2.0)]
        return tf.while_loop(c, b, [z, s])
      c = lambda x: tf.less(x, 128.0)
      b = lambda x: inner_loop2(inner_loop1(x)[1])[1]
      r = tf.while_loop(c, b, [v])

      r = tf.gradients(r, v)[0]
      self.assertAllClose(256.0, r.eval())

  def testNestedWhileGrad_ParallelInner(self):
    with self.test_session():
      v = tf.constant(1.0)
      def inner_loop1(s):
        z = tf.constant(0)
        c = lambda i, x: tf.less(i, 4)
        b = lambda i, x: [tf.add(i, 1), tf.mul(x, 2.0)]
        return tf.while_loop(c, b, [z, s])
      def inner_loop2(s):
        z = tf.constant(0)
        c = lambda i, x: tf.less(i, 4)
        b = lambda i, x: [tf.add(i, 1), tf.mul(x, 2.0)]
        return tf.while_loop(c, b, [z, s])
      c = lambda x: tf.less(x, 128.0)
      b = lambda x: tf.mul(inner_loop1(x)[1], inner_loop2(x)[1])
      r = tf.while_loop(c, b, [v])

      r = tf.gradients(r, v)[0]
      self.assertAllClose(512.0, r.eval())

  def _testWhileCondGrad_Simple(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      v = tf.convert_to_tensor(2.0, name="v")
      n = tf.convert_to_tensor(100.0, name="n")
      one = tf.convert_to_tensor(1.0, name="one")
      c = lambda x: tf.less(x, n)
      # pylint: disable=undefined-variable
      # for OSS build
      b = lambda x: control_flow_ops.cond(tf.constant(True),
                                          lambda: tf.square(x),
                                          lambda: tf.sub(x, one))
      # pylint: enable=undefined-variable
      r = tf.while_loop(c, b, [v])
      r = tf.gradients(r, v)[0]
      self.assertAllClose(1024.0, r.eval())

  def testWhileCondGrad_Simple(self):
    self._testWhileCondGrad_Simple(use_gpu=False)
    self._testWhileCondGrad_Simple(use_gpu=True)

  def testWhileCondGrad_UnknownShape(self):
    with self.test_session() as sess:
      v = tf.placeholder(tf.float32)
      n = tf.convert_to_tensor(100.0, name="n")
      one = tf.convert_to_tensor(1.0, name="one")
      c = lambda x: tf.less(x, n)
      # pylint: disable=undefined-variable
      # for OSS build
      b = lambda x: control_flow_ops.cond(tf.constant(True),
                                          lambda: tf.square(x),
                                          lambda: tf.sub(x, one))
      # pylint: enable=undefined-variable
      r = tf.while_loop(c, b, [v])
      r = tf.gradients(r, v)[0]
      r = sess.run(r, feed_dict={v: 2.0})
      self.assertAllClose(1024.0, r)

  def testWhileWithRefsWithGradients_1(self):
    with self.test_session() as sess:
      x = tf.Variable(0).ref()
      i = tf.constant(0)
      c = lambda i, x: tf.less(i, 10)

      self.assertEqual(x.dtype, tf.int32_ref)

      # pylint: disable=protected-access
      def body(i, x):
        self.assertEqual(x.dtype, tf.int32_ref)
        return [i+1, gen_array_ops._ref_identity(x)]
      # pylint: enable=protected-access

      r = tf.while_loop(c, body, [i, x], parallel_iterations=5)

      grad_ys = [tf.Variable(73).ref()]
      grad = tf.gradients([r[1]], [x], grad_ys=grad_ys)

      tf.initialize_all_variables().run()

      self.assertEqual(r[0].dtype, tf.int32)
      self.assertEqual(r[1].dtype, tf.int32_ref)

      value_i, value_x, value_x_grad = sess.run(r + grad)

    self.assertEqual(10, value_i)
    self.assertEqual(0, value_x)
    self.assertEqual(73, value_x_grad)

  def testWhileGrad_IndexedSlices(self):
    with self.test_session():
      values = tf.constant([2.0, 4.0], name="values")
      indices = tf.constant([0, 3], name="indices")
      shape = tf.constant([10], name="dense_shape")
      i = tf.constant(0)
      x = tf.IndexedSlices(values, indices, dense_shape=shape)
      def c(i, _):
        return i < 10
      def b(i, x):
        return [i + 1, tf.IndexedSlices(x.values * 2.0, x.indices,
                                        x.dense_shape)]
      _, r = tf.while_loop(c, b, [i, x])
      r = tf.gradients(r.values, values)[0]
      self.assertAllClose(np.array([1024.0, 1024.0]), r.eval())

  def testWhileGrad_SparseTensor(self):
    with self.test_session():
      values = tf.constant([2.0, 4.0], name="values")
      indices = tf.constant([[0], [3]], dtype=tf.int64, name="indices")
      shape = tf.constant([10], dtype=tf.int64, name="dense_shape")
      i = tf.constant(0)
      x = tf.SparseTensor(indices, values, shape=shape)
      def c(i, _):
        return i < 10
      def b(i, x):
        return [i + 1, tf.SparseTensor(x.indices, x.values * 2.0,
                                       x.shape)]
      _, r = tf.while_loop(c, b, [i, x])
      r = tf.gradients(r.values, values)[0]
      self.assertAllClose(np.array([1024.0, 1024.0]), r.eval())

  def testCallGradInLoop(self):
    with self.test_session() as sess:
      i0 = tf.constant(0)
      params = tf.constant(5.0)
      params_1 = tf.square(params)
      def c(i, _):
        return i < 10
      def b(i, x):
        data = tf.constant([1.0, 2.0, 3.0])
        data = tf.mul(data, params_1)
        x1 = x + tf.gradients(data, params)[0]
        return i + 1, x1
      output_grad = tf.while_loop(c, b, [i0, tf.constant(0.0)])
      self.assertAllClose(600.0, sess.run(output_grad)[1])

  def testWhileGrad_StopGrad(self):
    with self.test_session():
      x = tf.constant(3.0, name="x")
      y = tf.constant(2.0, name="y")

      c = lambda x, y: tf.less(x, 100.0)
      def b(x, y):
        y1 = tf.stop_gradient(tf.square(y))
        x1 = tf.add(tf.square(x), y1)
        return x1, y1
      rx, _ = tf.while_loop(c, b, [x, y])

      r = tf.gradients(rx, y)[0]
      self.assertAllClose(0.0, r.eval())
      r = tf.gradients(rx, x)[0]
      self.assertAllClose(156.0, r.eval())

  def testWhileGradGrad(self):
    theta = tf.Variable(initial_value=1.)
    def fn(prev, x):
      return prev + x * theta
    result = tf.scan(fn, np.array([1., 2., 3.], dtype=np.float32))
    grad_theta = tf.gradients(result, theta)
    with self.assertRaisesRegexp(TypeError, "Second-order gradient"):
      tf.gradients(grad_theta, theta)

  def testOneValueCond(self):
    with self.test_session():
      c = tf.placeholder(tf.int32, shape=[])
      one = tf.convert_to_tensor(1, name="one")
      two = tf.convert_to_tensor(2, name="two")
      p = tf.greater_equal(c, 1)
      i = tf.cond(p, lambda: one, lambda: two)
      self.assertTrue(isinstance(i, tf.Tensor))

      # True case: c = 2 is >= 1
      self.assertEqual([1], i.eval(feed_dict={c: 2}))

      # False case: c = 0 is not >= 1
      self.assertEqual([2], i.eval(feed_dict={c: 0}))

  def testExampleCond(self):
    with self.test_session():
      x = tf.convert_to_tensor([-2.0, 2.0], name="x")
      d = tf.placeholder(tf.int32, shape=[])

      def l2():
        return tf.sqrt(tf.reduce_sum(tf.square(x)))

      def l1():
        return tf.reduce_sum(tf.abs(x))

      i = tf.cond(tf.equal(d, 2), l2, l1)
      self.assertAllClose(4.0, i.eval(feed_dict={d: 1}))
      self.assertAllClose(2.0 * math.sqrt(2), i.eval(feed_dict={d: 2}))

  def testCase(self):
    with self.test_session():
      x = tf.constant(1)
      y = tf.constant(2)
      z = tf.constant(3)
      f1 = lambda: tf.constant(17)
      f2 = lambda: tf.constant(23)
      f3 = lambda: tf.constant(-1)

      r1 = tf.case({x < y: f1, x > z: f2}, default=f3, exclusive=True)
      self.assertAllEqual(r1.eval(), 17)

      r2 = tf.case([(y > z, f1), (y > x, f2)], default=f3)
      self.assertAllEqual(r2.eval(), 23)

      # Duplicate events can happen, first one is selected
      r3 = tf.case([(x < y, f1), (x < y, f2)], default=f3)
      self.assertAllEqual(r3.eval(), 17)

      # Duplicate events cause an error if exclusive = True
      r4 = tf.case([(x < y, f1), (x < y, f2)], default=f3, exclusive=True)
      with self.assertRaisesOpError(
          "More than one condition evaluated as True but exclusive=True."):
        r4.eval()

      # Check that the default is called if none of the others are
      r5 = tf.case({x > y: f1}, default=f3)
      self.assertAllEqual(r5.eval(), -1)

      ran_once = [False, False, False]

      def break_run_twice(ix):
        def _break():
          ran_once[ix] = True
          return tf.constant(ix)
        return _break

      # Should not fail - each conditional gets called exactly once
      # except default.  Default gets called twice: once to create an
      # empty output and once for the actual cond switch.
      r6 = tf.case([(x < y, break_run_twice(0)), (x > y, break_run_twice(1))],
                   default=lambda: tf.constant(2))

      self.assertAllEqual(r6.eval(), 0)

  def testCaseSideEffects(self):
    with self.test_session() as sess:
      v0 = tf.Variable(-1)
      v1 = tf.Variable(-1)
      v2 = tf.Variable(-1)

      a = lambda: control_flow_ops.with_dependencies([tf.assign(v0, 0)], 0)
      b = lambda: control_flow_ops.with_dependencies([tf.assign(v1, 1)], 1)
      c = lambda: control_flow_ops.with_dependencies([tf.assign(v2, 2)], 2)

      x = tf.constant(1)
      y = tf.constant(2)

      r0 = tf.case(((x < y, a), (x > y, b)), default=c, exclusive=True)
      r1 = tf.case(((x > y, a), (x < y, b)), default=c, exclusive=True)
      r2 = tf.case(((x > y, a), (x > y, b)), default=c, exclusive=True)

      tf.initialize_all_variables().run()
      self.assertAllEqual(sess.run([v0, v1, v2]), [-1] * 3)
      self.assertEqual(2, r2.eval())
      self.assertAllEqual(sess.run([v0, v1, v2]), [-1, -1, 2])

      tf.initialize_all_variables().run()
      self.assertAllEqual(sess.run([v0, v1, v2]), [-1] * 3)
      self.assertEqual(1, r1.eval())
      self.assertAllEqual(sess.run([v0, v1, v2]), [-1, 1, -1])

      tf.initialize_all_variables().run()
      self.assertAllEqual(sess.run([v0, v1, v2]), [-1] * 3)
      self.assertEqual(0, r0.eval())
      self.assertAllEqual(sess.run([v0, v1, v2]), [0, -1, -1])

  def testOneOpCond(self):
    with self.test_session():
      v = tf.Variable(0)
      c = tf.convert_to_tensor(0)
      one = tf.convert_to_tensor(1)
      two = tf.convert_to_tensor(2)
      p = tf.greater_equal(c, 1)

      def a():
        return tf.assign(v, one)

      def b():
        return tf.assign(v, two)

      i = tf.cond(p, a, b)
      self.assertTrue(isinstance(i, tf.Tensor))
      tf.initialize_all_variables().run()

      self.assertEqual(0, v.eval())

      # True case: c = 2 is >= 1, v is set to 1.
      self.assertEqual(1, i.eval(feed_dict={c.name: 2}))
      self.assertEqual(1, v.eval())

      # False case: c = 0 is not >= 1, v is set to 2.
      self.assertEqual(2, i.eval(feed_dict={c.name: 0}))
      self.assertEqual(2, v.eval())

  def testWithOpsDependencies(self):
    with self.test_session() as sess:
      v = tf.Variable(0.0)
      c = tf.constant(10)

      # Fetching v directly will result in an uninitialized error
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        sess.run([c, v])

      # Use a control dependency to ensure init_variable is run
      # while asking for c
      real_v = control_flow_ops.with_dependencies(
          name="real_tensor",
          output_tensor=v.ref(),
          dependencies=[v.initializer])
      c_val, real_v_val = sess.run([c, real_v])

    # Ensure the result of 'real_c' is the same as 'c'
    self.assertAllEqual(10, c_val)

    # Ensure that 'v' is initialized
    self.assertAllClose(0.0, real_v_val)

  def testWithTensorDependencies(self):
    with self.test_session():
      v = tf.Variable(0.0)
      c1 = tf.constant(10)
      c2 = tf.constant(20)

      # c1_with_init_v depends on the init op for v
      c1_with_init_v = control_flow_ops.with_dependencies(
          name="c1_with_init_v",
          output_tensor=c1,
          dependencies=[v.initializer])
      # c2_with_c1 depends on the value of c1_with_init_v
      c2_with_c1_dep = control_flow_ops.with_dependencies(
          name="c2_with_c1_dep",
          output_tensor=c2,
          dependencies=[c1_with_init_v])

      # Fetching v directly will result in an uninitialized error
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        v.eval()

      # Get the value of 'c2_with_c1_dep', which should cause 'v'
      # to be initialized.
      self.assertAllEqual(20, c2_with_c1_dep.eval())

      # Ensure that 'v' is initialized
      self.assertAllClose(0.0, v.eval())

  def testWithIndexedSlicesDependencies(self):
    with self.test_session():
      v = tf.Variable(
          np.array([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]]).astype(np.float32))
      v_at_1 = tf.IndexedSlices(v, tf.constant([1]))
      gather_v_at_1 = tf.gather(v_at_1.values, v_at_1.indices)
      v_at_1_after_init = control_flow_ops.with_dependencies([v.initializer],
                                                             v_at_1)
      gather_v_at_1_after_init = tf.gather(
          v_at_1_after_init.values, v_at_1_after_init.indices)

      # Fetching gather_v_at_1 will result in an uninitialized error
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        gather_v_at_1.eval()

      # Getting gather_v_at_1_after_init will work, and initialize v.
      self.assertAllEqual([[10.0, 11.0]], gather_v_at_1_after_init.eval())

      # Double check that 'v' is initialized
      self.assertAllClose([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]], v.eval())

  def testDependenciesDevice(self):
    with tf.Graph().as_default():
      # device set on tensor => same device on dep.
      with tf.device("/job:ps"):
        vd = tf.Variable([0.0])
      with_vd_dep = control_flow_ops.with_dependencies([vd.initializer], vd)
      self.assertTrue("/job:ps" in with_vd_dep.device)

      # No device set on tensor => no device on dep.
      vnod = tf.Variable([0.0])
      with_vnod_dep = control_flow_ops.with_dependencies([vnod.initializer],
                                                         vnod)
      self.assertDeviceEqual(None, with_vnod_dep.device)

      # device set on tensor, default device on graph => default device on dep.
      vdef = tf.Variable([0.0], name="vdef")
      with tf.device("/job:worker/gpu:1"):
        with_vdef_dep = control_flow_ops.with_dependencies([vdef.initializer],
                                                           vdef)
        # The device is empty, but the colocation constraint is set.
        self.assertDeviceEqual("", with_vdef_dep.device)
        self.assertEqual([b"loc:@vdef"],
                         with_vdef_dep.op.colocation_groups())

  def testGroup(self):
    with self.test_session() as sess:
      v1 = tf.Variable([0.0])
      v2 = tf.Variable([1.0])

      # Group init1 and init2 and run.
      init = control_flow_ops.group(v1.initializer, v2.initializer)
      # Fetching v1 directly will result in an uninitialized error
      with self.assertRaisesOpError("Attempting to use uninitialized value"):
        v1.eval()

      # Runs "init" before fetching v1 and v2.
      init.run()
      v1_val, v2_val = sess.run([v1, v2])

    # Ensure that v1 and v2 are initialized
    self.assertAllClose([0.0], v1_val)
    self.assertAllClose([1.0], v2_val)

  def testGroupEmpty(self):
    op = tf.group()
    self.assertEqual(op.type, "NoOp")
    self.assertEqual(op.control_inputs, [])

  def testMergeShapes(self):
    # All inputs unknown.
    p1 = tf.placeholder(tf.float32)
    p2 = tf.placeholder(tf.float32)
    p3 = tf.placeholder(tf.float32)
    m, index = control_flow_ops.merge([p1, p2, p3])
    self.assertIs(None, m.get_shape().ndims)
    self.assertEqual([], index.get_shape())

    # All inputs known with different ranks.
    p1 = tf.placeholder(tf.float32, shape=[1, 2])
    p2 = tf.placeholder(tf.float32, shape=[1, 2, 3])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertIs(None, m.get_shape().ndims)
    self.assertEqual([], index.get_shape())

    # All inputs known with some dimensions different.
    p1 = tf.placeholder(tf.float32, shape=[1, 2])
    p2 = tf.placeholder(tf.float32, shape=[2, 1])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([None, None], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

    p1 = tf.placeholder(tf.float32, shape=[1, 2])
    p2 = tf.placeholder(tf.float32, shape=[None, 2])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([None, 2], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

    p1 = tf.placeholder(tf.float32, shape=[1, 2])
    p2 = tf.placeholder(tf.float32, shape=[2, 2])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([None, 2], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

    # All inputs known with same dimensions.
    p1 = tf.placeholder(tf.float32, shape=[1, 2])
    p2 = tf.placeholder(tf.float32, shape=[1, 2])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([1, 2], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

    p1 = tf.placeholder(tf.float32, shape=[None, 2])
    p2 = tf.placeholder(tf.float32, shape=[None, 2])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([None, 2], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

    p1 = tf.placeholder(tf.float32, shape=[None, None])
    p2 = tf.placeholder(tf.float32, shape=[None, None])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([None, None], m.get_shape().as_list())
    self.assertEqual([], index.get_shape())

  def testRefSelect(self):
    index = tf.placeholder(tf.int32)

    # All inputs unknown.
    p1 = tf.placeholder(tf.float32)
    p2 = tf.placeholder(tf.float32)
    p3 = tf.placeholder(tf.float32)
    v1 = tf.Variable(p1, validate_shape=False)
    v2 = tf.Variable(p2, validate_shape=False)
    v3 = tf.Variable(p3, validate_shape=False)
    s = control_flow_ops.ref_select(index, [v1, v2, v3])
    self.assertIs(None, s.get_shape().ndims)

    # All inputs known but different.
    v1 = tf.Variable([[1, 2]])
    v2 = tf.Variable([[2], [1]])
    s = control_flow_ops.ref_select(index, [v1, v2])
    self.assertIs(None, s.get_shape().ndims)

    # All inputs known and same.
    v1 = tf.Variable([[1, 2]])
    v2 = tf.Variable([[1, 2]])
    s = control_flow_ops.ref_select(index, [v1, v2])
    self.assertEqual([1, 2], s.get_shape())

    # Possibly the same but not guaranteed.
    v1 = tf.Variable([[1., 2.]])
    p2 = tf.placeholder(tf.float32, shape=[None, 2])
    v2 = tf.Variable(p2, validate_shape=False)
    s = control_flow_ops.ref_select(index, [v1, v2])
    self.assertEqual(None, s.get_shape())

  def testRunLoopTensor(self):
    with self.test_session() as sess:
      tensor_list = []
      def condition(t):
        return t < tf.constant(5)
      def body(_):
        tensor_list.append(tf.constant(5))
        return tf.constant(10)
      result = tf.while_loop(condition, body, [tf.constant(4)])
      self.assertEqual(10, sess.run(result))

      # Ensure that we cannot run a tensor that escapes the loop body
      # accidentally.
      with self.assertRaises(ValueError):
        sess.run(tensor_list[0])


class TupleTest(tf.test.TestCase):

  def testTensors(self):
    for v1_first in [True, False]:
      with self.test_session():
        v1 = tf.Variable([1.0])
        add1 = tf.add(
            control_flow_ops.with_dependencies([v1.initializer], v1.ref()),
            2.0)
        v2 = tf.Variable([10.0])
        add2 = tf.add(
            control_flow_ops.with_dependencies([v2.initializer], v2.ref()),
            20.0)
        t1, _, t2 = control_flow_ops.tuple([add1, None, add2])

        # v1 is not initialized.
        with self.assertRaisesOpError("Attempting to use uninitialized value"):
          v1.eval()

        # v2 is not initialized.
        with self.assertRaisesOpError("Attempting to use uninitialized value"):
          v2.eval()

        if v1_first:
          # Getting t1 initializes v2.
          self.assertAllClose([3.0], t1.eval())
          self.assertAllClose([10.0], v2.eval())
        else:
          # Getting t2 initializes v1.
          self.assertAllClose([30.0], t2.eval())
          self.assertAllClose([1.0], v1.eval())

  def testIndexedSlices(self):
    for v1_first in [True, False]:
      with self.test_session():
        v1 = tf.Variable(
            np.array([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]]).astype(
                np.float32))
        v1_at_1 = tf.IndexedSlices(
            control_flow_ops.with_dependencies([v1.initializer], v1.ref()),
            tf.constant([1]))

        v2 = tf.Variable(
            np.array([[0.1, 1.1], [10.1, 11.1], [20.1, 21.1]]).astype(
                np.float32))
        v2_at_1 = tf.IndexedSlices(
            control_flow_ops.with_dependencies([v2.initializer], v2.ref()),
            tf.constant([1]))

        st1, st2 = control_flow_ops.tuple([v1_at_1, v2_at_1])
        g1 = tf.gather(st1.values, st1.indices)
        g2 = tf.gather(st2.values, st2.indices)

        # v1 is not initialized.
        with self.assertRaisesOpError("Attempting to use uninitialized value"):
          v1.eval()

        # v2 is not initialized.
        with self.assertRaisesOpError("Attempting to use uninitialized value"):
          v2.eval()

        if v1_first:
          # Getting g1 initializes v2.
          self.assertAllClose([[10.0, 11.0]], g1.eval())
          self.assertAllClose([[0.1, 1.1], [10.1, 11.1], [20.1, 21.1]],
                              v2.eval())
        else:
          # Getting g2 initializes v1.
          self.assertAllClose([[10.1, 11.1]], g2.eval())
          self.assertAllClose([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]],
                              v1.eval())

  def testAcceptTensorsAsControlInputs(self):
    with self.test_session():
      var = tf.Variable(0)
      assign = tf.assign(var, 1)
      t, = tf.tuple([tf.constant(0)], control_inputs=[assign])

      # Should trigger the assign.
      t.eval()

      self.assertEquals(1, var.eval())

  def testWhilePyFuncBasic(self):
    def func(x):
      return np.square(x)

    with self.test_session():
      r = tf.while_loop(
          lambda i, v: i < 4,
          lambda i, v: [i + 1, tf.py_func(func, [v], [tf.float32])[0]],
          [tf.constant(0), tf.constant(2.0, tf.float32)])
      self.assertEqual(r[1].eval(), 65536.0)

  def testWhileFuncBasic(self):
    @function.Defun(tf.float32)
    def func(x):
      return tf.square(tf.square(x))

    with self.test_session():
      x = tf.constant(2.0, tf.float32)
      r = tf.while_loop(
          lambda i, v: i < 2,
          lambda i, v: [i + 1, func(v)],
          [tf.constant(0), x])
      self.assertEqual(r[1].eval(), 65536.0)

      r = tf.gradients(r, x)[0]
      self.assertEqual(r.eval(), 524288.0)
      self.assertEqual(len([op for op in x.graph.get_operations()
                            if op.type == "Stack"]),
                       1)

if __name__ == "__main__":
  tf.test.main()
