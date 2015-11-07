# pylint: disable=g-long-lambda
"""Tests for tensorflow.ops.control_flow_ops."""
import math

import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.pywrap_tensorflow import StatusNotOK

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
  for k, v in consumer_count.iteritems():
    if len(k.consumers()) != v:
      return False
  return True


def isum(s):
  i = tf.constant(0, name="i")
  c = lambda i, s: tf.less(i, 10)
  b = lambda i, s: [tf.add(i, 1), tf.add(i, s)]
  _, r_s = control_flow_ops.While(c, b, [i, s])
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

      enter_v = control_flow_ops._Enter(v, "foo_1")
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
      v1 = control_flow_ops._SwitchRefOrTensor(v, p)
      v2 = tf.assign(v1[1], 9)
      tf.initialize_all_variables().run()
      self.assertEqual(9, v2.eval())

  def testEnterExit_1(self):
    with self.test_session():
      data = tf.constant([1, 2, 3, 4, 5, 6], name="data")
      enter_op = control_flow_ops.enter(data, "foo_1", False)
      exit_op = control_flow_ops.exit(enter_op)

      result = exit_op.eval()
    self.assertAllEqual(np.array([1, 2, 3, 4, 5, 6]), result)

  def testEnterMulExit_1(self):
    with self.test_session():
      data = tf.constant([1, 2, 3, 4, 5, 6], name="data")
      enter_data = control_flow_ops.enter(data, "foo_1", False)
      five = tf.constant(5)
      enter_five = control_flow_ops.enter(five, "foo_1", False)
      mul_op = tf.mul(enter_data, enter_five)
      exit_op = control_flow_ops.exit(mul_op)

      result = exit_op.eval()
    self.assertAllEqual(np.array([x * 5 for x in [1, 2, 3, 4, 5, 6]]), result)

  def testEnterNextExit_1(self):
    with self.test_session():
      data = tf.constant([1, 2, 3, 4, 5, 6], name="data")
      enter_op = control_flow_ops.enter(data, "foo_1", False)
      next_op = control_flow_ops.next_iteration(enter_op)
      exit_op = control_flow_ops.exit(next_op)

      result = exit_op.eval()
    self.assertAllEqual(np.array([1, 2, 3, 4, 5, 6]), result)

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

  def _testSwitchMerge_1(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      data = tf.constant([1, 2, 3, 4, 5, 6], name="data")
      ports = tf.convert_to_tensor(True, name="ports")
      switch_op = control_flow_ops.switch(data, ports)
      merge_op = control_flow_ops.merge(switch_op)[0]

      result = merge_op.eval()
    self.assertAllEqual(np.arange(1, 7), result)

  def testSwitchMerge_1(self):
    self._testSwitchMerge_1(use_gpu=False)
    self._testSwitchMerge_1(use_gpu=True)

  def testSwitchDeadBranch(self):
    with self.test_session():
      data = tf.constant([1, 2, 3, 4, 5, 6], name="data")
      ports = tf.convert_to_tensor(True, name="ports")
      switch_op = control_flow_ops.switch(data, ports)
      dead_branch = tf.identity(switch_op[0])

      with self.assertRaisesWithPredicateMatch(
          StatusNotOK, lambda e: 'The tensor returned for' in str(e)):
        dead_branch.eval()

  def testSwitchMergeIdentity_1(self):
    with self.test_session():
      data = tf.constant([1, 2, 3, 4, 5, 6], name="data")
      ports = tf.convert_to_tensor(True, name="ports")
      switch_op = control_flow_ops.switch(data, ports)
      merge_op = control_flow_ops.merge(switch_op)[0]
      id_op = tf.identity(merge_op)

      result = id_op.eval()
    self.assertAllEqual(np.arange(1, 7), result)

  def testSwitchMergeLess_0(self):
    with self.test_session():
      data = tf.constant([1, 2, 3, 4, 5, 6], name="data")
      zero = tf.constant(0)
      one = tf.constant(1)
      less_op = tf.less(zero, one)
      switch_op = control_flow_ops.switch(data, less_op)
      merge_op = control_flow_ops.merge(switch_op)[0]

      result = merge_op.eval()
    self.assertAllEqual(np.arange(1, 7), result)

  def testSwitchMergeLess_1(self):
    with self.test_session():
      data = tf.constant([1, 2, 3, 4, 5, 6], name="data")
      zero = tf.convert_to_tensor(0)
      one = tf.convert_to_tensor(1)
      less_op = tf.less(zero, one)
      switch_op = control_flow_ops.switch(data, less_op)
      merge_op = control_flow_ops.merge(switch_op)[0]

      result = merge_op.eval()
    self.assertAllEqual(np.arange(1, 7), result)

  def testSwitchMergeAddIdentity_0(self):
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

  def testSwitchMergeAddIdentity_1(self):
    with self.test_session():
      data = tf.constant([1, 2, 3, 4, 5, 6], name="data")
      ports = tf.convert_to_tensor(True, name="ports")
      switch_op = control_flow_ops.switch(data, ports)
      one = tf.constant(1)
      add_op = tf.add(switch_op[0], one)
      id_op = tf.identity(switch_op[1])
      merge_op = control_flow_ops.merge([add_op, id_op])[0]

      result = merge_op.eval()
    self.assertAllEqual(np.arange(1, 7), result)

  def testSwitchMergeAddMul_0(self):
    with self.test_session():
      data = tf.constant([1, 2, 3, 4, 5, 6], name="data")
      ports = tf.convert_to_tensor(False, name="ports")
      switch_op = control_flow_ops.switch(data, ports)
      one = tf.constant(1)
      add_op = tf.add(switch_op[0], one)
      five = tf.constant(5)
      mul_op = tf.mul(switch_op[1], five)
      merge_op = control_flow_ops.merge([add_op, mul_op])[0]

      result = merge_op.eval()
    self.assertAllEqual(np.array([x + 1 for x in [1, 2, 3, 4, 5, 6]]), result)

  def testSwitchMergeAddMul_1(self):
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

      merge_n = control_flow_ops.merge([enter_n], name="merge_n")[0]
      switch_n = control_flow_ops.switch(merge_n, enter_false)
      exit_n = control_flow_ops.exit(switch_n[0])

      result = exit_n.eval()
    self.assertAllEqual(10, result)

  def testLoop_false_1(self):
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
      zero = tf.convert_to_tensor(0)
      one = tf.convert_to_tensor(1)
      n = tf.constant(10)

      enter_zero = control_flow_ops.enter(zero, "foo_1", False)
      enter_one = control_flow_ops.enter(one, "foo_1", False)
      enter_n = control_flow_ops.enter(n, "foo_1", False)
      merge_zero = control_flow_ops.merge([enter_zero, enter_zero],
                                          name="merge_zero")[0]
      merge_one = control_flow_ops.merge([enter_one, enter_one],
                                         name="merge_one")[0]
      merge_n = control_flow_ops.merge([enter_n, enter_n], name="merge_n")[0]
      less_op = tf.less(merge_n, merge_n)
      cond_op = control_flow_ops.loop_cond(less_op)
      switch_zero = control_flow_ops.switch(merge_zero, cond_op)
      switch_one = control_flow_ops.switch(merge_one, cond_op)
      switch_n = control_flow_ops.switch(merge_n, cond_op)
      next_zero = control_flow_ops.next_iteration(switch_zero[1])
      next_one = control_flow_ops.next_iteration(switch_one[1])
      next_n = control_flow_ops.next_iteration(switch_n[1])
      merge_zero.op._update_input(1, next_zero)
      merge_one.op._update_input(1, next_one)
      merge_n.op._update_input(1, next_n)
      exit_n = control_flow_ops.exit(switch_n[0])

      result = exit_n.eval()
    self.assertAllEqual(10, result)

  def testCondIndexedSlices(self):
    with self.test_session():
      values = tf.constant(10)
      indices = tf.constant(0)
      x = tf.IndexedSlices(values, indices)
      pred = tf.less(1, 2)
      fn1 = lambda: tf.IndexedSlices(tf.add(x.values, 1), indices)
      fn2 = lambda: tf.IndexedSlices(tf.sub(x.values, 1), indices)
      r = control_flow_ops.cond(pred, fn1, fn2)

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
      r = control_flow_ops.cond(pred, fn1, fn2)

      val = r.values.eval()
      ind = r.indices.eval()
    self.assertTrue(check_op_order(x.values.graph))
    self.assertAllEqual(11, val)
    self.assertAllEqual(0, ind)
    self.assertTrue(ind.dtype == np.int64)

  def _testCond_1(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      x = tf.constant(10)
      pred = tf.less(1, 2)
      fn1 = lambda: tf.add(x, 1)
      fn2 = lambda: tf.sub(x, 1)
      r = control_flow_ops.cond(pred, fn1, fn2)

      result = r.eval()
    self.assertTrue(check_op_order(x.graph))
    self.assertAllEqual(11, result)

  def testCond_1(self):
    self._testCond_1(use_gpu=False)
    self._testCond_1(use_gpu=True)

  def testCond_2(self):
    with self.test_session():
      x = tf.constant(10)
      r = control_flow_ops.cond(tf.less(1, 0), lambda: tf.add(x, 1),
                                lambda: tf.sub(x, 1))
      result = r.eval()
    self.assertTrue(check_op_order(x.graph))
    self.assertAllEqual(9, result)

  def testCond_3(self):
    with self.test_session():
      x = tf.constant(10)
      pred = tf.less(1, 2)
      fn1 = lambda: tf.add(x, 1)
      fn2 = lambda: tf.sub(x, 1)
      fn3 = lambda: tf.add(control_flow_ops.cond(pred, fn1, fn2), 1)
      r = control_flow_ops.cond(pred, fn3, fn2)

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
      r = control_flow_ops.cond(pred, fn1, fn2)

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
        return control_flow_ops.cond(
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
      r = control_flow_ops.cond(pred, fn1, fn2)

      tf.initialize_all_variables().run()
      result = r.eval()
      self.assertAllEqual(np.array([7]), result)

  def testCondGrad_1(self):
    with self.test_session():
      x = tf.constant(10.0, name="x")
      pred = tf.less(1, 2)
      fn1 = lambda: tf.identity(x)
      fn2 = lambda: tf.identity(x)
      r = control_flow_ops.cond(pred, fn1, fn2)

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
      r = control_flow_ops.cond(pred, fn1, fn2)

      grad = tf.gradients(r, [x])[0]
      self.assertAllEqual(42.0, grad.eval(feed_dict={c: 1}))
      self.assertAllEqual(3.0, grad.eval(feed_dict={c: 3}))

  def testCondGrad_Gather(self):
    with self.test_session() as sess:
      v1 = tf.Variable([1.0, 42.0])
      c = tf.placeholder(tf.int32, shape=[])
      pred = tf.less(c, 2)
      fn1 = lambda: tf.identity(v1)
      fn2 = lambda: tf.gather(v1, [1, 1])
      r = control_flow_ops.cond(pred, fn1, fn2)
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

  def testWhileGrad_1(self):
    with self.test_session():
      v = tf.constant(2.0, name="v")
      c = lambda v: tf.less(v, 100.0)
      b = tf.square
      r = control_flow_ops.While(c, b, [v], parallel_iterations=1)

      r = tf.gradients(r, v)
      result = r[0].eval()
      self.assertEqual(1024.0, result)

  def testWhileGrad_2(self):
    with self.test_session():
      a = tf.constant(3.0, name="a")
      v = tf.constant(2.0, name="v")
      c = lambda v: tf.less(v, 100.0)
      b = lambda v: tf.mul(v, a)
      r = control_flow_ops.While(c, b, [v], parallel_iterations=1)

      r = tf.gradients(r, a)
      result = r[0].eval()
      self.assertEqual(216.0, result)

  def testWhileGrad_3(self):
    with self.test_session():
      a = tf.constant(3.0, name="a")
      v = tf.constant(2.0, name="v")
      c = lambda v: tf.less(v, 100.0)
      b = lambda v: tf.mul(v, a)
      r = control_flow_ops.While(c, b, [v], parallel_iterations=1)

      r = tf.gradients(r, v)
      result = r[0].eval()
      self.assertEqual(81.0, result)

  def testWhileGrad_4(self):
    with self.test_session():
      a = tf.Variable(3.0)
      v = tf.constant(2.0, name="v")
      c = lambda v: tf.less(v, 100.0)
      b = lambda v: tf.mul(v, a)
      r = control_flow_ops.While(c, b, [v], parallel_iterations=1)

      r = tf.gradients(r, a)
      tf.initialize_all_variables().run()
      result = r[0].eval()
      self.assertEqual(216.0, result)

  def testWhileGrad_5(self):
    with self.test_session():
      x = tf.constant(3.0, name="x")
      y = tf.constant(2.0, name="y")
      c = lambda x, y: tf.less(x, 100.0)

      def b(x, y):
        y1 = tf.add(x, y)
        x1 = tf.mul(x, y1)
        return x1, y1

      r = control_flow_ops.While(c, b, [x, y], parallel_iterations=1)

      # Must use the complete r.
      r = tf.gradients(r, x)
      result = r[0].eval()
      self.assertEqual(304.0, result)

  def testWhileGrad_6(self):
    with self.test_session():
      i = tf.constant(0, name="i")
      x = tf.constant(2.0, name="x")
      c = lambda i, x: tf.less(i, 10)

      def b(i, x):
        x = tf.mul(x, 2.0)
        i = tf.add(i, 1)
        return i, x

      r = control_flow_ops.While(c, b, [i, x], parallel_iterations=1)

      # Must use the complete r.
      r = tf.gradients(r, x)
      r = r[0].eval()
      self.assertEqual(1024.0, r)

  def testWhileGrad_7(self):
    with self.test_session():
      v = tf.constant(2.0, name="v")
      c = lambda v: tf.less(v, 100.0)
      b = tf.square
      r = control_flow_ops.While(c, b, [v], parallel_iterations=1,
                                 back_prop=False)
      r = tf.add(r, v)
      r = tf.gradients(r, v)
      result = r[0].eval()
      self.assertEqual(1.0, result)

  # Microbenchmark: 10,000 iterations took 0.21s.
  def testWhile_1(self):
    with self.test_session():
      n = tf.constant(0)
      c = lambda x: tf.less(x, 10000)
      b = lambda x: tf.add(x, 1)
      r = control_flow_ops.While(c, b, [n], parallel_iterations=20)

      result = r.eval()
    self.assertTrue(check_op_order(n.graph))
    self.assertEqual(10000, result)

  def testWhile_2(self):
    with self.test_session():
      s = tf.constant(0)
      r = isum(s)

      result = r.eval()
    self.assertTrue(check_op_order(s.graph))
    self.assertAllEqual(45, result)

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
      r = control_flow_ops.While(
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
      r = control_flow_ops.While(
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
      r = control_flow_ops.While(
          lambda i, c, o: tf.less(i, s), compute, [i, c, o])
      result = r[2].eval()
    self.assertTrue(check_op_order(i.graph))
    self.assertAllEqual(np.array([0, 1, 2, 3, 4, 5, 6]), result)

  def _testWhile_Gpu_1(self, use_gpu):
    with self.test_session(use_gpu=use_gpu):
      n = tf.constant(1.0)
      c = lambda x: tf.less(x, 10.0)
      b = lambda x: tf.add(x, 1.0)
      r = control_flow_ops.While(c, b, [n])

      result = r.eval()
    self.assertEqual(10.0, result)

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
      r = control_flow_ops.While(c, b, [n])

      result = r.eval()
    self.assertEqual(10.0, result)

  def testWhile_Gpu_2(self):
    self._testWhile_Gpu_1(use_gpu=False)
    self._testWhile_Gpu_1(use_gpu=True)

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

      res = control_flow_ops.While(condition,
                                   body,
                                   [n, r],
                                   parallel_iterations=1)
      result = res[1].eval()
    self.assertTrue(check_op_order(n.graph))
    self.assertAllEqual(12, result)

  def testWhileWithControl_2(self):
    with self.test_session():
      r = tf.constant(0)
      condition = lambda r_: tf.less(r_, 10)

      def body(r_):
        with r_.graph.control_dependencies([r_]):
          r_ = tf.constant(12)
        return [r_]

      res = control_flow_ops.While(condition, body, [r], parallel_iterations=1)
      result = res.eval()
    self.assertTrue(check_op_order(r.graph))
    self.assertAllEqual(12, result)

  def testCondWhile_1(self):
    with self.test_session():
      n = tf.convert_to_tensor(0, name="n")
      c = lambda x: tf.less(x, 10)
      b = lambda x: tf.add(x, 1)
      r = control_flow_ops.cond(tf.less(0, 1),
                                lambda: control_flow_ops.While(c, b, [n]),
                                lambda: n)

      result = r.eval()
    self.assertTrue(check_op_order(n.graph))
    self.assertAllEqual(10, result)

  def testCondWhile_2(self):
    with self.test_session():
      n = tf.convert_to_tensor(0)
      c = lambda x: tf.less(x, 10)
      b = lambda x: tf.add(x, 1)
      r = control_flow_ops.cond(tf.less(1, 0), lambda: tf.add(n, 1),
                                lambda: control_flow_ops.While(c, b, [n]))

      result = r.eval()
    self.assertTrue(check_op_order(n.graph))
    self.assertAllEqual(10, result)

  def testWhileCond_1(self):
    with self.test_session():
      i = tf.convert_to_tensor(0, name="i")
      n = tf.convert_to_tensor(10, name="n")
      one = tf.convert_to_tensor(1, name="one")
      c = lambda x: tf.less(x, n)
      b = lambda x: control_flow_ops.cond(tf.constant(True),
                                          lambda: tf.add(x, one),
                                          lambda: tf.sub(x, one))
      r = control_flow_ops.While(c, b, [i])

      result = r.eval()
    self.assertTrue(check_op_order(n.graph))
    self.assertAllEqual(10, result)

  def testWhileCond_2(self):
    with self.test_session():
      n = tf.convert_to_tensor(0, name="n")
      c = lambda x: tf.less(x, 10)
      b = lambda x: control_flow_ops.cond(tf.constant(True),
                                          lambda: tf.add(x, 1),
                                          lambda: n)
      r = control_flow_ops.While(c, b, [n])

      result = r.eval()
    self.assertTrue(check_op_order(n.graph))
    self.assertAllEqual(10, result)

  def testWhileCond_3(self):
    with self.test_session():
      n = tf.convert_to_tensor(0)
      c = lambda x: tf.less(x, 10)
      b = lambda x: control_flow_ops.cond(tf.less(0, 1),
                                          lambda: tf.add(x, 1),
                                          lambda: tf.sub(x, 1))
      r = control_flow_ops.While(c, b, [n])

      result = r.eval()
    self.assertTrue(check_op_order(n.graph))
    self.assertAllEqual(10, result)

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

      r = control_flow_ops.While(loop_iterator,
                                 loop_body,
                                 [n],
                                 parallel_iterations=1)
      self.assertTrue(check_op_order(n.graph))
      tf.initialize_all_variables().run()
      self.assertEqual(3, r.eval())
      result = select.eval()
      self.assertAllEqual(np.array([10.0, 10.0, 10.0]), result)

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

      r = control_flow_ops.While(loop_iterator,
                                 loop_body,
                                 [n],
                                 parallel_iterations=1)
      self.assertTrue(check_op_order(n.graph))
      tf.initialize_all_variables().run()
      self.assertEqual(3, r.eval())
      result1 = select1.eval()
      self.assertAllEqual(np.array([10.0, 10.0, 10.0]), result1)
      result2 = select2.eval()
      self.assertAllEqual(np.array([10.0, 10.0, 10.0]), result2)

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

      r = control_flow_ops.While(loop_iterator,
                                 loop_body,
                                 [n, tf.identity(select)],
                                 parallel_iterations=1)
      tf.initialize_all_variables().run()
      result = r[1].eval()
    self.assertTrue(check_op_order(n.graph))
    self.assertAllEqual(np.array([10.0, 10.0, 10.0]), result)

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

      lpa = control_flow_ops.While(pred, loop_body, [c],
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
      def pred(i):
        return tf.less(var_b, 10)

      # Change body to increment var_b
      def loop_body(i):
        asn1 = tf.assign_add(var_a, tf.constant(1), name="a_add")
        asn2 = tf.assign_add(var_b, tf.constant(1), name="b_add")
        with tf.control_dependencies([asn1, asn2]):
          inc_b = tf.identity(var_b)
        return inc_b

      lpa = control_flow_ops.While(pred, loop_body, [var_b], 1, name="loop")

      self.assertEqual(0, var_b.eval())
      lpa.eval()  # Run the loop
      self.assertEqual(10, var_a.eval())
      self.assertEqual(10, var_b.eval())

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

      r = control_flow_ops.While(c, b, [i], parallel_iterations=1)
      self.assertEqual([10], r.eval())
      for i in xrange(10):
        self.assertEqual([i], q.dequeue().eval())

  def testFold_1(self):
    with self.test_session():
      elems = tf.constant([1, 2, 3, 4, 5, 6], name="data")
      r = control_flow_ops.fold(
          lambda a, x: tf.mul(tf.add(a, x), 2), elems, [1])
      result = r.eval()
    self.assertTrue(check_op_order(elems.graph))
    self.assertAllEqual(np.array([208]), result)

  def testFold_2(self):
    with self.test_session():
      elems = tf.constant([1, 2, 3, 4, 5, 6], name="data")
      ten = tf.convert_to_tensor(10)

      def compute(a, x):
        r = tf.mul(x, ten)
        return tf.add(a, r)

      r = control_flow_ops.fold(compute, elems, [1])
      result = r.eval()
    self.assertTrue(check_op_order(elems.graph))
    self.assertAllEqual([201], result)

  def testOneValueCond(self):
    with self.test_session():
      c = tf.placeholder(tf.int32, shape=[])
      one = tf.convert_to_tensor(1, name="one")
      two = tf.convert_to_tensor(2, name="two")
      p = tf.greater_equal(c, 1)
      i = control_flow_ops.cond(p, lambda: one, lambda: two)
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

      i = control_flow_ops.cond(tf.equal(d, 2), l2, l1)
      self.assertEqual(4.0, i.eval(feed_dict={d: 1}))
      self.assertAllClose(2.0 * math.sqrt(2), i.eval(feed_dict={d: 2}))

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

      i = control_flow_ops.cond(p, a, b)
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
      real_v = control_flow_ops.with_dependencies(name="real_tensor",
                                                 output_tensor=v,
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
      self.assertEquals(None, with_vnod_dep.device)

      # device set on tensor, default device on graph => default device on dep.
      vdef = tf.Variable([0.0])
      with tf.device("/job:worker/gpu:1"):
        with_vdef_dep = control_flow_ops.with_dependencies([vdef.initializer],
                                                           vdef)
        self.assertEquals("/job:worker/gpu:1", with_vdef_dep.device)

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

  def testMergeShapes(self):
    # All inputs unknown.
    p1 = tf.placeholder(tf.float32)
    p2 = tf.placeholder(tf.float32)
    p3 = tf.placeholder(tf.float32)
    m, index = control_flow_ops.merge([p1, p2, p3])
    self.assertIs(None, m.get_shape().ndims)
    self.assertEqual([], index.get_shape())

    # All inputs known but different.
    p1 = tf.placeholder(tf.float32, shape=[1, 2])
    p2 = tf.placeholder(tf.float32, shape=[2, 1])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertIs(None, m.get_shape().ndims)
    self.assertEqual([], index.get_shape())

    # All inputs known but same.
    p1 = tf.placeholder(tf.float32, shape=[1, 2])
    p2 = tf.placeholder(tf.float32, shape=[1, 2])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertEqual([1, 2], m.get_shape())
    self.assertEqual([], index.get_shape())

    # Possibly the same but not guaranteed.
    p1 = tf.placeholder(tf.float32, shape=[1, 2])
    p2 = tf.placeholder(tf.float32)
    p2.set_shape([None, 2])
    m, index = control_flow_ops.merge([p1, p2])
    self.assertIs(None, m.get_shape().ndims)
    self.assertEqual([], index.get_shape())

  def testRefSelect(self):
    index = tf.placeholder(tf.int32)

    # All inputs unknown.
    p1 = tf.placeholder(tf.float32_ref)
    p2 = tf.placeholder(tf.float32_ref)
    p3 = tf.placeholder(tf.float32_ref)
    s = control_flow_ops.ref_select(index, [p1, p2, p3])
    self.assertIs(None, s.get_shape().ndims)

    # All inputs known but different.
    p1 = tf.placeholder(tf.float32_ref, shape=[1, 2])
    p2 = tf.placeholder(tf.float32_ref, shape=[2, 1])
    s = control_flow_ops.ref_select(index, [p1, p2])
    self.assertIs(None, s.get_shape().ndims)

    # All inputs known but same.
    p1 = tf.placeholder(tf.float32_ref, shape=[1, 2])
    p2 = tf.placeholder(tf.float32_ref, shape=[1, 2])
    s = control_flow_ops.ref_select(index, [p1, p2])
    self.assertEqual([1, 2], s.get_shape())

    # Possibly the same but not guaranteed.
    p1 = tf.placeholder(tf.float32_ref, shape=[1, 2])
    p2 = tf.placeholder(tf.float32_ref)
    p2.set_shape([None, 2])
    s = control_flow_ops.ref_select(index, [p1, p2])
    self.assertEqual(None, s.get_shape())


class TupleTest(tf.test.TestCase):

  def testTensors(self):
    for v1_first in [True, False]:
      with self.test_session():
        v1 = tf.Variable([1.0])
        add1 = tf.add(
            control_flow_ops.with_dependencies([v1.initializer], v1),
            2.0)
        v2 = tf.Variable([10.0])
        add2 = tf.add(control_flow_ops.with_dependencies([v2.initializer],
                                                               v2),
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
            control_flow_ops.with_dependencies([v1.initializer], v1),
            tf.constant([1]))

        v2 = tf.Variable(
            np.array([[0.1, 1.1], [10.1, 11.1], [20.1, 21.1]]).astype(
                np.float32))
        v2_at_1 = tf.IndexedSlices(
            control_flow_ops.with_dependencies([v2.initializer], v2),
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

if __name__ == "__main__":
  tf.test.main()
