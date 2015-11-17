"""Tests for tensorflow.python.client.session.Session."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import time

import tensorflow.python.platform

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.framework import config_pb2
from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework import types
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


# NOTE(mrry): Dummy shape registration for op used in the tests.
ops.RegisterShape('ConstructionFails')(None)


class SessionTest(test_util.TensorFlowTestCase):

  def testUseExistingGraph(self):
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      a = constant_op.constant(6.0, shape=[1, 1])
      b = constant_op.constant(7.0, shape=[1, 1])
      c = math_ops.matmul(a, b, name='matmul')
    with session.Session(graph=g):
      result = c.eval()
      self.assertAllEqual(result, [[42.0]])

  def testUseDefaultGraph(self):
    with ops.Graph().as_default(), ops.device('/cpu:0'):
      a = constant_op.constant(6.0, shape=[1, 1])
      b = constant_op.constant(7.0, shape=[1, 1])
      c = math_ops.matmul(a, b, name='matmul')
      with session.Session():
        result = c.eval()
        self.assertAllEqual(result, [[42.0]])

  def testCreate(self):
    with session.Session():
      inp = constant_op.constant(10.0, name='W1')
      copy = array_ops.identity(inp)
      # Test with feed.
      # TODO(mrry): Investigate why order='F' didn't work.
      arr = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.float32, order='C')
      copy_val = copy.eval({'W1:0': arr})
      self.assertAllEqual(arr, copy_val)
      # Test without feed.
      copy_val = copy.eval()
      self.assertAllEqual(np.asarray(10.0, dtype=np.float32), copy_val)

  def testManyCPUs(self):
    # TODO(keveman): Implement ListDevices and test for the number of
    # devices returned by ListDevices.
    with session.Session(
        config=config_pb2.ConfigProto(device_count={'CPU': 2})):
      inp = constant_op.constant(10.0, name='W1')
      self.assertAllEqual(inp.eval(), 10.0)

  def testErrorsReported(self):
    with session.Session() as s:
      constant_op.constant(10.0, name='W1')
      with self.assertRaises(ValueError):
        s.run('foo:0')

  def testErrorPayload(self):
    with session.Session():
      a = array_ops.placeholder(types.float32)
      with self.assertRaisesOpError(lambda e: e.op == a.op):
        a.eval()

  def testOpConstructionErrorPayload(self):
    with session.Session():
      failing_op = ops.get_default_graph().create_op(
          'ConstructionFails', [], [], name='f')

      def exc_predicate(e):
        return (e.op == failing_op
                and e.error_code == error_codes_pb2.INVALID_ARGUMENT)
      with self.assertRaisesOpError(exc_predicate):
        failing_op.run()

  def testErrorBasedOn(self):
    with session.Session() as sess:
      a = constant_op.constant(0.0, shape=[2, 3])
      # NOTE(mrry): The original_op is nonsense, but used here to test that the
      #   errors are reported correctly.
      # pylint: disable=protected-access
      with sess.graph._original_op(a.op):
        b = array_ops.identity(a, name='id')
      with sess.graph._original_op(b.op):
        c = array_ops.placeholder(types.float32)
      # pylint: enable=protected-access

      def exc_predicate(e):
        return (e.op == c.op
                and e.op._original_op == b.op
                and e.op._original_op._original_op == a.op)
      with self.assertRaisesOpError(exc_predicate):
        c.eval()

  def testFetchTensorObject(self):
    with session.Session() as s:
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)
      results_with_list = s.run([c])
      self.assertAllEqual([[4.0, 4.0, 4.0]], results_with_list[0])
      results_with_single = s.run(c)
      self.assertAllEqual([[4.0, 4.0, 4.0]], results_with_single)
      results_with_get = c.eval()
      self.assertAllEqual([[4.0, 4.0, 4.0]], results_with_get)
      a_val, b_val = s.run([a, b])  # Test multiple fetches.
      self.assertAllEqual([[1.0, 1.0]], a_val)
      self.assertAllEqual([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]], b_val)

  def testFetchScalar(self):
    with session.Session() as s:
      for scalar in np.int32, np.int64, np.float32, np.float64:
        x = scalar(7)
        y = scalar(8)
        tf_x = constant_op.constant(x, shape=[])
        tf_y = constant_op.constant(y)
        tf_xy = math_ops.add(tf_x, tf_y)
        # Single fetch
        xy = s.run(tf_xy)
        self.assertEqual(scalar, type(xy))
        self.assertEqual(x + y, xy)
        # List fetch
        xy, = s.run([tf_xy])
        self.assertEqual(scalar, type(xy))
        self.assertEqual(x + y, xy)

  def testFetchOperationObject(self):
    with session.Session() as s:
      a = constant_op.constant(1.0, shape=[1, 2])
      v = variables.Variable(a, name='testFetchOperationObject_v')
      s.run(v.initializer)
      v_val = s.run(v)
      self.assertAllEqual([[1.0, 1.0]], v_val)

  def testFetchSparseTensor(self):
    with session.Session() as s:
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      values = np.array([1.0, 2.0]).astype(np.float32)
      shape = np.array([7, 9, 2]).astype(np.int64)
      sp = ops.SparseTensor(
          constant_op.constant(indices),
          constant_op.constant(values),
          constant_op.constant(shape))
      # Single fetch, use as tuple
      sp_out = s.run(sp)
      indices_out, values_out, shape_out = sp_out
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Single fetch, use as SparseTensorValue
      sp_out = s.run(sp)
      self.assertAllEqual(sp_out.indices, indices)
      self.assertAllEqual(sp_out.values, values)
      self.assertAllEqual(sp_out.shape, shape)
      # Tuple fetch, use as tuple
      indices_out, values_out, shape_out = s.run(sp)
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # List fetch, use as tuple
      (indices_out, values_out, shape_out), = s.run([sp])
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # List fetch, use as SparseTensorValue
      sp_out, = s.run([sp])
      self.assertAllEqual(sp_out.indices, indices)
      self.assertAllEqual(sp_out.values, values)
      self.assertAllEqual(sp_out.shape, shape)

  def testFeedSparseTensor(self):
    with session.Session() as s:
      indices = np.array([[3, 2, 0], [4, 5, 1]]).astype(np.int64)
      values = np.array([1.0, 2.0]).astype(np.float32)
      shape = np.array([7, 9, 2]).astype(np.int64)
      sp = ops.SparseTensor(
          array_ops.placeholder(dtype=np.int64, shape=(2, 3)),
          array_ops.placeholder(dtype=np.float32, shape=(2,)),
          array_ops.placeholder(dtype=np.int64, shape=(3,)),)
      sp_indices = array_ops.identity(sp.indices)
      sp_values = array_ops.identity(sp.values)
      sp_shape = array_ops.identity(sp.shape)
      sp2 = ops.SparseTensor(sp_indices, sp_values, sp_shape)
      # Feed with tuple
      indices_out, values_out, shape_out = s.run(
          [sp_indices, sp_values, sp_shape], {sp: (indices, values, shape)})
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Feed with SparseTensorValue
      indices_out, values_out, shape_out = s.run(
          [sp_indices, sp_values, sp_shape],
          {sp: ops.SparseTensorValue(indices, values, shape)})
      self.assertAllEqual(indices_out, indices)
      self.assertAllEqual(values_out, values)
      self.assertAllEqual(shape_out, shape)
      # Feed with SparseTensorValue, fetch SparseTensorValue
      sp2_out = s.run(sp2, {sp: ops.SparseTensorValue(indices, values, shape)})
      self.assertAllEqual(sp2_out.indices, indices)
      self.assertAllEqual(sp2_out.values, values)
      self.assertAllEqual(sp2_out.shape, shape)

  def testExtendWithStatelessOperations(self):
    with session.Session() as s:
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)
      c_val = s.run(c)
      self.assertAllEqual([[4.0, 4.0, 4.0]], c_val)
      d = constant_op.constant([1.0, 2.0, 3.0], shape=[3, 1])
      e = math_ops.matmul(c, d)
      # Extend will happen here.
      e_val = s.run(e)
      self.assertAllEqual([[24.0]], e_val)

  def testExtendWithStatefulOperations(self):
    with session.Session() as s:
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)
      v = variables.Variable(c, name='testExtendWithStatefulOperations_v')
      v.initializer.run()
      v_val = v.eval()
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      d = constant_op.constant(3.0, shape=[2, 3])
      e = math_ops.matmul(a, d)
      assign_e_to_v = state_ops.assign(v, e)
      # Extend will happen here.
      e_val = e.eval()
      self.assertAllEqual([[6.0, 6.0, 6.0]], e_val)
      v_val = v.eval()
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      s.run(assign_e_to_v)
      v_val = v.eval()
      self.assertAllEqual([[6.0, 6.0, 6.0]], v_val)

  def testExtendWithGroupBy(self):
    with session.Session() as s:
      a = constant_op.constant(1.0, shape=[1, 2])
      p = variables.Variable(a, name='testExtendWithGroupBy_p')
      a_val = a.eval()  # Force an Extend after this op.
      self.assertAllEqual([[1.0, 1.0]], a_val)

      b = constant_op.constant(2.0, shape=[1, 2])
      q = variables.Variable(b, name='testExtendWithGroupBy_q')
      # Extend will happen here.
      init = control_flow_ops.group(p.initializer, q.initializer)
      s.run(init)
      p_val, q_val = s.run([p, q])

      self.assertAllEqual([[1.0, 1.0]], p_val)
      self.assertAllEqual([[2.0, 2.0]], q_val)

  def testTensorGetMethod(self):
    with session.Session():
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)

      c_val = c.eval()
      self.assertAllEqual([[4.0, 4.0, 4.0]], c_val)

      fed_c_val = c.eval(feed_dict={a.name: [[4.0, 4.0]]})
      self.assertAllEqual([[16.0, 16.0, 16.0]], fed_c_val)

  def testOperationRunMethod(self):
    with session.Session():
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[1, 2], name='b')
      v = variables.Variable(a, a.dtype)
      assign_a_to_v = state_ops.assign(v, a)

      assign_a_to_v.eval()

      v_val = v.eval()
      self.assertAllEqual([[1.0, 1.0]], v_val)

      assign_b_to_v = state_ops.assign(v, b)

      assign_b_to_v.eval()
      v_val = v.eval()
      self.assertAllEqual([[2.0, 2.0]], v_val)

      assign_b_to_v.eval(feed_dict={'b:0': [[3.0, 3.0]]})
      v_val = v.eval()
      self.assertAllEqual([[3.0, 3.0]], v_val)

  def testDefaultGraph(self):
    with session.Session() as s:
      self.assertEqual(ops.get_default_graph(), s.graph)
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      self.assertEqual(ops.get_default_graph(), a.graph)
      self.assertEqual(ops.get_default_graph(), b.graph)
      c = math_ops.matmul(a, b)
      v = variables.Variable(c, name='testDefaultGraph_v')
      v.initializer.run()
      v_val = v.eval()
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      d = constant_op.constant(3.0, shape=[2, 3])
      e = math_ops.matmul(a, d)
      assign_e_to_v = state_ops.assign(v, e)
      e_val = e.eval()
      self.assertAllEqual([[6.0, 6.0, 6.0]], e_val)
      v_val = v.eval()
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      s.run(assign_e_to_v)
      v_val = v.eval()
      self.assertAllEqual([[6.0, 6.0, 6.0]], v_val)
      self.assertEqual(ops.get_default_graph(), s.graph)

  def _testDefaultGraphInThread(self, constructed_event, continue_event, i):
    with session.Session() as s:
      self.assertEqual(ops.get_default_graph(), s.graph)
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)
      v = variables.Variable(c, name='var_%d' % i)

      # Block here until all threads have constructed their graph.
      constructed_event.set()
      continue_event.wait()

      assign_c_to_v = state_ops.assign(v, c)
      v.initializer.run()
      assign_c_to_v.eval()
      v_val = v.eval()
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      d = constant_op.constant(3.0, shape=[2, 3])
      e = math_ops.matmul(a, d)
      assign_e_to_v = state_ops.assign(v, e)
      e_val = e.eval()
      self.assertAllEqual([[6.0, 6.0, 6.0]], e_val)
      v_val = v.eval()
      self.assertAllEqual([[4.0, 4.0, 4.0]], v_val)
      s.run(assign_e_to_v)
      v_val = v.eval()
      self.assertAllEqual([[6.0, 6.0, 6.0]], v_val)
      self.assertEqual(ops.get_default_graph(), s.graph)

  def testDefaultGraphWithThreads(self):
    # Fork ten threads that use their thread-local default graph.
    threads = []
    constructed_events = [threading.Event() for _ in range(10)]
    continue_event = threading.Event()
    for i, constructed_event in enumerate(constructed_events):
      t = self.checkedThread(target=self._testDefaultGraphInThread,
                             args=(constructed_event, continue_event, i))
      threads.append(t)
    for t in threads:
      t.start()
    for constructed_event in constructed_events:
      constructed_event.wait()
    continue_event.set()
    for t in threads:
      t.join()

  def testParallelRun(self):
    with session.Session() as sess:
      c = constant_op.constant(5.0)
      ev = threading.Event()

      def run_step():
        ev.wait()
        val = c.eval(session=sess)
        self.assertEqual(val, 5.0)
      threads = [self.checkedThread(target=run_step) for _ in range(100)]
      for t in threads:
        t.start()
      ev.set()
      for t in threads:
        t.join()

  def testRunFeedDict(self):
    with session.Session() as s:
      x = array_ops.zeros([2])

      y = s.run(2 * x, feed_dict={x: np.ones(2).astype(np.float32)})
      self.assertAllEqual(y, 2 * np.ones(2))

      y = s.run(2 * x, feed_dict={x.name: np.ones(2).astype(np.float32)})
      self.assertAllEqual(y, 2 * np.ones(2))

      y = s.run(2 * x, feed_dict={x: [1, 1]})
      assert (y == 2 * np.ones(2)).all()

  def testGraphDef(self):
    with session.Session() as sess:
      self.assertProtoEquals('', sess.graph_def)
      c = constant_op.constant(5.0, name='c')
      self.assertEquals(len(sess.graph_def.node), 1)
      d = constant_op.constant(6.0, name='d')
      self.assertEquals(len(sess.graph_def.node), 2)
      self.assertAllEqual(c.eval(), 5.0)
      self.assertAllEqual(d.eval(), 6.0)
      e = constant_op.constant(7.0, name='e')
      self.assertEquals(len(sess.graph_def.node), 3)
      self.assertAllEqual(e.eval(), 7.0)

  def testUseAfterClose(self):
    with session.Session() as sess:
      c = constant_op.constant(5.0)
      self.assertAllEqual(sess.run(c), 5.0)
    with self.assertRaisesWithPredicateMatch(
        RuntimeError, lambda e: 'Attempted to use a closed Session.' in str(e)):
      sess.run(c)

  def testUseAfterCloseConcurrent(self):
    with session.Session() as sess:
      c = constant_op.constant(5.0)
      self.assertAllEqual(sess.run(c), 5.0)

      def update_thread():
        with self.assertRaisesWithPredicateMatch(
            RuntimeError,
            lambda e: 'Attempted to use a closed Session.' in str(e)):
          while True:
            sess.run(c)
      t = threading.Thread(target=update_thread)
      t.start()
      time.sleep(0.1)
      sess.close()
      t.join()

  def testUseEmptyGraph(self):
    with session.Session() as sess:
      with self.assertRaisesWithPredicateMatch(
          RuntimeError, lambda e: 'The Session graph is empty.' in str(e)):
        sess.run([])

  def testNotEntered(self):
    # pylint: disable=protected-access
    self.assertEqual(ops._default_session_stack.get_default(), None)
    # pylint: enable=protected-access
    with ops.device('/cpu:0'):
      sess = session.Session()
      c_1 = constant_op.constant(5.0)
      with sess.graph.as_default():
        c_2 = constant_op.constant(5.0)
      self.assertEqual(c_1.graph, c_2.graph)
      self.assertEqual(sess.run(c_2), 5.0)
      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: 'No default session is registered.' in str(e)):
        c_2.eval()

  def testInteractive(self):
    with ops.device('/cpu:0'):
      sess = session.InteractiveSession()
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)
      self.assertAllEqual([[4.0, 4.0, 4.0]], c.eval())
      d = constant_op.constant([1.0, 2.0, 3.0], shape=[3, 1])
      e = math_ops.matmul(c, d)
      self.assertAllEqual([[24.0]], e.eval())
      sess.close()

  def testSharedGraph(self):
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[2, 3])
      c = math_ops.matmul(a, b)

    with session.Session(graph=g) as sess1:
      with session.Session(graph=g) as sess2:
        self.assertAllEqual(sess1.run(c), sess2.run(c))

  def testDuplicatedInputs(self):
    with session.Session() as sess:
      a = constant_op.constant(1.0, shape=[1, 2])
      b = constant_op.constant(2.0, shape=[1, 3])
      a_val, b_val, a2_val = sess.run([a, b, a])
      self.assertAllEqual(a_val, [[1.0, 1.0]])
      self.assertAllEqual(b_val, [[2.0, 2.0, 2.0]])
      self.assertAllEqual(a2_val, [[1.0, 1.0]])

  def testFeedAndFetch(self):
    with session.Session():
      for dtype in [types.float32,
                    types.float64,
                    types.int32,
                    types.uint8,
                    types.int16,
                    types.int8,
                    types.int64,
                    types.bool,
                    types.complex64]:
        for shape in [(32, 4, 128), (37,), (2, 0, 6), (0, 0, 0)]:
          np_dtype = dtype.as_numpy_dtype

          feed_t = array_ops.placeholder(dtype=dtype, shape=shape)
          out_t = array_ops.identity(feed_t)

          np_array = np.random.randint(-10, 10, shape)

          if dtype == types.bool:
            np_array = np_array > 0
          elif dtype == types.complex64:
            np_array = np.sqrt(np_array.astype(np_dtype))
          else:
            np_array = np_array.astype(np_dtype)

          self.assertAllEqual(np_array,
                              out_t.eval(feed_dict={feed_t: np_array}))

  def testStringFetch(self):
    with session.Session():
      for shape in [(32, 4, 128), (37,), (2, 0, 6), (0, 0, 0)]:
        size = 1
        for s in shape:
          size *= s
        c_list = np.array([str(i) for i in xrange(size)],
                          dtype=np.object).reshape(shape) if size > 0 else []
        c = constant_op.constant(c_list)
        self.assertAllEqual(c.eval(), c_list)

  def testStringFeed(self):
    with session.Session():
      for shape in [(32, 4, 128), (37,), (2, 0, 6), (0, 0, 0)]:
        size = 1
        for s in shape:
          size *= s
        c_list = np.array([str(i) for i in xrange(size)],
                          dtype=np.object).reshape(shape)
        feed_t = array_ops.placeholder(dtype=types.string, shape=shape)
        c = array_ops.identity(feed_t)
        self.assertAllEqual(c.eval(feed_dict={feed_t: c_list}), c_list)

  def testStringFeedWithNullCharacters(self):
    with session.Session():
      c_list = ['\n\x01\x00', '\n\x00\x01']
      feed_t = array_ops.placeholder(dtype=types.string, shape=[2])
      c = array_ops.identity(feed_t)
      out = c.eval(feed_dict={feed_t: c_list})
      self.assertEqual(c_list[0], out[0])
      self.assertEqual(c_list[1], out[1])

  def testStringFeedWithUnicode(self):
    with session.Session():
      c_list = [u'\n\x01\x00', u'\n\x00\x01']
      feed_t = array_ops.placeholder(dtype=types.string, shape=[2])
      c = array_ops.identity(feed_t)

      out = c.eval(feed_dict={feed_t: c_list})
      self.assertEqual(c_list[0], out[0].decode('utf-8'))
      self.assertEqual(c_list[1], out[1].decode('utf-8'))

      out = c.eval(feed_dict={feed_t: np.array(c_list, dtype=np.object)})
      self.assertEqual(c_list[0], out[0].decode('utf-8'))
      self.assertEqual(c_list[1], out[1].decode('utf-8'))

  def testInvalidTargetFails(self):
    with self.assertRaises(RuntimeError):
      session.Session('INVALID_TARGET')

  def testFetchByNameDifferentStringTypes(self):
    with session.Session() as sess:
      c = constant_op.constant(42.0, name='c')
      d = constant_op.constant(43.0, name=u'd')
      e = constant_op.constant(44.0, name=b'e')
      f = constant_op.constant(45.0, name=r'f')

      self.assertTrue(isinstance(c.name, six.text_type))
      self.assertTrue(isinstance(d.name, six.text_type))
      self.assertTrue(isinstance(e.name, six.text_type))
      self.assertTrue(isinstance(f.name, six.text_type))

      self.assertEqual(42.0, sess.run('c:0'))
      self.assertEqual(42.0, sess.run(u'c:0'))
      self.assertEqual(42.0, sess.run(b'c:0'))
      self.assertEqual(42.0, sess.run(r'c:0'))

      self.assertEqual(43.0, sess.run('d:0'))
      self.assertEqual(43.0, sess.run(u'd:0'))
      self.assertEqual(43.0, sess.run(b'd:0'))
      self.assertEqual(43.0, sess.run(r'd:0'))

      self.assertEqual(44.0, sess.run('e:0'))
      self.assertEqual(44.0, sess.run(u'e:0'))
      self.assertEqual(44.0, sess.run(b'e:0'))
      self.assertEqual(44.0, sess.run(r'e:0'))

      self.assertEqual(45.0, sess.run('f:0'))
      self.assertEqual(45.0, sess.run(u'f:0'))
      self.assertEqual(45.0, sess.run(b'f:0'))
      self.assertEqual(45.0, sess.run(r'f:0'))

  def testIncorrectGraph(self):
    with ops.Graph().as_default() as g_1:
      c_1 = constant_op.constant(1.0, name='c')

    with ops.Graph().as_default() as g_2:
      c_2 = constant_op.constant(2.0, name='c')

    self.assertEqual('c', c_1.op.name)
    self.assertEqual('c', c_2.op.name)

    with session.Session(graph=g_1) as sess_1:
      self.assertEqual(1.0, sess_1.run(c_1))
      with self.assertRaises(ValueError):
        sess_1.run(c_2)
      with self.assertRaises(ValueError):
        sess_1.run(c_2.op)

    with session.Session(graph=g_2) as sess_2:
      with self.assertRaises(ValueError):
        sess_2.run(c_1)
      with self.assertRaises(ValueError):
        sess_2.run(c_1.op)
      self.assertEqual(2.0, sess_2.run(c_2))

if __name__ == '__main__':
  googletest.main()
