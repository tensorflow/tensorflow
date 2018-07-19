# Copyright 2017 Graphcore Ltd
#

import numpy as np

from tensorflow.python.platform import googletest
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import gradient_descent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent

import test_utils as tu

class IpuXlaVariableTest(test_util.TensorFlowTestCase):

  def testInitializeSimpleVariables(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:

        x = resource_variable_ops.ResourceVariable(
          random_ops.random_normal([5,5], stddev=0.1), name="x")
        y = resource_variable_ops.ResourceVariable(
          random_ops.random_normal([1], stddev=0.1), name="y")

        sess.run(variables.global_variables_initializer())

        r1, r2 = sess.run([x,y])

        self.assertAllClose(r1, np.zeros([5,5]), atol=1.0)
        self.assertAllClose(r2, [0.0], atol=1.0)

  def testInitializeSharedVariables(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        with variable_scope.variable_scope("vs", use_resource=True):
          x = variable_scope.get_variable("x", shape=[], dtype=np.float32,
                              initializer=init_ops.constant_initializer(1))

          y = variable_scope.get_variable("y", shape=[], dtype=np.float32,
                               initializer=init_ops.constant_initializer(2))

        sess.run(variables.global_variables_initializer())

        r1, r2 = sess.run([x,y])

        self.assertAllClose(r1, 1)
        self.assertAllClose(r2, 2)

  def testRead(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        with variable_scope.variable_scope("vs", use_resource=True):
          z = variable_scope.get_variable("z", shape=[], dtype=np.float32,
                              initializer=init_ops.constant_initializer(3))

        sess.run(variables.global_variables_initializer())

        r = sess.run(z.read_value())

        self.assertAllClose(r, 3)

  def testAssign(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        with variable_scope.variable_scope("vs", use_resource=True):
          z = variable_scope.get_variable("z", shape=[], dtype=np.float32,
                              initializer=init_ops.constant_initializer(0))

        sess.run(variables.global_variables_initializer())

        sess.run(state_ops.assign(z, 2))
        r = sess.run(z)
        self.assertAllClose(r, 2)

        sess.run(state_ops.assign_add(z, 6))
        r = sess.run(z)
        self.assertAllClose(r, 8)

  def testGradientDescent(self):
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("vs", use_resource=True):

        w = variable_scope.get_variable("w", shape=[4, 2], dtype=np.float32,
          initializer=init_ops.constant_initializer(
            np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)))
        b = variable_scope.get_variable("b", shape=[2], dtype=np.float32,
          initializer=init_ops.constant_initializer(
            np.array([2, 3], dtype=np.float32)))

      x = array_ops.placeholder(np.float32, shape=[1, 4])
      y = math_ops.matmul(x, w) + b

      loss = math_ops.reduce_sum(y)
      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(loss)

    with session_lib.Session() as sess:

      sess.run(variables.global_variables_initializer())
      sess.run(train, {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})

      vw, vb = sess.run([w, b])

      self.assertAllClose(
        np.array([[0.3, 1.3], [2.7, 3.7], [4.5, 5.5], [6.1, 7.1]],
                 dtype=np.float32),
        vw, rtol=1e-4)

      self.assertAllClose(np.array([1.9, 2.9], dtype=np.float32), vb, rtol=1e-4)

  def testRepeatedGradientDescent(self):
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("vs", use_resource=True):

        w = variable_scope.get_variable("w", shape=[4, 2], dtype=np.float32,
                            initializer=init_ops.constant_initializer(
                              np.array([[1, 2], [3, 4], [5, 6], [7, 8]],
                                       dtype=np.float32)))
        b = variable_scope.get_variable("b", shape=[2], dtype=np.float32,
                            initializer=init_ops.constant_initializer(
                              np.array([2, 3], dtype=np.float32)))

      x = array_ops.placeholder(np.float32, shape=[1, 4])
      y = math_ops.matmul(x, w) + b

      loss = math_ops.reduce_sum(y)
      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(loss)

    with session_lib.Session() as sess:

      sess.run(variables.global_variables_initializer())
      sess.run(train, {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      sess.run(train, {x: np.array([[1, 2, 3, 4]], dtype=np.float32)})
      sess.run(train, {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      sess.run(train, {x: np.array([[1, 2, 3, 4]], dtype=np.float32)})
      sess.run(train, {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})

      vw, vb = sess.run([w, b])

      self.assertAllClose(
        np.array([[-1.3, -0.3], [1.7, 2.7], [2.9, 3.9], [3.5, 4.5]],
                 dtype=np.float32),
        vw, rtol=1e-4)

      self.assertAllClose(np.array([1.5, 2.5], dtype=np.float32), vb, rtol=1e-4)


  def testMultipleUpdate(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        with variable_scope.variable_scope("vs", use_resource=True):
          z = variable_scope.get_variable("z", shape=[], dtype=np.float32,
                              initializer=init_ops.constant_initializer(0))

        updater = state_ops.assign_add(z, 1.0)

        sess.run(variables.global_variables_initializer())

        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)
        sess.run(updater)

        r = sess.run(z)
        self.assertAllClose(r, 10.0)

  def testRandomNormalInitalizer(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        with variable_scope.variable_scope("vs", use_resource=True):
          i = init_ops.random_normal_initializer(mean=2.0, stddev=0.01)
          z = variable_scope.get_variable(
              "z1", shape=[], dtype=np.float32, initializer=i)

        sess.run(variables.global_variables_initializer())
        o = sess.run(z)
        self.assertAllClose(o, 2.0, 0.2, 0.2)

  def testDefaultRandomNormalInitalizer(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        with variable_scope.variable_scope("vs", use_resource=True):
          i = init_ops.random_normal_initializer()
          z = variable_scope.get_variable(
              "z1", shape=[], dtype=np.float32, initializer=i)

        sess.run(variables.global_variables_initializer())
        o = sess.run(z)
        self.assertAllClose(o, 0.0, 1.0, 3.0)

  def testTruncatedNormalScalarInitalizer(self):

    with ops.device('cpu'):
     report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("", use_resource=True):
        i = init_ops.truncated_normal_initializer(mean=1.0, stddev=0.01)
        z = variable_scope.get_variable(
          "z1", shape=[], dtype=np.float32, initializer=i)

    with tu.ipu_session() as sess:
      # Clean existing reports
      sess.run(report)

      sess.run(variables.global_variables_initializer())
      o = sess.run(z)
      self.assertAllClose(o, 1.0, 0.2, 0.2)

      # Find of the names of compute sets
      r = sess.run(report)
      self.assertTrue(len(r) == 5)
      cs_list = tu.get_compute_sets_from_report(r[2])

      ok = ['z1/Initializer/truncated_normal/TruncatedNormal/call',
            'z1/Initializer/truncated_normal/mul',
            'z1/Initializer/truncated_normal/add']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testTruncatedNormalInitalizer(self):

    with ops.device('cpu'):
     report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("", use_resource=True):
        i = init_ops.truncated_normal_initializer(mean=1.0, stddev=0.01)
        z = variable_scope.get_variable(
          "z1", shape=[2, 4], dtype=np.float32, initializer=i)

    with tu.ipu_session() as sess:
      # Clean existing reports
      sess.run(report)

      sess.run(variables.global_variables_initializer())
      o = sess.run(z)
      self.assertAllClose(o, np.ones((2,4)), 0.2, 0.2)

      # Find of the names of compute sets
      r = sess.run(report)
      self.assertTrue(len(r) == 5)
      cs_list = tu.get_compute_sets_from_report(r[2])

      ok = ['z1/Initializer/truncated_normal/TruncatedNormal/call',
            'z1/Initializer/truncated_normal/mul',
            'z1/Initializer/truncated_normal/add']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))


  def testDefaultTruncatedNormalScalarInitalizer(self):

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          i = init_ops.truncated_normal_initializer()
          z = variable_scope.get_variable(
              "z1", shape=[], dtype=np.float32, initializer=i)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      o = sess.run(z)
      self.assertAllClose(o, 1.0, 2.0, 2.0)

      # Find of the names of compute sets
      r = sess.run(report)
      self.assertTrue(len(r) == 5)
      cs_list = tu.get_compute_sets_from_report(r[2])

      ok = ['z1/Initializer/truncated_normal/TruncatedNormal/call']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))


  def testDefaultTruncatedNormalInitalizer(self):

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
        with variable_scope.variable_scope("", use_resource=True):
          i = init_ops.truncated_normal_initializer()
          z = variable_scope.get_variable(
              "z1", shape=[2, 4], dtype=np.float32, initializer=i)

    with tu.ipu_session() as sess:
      sess.run(variables.global_variables_initializer())
      o = sess.run(z)
      self.assertAllClose(o, np.ones((2,4)), 2.0, 2.0)

      # Find of the names of compute sets
      r = sess.run(report)
      self.assertTrue(len(r) == 5)
      cs_list = tu.get_compute_sets_from_report(r[2])

      ok = ['z1/Initializer/truncated_normal/TruncatedNormal/call']
      self.assertTrue(tu.check_all_compute_sets_in_list(cs_list, ok))

  def testUniformRandomInitalizer(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        with variable_scope.variable_scope("vs", use_resource=True):
          i = init_ops.random_uniform_initializer(minval=-2.0, maxval=2.0)
          z = variable_scope.get_variable(
              "z1", shape=[], dtype=np.float32, initializer=i)

        sess.run(variables.global_variables_initializer())
        o = sess.run(z)
        self.assertAllClose(o, 0.0, 2.0, 2.0)

  def testDefaultUniformRandomInitalizer(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        with variable_scope.variable_scope("vs", use_resource=True):
          i = init_ops.random_uniform_initializer()
          z = variable_scope.get_variable(
              "z1", shape=[], dtype=np.float32, initializer=i)

        sess.run(variables.global_variables_initializer())
        o = sess.run(z)
        self.assertAllClose(o, 0.5, 0.5, 0.5)

  def testVariablesRemainResident(self):
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("vs", use_resource=True):

        w = variable_scope.get_variable("w", shape=[4, 2], dtype=np.float32,
                            initializer=init_ops.constant_initializer(
                              np.array([[1, 2], [3, 4], [5, 6], [7, 8]],
                                       dtype=np.float32)))
        b = variable_scope.get_variable("b", shape=[2], dtype=np.float32,
                            initializer=init_ops.constant_initializer(
                              np.array([2, 3], dtype=np.float32)))

      x = array_ops.placeholder(np.float32, shape=[1, 4])
      y = math_ops.matmul(x, w) + b

      loss = math_ops.reduce_sum(y)
      optimizer = gradient_descent.GradientDescentOptimizer(0.1)
      train = optimizer.minimize(loss)

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session(True, True, True) as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)

      sess.run([train,loss], {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      sess.run([train,loss], {x: np.array([[1, 2, 3, 4]], dtype=np.float32)})
      sess.run([train,loss], {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})
      sess.run([train,loss], {x: np.array([[1, 2, 3, 4]], dtype=np.float32)})
      sess.run([train,loss], {x: np.array([[7, 3, 5, 9]], dtype=np.float32)})

      d_dl = "0.0"
      d_ul = "0"
      w_dl = "1.0"
      w_ul = "1"
      b_dl = "2.0"
      b_ul = "2"

      rep = sess.run(report)
      io_evts = tu.extract_all_io_events(rep)

      # The initialization is constant, so there are no events generated on the
      # IPU.

      host_to_device = filter(
        lambda x:x[0]==IpuTraceEvent.HOST_TO_DEVICE_TRANSFER, io_evts)
      device_to_host = filter(
        lambda x:x[0]==IpuTraceEvent.DEVICE_TO_HOST_TRANSFER, io_evts)

      # Weights/biases should be downloaded once, and the input no times
      # because it is streamed
      self.assertEqual(len(filter(lambda x:x[1]==d_dl, host_to_device)), 0)
      self.assertEqual(len(filter(lambda x:x[1]==w_dl, host_to_device)), 1)
      self.assertEqual(len(filter(lambda x:x[1]==b_dl, host_to_device)), 1)

      # Weights/biases should not be uploaded, and the loss is streamed
      self.assertEqual(len(filter(lambda x:x[1]==d_ul, device_to_host)), 0)
      self.assertEqual(len(filter(lambda x:x[1]==w_ul, device_to_host)), 0)
      self.assertEqual(len(filter(lambda x:x[1]==b_ul, device_to_host)), 0)

      # Explicitly fetch the weights
      vw, vb = sess.run([w, b])

      self.assertAllClose(
        np.array([[-1.3, -0.3], [1.7, 2.7], [2.9, 3.9], [3.5, 4.5]],
                 dtype=np.float32),
        vw, rtol=1e-4)

      self.assertAllClose(np.array([1.5, 2.5], dtype=np.float32), vb, rtol=1e-4)

      rep = sess.run(report)
      io_evts = tu.extract_all_io_events(rep)

      host_to_device = filter(
        lambda x:x[0]==IpuTraceEvent.HOST_TO_DEVICE_TRANSFER, io_evts)
      device_to_host = filter(
        lambda x:x[0]==IpuTraceEvent.DEVICE_TO_HOST_TRANSFER, io_evts)

      # Weights/biases/inputs should not be downloaded at all
      self.assertEqual(len(filter(lambda x:x[1]==d_dl, host_to_device)), 0)
      self.assertEqual(len(filter(lambda x:x[1]==w_dl, host_to_device)), 0)
      self.assertEqual(len(filter(lambda x:x[1]==b_dl, host_to_device)), 0)

      # Weights/biases should be uploaded once (explicitly fetched)
      self.assertEqual(len(filter(lambda x:x[1]==d_ul, device_to_host)), 0)
      self.assertEqual(len(filter(lambda x:x[1]==w_ul, device_to_host)), 1)
      self.assertEqual(len(filter(lambda x:x[1]==b_ul, device_to_host)), 1)

if __name__ == "__main__":
    googletest.main()
