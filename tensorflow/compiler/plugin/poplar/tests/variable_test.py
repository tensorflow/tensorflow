# Copyright 2017 Graphcore Ltd
#

import os
import numpy as np

from tensorflow.python.platform import googletest
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
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
    with ops.device('cpu'):
     report = gen_ipu_ops.ipu_event_trace()
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("vs", use_resource=True):
        i = init_ops.random_normal_initializer(mean=2.0, stddev=0.01)
        z = variable_scope.get_variable(
            "z1", shape=[], dtype=np.float32, initializer=i)
    with tu.ipu_session() as sess:
      # Clean existing reports
      sess.run(report)
      sess.run(variables.global_variables_initializer())
      r = sess.run(report)

      o = sess.run(z)
      self.assertAllClose(o, 2.0, 0.2, 0.2)

      s = tu.extract_all_strings_from_event_trace(r)
      cs_list = tu.get_compute_sets_from_report(s)
      ok = ['vs/z1/Initializer/random_normal/RandomStandardNormal/fusion/Normal']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testRandomNormalNonScalarInitalizer(self):
    with ops.device('cpu'):
     report = gen_ipu_ops.ipu_event_trace()
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("vs", use_resource=True):
        i = init_ops.random_normal_initializer(mean=2.0, stddev=0.01)
        z = variable_scope.get_variable(
            "z1", shape=[2], dtype=np.float32, initializer=i)
    with tu.ipu_session() as sess:
      # Clean existing reports
      sess.run(report)
      sess.run(variables.global_variables_initializer())
      r = sess.run(report)

      o = sess.run(z)
      self.assertAllClose(o, [2.0, 2.0], 0.2, 0.2)

      s = tu.extract_all_strings_from_event_trace(r)
      cs_list = tu.get_compute_sets_from_report(s)
      ok = ['vs/z1/Initializer/random_normal/RandomStandardNormal/fusion/Normal']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

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
      self.assertTrue(len(r) == 3) # compile,load,execute

      s = tu.extract_all_strings_from_event_trace(r)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['z1/Initializer/truncated_normal/TruncatedNormal/fusion',
            'z1/Initializer/truncated_normal/mul',
            'z1/Initializer/truncated_normal/add.*/AddTo']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

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
      self.assertTrue(len(r) == 3) # compile,load,execute

      s = tu.extract_all_strings_from_event_trace(r)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['z1/Initializer/truncated_normal/TruncatedNormal/fusion',
            'z1/Initializer/truncated_normal/mul',
            'z1/Initializer/truncated_normal/add.*/AddTo']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))


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
      self.assertTrue(len(r) == 3) # compile,load,execute

      s = tu.extract_all_strings_from_event_trace(r)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['z1/Initializer/truncated_normal/TruncatedNormal/fusion']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))


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
      self.assertTrue(len(r) == 3) # compile,load,execute

      s = tu.extract_all_strings_from_event_trace(r)
      cs_list = tu.get_compute_sets_from_report(s)

      ok = ['z1/Initializer/truncated_normal/TruncatedNormal/fusion']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testUniformRandomInitalizer(self):
    with ops.device('cpu'):
     report = gen_ipu_ops.ipu_event_trace()
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("vs", use_resource=True):
        i = init_ops.random_uniform_initializer(minval=-2.0, maxval=2.0)
        z = variable_scope.get_variable(
            "z1", shape=[], dtype=np.float32, initializer=i)
    with tu.ipu_session() as sess:
      # Clean existing reports
      sess.run(report)
      sess.run(variables.global_variables_initializer())
      r = sess.run(report)

      o = sess.run(z)
      self.assertAllClose(o, 0.0, 2.0, 2.0)

      s = tu.extract_all_strings_from_event_trace(r)
      cs_list = tu.get_compute_sets_from_report(s)
      ok = ['vs/z1/Initializer/random_uniform/RandomUniform/fusion/Uniform']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))

  def testUniformRandomNonScalarInitalizer(self):
    with ops.device('cpu'):
     report = gen_ipu_ops.ipu_event_trace()
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("vs", use_resource=True):
        i = init_ops.random_uniform_initializer(minval=-2.0, maxval=2.0)
        z = variable_scope.get_variable(
            "z1", shape=[2], dtype=np.float32, initializer=i)
    with tu.ipu_session() as sess:
      # Clean existing reports
      sess.run(report)
      sess.run(variables.global_variables_initializer())
      r = sess.run(report)

      o = sess.run(z)
      self.assertAllClose(o, [0.0, 0.0] , 2.0, 2.0)

      s = tu.extract_all_strings_from_event_trace(r)
      cs_list = tu.get_compute_sets_from_report(s)
      ok = ['vs/z1/Initializer/random_uniform/RandomUniform/fusion/Uniform']
      self.assertTrue(tu.check_all_compute_sets_and_list(cs_list, ok))


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
      d_ul = "out_0.0"
      w_dl = "1.0"
      w_ul = "out_1.0"
      b_dl = "2.0"
      b_ul = "out_2.0"

      rep = sess.run(report)
      io_evts = tu.extract_all_io_events(rep)
      self.assertEqual(len(list(io_evts)), 2)
      # The initialization is constant, so there are no events generated on the
      # IPU.

      host_to_device = list(filter(
        lambda x:x[0]==IpuTraceEvent.HOST_TO_DEVICE_TRANSFER, io_evts))
      device_to_host = list(filter(
        lambda x:x[0]==IpuTraceEvent.DEVICE_TO_HOST_TRANSFER, io_evts))

      # Weights/biases should be downloaded once, and the input no times
      # because it is streamed
      self.assertEqual(len(list(filter(lambda x:x[1]==d_dl, host_to_device))), 0)
      self.assertEqual(len(list(filter(lambda x:x[1]==w_dl, host_to_device))), 1)
      self.assertEqual(len(list(filter(lambda x:x[1]==b_dl, host_to_device))), 1)

      # Weights/biases should not be uploaded, and the loss is streamed
      self.assertEqual(len(list(filter(lambda x:x[1]==d_ul, device_to_host))), 0)
      self.assertEqual(len(list(filter(lambda x:x[1]==w_ul, device_to_host))), 0)
      self.assertEqual(len(list(filter(lambda x:x[1]==b_ul, device_to_host))), 0)

      # Explicitly fetch the weights
      vw, vb = sess.run([w, b])

      self.assertAllClose(
        np.array([[-1.3, -0.3], [1.7, 2.7], [2.9, 3.9], [3.5, 4.5]],
                 dtype=np.float32),
        vw, rtol=1e-4)

      self.assertAllClose(np.array([1.5, 2.5], dtype=np.float32), vb, rtol=1e-4)

      rep = sess.run(report)
      io_evts = tu.extract_all_io_events(rep)

      host_to_device = list(filter(
        lambda x:x[0]==IpuTraceEvent.HOST_TO_DEVICE_TRANSFER, io_evts))
      device_to_host = list(filter(
        lambda x:x[0]==IpuTraceEvent.DEVICE_TO_HOST_TRANSFER, io_evts))
      self.assertEqual(len(list(io_evts)), 2)

      # Weights/biases/inputs should not be downloaded at all
      self.assertEqual(len(list(filter(lambda x:x[1]==d_dl, host_to_device))), 0)
      self.assertEqual(len(list(filter(lambda x:x[1]==w_dl, host_to_device))), 0)
      self.assertEqual(len(list(filter(lambda x:x[1]==b_dl, host_to_device))), 0)

      # Weights/biases should be uploaded once (explicitly fetched)
      self.assertEqual(len(list(filter(lambda x:x[1]==d_ul, device_to_host))), 0)
      self.assertEqual(len(list(filter(lambda x:x[1]==w_ul, device_to_host))), 1)
      self.assertEqual(len(list(filter(lambda x:x[1]==b_ul, device_to_host))), 1)

  def testResourceCountsAreCorrect(self):
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("vs", use_resource=True):
        w1 = variable_scope.get_variable("w1", shape=[4, 2], dtype=np.float32,
                                         initializer=init_ops.constant_initializer(
                                           np.array(
                                             [[1, 2], [3, 4], [5, 6], [7, 8]],
                                             dtype=np.float32)))
        b1 = variable_scope.get_variable("b1", shape=[2], dtype=np.float32,
                                         trainable=False,
                                         initializer=init_ops.constant_initializer(
                                           np.array([2, 3], dtype=np.float32)))
        w2 = variable_scope.get_variable("w2", shape=[2, 2], dtype=np.float32,
                                         initializer=init_ops.constant_initializer(
                                           np.array(
                                             [[1, 2], [3, 4]],
                                             dtype=np.float32)))
        b2 = variable_scope.get_variable("b2", shape=[2], dtype=np.float32,
                                         trainable=False,
                                         initializer=init_ops.constant_initializer(
                                           np.array([2, 3], dtype=np.float32)))

      x = array_ops.placeholder(np.float32, shape=[1, 4])
      y = math_ops.matmul(x, w1) + b1
      y = math_ops.matmul(y, w2) + b2

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
      w1_dl = "1.0"
      b1_dl = "2.0"
      w2_dl = "3.0"
      b2_dl = "4.0"

      # biases are not outputs of the graph
      d_ul = "out_0.0"
      w1_ul = "out_1.0"
      w2_ul = "out_2.0"

      rep = sess.run(report)
      io_evts = tu.extract_all_io_events(rep)

      # The initialization is constant, so there are no events generated on the
      # IPU.

      host_to_device = list(filter(
        lambda x:x[0]==IpuTraceEvent.HOST_TO_DEVICE_TRANSFER, io_evts))
      device_to_host = list(filter(
        lambda x:x[0]==IpuTraceEvent.DEVICE_TO_HOST_TRANSFER, io_evts))
      self.assertEqual(len(list(io_evts)), 4)

      # Weights/biases should be downloaded once, and the input no times
      # because it is streamed
      self.assertEqual(len(list(filter(lambda x:x[1]==d_dl, host_to_device))), 0)
      self.assertEqual(len(list(filter(lambda x:x[1]==w1_dl, host_to_device))), 1)
      self.assertEqual(len(list(filter(lambda x:x[1]==b1_dl, host_to_device))), 1)
      self.assertEqual(len(list(filter(lambda x:x[1]==w2_dl, host_to_device))), 1)
      self.assertEqual(len(list(filter(lambda x:x[1]==b2_dl, host_to_device))), 1)

      # Weights should not be uploaded, and the loss is streamed
      self.assertEqual(len(list(filter(lambda x:x[1]==d_ul, device_to_host))), 0)
      self.assertEqual(len(list(filter(lambda x:x[1]==w1_ul, device_to_host))), 0)
      self.assertEqual(len(list(filter(lambda x:x[1]==w2_ul, device_to_host))), 0)

      # Explicitly fetch the first set of weights and biases
      vw, vb = sess.run([w1, b1])

      self.assertAllClose(
        np.array([[100.00576782, 86.60944366],[57.62784195,51.23856354],
                  [93.45920563,82.40240479],[155.36032104,135.74447632]],
                 dtype=np.float32),
        vw, rtol=1e-4)

      self.assertAllClose(np.array([2, 3], dtype=np.float32), vb, rtol=1e-4)

      rep = sess.run(report)
      io_evts = tu.extract_all_io_events(rep)

      host_to_device = list(filter(
        lambda x:x[0]==IpuTraceEvent.HOST_TO_DEVICE_TRANSFER, io_evts))
      device_to_host = list(filter(
        lambda x:x[0]==IpuTraceEvent.DEVICE_TO_HOST_TRANSFER, io_evts))
      self.assertEqual(len(list(io_evts)), 2)

      # Weights/biases/inputs should not be downloaded at all
      self.assertEqual(len(list(filter(lambda x:x[1]==d_dl, host_to_device))), 0)
      self.assertEqual(len(list(filter(lambda x:x[1]==w1_dl, host_to_device))), 0)
      self.assertEqual(len(list(filter(lambda x:x[1]==b1_dl, host_to_device))), 0)
      self.assertEqual(len(list(filter(lambda x:x[1]==w2_dl, host_to_device))), 0)
      self.assertEqual(len(list(filter(lambda x:x[1]==b2_dl, host_to_device))), 0)

      # Weights should be uploaded once (explicitly fetched)
      # Note all weights are fetched as a group
      self.assertEqual(len(list(filter(lambda x:x[1]==d_ul, device_to_host))), 0)
      self.assertEqual(len(list(filter(lambda x:x[1]==w1_ul, device_to_host))), 1)
      self.assertEqual(len(list(filter(lambda x:x[1]==w2_ul, device_to_host))), 1)

  def testTuplesOfTuplesAreStreamed(self):
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("vs", use_resource=True):
        pa = array_ops.placeholder(np.int64, [2,2], name="a")
        pb = array_ops.placeholder(np.int64, [2,2], name="b")
        pc = array_ops.placeholder(np.int64, [2,2], name="c")
        c = control_flow_ops.tuple((pa + pc, pb + pc))

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session(True, True, True) as sess:
      sess.run(report)
      in0 = np.full((2,2), 7)
      in1 = np.full((2,2), 6)
      in2 = np.full((2,2), 5)
      fd = {
        pa : in0,
        pb : in1,
        pc : in2,
      }
      out = sess.run(c, fd)
      self.assertEqual(len(out), 2)
      self.assertAllClose(out, (np.full((2,2), 12), np.full((2,2), 11)))

      rep = sess.run(report)
      io_evts = tu.extract_all_io_events(rep)
      # No io_events implies the data was streamed
      self.assertEqual(len(list(io_evts)), 0)

  def testNonModifiedResourceIsNotOverwrittenInPlaceOp(self):
    # This test verifies that if we have a resource varaible (w) which is marked
    # as not modified then a copy is inserted to make sure it is not overwritten
    # between executions if it is used by an inplace op
    w_val = [1, 2, 3, 4]
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("vs", use_resource=True):
        w = variable_scope.get_variable("w", shape=[4], dtype=np.float32,
                                         initializer=init_ops.constant_initializer(
                                           np.array(w_val,
                                             dtype=np.float32)))

      px = array_ops.placeholder(np.float32, shape=[4])
      y = w + px

      with ops.device('cpu'):
        report = gen_ipu_ops.ipu_event_trace()

    with tu.ipu_session(True, True, True) as sess:
      sess.run(variables.global_variables_initializer())

      sess.run(report)
      xs = [np.array([7, 3, 5, 9], dtype=np.float32),
            np.array([1, 8, 3, 4], dtype=np.float32),
            np.array([9, 2, 2, 6], dtype=np.float32)]
      for x in xs:
        out = sess.run(y, {px: x})
        self.assertAllClose(out, x + w_val)


      rep = sess.run(report)
      io_evts = tu.extract_all_io_events(rep)

      host_to_device = list(filter(
        lambda x:x[0]==IpuTraceEvent.HOST_TO_DEVICE_TRANSFER, io_evts))
      self.assertEqual(len(list(host_to_device)), 1)
      device_to_host = list(filter(
        lambda x:x[0]==IpuTraceEvent.DEVICE_TO_HOST_TRANSFER, io_evts))
      self.assertEqual(len(list(device_to_host)), 0)

      # w should be copied to device once and that should be the only io event
      w_dl = "1.0"
      self.assertEqual(len(list(filter(lambda x:x[1]==w_dl, host_to_device))), 1)

class IpuXlaVariableTestSyntheticData(test_util.TensorFlowTestCase):
  # This test is in its separate class to prevent messing up the enviroment for
  # other tests.

  def testResourceCountsAreCorrect(self):
    # Same test as above, but no copying to and form device should occur.
    os.environ["TF_POPLAR_USE_SYNTHETIC_DATA"] = "true"
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("vs", use_resource=True):
        w1 = variable_scope.get_variable("w1", shape=[4, 2], dtype=np.float32,
                                         initializer=init_ops.constant_initializer(
                                           np.array(
                                             [[1, 2], [3, 4], [5, 6], [7, 8]],
                                             dtype=np.float32)))
        b1 = variable_scope.get_variable("b1", shape=[2], dtype=np.float32,
                                         trainable=False,
                                         initializer=init_ops.constant_initializer(
                                           np.array([2, 3], dtype=np.float32)))
        w2 = variable_scope.get_variable("w2", shape=[2, 2], dtype=np.float32,
                                         initializer=init_ops.constant_initializer(
                                           np.array(
                                             [[1, 2], [3, 4]],
                                             dtype=np.float32)))
        b2 = variable_scope.get_variable("b2", shape=[2], dtype=np.float32,
                                         trainable=False,
                                         initializer=init_ops.constant_initializer(
                                           np.array([2, 3], dtype=np.float32)))

      x = array_ops.placeholder(np.float32, shape=[1, 4])
      y = math_ops.matmul(x, w1) + b1
      y = math_ops.matmul(y, w2) + b2

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

      rep = sess.run(report)
      io_evts = tu.extract_all_io_events(rep)
      self.assertEqual(len(list(io_evts)), 0)

      # Explicitly fetch the first set of weights and biases
      sess.run([w1, b1])

      rep = sess.run(report)
      io_evts = tu.extract_all_io_events(rep)
      self.assertEqual(len(list(io_evts)), 0)

if __name__ == "__main__":
    googletest.main()
