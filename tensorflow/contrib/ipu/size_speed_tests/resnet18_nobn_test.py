from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.contrib.ipu import utils
from tensorflow.contrib.ipu import ops as ipu_ops
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session as sl
from tensorflow.python.framework import test_util
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent

datatype = np.float16


def _get_variable(name, shape, init):
  return vs.get_variable(name, shape, initializer=init, dtype=datatype)


def inference(x):

  with vs.variable_scope('all', use_resource=True):
    x = conv(x, 7, 2, 64)
    x = nn_ops.relu(x)
    x = max_pool(x, ksize=3, stride=2)
    x = block("b1", 64, 1, 2, x)
    x = block("b2", 128, 2, 2, x)
    x = block("b3", 256, 2, 2, x)
    x = block("b4", 512, 2, 2, x)
    x = nn_ops.max_pool(x, [1, x.shape[1], x.shape[2], 1], [1, 1, 1, 1],
                        'VALID')
    x = array_ops.reshape(x, [x.shape[0], x.shape[3]])
    x = fc("fc1", x, 1000)

  return x


def block(name, out_filters, first_stride, count, x):

  for i in range(count):
    sc = x
    shape_in = x.shape
    stride = (first_stride if (i == 0) else 1)

    with vs.variable_scope(name + "/" + str(i) + "/1"):
      x = conv(x, 3, stride, out_filters)
      x = nn_ops.relu(x)

    with vs.variable_scope(name + "/" + str(i) + "/2"):
      x = conv(x, 3, 1, out_filters)

      # shortcut
      if (stride != 1):
        sc = array_ops.strided_slice(
            sc, [0, 0, 0, 0], sc.shape, strides=[1, stride, stride, 1])
      pad = int(x.shape[3] - shape_in[3])
      if (pad != 0):
        sc = array_ops.pad(sc, paddings=[[0, 0], [0, 0], [0, 0], [0, pad]])

      x = nn_ops.relu(x + sc)

  return x


def fc(name, x, num_units_out):
  num_units_in = x.shape[1]
  weights_initializer = init_ops.truncated_normal_initializer(stddev=0.01)

  with vs.variable_scope(name):
    weights = _get_variable(
        'weights',
        shape=[num_units_in, num_units_out],
        init=weights_initializer)
    biases = _get_variable(
        'biases',
        shape=[num_units_out],
        init=init_ops.constant_initializer(0.0))

    x = nn_ops.xw_plus_b(x, weights, biases)

  return x


def conv(x, ksize, stride, filters_out):

  filters_in = x.shape[-1]

  wshape = [ksize, ksize, filters_in, filters_out]
  winitializer = init_ops.truncated_normal_initializer(stddev=0.1)
  bshape = [filters_out]
  binitializer = init_ops.zeros_initializer()

  weights = _get_variable('weights', shape=wshape, init=winitializer)
  biases = _get_variable('biases', shape=bshape, init=binitializer)
  stride = [1, stride, stride, 1]
  return nn_ops.conv2d(x, weights, strides=stride, padding='SAME') + biases


def max_pool(x, ksize=3, stride=2):
  return nn_ops.max_pool(
      x,
      ksize=[1, ksize, ksize, 1],
      strides=[1, stride, stride, 1],
      padding='SAME')


class Resnet18_No_Batchnorm(test_util.TensorFlowTestCase):
  def testInference(self):
    x = array_ops.placeholder(datatype, shape=[1, 224, 224, 4])
    y_ = array_ops.placeholder(datatype, shape=[1, 1000])

    with ipu_ops.ipu_scope("/device:IPU:0"):
      logits = inference(x)

      loss = math_ops.reduce_mean(
          nn_ops.softmax_cross_entropy_with_logits(logits=logits, labels=y_))

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    opts = utils.create_ipu_config(profiling=True, use_poplar_text_report=True)
    utils.configure_ipu_system(opts)
    sess = sl.Session()

    sess.run(variables.global_variables_initializer())
    sess.run(report)

    data = np.zeros([1, 224, 224, 4])
    labels = np.zeros([1, 1000])

    sess.run(loss, feed_dict={x: data, y_: labels})
    out = sess.run(report)

    sess.close()

    evts = utils.extract_all_events(out)
    size = utils.get_memory_size_from_events(evts)
    self.assertTrue(size < 227000000)

  def testTraining(self):
    x = array_ops.placeholder(datatype, shape=[1, 224, 224, 4])
    y_ = array_ops.placeholder(datatype, shape=[1, 1000])

    with ipu_ops.ipu_scope("/device:IPU:0"):
      logits = inference(x)

      loss = math_ops.reduce_mean(
          nn_ops.softmax_cross_entropy_with_logits(logits=logits, labels=y_))

      train = gradient_descent.GradientDescentOptimizer(0.01).minimize(loss)

    with ops.device('cpu'):
      report = gen_ipu_ops.ipu_event_trace()

    opts = utils.create_ipu_config(profiling=True, use_poplar_text_report=True)
    utils.configure_ipu_system(opts)

    sess = sl.Session()

    sess.run(variables.global_variables_initializer())
    sess.run(report)

    data = np.zeros([1, 224, 224, 4])
    labels = np.zeros([1, 1000])

    sess.run(train, feed_dict={x: data, y_: labels})
    out = sess.run(report)

    sess.close()

    evts = utils.extract_all_events(out)
    size = utils.get_memory_size_from_events(evts)
    self.assertTrue(size < 243000000)


if __name__ == "__main__":
  googletest.main()
