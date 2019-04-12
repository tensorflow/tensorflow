# Copyright 2019 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import test_utils as tu

from tensorflow.python.platform import googletest
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops


class OneHotTopK(test_util.TensorFlowTestCase):
  def testOneHot(self):

    n_classes = 1200

    def model(a):
      return array_ops.one_hot(a, n_classes, dtype=dtypes.float32)

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.int32, [4], name="a")
      report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
      out = model(pa)

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      in_data = np.array([1, 4, 3, 5])

      fd = {pa: in_data}
      result = sess.run(out, fd)

      expected = np.eye(n_classes)[in_data]
      self.assertAllClose(result, expected)

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

  def testTopK(self):

    n_categories = 1200
    topn = 24

    def model(a):
      values, indices = nn.top_k(a, topn)
      return indices

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.float32, [n_categories], name="a")
      report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
      out = model(pa)

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      input = np.random.random(n_categories)
      expected = (-input).argsort()[:topn]

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result, expected)

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

  def testArgMax(self):

    batchsize = 4
    n_categories = 1200

    def model(a):
      return math_ops.argmax(a, axis=1, output_type=dtypes.int32)

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.float32, [batchsize, n_categories])
      report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
      out = model(pa)

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      input = np.random.rand(batchsize, n_categories)

      fd = {pa: input}
      result = sess.run(out, fd)
      self.assertAllClose(result, np.argmax(input, axis=1))

      result = sess.run(report)
      self.assertTrue(len(result) == 3)

  def testInTopK(self):

    batchsize = 4
    n_categories = 1200
    topn = 8

    def model(a, b):
      return nn.in_top_k(a, b, topn)

    with ops.device('cpu'):
      pa = array_ops.placeholder(np.float32, [batchsize, n_categories])
      pb = array_ops.placeholder(np.int32, [batchsize])
      report = gen_ipu_ops.ipu_event_trace()

    with ops.device("/device:IPU:0"):
      out = model(pa, pb)

    tu.configure_ipu_system()

    with tu.ipu_session() as sess:
      sess.run(report)

      input = np.random.rand(batchsize, n_categories)
      input = input / np.sqrt(np.sum(input**2))

      ref = (-input).argsort(axis=1)[:, :1]
      ref = ref.reshape([batchsize])

      expected = [True] * batchsize

      fd = {pa: input, pb: ref}
      result = sess.run(out, fd)
      self.assertAllClose(result, [True, True, True, True])

      result = sess.run(report)
      self.assertTrue(len(result) == 3)


if __name__ == "__main__":
  os.environ['TF_XLA_FLAGS'] = (
      '--tf_xla_min_cluster_size=1 ' + os.environ.get('TF_XLA_FLAGS', ''))
  googletest.main()
