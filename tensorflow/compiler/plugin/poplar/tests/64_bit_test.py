# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import googletest
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops


class Ipu64BitTest(test_util.TensorFlowTestCase):
  def testAdd(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.int64, [2, 2], name="a")
        pb = array_ops.placeholder(np.int64, [2, 2], name="b")
        output = pa + pb

        fd = {pa: [[1, 1], [2, 3]], pb: [[0, 1], [4, 5]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[1, 2], [6, 8]])

        fd = {pa: [[0, 0], [1, 1]], pb: [[2, 1], [4, 5]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[2, 1], [5, 6]])

  def testSubConstant(self):
    with ops.device("/device:IPU:0"):
      with session_lib.Session() as sess:
        pa = array_ops.placeholder(np.int64, [2, 2], name="a")
        pb = array_ops.constant([[1, 2], [3, 4]], dtype=np.int64)
        output = pa - pb

        fd = {pa: [[1, 1], [2, 3]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[0, -1], [-1, -1]])

        fd = {pa: [[0, 0], [4, 4]]}
        result = sess.run(output, fd)
        self.assertAllClose(result, [[-1, -2], [1, 0]])


if __name__ == "__main__":
  googletest.main()
