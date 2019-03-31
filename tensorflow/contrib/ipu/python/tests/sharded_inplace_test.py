# Copyright 2019 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json

from tensorflow.contrib import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.contrib.ipu import ipu_compiler


class MappingTest(test_util.TensorFlowTestCase):
  def testConcat(self):
    def my_net(a, b, c, d):
      with ipu.ops.ipu_shard(0):
        c1 = array_ops.concat([a, b, d], axis=0)
        c2 = array_ops.concat([a, b, c], axis=0)
      return [c1, c2]

    with ops.device('cpu'):
      a = array_ops.placeholder(np.int32, [1])
      b = array_ops.placeholder(np.int32, [1])
      c = array_ops.placeholder(np.int32, [1])
      d = array_ops.placeholder(np.int32, [1])

    with ipu.ops.ipu_scope("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[a, b, c, d])

    cfg = ipu.utils.create_ipu_config()
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)
    with sl.Session() as sess:

      result = sess.run(r, {a: [0], b: [1], c: [2], d: [3]})
      self.assertAllClose(result[0], [0, 1, 3])
      self.assertAllClose(result[1], [0, 1, 2])


if __name__ == "__main__":
  googletest.main()
