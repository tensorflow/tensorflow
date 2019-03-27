# Copyright 2017 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os

import test_utils as tu

from tensorflow.compiler.plugin.poplar.driver.config_pb2 import IpuOptions
from tensorflow.python.client import session as session_lib
from tensorflow.python.platform import googletest
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops


class IpuEngineCacheTest(test_util.TensorFlowTestCase):
  def testWriteToCache(self):
    cache_dir = self.get_temp_dir()

    with ops.device("/device:IPU:0"):
      pa = array_ops.placeholder(np.float32, [2, 2], name="a")
      pb = array_ops.placeholder(np.float32, [2, 2], name="b")
      output = pa + pb

    tu.configure_ipu_system()

    os.environ['TF_POPLAR_ENGINE_CACHE'] = cache_dir

    with session_lib.Session() as sess:

      fd = {pa: [[1., 1.], [2., 3.]], pb: [[0., 1.], [4., 5.]]}
      sess.run(output, fd)

    files = []
    for fn in os.listdir(cache_dir):
      if fn.endswith(".xla_engine"):
        files += [fn]

    self.assertEqual(len(files), 1)

  def testRestoreFromCache(self):
    # TODO figure out how to generate a cache file without filling the in
    # TODO memory cache too
    pass


if __name__ == "__main__":
  googletest.main()
