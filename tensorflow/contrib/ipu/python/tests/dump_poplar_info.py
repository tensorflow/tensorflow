# Copyright 2019 Graphcore Ltd
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import numpy as np
import os

from tensorflow.contrib.ipu.python import ipu_compiler
from tensorflow.contrib import ipu
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class DumpPoplarInfo(test_util.TensorFlowTestCase):
  def testVertexGraphAndIntervalReport(self):
    tempdir = test.get_temp_dir()
    os.environ['TF_POPLAR_FLAGS'] = (
        '--save_vertex_graph=' + tempdir + " " + '--save_interval_report=' +
        tempdir + " " + os.environ.get('TF_POPLAR_FLAGS', ''))

    def my_model(pa, pb, pc):
      output = pa + pb + pc
      return [output]

    with ops.device("cpu"):
      pa = array_ops.placeholder(np.float32, [2048], name="a")
      pb = array_ops.placeholder(np.float32, [2048], name="b")
      pc = array_ops.placeholder(np.float32, [2048], name="c")

    with ops.device("/device:IPU:0"):
      r = ipu_compiler.compile(my_model, inputs=[pa, pb, pc])

    cfg = ipu.utils.create_ipu_config(profiling=False)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)

    with session_lib.Session() as sess:

      fd = {pa: [1.] * 2048, pb: [2.] * 2048, pc: [3.] * 2048}
      result = sess.run(r[0], fd)
      self.assertAllClose(result, [6.] * 2048)

      vertex_graphs = glob.glob(os.path.join(tempdir, "*.vertex_graph"))
      interval_reports = glob.glob(os.path.join(tempdir, "*.csv"))
      self.assertEqual(len(vertex_graphs), 1)
      self.assertEqual(len(interval_reports), 1)


if __name__ == "__main__":
  googletest.main()
