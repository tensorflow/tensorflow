# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json

from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.contrib.compiler import xla
from tensorflow.contrib import ipu
from tensorflow.python.client import session as sl
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.contrib.ipu import ipu_compiler
from tensorflow.contrib.ipu import popnn_embedding


class EmbeddingLookupTest(test_util.TensorFlowTestCase):
  def testGather(self):
    def my_net(w, i):
      out = popnn_embedding.embedding_lookup(w, i)
      return [out]

    with ops.device('cpu'):
      i = array_ops.placeholder(np.int32, [8])
      w = array_ops.placeholder(np.float32, [100002, 200])
      report = gen_ipu_ops.ipu_event_trace()

    with ipu.ops.ipu_scope("/device:IPU:0"):
      r = ipu_compiler.compile(my_net, inputs=[w, i])

    cfg = ipu.utils.create_ipu_config(profiling=True)
    cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    ipu.utils.configure_ipu_system(cfg)
    with sl.Session() as sess:
      i_h = np.arange(0, 8)
      w_h = np.arange(20000400).reshape([100002, 200])

      result = sess.run(r, {i: i_h, w: w_h})
      self.assertAllClose(result[0], np.take(w_h, i_h, axis=0))


if __name__ == "__main__":
  googletest.main()
