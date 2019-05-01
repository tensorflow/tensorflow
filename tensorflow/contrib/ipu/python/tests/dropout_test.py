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
# =============================================================================

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
from tensorflow.contrib.ipu.python import poprand

import tensorflow as tf


class PopnnRandomDropoutTest(test_util.TensorFlowTestCase):
  def testDropout(self):
    def testDropoutImpl(rate):
      def ipu_dropout(w):
        output = poprand.dropout(w, rate=rate)
        return [output]

      with ops.device('cpu'):
        input_data = array_ops.placeholder(np.float32, [1024, 1024, 4])
        report = gen_ipu_ops.ipu_event_trace()

      with ipu.ops.ipu_scope("/device:IPU:0"):
        r = ipu_compiler.compile(ipu_dropout, inputs=[input_data])

      cfg = ipu.utils.create_ipu_config()
      cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
      ipu.utils.configure_ipu_system(cfg)
      with sl.Session() as sess:
        in_data = np.random.rand(1024, 1024, 4)

        result = sess.run(r, {input_data: in_data})

        percent_kept = np.count_nonzero(result) / np.count_nonzero(in_data)

        # There's a considerable amount for randomness so we have a reasonably large
        # dimensionality of test data to make sure the error is smaller.
        is_roughly_close = abs(percent_kept - (1.0 - rate))

        # The observed error is actually a lot less than this (>1%) but we don't want to cause
        # random regressions and 3% is probably still acceptable for any outlier randoms.
        self.assertTrue(is_roughly_close < 0.03)

    # We want to test the internal seed is working.
    for i in range(0, 6):
      testDropoutImpl(np.random.uniform())

  # Check user provided seed works
  def testDropoutUserSeed(self):
    def testDropoutImpl(rate, seed, in_data):
      def ipu_dropout(w):
        output = poprand.dropout(w, rate=rate, seed=seed)
        return [output]

      with ops.device('cpu'):
        input_data = array_ops.placeholder(np.float32, [32, 4])
        report = gen_ipu_ops.ipu_event_trace()

      with ipu.ops.ipu_scope("/device:IPU:0"):
        r = ipu_compiler.compile(ipu_dropout, inputs=[input_data])

      cfg = ipu.utils.create_ipu_config()
      cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
      ipu.utils.configure_ipu_system(cfg)

      with sl.Session() as sess:
        return sess.run(r, {input_data: in_data})

    # Randomize the parameters but keep them the same for all runs
    int32_limits = np.iinfo(np.int32)
    seed = np.random.randint(int32_limits.max, size=[2], dtype=np.int32)
    seed_tensor = tf.constant(seed)
    rate = np.random.uniform()
    in_data = np.random.rand(32, 4)

    outs = []
    # Run with the same seed multiple times then check they are the same.
    for i in range(0, 6):
      outs.append(testDropoutImpl(rate, seed_tensor, in_data))

    for i in range(1, 6):
      self.assertAllEqual(outs[0], outs[i])


# Check user provided seed works

  def testDropoutBackwardPass(self):
    def testDropoutImpl():
      def ipu_dropout_back(w):
        output = poprand.dropout(w, rate=0.4)

        largest = output
        cost = tf.square(largest)

        opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)

        gradients = opt.compute_gradients(cost, w)

        return [output, gradients]

      with ops.device('cpu'):
        input_data = array_ops.placeholder(np.float32, [32])
        report = gen_ipu_ops.ipu_event_trace()

      with ipu.ops.ipu_scope("/device:IPU:0"):
        r = ipu_compiler.compile(ipu_dropout_back, inputs=[input_data])

      cfg = ipu.utils.create_ipu_config()
      cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
      ipu.utils.configure_ipu_system(cfg)

      with sl.Session() as sess:
        in_data = np.random.rand(32)
        out = sess.run(r, {input_data: in_data})

        dropout_out = out[0]
        gradients = out[1][0][0]

        # Check we have the same number of zeros.
        self.assertAllEqual(
            np.count_nonzero(dropout_out), np.count_nonzero(gradients))

    # Run with the same seed multiple times then check they are the same.
    for i in range(0, 6):
      testDropoutImpl()

if __name__ == "__main__":
  googletest.main()
