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
"""
  Tests to verify if the converion of python ops to MKL graph ops happen when
  MKL is enabled. These tests create graphs to verify the rewrite, and
  therefore will not test eager rewrites. The tests will be skipped if MKL is
  not enabled.
"""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2

config = config_pb2.ConfigProto()
run_options = config_pb2.RunOptions(
    trace_level=config_pb2.RunOptions.FULL_TRACE, output_partition_graphs=True)
config.graph_options.rewrite_options.constant_folding = (
    rewriter_config_pb2.RewriterConfig.OFF)
config.graph_options.optimizer_options.opt_level = -1

def RunAndTest(sess, op, mkl_nodename):
  run_meta = config_pb2.RunMetadata()
  sess.run(op, options=run_options, run_metadata=run_meta)
  graph = run_meta.partition_graphs
  for node in graph:
    for name in node.node:
      if name.op == mkl_nodename:
        return True
  return False

class TestForwardOps(test.TestCase):
  @test_util.run_deprecated_v1
  def testConv2D(self):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    y = [1, 4, 7, 2, 5, 8, 3, 6, 9]
    inp = constant_op.constant(x, dtype=dtypes.float32, shape=(1, 3, 4, 1))
    weights = constant_op.constant(y, dtype=dtypes.float32, shape=(3, 3, 1, 1))
    strides = [1, 1, 1, 1]
    conv = nn_ops.conv2d(inp, weights, padding="VALID", strides=strides)
    with self.cached_session(config=config) as sess:
      ret = RunAndTest(sess, conv, "_MklConv2D")
    self.assertEqual(ret, True)

class TestGradOps(test.TestCase):
  @test_util.run_deprecated_v1
  def testReluGrad(self):
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    inp = constant_op.constant(x, dtype=dtypes.float32, shape=(1, 3, 4, 1))
    grad = constant_op.constant(y, dtype=dtypes.float32, shape=(1, 3, 4, 1))
    relu_grad = nn_ops.relu_grad(grad, inp)
    with self.cached_session(config=config) as sess:
      ret = RunAndTest(sess, relu_grad, "_MklReluGrad")
    self.assertEqual(ret, True)

if __name__ == "__main__":
  if test_util.IsMklEnabled():
    test.main()
