# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for StatSummarizer Python wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class StatSummarizerTest(test.TestCase):

  def testStatSummarizer(self):
    with ops.Graph().as_default() as graph:
      matrix1 = constant_op.constant([[3., 3.]], name=r"m1")
      matrix2 = constant_op.constant([[2.], [2.]], name=r"m2")
      product = math_ops.matmul(matrix1, matrix2, name=r"product")

      graph_def = graph.as_graph_def()
      ss = pywrap_tensorflow.NewStatSummarizer(graph_def.SerializeToString())

      with self.cached_session() as sess:
        sess.run(variables.global_variables_initializer())

        for _ in range(20):
          run_metadata = config_pb2.RunMetadata()
          run_options = config_pb2.RunOptions(
              trace_level=config_pb2.RunOptions.FULL_TRACE)
          sess.run(product, options=run_options, run_metadata=run_metadata)

          ss.ProcessStepStatsStr(run_metadata.step_stats.SerializeToString())

      output_string = ss.GetOutputString()

      print(output_string)

      # Test it recorded running the expected number of times.
      self.assertRegexpMatches(output_string, r"count=20")

      # Test that a header line got printed.
      self.assertRegexpMatches(output_string, r"====== .* ======")

      # Test that the nodes we added were analyzed.
      # The line for the op should contain both the op type (MatMul)
      # and the name of the node (product)
      self.assertRegexpMatches(output_string, r"MatMul.*product")
      self.assertRegexpMatches(output_string, r"Const.*m1")
      self.assertRegexpMatches(output_string, r"Const.*m2")

      # Test that a CDF summed to 100%
      self.assertRegexpMatches(output_string, r"100\.")

      pywrap_tensorflow.DeleteStatSummarizer(ss)


if __name__ == "__main__":
  test.main()
