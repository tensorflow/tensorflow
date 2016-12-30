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

import tensorflow as tf


class StatSummarizerTest(tf.test.TestCase):

  def testStatSummarizer(self):
    with tf.Graph().as_default() as graph:
      matrix1 = tf.constant([[3., 3.]])
      matrix2 = tf.constant([[2.], [2.]])
      product = tf.matmul(matrix1, matrix2)

      graph_def = graph.as_graph_def()
      ss = tf.contrib.stat_summarizer.NewStatSummarizer(
          graph_def.SerializeToString())

      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())

        for _ in range(20):
          run_metadata = tf.RunMetadata()
          run_options = tf.RunOptions(
              trace_level=tf.RunOptions.FULL_TRACE)
          sess.run(product, options=run_options, run_metadata=run_metadata)

          ss.ProcessStepStatsStr(run_metadata.step_stats.SerializeToString())

      output_string = ss.GetOutputString()

      print(output_string)

      # Test it recorded running the expected number of times.
      self.assertRegexpMatches(output_string, r"count=20")

      # Test that a header line got printed.
      self.assertRegexpMatches(output_string, r"====== .* ======")

      # Test that the MatMul node we added was analyzed.
      self.assertRegexpMatches(output_string, r"MatMul")

      # Test that a CDF summed to 100%
      self.assertRegexpMatches(output_string, r"100\.")

      tf.contrib.stat_summarizer.DeleteStatSummarizer(ss)

if __name__ == "__main__":
  tf.test.main()
