# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

import tensorflow as tf


class TFProfLoggerTest(tf.test.TestCase):

  def _BuildSmallPlaceholderlModel(self):
    a = tf.placeholder(tf.int32, [2, 2])
    b = tf.placeholder(tf.int32, [2, 2])
    y = tf.matmul(a, b)
    return a, b, y

  def _BuildSmallModel(self):
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[1, 2], [3, 4]])
    return tf.matmul(a, b)

  def testFillMissingShape(self):
    a, b, y = self._BuildSmallPlaceholderlModel()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess = tf.Session()
    sess.run(y,
             options=run_options,
             run_metadata=run_metadata,
             feed_dict={a: [[1, 2], [2, 3]],
                        b: [[1, 2], [2, 3]]})

    graph2 = tf.Graph()
    # Use copy_op_to_graph to remove shape information.
    y2 = tf.contrib.copy_graph.copy_op_to_graph(y, graph2, [])
    self.assertEquals('<unknown>', str(y2.get_shape()))

    tf.contrib.tfprof.tfprof_logger._fill_missing_graph_shape(graph2,
                                                              run_metadata)
    self.assertEquals('(2, 2)', str(y2.get_shape()))

  def testFailedFillMissingShape(self):
    y = self._BuildSmallModel()
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    sess = tf.Session()
    sess.run(y, options=run_options, run_metadata=run_metadata)

    graph2 = tf.Graph()
    y2 = tf.contrib.copy_graph.copy_op_to_graph(y, graph2, [])
    self.assertEquals('<unknown>', str(y2.get_shape()))
    # run_metadata has special name for MatMul, hence failed to fill shape.
    tf.contrib.tfprof.tfprof_logger._fill_missing_graph_shape(graph2,
                                                              run_metadata)
    self.assertEquals('<unknown>', str(y2.get_shape()))


if __name__ == '__main__':
  tf.test.main()
