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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class TFProfLoggerTest(test.TestCase):

  def _BuildSmallPlaceholderlModel(self):
    a = array_ops.placeholder(dtypes.int32, [2, 2])
    b = array_ops.placeholder(dtypes.int32, [2, 2])
    y = math_ops.matmul(a, b)
    return a, b, y

  def _BuildSmallModel(self):
    a = constant_op.constant([[1, 2], [3, 4]])
    b = constant_op.constant([[1, 2], [3, 4]])
    return math_ops.matmul(a, b)

  # pylint: disable=pointless-string-statement
  """# TODO(xpan): This this out of core so it doesn't depend on contrib.
  def testFillMissingShape(self):
    a, b, y = self._BuildSmallPlaceholderlModel()
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()
    sess = session.Session()
    sess.run(y,
             options=run_options,
             run_metadata=run_metadata,
             feed_dict={a: [[1, 2], [2, 3]],
                        b: [[1, 2], [2, 3]]})

    graph2 = ops.Graph()
    # Use copy_op_to_graph to remove shape information.
    y2 = copy_elements.copy_op_to_graph(y, graph2, [])
    self.assertEquals('<unknown>', str(y2.get_shape()))

    tfprof_logger._fill_missing_graph_shape(graph2, run_metadata)
    self.assertEquals('(2, 2)', str(y2.get_shape()))

  def testFailedFillMissingShape(self):
    y = self._BuildSmallModel()
    run_options = config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
    run_metadata = config_pb2.RunMetadata()
    sess = session.Session()
    sess.run(y, options=run_options, run_metadata=run_metadata)

    graph2 = ops.Graph()
    y2 = copy_elements.copy_op_to_graph(y, graph2, [])
    self.assertEquals('<unknown>', str(y2.get_shape()))
    # run_metadata has special name for MatMul, hence failed to fill shape.
    tfprof_logger._fill_missing_graph_shape(graph2, run_metadata)
    self.assertEquals('<unknown>', str(y2.get_shape()))
  """


if __name__ == '__main__':
  test.main()
