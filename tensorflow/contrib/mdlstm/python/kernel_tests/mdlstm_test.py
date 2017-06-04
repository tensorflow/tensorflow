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
"""Tests for MDLSTM cells."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.contrib.mdlstm.python.ops import mdlstm
import numpy as np

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class MDLSTMCellTest(test.TestCase):

  def testMDLSTMCell(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.2)) as root_scope:
        x = array_ops.zeros([1,2,2,6])
        m = array_ops.zeros([1,2]), array_ops.zeros([1,2]), array_ops.zeros([1,2]), array_ops.zeros([1,2])
        cell = mdlstm.MultiDimentionalLSTMCell(2)
        
        g, s = cell(x, m)
        self.assertEqual(g[0].get_shape(), (1, 2))
        self.assertEqual(g[1].get_shape(), (1, 2))
        self.assertEqual(s[0].get_shape(), (1, 2))
        self.assertEqual(s[1].get_shape(), (1, 2))

        sess.run([variables.global_variables_initializer()])
        res = sess.run([g, s], {
            x: np.ones((1,2,2,6)),
            m: [np.ones((1,2)),np.ones((1,2)),np.ones((1,2)),np.ones((1,2))],
        })
        self.assertEqual(res[0][0].shape, (1, 2))
        self.assertEqual(res[0][1].shape, (1, 2))
        self.assertEqual(res[1][0].shape, (1, 2))
        self.assertEqual(res[1][1].shape, (1, 2))

if __name__ == '__main__':
  test.main()