# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow Lite Python Interface: Sanity check."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.lite.python import lite
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class LiteTest(test_util.TensorFlowTestCase):

  def testBasic(self):
    in_tensor = array_ops.placeholder(shape=[1, 16, 16, 3],
                                      dtype=dtypes.float32)
    out_tensor = in_tensor + in_tensor
    sess = session.Session()
    # Try running on valid graph
    result = lite.toco_convert(sess.graph_def, [in_tensor], [out_tensor])
    self.assertTrue(result)
    # TODO(aselle): remove tests that fail.
    # Try running on identity graph (known fail)
    # with self.assertRaisesRegexp(RuntimeError, "!model->operators.empty()"):
    #   result = lite.toco_convert(sess.graph_def, [in_tensor], [in_tensor])


if __name__ == "__main__":
  test.main()
