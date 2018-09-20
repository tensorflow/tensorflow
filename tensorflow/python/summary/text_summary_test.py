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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest
from tensorflow.python.summary import text_summary


class TextPluginTest(test_util.TensorFlowTestCase):
  """Test the Text Summary API.

  These tests are focused on testing the API design of the text_summary method.
  It doesn't test the PluginAsset and tensors registry functionality, because
  that is better tested by the text_plugin test that actually consumes that
  metadata.
  """

  def testTextSummaryAPI(self):
    with self.cached_session():

      with self.assertRaises(ValueError):
        num = array_ops.constant(1)
        text_summary.text_summary("foo", num)

      # The API accepts vectors.
      arr = array_ops.constant(["one", "two", "three"])
      summ = text_summary.text_summary("foo", arr)
      self.assertEqual(summ.op.type, "TensorSummaryV2")

      # the API accepts scalars
      summ = text_summary.text_summary("foo", array_ops.constant("one"))
      self.assertEqual(summ.op.type, "TensorSummaryV2")


if __name__ == "__main__":
  googletest.main()
