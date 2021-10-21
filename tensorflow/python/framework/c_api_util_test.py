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

"""Tests for c_api utils."""
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class ApiDefMapTest(test_util.TensorFlowTestCase):

  def testApiDefMapOpNames(self):
    api_def_map = c_api_util.ApiDefMap()
    self.assertIn("Add", api_def_map.op_names())

  def testApiDefMapGet(self):
    api_def_map = c_api_util.ApiDefMap()
    op_def = api_def_map.get_op_def("Add")
    self.assertEqual(op_def.name, "Add")
    api_def = api_def_map.get_api_def("Add")
    self.assertEqual(api_def.graph_op_name, "Add")

  def testApiDefMapPutThenGet(self):
    api_def_map = c_api_util.ApiDefMap()
    api_def_text = """
op {
  graph_op_name: "Add"
  summary: "Returns x + y element-wise."
  description: <<END
*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
END
}
"""
    api_def_map.put_api_def(api_def_text)
    api_def = api_def_map.get_api_def("Add")
    self.assertEqual(api_def.graph_op_name, "Add")
    self.assertEqual(api_def.summary, "Returns x + y element-wise.")


if __name__ == "__main__":
  googletest.main()

