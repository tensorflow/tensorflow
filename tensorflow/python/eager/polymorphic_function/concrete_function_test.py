# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

from absl.testing import parameterized

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager.polymorphic_function import concrete_function as cf
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class ConcreteFunctionTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.func_graph = func_graph_module.FuncGraph("f")

  @parameterized.parameters(
      ({"api_implements": True}, attr_value_pb2.AttrValue(b=True)),
      ({"api_implements": 1}, attr_value_pb2.AttrValue(i=1)),
      ({"api_implements": 1.0}, attr_value_pb2.AttrValue(f=1.0)),
      (
          {"api_implements": "test"},
          attr_value_pb2.AttrValue(s=compat.as_bytes("test")),
      ),
  )
  def test_parses_func_attr_scalar_values(self, attrs, expected):
    self.assertEqual(
        cf.ConcreteFunction(self.func_graph, attrs=attrs).function_def.attr[
            "api_implements"
        ],
        expected,
    )

  def test_parses_func_attr_list_values(self):
    self.assertProtoEquals(
        r"""
        list {
            s: 'test'
            b: True
            i: 1
            f: 1.0
        }
        """,
        cf.ConcreteFunction(
            self.func_graph, attrs={"api_implements": ["test", True, 1, 1.0]}
        ).function_def.attr["api_implements"],
    )

  def test_raises_value_error_for_invalid_attr(self):
    with self.assertRaisesRegex(ValueError, "Attribute api_implements must be"):
      cf.ConcreteFunction(self.func_graph, attrs={"api_implements": None})


if __name__ == "__main__":
  test.main()
