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
"""Tests for ExampleParserConfiguration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.protobuf import text_format

from tensorflow.core.example import example_parser_configuration_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test
from tensorflow.python.util.example_parser_configuration import extract_example_parser_configuration

EXPECTED_CONFIG_V1 = """
feature_map {
  key: "x"
  value {
    fixed_len_feature {
      dtype: DT_FLOAT
      shape {
        dim {
          size: 1
        }
      }
      default_value {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
        }
        float_val: 33.0
      }
      values_output_tensor_name: "ParseExample/ParseExample:3"
    }
  }
}
feature_map {
  key: "y"
  value {
    var_len_feature {
      dtype: DT_STRING
      values_output_tensor_name: "ParseExample/ParseExample:1"
      indices_output_tensor_name: "ParseExample/ParseExample:0"
      shapes_output_tensor_name: "ParseExample/ParseExample:2"
    }
  }
}
"""


EXPECTED_CONFIG_V2 = EXPECTED_CONFIG_V1.replace(
    'ParseExample/ParseExample:', 'ParseExample/ParseExampleV2:')


class ExampleParserConfigurationTest(test.TestCase):

  def getExpectedConfig(self, op_type):
    expected = example_parser_configuration_pb2.ExampleParserConfiguration()
    if op_type == 'ParseExampleV2':
      text_format.Parse(EXPECTED_CONFIG_V2, expected)
    else:
      text_format.Parse(EXPECTED_CONFIG_V1, expected)
    return expected

  def testBasic(self):
    with session.Session() as sess:
      examples = array_ops.placeholder(dtypes.string, shape=[1])
      feature_to_type = {
          'x': parsing_ops.FixedLenFeature([1], dtypes.float32, 33.0),
          'y': parsing_ops.VarLenFeature(dtypes.string)
      }
      result = parsing_ops.parse_example(examples, feature_to_type)
      parse_example_op = result['x'].op
      config = extract_example_parser_configuration(parse_example_op, sess)
      expected = self.getExpectedConfig(parse_example_op.type)
      self.assertProtoEquals(expected, config)


if __name__ == '__main__':
  test.main()
