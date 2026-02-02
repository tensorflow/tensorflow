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

import tensorflow as tf

from google.protobuf import text_format
from tensorflow.core.example import example_parser_configuration_pb2
from tensorflow.python.util.example_parser_configuration import extract_example_parser_configuration

BASIC_PROTO = """
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


class ExampleParserConfigurationTest(tf.test.TestCase):

  def testBasic(self):
    golden_config = example_parser_configuration_pb2.ExampleParserConfiguration(
    )
    text_format.Parse(BASIC_PROTO, golden_config)
    with tf.Session() as sess:
      examples = tf.placeholder(tf.string, shape=[1])
      feature_to_type = {
          'x': tf.FixedLenFeature([1], tf.float32, 33.0),
          'y': tf.VarLenFeature(tf.string)
      }
      _ = tf.parse_example(examples, feature_to_type)
      parse_example_op = sess.graph.get_operation_by_name(
          'ParseExample/ParseExample')
      config = extract_example_parser_configuration(parse_example_op, sess)
      self.assertProtoEquals(golden_config, config)


if __name__ == '__main__':
  tf.test.main()
