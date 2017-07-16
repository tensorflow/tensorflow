# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for learn.dataframe.transforms.example_parser."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from tensorflow.contrib.learn.python.learn.dataframe.transforms import example_parser
from tensorflow.contrib.learn.python.learn.tests.dataframe import mocks
from tensorflow.core.example import example_pb2


class ExampleParserTestCase(tf.test.TestCase):
  """Test class for `ExampleParser`."""

  def setUp(self):
    super(ExampleParserTestCase, self).setUp()
    self.example1 = example_pb2.Example()
    text_format.Parse("features: { "
                      "  feature: { "
                      "    key: 'int_feature' "
                      "    value: { "
                      "      int64_list: { "
                      "        value: [ 21, 2, 5 ] "
                      "      } "
                      "    } "
                      "  } "
                      "  feature: { "
                      "    key: 'string_feature' "
                      "    value: { "
                      "      bytes_list: { "
                      "        value: [ 'armadillo' ] "
                      "      } "
                      "    } "
                      "  } "
                      "} ", self.example1)
    self.example2 = example_pb2.Example()
    text_format.Parse("features: { "
                      "  feature: { "
                      "    key: 'int_feature' "
                      "    value: { "
                      "      int64_list: { "
                      "        value: [ 4, 5, 6 ] "
                      "      } "
                      "    } "
                      "  } "
                      "  feature: { "
                      "    key: 'string_feature' "
                      "    value: { "
                      "      bytes_list: { "
                      "        value: [ 'car', 'train' ] "
                      "      } "
                      "    } "
                      "  } "
                      "} ", self.example2)
    self.example_column = mocks.MockSeries(
        "example",
        tf.constant(
            [self.example1.SerializeToString(),
             self.example2.SerializeToString()],
            dtype=tf.string,
            shape=[2]))
    self.features = (("string_feature", tf.VarLenFeature(dtype=tf.string)),
                     ("int_feature",
                      tf.FixedLenFeature(shape=[3],
                                         dtype=tf.int64,
                                         default_value=[0, 0, 0])))

    self.expected_string_values = np.array(list(self.example1.features.feature[
        "string_feature"].bytes_list.value) + list(
            self.example2.features.feature["string_feature"].bytes_list.value))
    self.expected_string_indices = np.array([[0, 0], [1, 0], [1, 1]])
    self.expected_int_feature = np.array([list(self.example1.features.feature[
        "int_feature"].int64_list.value), list(self.example2.features.feature[
            "int_feature"].int64_list.value)])

  def testParseWithTupleDefinition(self):
    parser = example_parser.ExampleParser(self.features)
    output_columns = parser(self.example_column)
    self.assertEqual(2, len(output_columns))
    cache = {}
    output_tensors = [o.build(cache) for o in output_columns]
    self.assertEqual(2, len(output_tensors))

    with self.test_session() as sess:
      string_feature, int_feature = sess.run(output_tensors)
      np.testing.assert_array_equal(
          string_feature.dense_shape, np.array([2, 2]))
      np.testing.assert_array_equal(int_feature.shape, np.array([2, 3]))
      np.testing.assert_array_equal(self.expected_string_values,
                                    string_feature.values)
      np.testing.assert_array_equal(self.expected_string_indices,
                                    string_feature.indices)
      np.testing.assert_array_equal(self.expected_int_feature,
                                    int_feature)

  def testParseWithDictDefinition(self):
    parser = example_parser.ExampleParser(dict(self.features))
    output_columns = parser(self.example_column)
    self.assertEqual(2, len(output_columns))
    cache = {}
    output_tensors = [o.build(cache) for o in output_columns]
    self.assertEqual(2, len(output_tensors))

    with self.test_session() as sess:
      int_feature, string_feature = sess.run(output_tensors)
      np.testing.assert_array_equal(
          string_feature.dense_shape, np.array([2, 2]))
      np.testing.assert_array_equal(int_feature.shape, np.array([2, 3]))
      np.testing.assert_array_equal(self.expected_string_values,
                                    string_feature.values)
      np.testing.assert_array_equal(self.expected_string_indices,
                                    string_feature.indices)
      np.testing.assert_array_equal(self.expected_int_feature,
                                    int_feature)

if __name__ == "__main__":
  tf.test.main()
