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
"""Tests for export."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.protobuf import text_format
from tensorflow.core.example import example_pb2
from tensorflow.python.estimator import export
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import googletest


class ExportTest(test_util.TensorFlowTestCase):

  def test_single_feature_single_receiver(self):
    feature = constant_op.constant(5)
    receiver_tensor = array_ops.placeholder(dtypes.string)
    input_receiver = export.ServingInputReceiver(
        feature, receiver_tensor)
    # single feature is automatically named
    feature_key, = input_receiver.features.keys()
    self.assertEqual('feature', feature_key)
    # single receiver is automatically named
    receiver_key, = input_receiver.receiver_tensors.keys()
    self.assertEqual('input', receiver_key)

  def test_multi_feature_single_receiver(self):
    features = {'foo': constant_op.constant(5),
                'bar': constant_op.constant(6)}
    receiver_tensor = array_ops.placeholder(dtypes.string)
    _ = export.ServingInputReceiver(features, receiver_tensor)

  def test_multi_feature_multi_receiver(self):
    features = {'foo': constant_op.constant(5),
                'bar': constant_op.constant(6)}
    receiver_tensors = {'baz': array_ops.placeholder(dtypes.int64),
                        'qux': array_ops.placeholder(dtypes.float32)}
    _ = export.ServingInputReceiver(features, receiver_tensors)

  def test_feature_wrong_type(self):
    feature = 'not a tensor'
    receiver_tensor = array_ops.placeholder(dtypes.string)
    with self.assertRaises(ValueError):
      _ = export.ServingInputReceiver(feature, receiver_tensor)

  def test_receiver_wrong_type(self):
    feature = constant_op.constant(5)
    receiver_tensor = 'not a tensor'
    with self.assertRaises(ValueError):
      _ = export.ServingInputReceiver(feature, receiver_tensor)

  def test_build_parsing_serving_input_receiver_fn(self):
    feature_spec = {'int_feature': parsing_ops.VarLenFeature(dtypes.int64),
                    'float_feature': parsing_ops.VarLenFeature(dtypes.float32)}
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    with ops.Graph().as_default():
      serving_input_receiver = serving_input_receiver_fn()
      self.assertEqual(set(['int_feature', 'float_feature']),
                       set(serving_input_receiver.features.keys()))
      self.assertEqual(set(['examples']),
                       set(serving_input_receiver.receiver_tensors.keys()))

      example = example_pb2.Example()
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
                        "    key: 'float_feature' "
                        "    value: { "
                        "      float_list: { "
                        "        value: [ 525.25 ] "
                        "      } "
                        "    } "
                        "  } "
                        "} ", example)

      with self.test_session() as sess:
        sparse_result = sess.run(
            serving_input_receiver.features,
            feed_dict={
                serving_input_receiver.receiver_tensors['examples'].name:
                [example.SerializeToString()]})
        self.assertAllEqual([[0, 0], [0, 1], [0, 2]],
                            sparse_result['int_feature'].indices)
        self.assertAllEqual([21, 2, 5],
                            sparse_result['int_feature'].values)
        self.assertAllEqual([[0, 0]],
                            sparse_result['float_feature'].indices)
        self.assertAllEqual([525.25],
                            sparse_result['float_feature'].values)

  def test_build_raw_serving_input_receiver_fn(self):
    features = {'feature_1': constant_op.constant(['hello']),
                'feature_2': constant_op.constant([42])}
    serving_input_receiver_fn = export.build_raw_serving_input_receiver_fn(
        features)
    with ops.Graph().as_default():
      serving_input_receiver = serving_input_receiver_fn()
      self.assertEqual(set(['feature_1', 'feature_2']),
                       set(serving_input_receiver.features.keys()))
      self.assertEqual(set(['feature_1', 'feature_2']),
                       set(serving_input_receiver.receiver_tensors.keys()))
      self.assertEqual(
          dtypes.string,
          serving_input_receiver.receiver_tensors['feature_1'].dtype)
      self.assertEqual(
          dtypes.int32,
          serving_input_receiver.receiver_tensors['feature_2'].dtype)


if __name__ == '__main__':
  googletest.main()
