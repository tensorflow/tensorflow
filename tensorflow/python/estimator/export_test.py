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

from tensorflow.python.estimator import export
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
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


if __name__ == '__main__':
  googletest.main()
