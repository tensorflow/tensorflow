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

import os
import tempfile
import time

from google.protobuf import text_format

from tensorflow.core.example import example_pb2
from tensorflow.python.estimator.export import export
from tensorflow.python.estimator.export import export_output
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils


class ExportTest(test_util.TensorFlowTestCase):

  def test_serving_input_receiver_constructor(self):
    """Tests that no errors are raised when input is expected."""
    features = {
        "feature0": constant_op.constant([0]),
        u"feature1": constant_op.constant([1]),
        "feature2": sparse_tensor.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1]),
    }
    receiver_tensors = {
        "example0": array_ops.placeholder(dtypes.string, name="example0"),
        u"example1": array_ops.placeholder(dtypes.string, name="example1"),
    }
    export.ServingInputReceiver(features, receiver_tensors)

  def test_serving_input_receiver_features_invalid(self):
    receiver_tensors = {
        "example0": array_ops.placeholder(dtypes.string, name="example0"),
        u"example1": array_ops.placeholder(dtypes.string, name="example1"),
    }

    with self.assertRaisesRegexp(ValueError, "features must be defined"):
      export.ServingInputReceiver(
          features=None,
          receiver_tensors=receiver_tensors)

    with self.assertRaisesRegexp(ValueError, "feature keys must be strings"):
      export.ServingInputReceiver(
          features={1: constant_op.constant([1])},
          receiver_tensors=receiver_tensors)

    with self.assertRaisesRegexp(
        ValueError, "feature feature1 must be a Tensor or SparseTensor"):
      export.ServingInputReceiver(
          features={"feature1": [1]},
          receiver_tensors=receiver_tensors)

  def test_serving_input_receiver_receiver_tensors_invalid(self):
    features = {
        "feature0": constant_op.constant([0]),
        u"feature1": constant_op.constant([1]),
        "feature2": sparse_tensor.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1]),
    }

    with self.assertRaisesRegexp(
        ValueError, "receiver_tensors must be defined"):
      export.ServingInputReceiver(
          features=features,
          receiver_tensors=None)

    with self.assertRaisesRegexp(
        ValueError, "receiver_tensors keys must be strings"):
      export.ServingInputReceiver(
          features=features,
          receiver_tensors={
              1: array_ops.placeholder(dtypes.string, name="example0")})

    with self.assertRaisesRegexp(
        ValueError, "receiver_tensor example1 must be a Tensor"):
      export.ServingInputReceiver(
          features=features,
          receiver_tensors={"example1": [1]})

  def test_single_feature_single_receiver(self):
    feature = constant_op.constant(5)
    receiver_tensor = array_ops.placeholder(dtypes.string)
    input_receiver = export.ServingInputReceiver(
        feature, receiver_tensor)
    # single feature is automatically named
    feature_key, = input_receiver.features.keys()
    self.assertEqual("feature", feature_key)
    # single receiver is automatically named
    receiver_key, = input_receiver.receiver_tensors.keys()
    self.assertEqual("input", receiver_key)

  def test_multi_feature_single_receiver(self):
    features = {"foo": constant_op.constant(5),
                "bar": constant_op.constant(6)}
    receiver_tensor = array_ops.placeholder(dtypes.string)
    _ = export.ServingInputReceiver(features, receiver_tensor)

  def test_multi_feature_multi_receiver(self):
    features = {"foo": constant_op.constant(5),
                "bar": constant_op.constant(6)}
    receiver_tensors = {"baz": array_ops.placeholder(dtypes.int64),
                        "qux": array_ops.placeholder(dtypes.float32)}
    _ = export.ServingInputReceiver(features, receiver_tensors)

  def test_feature_wrong_type(self):
    feature = "not a tensor"
    receiver_tensor = array_ops.placeholder(dtypes.string)
    with self.assertRaises(ValueError):
      _ = export.ServingInputReceiver(feature, receiver_tensor)

  def test_receiver_wrong_type(self):
    feature = constant_op.constant(5)
    receiver_tensor = "not a tensor"
    with self.assertRaises(ValueError):
      _ = export.ServingInputReceiver(feature, receiver_tensor)

  def test_build_parsing_serving_input_receiver_fn(self):
    feature_spec = {"int_feature": parsing_ops.VarLenFeature(dtypes.int64),
                    "float_feature": parsing_ops.VarLenFeature(dtypes.float32)}
    serving_input_receiver_fn = export.build_parsing_serving_input_receiver_fn(
        feature_spec)
    with ops.Graph().as_default():
      serving_input_receiver = serving_input_receiver_fn()
      self.assertEqual(set(["int_feature", "float_feature"]),
                       set(serving_input_receiver.features.keys()))
      self.assertEqual(set(["examples"]),
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
                serving_input_receiver.receiver_tensors["examples"].name:
                [example.SerializeToString()]})
        self.assertAllEqual([[0, 0], [0, 1], [0, 2]],
                            sparse_result["int_feature"].indices)
        self.assertAllEqual([21, 2, 5],
                            sparse_result["int_feature"].values)
        self.assertAllEqual([[0, 0]],
                            sparse_result["float_feature"].indices)
        self.assertAllEqual([525.25],
                            sparse_result["float_feature"].values)

  def test_build_raw_serving_input_receiver_fn(self):
    features = {"feature_1": constant_op.constant(["hello"]),
                "feature_2": constant_op.constant([42])}
    serving_input_receiver_fn = export.build_raw_serving_input_receiver_fn(
        features)
    with ops.Graph().as_default():
      serving_input_receiver = serving_input_receiver_fn()
      self.assertEqual(set(["feature_1", "feature_2"]),
                       set(serving_input_receiver.features.keys()))
      self.assertEqual(set(["feature_1", "feature_2"]),
                       set(serving_input_receiver.receiver_tensors.keys()))
      self.assertEqual(
          dtypes.string,
          serving_input_receiver.receiver_tensors["feature_1"].dtype)
      self.assertEqual(
          dtypes.int32,
          serving_input_receiver.receiver_tensors["feature_2"].dtype)

  def test_build_all_signature_defs_explicit_default(self):
    receiver_tensor = constant_op.constant(["11"])
    output_1 = constant_op.constant([1.])
    output_2 = constant_op.constant(["2"])
    output_3 = constant_op.constant(["3"])
    export_outputs = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            export_output.RegressionOutput(value=output_1),
        "head-2": export_output.ClassificationOutput(classes=output_2),
        "head-3": export_output.PredictOutput(outputs={
            "some_output_3": output_3
        }),
    }

    signature_defs = export.build_all_signature_defs(
        receiver_tensor, export_outputs)

    expected_signature_defs = {
        "serving_default":
            signature_def_utils.regression_signature_def(receiver_tensor,
                                                         output_1),
        "head-2":
            signature_def_utils.classification_signature_def(receiver_tensor,
                                                             output_2, None),
        "head-3":
            signature_def_utils.predict_signature_def({
                "receiver": receiver_tensor
            }, {"some_output_3": output_3})
    }

    self.assertDictEqual(expected_signature_defs, signature_defs)

  def test_build_all_signature_defs_export_outputs_required(self):
    receiver_tensor = constant_op.constant(["11"])

    with self.assertRaises(ValueError) as e:
      export.build_all_signature_defs(receiver_tensor, None)

    self.assertEqual("export_outputs must be a dict.", str(e.exception))

  def test_get_timestamped_export_dir(self):
    export_dir_base = tempfile.mkdtemp() + "export/"
    export_dir_1 = export.get_timestamped_export_dir(
        export_dir_base)
    time.sleep(2)
    export_dir_2 = export.get_timestamped_export_dir(
        export_dir_base)
    time.sleep(2)
    export_dir_3 = export.get_timestamped_export_dir(
        export_dir_base)

    # Export directories should be named using a timestamp that is seconds
    # since epoch.  Such a timestamp is 10 digits long.
    time_1 = os.path.basename(export_dir_1)
    self.assertEqual(10, len(time_1))
    time_2 = os.path.basename(export_dir_2)
    self.assertEqual(10, len(time_2))
    time_3 = os.path.basename(export_dir_3)
    self.assertEqual(10, len(time_3))

    self.assertTrue(int(time_1) < int(time_2))
    self.assertTrue(int(time_2) < int(time_3))


if __name__ == "__main__":
  test.main()
