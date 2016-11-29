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

"""Tests of utilities supporting export to SavedModel."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import time

import tensorflow as tf

from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils


class SavedModelExportUtilsTest(tf.test.TestCase):

  def test_build_standardized_signature_def(self):
    input_tensors = {
        "input-1": tf.placeholder(tf.float32, 1, name="input-tensor-1")}
    output_tensors = {
        "output-1": tf.placeholder(tf.float32, 1, name="output-tensor-1")}
    problem_type = constants.ProblemType.LINEAR_REGRESSION
    regression_signature_def = (
        saved_model_export_utils.build_standardized_signature_def(
            input_tensors, output_tensors, problem_type))
    expected_regression_signature_def = meta_graph_pb2.SignatureDef()
    shape = tensor_shape_pb2.TensorShapeProto(
        dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])
    dtype = types_pb2.DataType.Value("DT_FLOAT")
    expected_regression_signature_def.inputs[
        signature_constants.REGRESS_INPUTS].CopyFrom(
            meta_graph_pb2.TensorInfo(name="input-tensor-1:0",
                                      dtype=dtype,
                                      tensor_shape=shape))
    expected_regression_signature_def.outputs[
        signature_constants.REGRESS_OUTPUTS].CopyFrom(
            meta_graph_pb2.TensorInfo(name="output-tensor-1:0",
                                      dtype=dtype,
                                      tensor_shape=shape))

    expected_regression_signature_def.method_name = (
        signature_constants.REGRESS_METHOD_NAME)
    self.assertEqual(regression_signature_def,
                     expected_regression_signature_def)

  def test_get_input_alternatives(self):
    input_ops = input_fn_utils.InputFnOps("bogus features dict", None,
                                          "bogus default input dict")

    input_alternatives, _ = saved_model_export_utils.get_input_alternatives(
        input_ops)
    self.assertEqual(
        input_alternatives[
            saved_model_export_utils.DEFAULT_INPUT_ALTERNATIVE_KEY],
        "bogus default input dict")
    self.assertEqual(
        input_alternatives[
            saved_model_export_utils.FEATURES_INPUT_ALTERNATIVE_KEY],
        "bogus features dict")

  def test_get_output_alternatives_explicit(self):
    provided_output_alternatives = {
        "head-1": (constants.ProblemType.LINEAR_REGRESSION,
                   "bogus output dict"),
        "head-2": (constants.ProblemType.CLASSIFICATION,
                   "bogus output dict 2"),
        "head-3": (constants.ProblemType.UNSPECIFIED,
                   "bogus output dict 3"),
    }
    model_fn_ops = model_fn.ModelFnOps(
        model_fn.ModeKeys.INFER,
        predictions={"some_output": "bogus_tensor"},
        output_alternatives=provided_output_alternatives)
    output_alternatives, _ = saved_model_export_utils.get_output_alternatives(
        model_fn_ops, "head-1")

    self.assertEqual(provided_output_alternatives, output_alternatives)

  def test_get_output_alternatives_implicit(self):
    prediction_tensor = tf.constant(["bogus"])
    model_fn_ops = model_fn.ModelFnOps(
        model_fn.ModeKeys.INFER,
        predictions={"some_output": prediction_tensor},
        output_alternatives=None)

    output_alternatives, _ = saved_model_export_utils.get_output_alternatives(
        model_fn_ops, "some_output")
    self.assertEqual(
        {"default_output_alternative": (constants.ProblemType.UNSPECIFIED,
                                        {"some_output": prediction_tensor})},
        output_alternatives)

  def test_build_all_signature_defs(self):
    input_features = tf.constant(["10"])
    input_example = tf.constant(["11"])
    input_ops = input_fn_utils.InputFnOps(
        {"features": input_features},
        None,
        {"default input": input_example})
    input_alternatives, _ = (
        saved_model_export_utils.get_input_alternatives(input_ops))
    output_1 = tf.constant(["1"])
    output_2 = tf.constant(["2"])
    output_3 = tf.constant(["3"])
    provided_output_alternatives = {
        "head-1": (constants.ProblemType.LINEAR_REGRESSION,
                   {"some_output_1": output_1}),
        "head-2": (constants.ProblemType.CLASSIFICATION,
                   {"some_output_2": output_2}),
        "head-3": (constants.ProblemType.UNSPECIFIED,
                   {"some_output_3": output_3}),
    }
    model_fn_ops = model_fn.ModelFnOps(
        model_fn.ModeKeys.INFER,
        predictions={"some_output": tf.constant(["4"])},
        output_alternatives=provided_output_alternatives)
    output_alternatives, _ = (
        saved_model_export_utils.get_output_alternatives(model_fn_ops,
                                                         "head-1"))

    signature_defs = saved_model_export_utils.build_all_signature_defs(
        input_alternatives, output_alternatives, "head-1")

    expected_signature_defs = {
        "serving_default":
            signature_def_utils.regression_signature_def(
                input_example, output_1),
        "default_input_alternative:head-1":
            signature_def_utils.regression_signature_def(
                input_example, output_1),
        "default_input_alternative:head-2":
            signature_def_utils.classification_signature_def(
                input_example, output_2, None),
        "default_input_alternative:head-3":
            signature_def_utils.predict_signature_def(
                {"input": input_example}, {"output": output_3}),
        "features_input_alternative:head-1":
            signature_def_utils.regression_signature_def(
                input_features, output_1),
        "features_input_alternative:head-2":
            signature_def_utils.classification_signature_def(
                input_features, output_2, None),
        "features_input_alternative:head-3":
            signature_def_utils.predict_signature_def(
                {"input": input_features}, {"output": output_3}),
    }

    self.assertDictEqual(expected_signature_defs, signature_defs)

  def test_get_timestamped_export_dir(self):
    export_dir_base = tempfile.mkdtemp() + "export/"
    export_dir_1 = saved_model_export_utils.get_timestamped_export_dir(
        export_dir_base)
    time.sleep(0.001)
    export_dir_2 = saved_model_export_utils.get_timestamped_export_dir(
        export_dir_base)
    time.sleep(0.001)
    export_dir_3 = saved_model_export_utils.get_timestamped_export_dir(
        export_dir_base)

    # Export directories should be named using a timestamp that is milliseconds
    # since epoch.  Such a timestamp is 13 digits long.
    time_1 = os.path.basename(export_dir_1)
    self.assertEqual(13, len(time_1))
    time_2 = os.path.basename(export_dir_2)
    self.assertEqual(13, len(time_2))
    time_3 = os.path.basename(export_dir_3)
    self.assertEqual(13, len(time_3))

    self.assertTrue(int(time_1) < int(time_2))
    self.assertTrue(int(time_2) < int(time_3))

  def test_garbage_collect_exports(self):
    export_dir_base = tempfile.mkdtemp() + "export/"
    tf.gfile.MkDir(export_dir_base)
    export_dir_1 = _create_test_export_dir(export_dir_base)
    export_dir_2 = _create_test_export_dir(export_dir_base)
    export_dir_3 = _create_test_export_dir(export_dir_base)
    export_dir_4 = _create_test_export_dir(export_dir_base)

    self.assertTrue(tf.gfile.Exists(export_dir_1))
    self.assertTrue(tf.gfile.Exists(export_dir_2))
    self.assertTrue(tf.gfile.Exists(export_dir_3))
    self.assertTrue(tf.gfile.Exists(export_dir_4))

    # Garbage collect all but the most recent 2 exports,
    # where recency is determined based on the timestamp directory names.
    saved_model_export_utils.garbage_collect_exports(export_dir_base, 2)

    self.assertFalse(tf.gfile.Exists(export_dir_1))
    self.assertFalse(tf.gfile.Exists(export_dir_2))
    self.assertTrue(tf.gfile.Exists(export_dir_3))
    self.assertTrue(tf.gfile.Exists(export_dir_4))


def _create_test_export_dir(export_dir_base):
  export_dir = saved_model_export_utils.get_timestamped_export_dir(
      export_dir_base)
  tf.gfile.MkDir(export_dir)
  time.sleep(0.001)
  return export_dir


if __name__ == "__main__":
  tf.test.main()
