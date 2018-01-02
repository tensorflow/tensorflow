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

from tensorflow.contrib.layers.python.layers import feature_column as fc
from tensorflow.contrib.learn.python.learn import export_strategy as export_strategy_lib
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators import estimator as core_estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.util import compat


class TestEstimator(core_estimator.Estimator):

  def __init__(self, *args, **kwargs):
    super(TestEstimator, self).__init__(*args, **kwargs)
    self.last_exported_checkpoint = ""
    self.last_exported_dir = ""

  # @Override
  def export_savedmodel(self,
                        export_dir,
                        serving_input_fn,
                        default_output_alternative_key=None,
                        assets_extra=None,
                        as_text=False,
                        checkpoint_path=None):

    if not os.path.exists(export_dir):
      os.makedirs(export_dir)

    open(os.path.join(export_dir, "placeholder.txt"), "a").close()

    self.last_exported_checkpoint = checkpoint_path
    self.last_exported_dir = export_dir

    return export_dir


class SavedModelExportUtilsTest(test.TestCase):

  def test_build_standardized_signature_def_regression(self):
    input_tensors = {
        "input-1":
            array_ops.placeholder(dtypes.string, 1, name="input-tensor-1")
    }
    output_tensors = {
        "output-1":
            array_ops.placeholder(dtypes.float32, 1, name="output-tensor-1")
    }
    problem_type = constants.ProblemType.LINEAR_REGRESSION
    actual_signature_def = (
        saved_model_export_utils.build_standardized_signature_def(
            input_tensors, output_tensors, problem_type))
    expected_signature_def = meta_graph_pb2.SignatureDef()
    shape = tensor_shape_pb2.TensorShapeProto(
        dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])
    dtype_float = types_pb2.DataType.Value("DT_FLOAT")
    dtype_string = types_pb2.DataType.Value("DT_STRING")
    expected_signature_def.inputs[signature_constants.REGRESS_INPUTS].CopyFrom(
        meta_graph_pb2.TensorInfo(
            name="input-tensor-1:0", dtype=dtype_string, tensor_shape=shape))
    expected_signature_def.outputs[
        signature_constants.REGRESS_OUTPUTS].CopyFrom(
            meta_graph_pb2.TensorInfo(name="output-tensor-1:0",
                                      dtype=dtype_float,
                                      tensor_shape=shape))

    expected_signature_def.method_name = signature_constants.REGRESS_METHOD_NAME
    self.assertEqual(actual_signature_def, expected_signature_def)

  def test_build_standardized_signature_def_classification(self):
    """Tests classification with one output tensor."""
    input_tensors = {
        "input-1":
            array_ops.placeholder(dtypes.string, 1, name="input-tensor-1")
    }
    output_tensors = {
        "output-1":
            array_ops.placeholder(dtypes.string, 1, name="output-tensor-1")
    }
    problem_type = constants.ProblemType.CLASSIFICATION
    actual_signature_def = (
        saved_model_export_utils.build_standardized_signature_def(
            input_tensors, output_tensors, problem_type))
    expected_signature_def = meta_graph_pb2.SignatureDef()
    shape = tensor_shape_pb2.TensorShapeProto(
        dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])
    dtype_string = types_pb2.DataType.Value("DT_STRING")
    expected_signature_def.inputs[signature_constants.CLASSIFY_INPUTS].CopyFrom(
        meta_graph_pb2.TensorInfo(
            name="input-tensor-1:0", dtype=dtype_string, tensor_shape=shape))
    expected_signature_def.outputs[
        signature_constants.CLASSIFY_OUTPUT_CLASSES].CopyFrom(
            meta_graph_pb2.TensorInfo(
                name="output-tensor-1:0",
                dtype=dtype_string,
                tensor_shape=shape))

    expected_signature_def.method_name = (
        signature_constants.CLASSIFY_METHOD_NAME)
    self.assertEqual(actual_signature_def, expected_signature_def)

  def test_build_standardized_signature_def_classification2(self):
    """Tests multiple output tensors that include classes and probabilities."""
    input_tensors = {
        "input-1":
            array_ops.placeholder(dtypes.string, 1, name="input-tensor-1")
    }
    output_tensors = {
        "classes":
            array_ops.placeholder(
                dtypes.string, 1, name="output-tensor-classes"),
        # Will be used for CLASSIFY_OUTPUT_SCORES.
        "probabilities":
            array_ops.placeholder(
                dtypes.float32, 1, name="output-tensor-proba"),
        "logits":
            array_ops.placeholder(
                dtypes.float32, 1, name="output-tensor-logits-unused"),
    }
    problem_type = constants.ProblemType.CLASSIFICATION
    actual_signature_def = (
        saved_model_export_utils.build_standardized_signature_def(
            input_tensors, output_tensors, problem_type))
    expected_signature_def = meta_graph_pb2.SignatureDef()
    shape = tensor_shape_pb2.TensorShapeProto(
        dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])
    dtype_float = types_pb2.DataType.Value("DT_FLOAT")
    dtype_string = types_pb2.DataType.Value("DT_STRING")
    expected_signature_def.inputs[signature_constants.CLASSIFY_INPUTS].CopyFrom(
        meta_graph_pb2.TensorInfo(
            name="input-tensor-1:0", dtype=dtype_string, tensor_shape=shape))
    expected_signature_def.outputs[
        signature_constants.CLASSIFY_OUTPUT_CLASSES].CopyFrom(
            meta_graph_pb2.TensorInfo(
                name="output-tensor-classes:0",
                dtype=dtype_string,
                tensor_shape=shape))
    expected_signature_def.outputs[
        signature_constants.CLASSIFY_OUTPUT_SCORES].CopyFrom(
            meta_graph_pb2.TensorInfo(
                name="output-tensor-proba:0",
                dtype=dtype_float,
                tensor_shape=shape))

    expected_signature_def.method_name = (
        signature_constants.CLASSIFY_METHOD_NAME)
    self.assertEqual(actual_signature_def, expected_signature_def)

  def test_build_standardized_signature_def_classification3(self):
    """Tests multiple output tensors that include classes and scores."""
    input_tensors = {
        "input-1":
            array_ops.placeholder(dtypes.string, 1, name="input-tensor-1")
    }
    output_tensors = {
        "classes":
            array_ops.placeholder(
                dtypes.string, 1, name="output-tensor-classes"),
        "scores":
            array_ops.placeholder(
                dtypes.float32, 1, name="output-tensor-scores"),
        "logits":
            array_ops.placeholder(
                dtypes.float32, 1, name="output-tensor-logits-unused"),
    }
    problem_type = constants.ProblemType.CLASSIFICATION
    actual_signature_def = (
        saved_model_export_utils.build_standardized_signature_def(
            input_tensors, output_tensors, problem_type))
    expected_signature_def = meta_graph_pb2.SignatureDef()
    shape = tensor_shape_pb2.TensorShapeProto(
        dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])
    dtype_float = types_pb2.DataType.Value("DT_FLOAT")
    dtype_string = types_pb2.DataType.Value("DT_STRING")
    expected_signature_def.inputs[signature_constants.CLASSIFY_INPUTS].CopyFrom(
        meta_graph_pb2.TensorInfo(
            name="input-tensor-1:0", dtype=dtype_string, tensor_shape=shape))
    expected_signature_def.outputs[
        signature_constants.CLASSIFY_OUTPUT_CLASSES].CopyFrom(
            meta_graph_pb2.TensorInfo(
                name="output-tensor-classes:0",
                dtype=dtype_string,
                tensor_shape=shape))
    expected_signature_def.outputs[
        signature_constants.CLASSIFY_OUTPUT_SCORES].CopyFrom(
            meta_graph_pb2.TensorInfo(
                name="output-tensor-scores:0",
                dtype=dtype_float,
                tensor_shape=shape))

    expected_signature_def.method_name = (
        signature_constants.CLASSIFY_METHOD_NAME)
    self.assertEqual(actual_signature_def, expected_signature_def)

  def test_build_standardized_signature_def_classification4(self):
    """Tests classification without classes tensor."""
    input_tensors = {
        "input-1":
            array_ops.placeholder(dtypes.string, 1, name="input-tensor-1")
    }
    output_tensors = {
        "probabilities":
            array_ops.placeholder(
                dtypes.float32, 1, name="output-tensor-proba"),
        "logits":
            array_ops.placeholder(
                dtypes.float32, 1, name="output-tensor-logits-unused"),
    }
    problem_type = constants.ProblemType.CLASSIFICATION
    actual_signature_def = (
        saved_model_export_utils.build_standardized_signature_def(
            input_tensors, output_tensors, problem_type))
    expected_signature_def = meta_graph_pb2.SignatureDef()
    shape = tensor_shape_pb2.TensorShapeProto(
        dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])
    dtype_float = types_pb2.DataType.Value("DT_FLOAT")
    dtype_string = types_pb2.DataType.Value("DT_STRING")
    expected_signature_def.inputs[signature_constants.CLASSIFY_INPUTS].CopyFrom(
        meta_graph_pb2.TensorInfo(
            name="input-tensor-1:0", dtype=dtype_string, tensor_shape=shape))
    expected_signature_def.outputs[
        signature_constants.CLASSIFY_OUTPUT_SCORES].CopyFrom(
            meta_graph_pb2.TensorInfo(
                name="output-tensor-proba:0",
                dtype=dtype_float,
                tensor_shape=shape))

    expected_signature_def.method_name = (
        signature_constants.CLASSIFY_METHOD_NAME)
    self.assertEqual(actual_signature_def, expected_signature_def)

  def test_build_standardized_signature_def_classification5(self):
    """Tests multiple output tensors that include integer classes and scores.

    Integer classes are dropped out, because Servo classification can only serve
    string classes. So, only scores are present in the signature.
    """
    input_tensors = {
        "input-1":
            array_ops.placeholder(dtypes.string, 1, name="input-tensor-1")
    }
    output_tensors = {
        "classes":
            array_ops.placeholder(
                dtypes.int64, 1, name="output-tensor-classes"),
        "scores":
            array_ops.placeholder(
                dtypes.float32, 1, name="output-tensor-scores"),
        "logits":
            array_ops.placeholder(
                dtypes.float32, 1, name="output-tensor-logits-unused"),
    }
    problem_type = constants.ProblemType.CLASSIFICATION
    actual_signature_def = (
        saved_model_export_utils.build_standardized_signature_def(
            input_tensors, output_tensors, problem_type))
    expected_signature_def = meta_graph_pb2.SignatureDef()
    shape = tensor_shape_pb2.TensorShapeProto(
        dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])
    dtype_float = types_pb2.DataType.Value("DT_FLOAT")
    dtype_string = types_pb2.DataType.Value("DT_STRING")
    expected_signature_def.inputs[signature_constants.CLASSIFY_INPUTS].CopyFrom(
        meta_graph_pb2.TensorInfo(
            name="input-tensor-1:0", dtype=dtype_string, tensor_shape=shape))
    expected_signature_def.outputs[
        signature_constants.CLASSIFY_OUTPUT_SCORES].CopyFrom(
            meta_graph_pb2.TensorInfo(
                name="output-tensor-scores:0",
                dtype=dtype_float,
                tensor_shape=shape))

    expected_signature_def.method_name = (
        signature_constants.CLASSIFY_METHOD_NAME)
    self.assertEqual(actual_signature_def, expected_signature_def)

  def test_build_standardized_signature_def_classification6(self):
    """Tests multiple output tensors that with integer classes and no scores.

    Servo classification cannot serve integer classes, but no scores are
    available. So, we fall back to predict signature.
    """
    input_tensors = {
        "input-1":
            array_ops.placeholder(dtypes.string, 1, name="input-tensor-1")
    }
    output_tensors = {
        "classes":
            array_ops.placeholder(
                dtypes.int64, 1, name="output-tensor-classes"),
        "logits":
            array_ops.placeholder(
                dtypes.float32, 1, name="output-tensor-logits"),
    }
    problem_type = constants.ProblemType.CLASSIFICATION
    actual_signature_def = (
        saved_model_export_utils.build_standardized_signature_def(
            input_tensors, output_tensors, problem_type))
    expected_signature_def = meta_graph_pb2.SignatureDef()
    shape = tensor_shape_pb2.TensorShapeProto(
        dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=1)])
    dtype_int64 = types_pb2.DataType.Value("DT_INT64")
    dtype_float = types_pb2.DataType.Value("DT_FLOAT")
    dtype_string = types_pb2.DataType.Value("DT_STRING")
    expected_signature_def.inputs["input-1"].CopyFrom(
        meta_graph_pb2.TensorInfo(
            name="input-tensor-1:0", dtype=dtype_string, tensor_shape=shape))
    expected_signature_def.outputs["classes"].CopyFrom(
        meta_graph_pb2.TensorInfo(
            name="output-tensor-classes:0",
            dtype=dtype_int64,
            tensor_shape=shape))
    expected_signature_def.outputs["logits"].CopyFrom(
        meta_graph_pb2.TensorInfo(
            name="output-tensor-logits:0",
            dtype=dtype_float,
            tensor_shape=shape))

    expected_signature_def.method_name = (
        signature_constants.PREDICT_METHOD_NAME)
    self.assertEqual(actual_signature_def, expected_signature_def)

  def test_get_input_alternatives(self):
    input_ops = input_fn_utils.InputFnOps("bogus features dict", None,
                                          "bogus default input dict")

    input_alternatives, _ = saved_model_export_utils.get_input_alternatives(
        input_ops)
    self.assertEqual(input_alternatives[
        saved_model_export_utils.DEFAULT_INPUT_ALTERNATIVE_KEY],
                     "bogus default input dict")
    # self.assertEqual(input_alternatives[
    #     saved_model_export_utils.FEATURES_INPUT_ALTERNATIVE_KEY],
    #                  "bogus features dict")

  def test_get_output_alternatives_explicit_default(self):
    provided_output_alternatives = {
        "head-1": (constants.ProblemType.LINEAR_REGRESSION,
                   "bogus output dict"),
        "head-2": (constants.ProblemType.CLASSIFICATION, "bogus output dict 2"),
        "head-3": (constants.ProblemType.UNSPECIFIED, "bogus output dict 3"),
    }
    model_fn_ops = model_fn.ModelFnOps(
        model_fn.ModeKeys.INFER,
        predictions={"some_output": "bogus_tensor"},
        output_alternatives=provided_output_alternatives)

    output_alternatives, _ = saved_model_export_utils.get_output_alternatives(
        model_fn_ops, "head-1")

    self.assertEqual(provided_output_alternatives, output_alternatives)

  def test_get_output_alternatives_wrong_default(self):
    provided_output_alternatives = {
        "head-1": (constants.ProblemType.LINEAR_REGRESSION,
                   "bogus output dict"),
        "head-2": (constants.ProblemType.CLASSIFICATION, "bogus output dict 2"),
        "head-3": (constants.ProblemType.UNSPECIFIED, "bogus output dict 3"),
    }
    model_fn_ops = model_fn.ModelFnOps(
        model_fn.ModeKeys.INFER,
        predictions={"some_output": "bogus_tensor"},
        output_alternatives=provided_output_alternatives)

    with self.assertRaises(ValueError) as e:
      saved_model_export_utils.get_output_alternatives(model_fn_ops, "WRONG")

    self.assertEqual("Requested default_output_alternative: WRONG, but "
                     "available output_alternatives are: ['head-1', 'head-2', "
                     "'head-3']", str(e.exception))

  def test_get_output_alternatives_single_no_default(self):
    prediction_tensor = constant_op.constant(["bogus"])
    provided_output_alternatives = {
        "head-1": (constants.ProblemType.LINEAR_REGRESSION, {
            "output": prediction_tensor
        }),
    }
    model_fn_ops = model_fn.ModelFnOps(
        model_fn.ModeKeys.INFER,
        predictions=prediction_tensor,
        output_alternatives=provided_output_alternatives)

    output_alternatives, _ = saved_model_export_utils.get_output_alternatives(
        model_fn_ops)

    self.assertEqual({
        "head-1": (constants.ProblemType.LINEAR_REGRESSION, {
            "output": prediction_tensor
        })
    }, output_alternatives)

  def test_get_output_alternatives_multi_no_default(self):
    provided_output_alternatives = {
        "head-1": (constants.ProblemType.LINEAR_REGRESSION,
                   "bogus output dict"),
        "head-2": (constants.ProblemType.CLASSIFICATION, "bogus output dict 2"),
        "head-3": (constants.ProblemType.UNSPECIFIED, "bogus output dict 3"),
    }
    model_fn_ops = model_fn.ModelFnOps(
        model_fn.ModeKeys.INFER,
        predictions={"some_output": "bogus_tensor"},
        output_alternatives=provided_output_alternatives)

    with self.assertRaises(ValueError) as e:
      saved_model_export_utils.get_output_alternatives(model_fn_ops)

    self.assertEqual("Please specify a default_output_alternative.  Available "
                     "output_alternatives are: ['head-1', 'head-2', 'head-3']",
                     str(e.exception))

  def test_get_output_alternatives_none_provided(self):
    prediction_tensor = constant_op.constant(["bogus"])
    model_fn_ops = model_fn.ModelFnOps(
        model_fn.ModeKeys.INFER,
        predictions={"some_output": prediction_tensor},
        output_alternatives=None)

    output_alternatives, _ = saved_model_export_utils.get_output_alternatives(
        model_fn_ops)

    self.assertEqual({
        "default_output_alternative": (constants.ProblemType.UNSPECIFIED, {
            "some_output": prediction_tensor
        })
    }, output_alternatives)

  def test_get_output_alternatives_empty_provided_with_default(self):
    prediction_tensor = constant_op.constant(["bogus"])
    model_fn_ops = model_fn.ModelFnOps(
        model_fn.ModeKeys.INFER,
        predictions={"some_output": prediction_tensor},
        output_alternatives={})

    with self.assertRaises(ValueError) as e:
      saved_model_export_utils.get_output_alternatives(model_fn_ops, "WRONG")

    self.assertEqual("Requested default_output_alternative: WRONG, but "
                     "available output_alternatives are: []", str(e.exception))

  def test_get_output_alternatives_empty_provided_no_default(self):
    prediction_tensor = constant_op.constant(["bogus"])
    model_fn_ops = model_fn.ModelFnOps(
        model_fn.ModeKeys.INFER,
        predictions={"some_output": prediction_tensor},
        output_alternatives={})

    output_alternatives, _ = saved_model_export_utils.get_output_alternatives(
        model_fn_ops)

    self.assertEqual({
        "default_output_alternative": (constants.ProblemType.UNSPECIFIED, {
            "some_output": prediction_tensor
        })
    }, output_alternatives)

  def test_get_output_alternatives_implicit_single(self):
    prediction_tensor = constant_op.constant(["bogus"])
    model_fn_ops = model_fn.ModelFnOps(
        model_fn.ModeKeys.INFER,
        predictions=prediction_tensor,
        output_alternatives=None)

    output_alternatives, _ = saved_model_export_utils.get_output_alternatives(
        model_fn_ops)
    self.assertEqual({
        "default_output_alternative": (constants.ProblemType.UNSPECIFIED, {
            "output": prediction_tensor
        })
    }, output_alternatives)

  def test_build_all_signature_defs(self):
    input_features = constant_op.constant(["10"])
    input_example = constant_op.constant(["input string"])
    input_ops = input_fn_utils.InputFnOps({
        "features": input_features
    }, None, {"default input": input_example})
    input_alternatives, _ = (
        saved_model_export_utils.get_input_alternatives(input_ops))
    output_1 = constant_op.constant([1.0])
    output_2 = constant_op.constant(["2"])
    output_3 = constant_op.constant(["3"])
    provided_output_alternatives = {
        "head-1": (constants.ProblemType.LINEAR_REGRESSION, {
            "some_output_1": output_1
        }),
        "head-2": (constants.ProblemType.CLASSIFICATION, {
            "some_output_2": output_2
        }),
        "head-3": (constants.ProblemType.UNSPECIFIED, {
            "some_output_3": output_3
        }),
    }
    model_fn_ops = model_fn.ModelFnOps(
        model_fn.ModeKeys.INFER,
        predictions={"some_output": constant_op.constant(["4"])},
        output_alternatives=provided_output_alternatives)
    output_alternatives, _ = (saved_model_export_utils.get_output_alternatives(
        model_fn_ops, "head-1"))

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
            signature_def_utils.predict_signature_def({
                "default input": input_example
            }, {"some_output_3": output_3}),
        # "features_input_alternative:head-1":
        #     signature_def_utils.regression_signature_def(input_features,
        #                                                  output_1),
        # "features_input_alternative:head-2":
        #     signature_def_utils.classification_signature_def(input_features,
        #                                                      output_2, None),
        # "features_input_alternative:head-3":
        #     signature_def_utils.predict_signature_def({
        #         "input": input_features
        #     }, {"output": output_3}),
    }

    self.assertDictEqual(expected_signature_defs, signature_defs)

  def test_build_all_signature_defs_legacy_input_fn_not_supported(self):
    """Tests that legacy input_fn returning (features, labels) raises error.

    serving_input_fn must return InputFnOps including a default input
    alternative.
    """
    input_features = constant_op.constant(["10"])
    input_ops = ({"features": input_features}, None)
    input_alternatives, _ = (
        saved_model_export_utils.get_input_alternatives(input_ops))
    output_1 = constant_op.constant(["1"])
    output_2 = constant_op.constant(["2"])
    output_3 = constant_op.constant(["3"])
    provided_output_alternatives = {
        "head-1": (constants.ProblemType.LINEAR_REGRESSION, {
            "some_output_1": output_1
        }),
        "head-2": (constants.ProblemType.CLASSIFICATION, {
            "some_output_2": output_2
        }),
        "head-3": (constants.ProblemType.UNSPECIFIED, {
            "some_output_3": output_3
        }),
    }
    model_fn_ops = model_fn.ModelFnOps(
        model_fn.ModeKeys.INFER,
        predictions={"some_output": constant_op.constant(["4"])},
        output_alternatives=provided_output_alternatives)
    output_alternatives, _ = (saved_model_export_utils.get_output_alternatives(
        model_fn_ops, "head-1"))

    with self.assertRaisesRegexp(
        ValueError, "A default input_alternative must be provided"):
      saved_model_export_utils.build_all_signature_defs(
          input_alternatives, output_alternatives, "head-1")

  def test_get_timestamped_export_dir(self):
    export_dir_base = tempfile.mkdtemp() + "export/"
    export_dir_1 = saved_model_export_utils.get_timestamped_export_dir(
        export_dir_base)
    time.sleep(2)
    export_dir_2 = saved_model_export_utils.get_timestamped_export_dir(
        export_dir_base)
    time.sleep(2)
    export_dir_3 = saved_model_export_utils.get_timestamped_export_dir(
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

  def test_garbage_collect_exports(self):
    export_dir_base = tempfile.mkdtemp() + "export/"
    gfile.MkDir(export_dir_base)
    export_dir_1 = _create_test_export_dir(export_dir_base)
    export_dir_2 = _create_test_export_dir(export_dir_base)
    export_dir_3 = _create_test_export_dir(export_dir_base)
    export_dir_4 = _create_test_export_dir(export_dir_base)

    self.assertTrue(gfile.Exists(export_dir_1))
    self.assertTrue(gfile.Exists(export_dir_2))
    self.assertTrue(gfile.Exists(export_dir_3))
    self.assertTrue(gfile.Exists(export_dir_4))

    # Garbage collect all but the most recent 2 exports,
    # where recency is determined based on the timestamp directory names.
    saved_model_export_utils.garbage_collect_exports(export_dir_base, 2)

    self.assertFalse(gfile.Exists(export_dir_1))
    self.assertFalse(gfile.Exists(export_dir_2))
    self.assertTrue(gfile.Exists(export_dir_3))
    self.assertTrue(gfile.Exists(export_dir_4))

  def test_get_most_recent_export(self):
    export_dir_base = tempfile.mkdtemp() + "export/"
    gfile.MkDir(export_dir_base)
    _create_test_export_dir(export_dir_base)
    _create_test_export_dir(export_dir_base)
    _create_test_export_dir(export_dir_base)
    export_dir_4 = _create_test_export_dir(export_dir_base)

    (most_recent_export_dir, most_recent_export_version) = (
        saved_model_export_utils.get_most_recent_export(export_dir_base))

    self.assertEqual(
        compat.as_bytes(export_dir_4), compat.as_bytes(most_recent_export_dir))
    self.assertEqual(
        compat.as_bytes(export_dir_4),
        os.path.join(
            compat.as_bytes(export_dir_base),
            compat.as_bytes(str(most_recent_export_version))))

  def test_make_export_strategy(self):
    """Only tests that an ExportStrategy instance is created."""

    def _serving_input_fn():
      return array_ops.constant([1]), None

    export_strategy = saved_model_export_utils.make_export_strategy(
        serving_input_fn=_serving_input_fn,
        default_output_alternative_key="default",
        assets_extra={"from/path": "to/path"},
        as_text=False,
        exports_to_keep=5)
    self.assertTrue(
        isinstance(export_strategy, export_strategy_lib.ExportStrategy))

  def test_make_parsing_export_strategy(self):
    """Only tests that an ExportStrategy instance is created."""
    sparse_col = fc.sparse_column_with_hash_bucket(
        "sparse_column", hash_bucket_size=100)
    embedding_col = fc.embedding_column(
        fc.sparse_column_with_hash_bucket(
            "sparse_column_for_embedding", hash_bucket_size=10),
        dimension=4)
    real_valued_col1 = fc.real_valued_column("real_valued_column1")
    bucketized_col1 = fc.bucketized_column(
        fc.real_valued_column("real_valued_column_for_bucketization1"), [0, 4])
    feature_columns = [
        sparse_col, embedding_col, real_valued_col1, bucketized_col1
    ]

    export_strategy = saved_model_export_utils.make_parsing_export_strategy(
        feature_columns=feature_columns)
    self.assertTrue(
        isinstance(export_strategy, export_strategy_lib.ExportStrategy))

  def test_make_best_model_export_strategy(self):
    export_dir_base = tempfile.mkdtemp() + "export/"
    gfile.MkDir(export_dir_base)

    test_estimator = TestEstimator()
    export_strategy = saved_model_export_utils.make_best_model_export_strategy(
        serving_input_fn=None, exports_to_keep=3, compare_fn=None)

    self.assertNotEqual("",
                        export_strategy.export(test_estimator, export_dir_base,
                                               "fake_ckpt_0", {"loss": 100}))
    self.assertNotEqual("", test_estimator.last_exported_dir)
    self.assertNotEqual("", test_estimator.last_exported_checkpoint)

    self.assertEqual("",
                     export_strategy.export(test_estimator, export_dir_base,
                                            "fake_ckpt_1", {"loss": 101}))
    self.assertEqual(test_estimator.last_exported_dir,
                     os.path.join(export_dir_base, "fake_ckpt_0"))

    self.assertNotEqual("",
                        export_strategy.export(test_estimator, export_dir_base,
                                               "fake_ckpt_2", {"loss": 10}))
    self.assertEqual(test_estimator.last_exported_dir,
                     os.path.join(export_dir_base, "fake_ckpt_2"))

    self.assertEqual("",
                     export_strategy.export(test_estimator, export_dir_base,
                                            "fake_ckpt_3", {"loss": 20}))
    self.assertEqual(test_estimator.last_exported_dir,
                     os.path.join(export_dir_base, "fake_ckpt_2"))

  def test_make_best_model_export_strategy_exceptions(self):
    export_dir_base = tempfile.mkdtemp() + "export/"

    test_estimator = TestEstimator()
    export_strategy = saved_model_export_utils.make_best_model_export_strategy(
        serving_input_fn=None, exports_to_keep=3, compare_fn=None)

    with self.assertRaises(ValueError):
      export_strategy.export(test_estimator, export_dir_base, "", {"loss": 200})

    with self.assertRaises(ValueError):
      export_strategy.export(test_estimator, export_dir_base, "fake_ckpt_1",
                             None)

  def test_extend_export_strategy(self):

    def _base_export_fn(unused_estimator,
                        export_dir_base,
                        unused_checkpoint_path=None):
      base_path = os.path.join(export_dir_base, "e1")
      gfile.MkDir(base_path)
      return base_path

    def _post_export_fn(orig_path, new_path):
      assert orig_path.endswith("/e1")
      post_export_path = os.path.join(new_path, "rewrite")
      gfile.MkDir(post_export_path)
      return post_export_path

    base_export_strategy = export_strategy_lib.ExportStrategy(
        "Servo", _base_export_fn)

    final_export_strategy = saved_model_export_utils.extend_export_strategy(
        base_export_strategy, _post_export_fn, "Servo2")
    self.assertEqual(final_export_strategy.name, "Servo2")

    test_estimator = TestEstimator()
    tmpdir = tempfile.mkdtemp()
    export_model_dir = os.path.join(tmpdir, "model")
    checkpoint_path = os.path.join(tmpdir, "checkpoint")
    final_path = final_export_strategy.export(test_estimator, export_model_dir,
                                              checkpoint_path)
    self.assertEqual(os.path.join(export_model_dir, "rewrite"), final_path)

  def test_extend_export_strategy_same_name(self):

    def _base_export_fn(unused_estimator,
                        export_dir_base,
                        unused_checkpoint_path=None):
      base_path = os.path.join(export_dir_base, "e1")
      gfile.MkDir(base_path)
      return base_path

    def _post_export_fn(orig_path, new_path):
      assert orig_path.endswith("/e1")
      post_export_path = os.path.join(new_path, "rewrite")
      gfile.MkDir(post_export_path)
      return post_export_path

    base_export_strategy = export_strategy_lib.ExportStrategy(
        "Servo", _base_export_fn)

    final_export_strategy = saved_model_export_utils.extend_export_strategy(
        base_export_strategy, _post_export_fn)
    self.assertEqual(final_export_strategy.name, "Servo")

    test_estimator = TestEstimator()
    tmpdir = tempfile.mkdtemp()
    export_model_dir = os.path.join(tmpdir, "model")
    checkpoint_path = os.path.join(tmpdir, "checkpoint")
    final_path = final_export_strategy.export(test_estimator, export_model_dir,
                                              checkpoint_path)
    self.assertEqual(os.path.join(export_model_dir, "rewrite"), final_path)

  def test_extend_export_strategy_raises_error(self):

    def _base_export_fn(unused_estimator,
                        export_dir_base,
                        unused_checkpoint_path=None):
      base_path = os.path.join(export_dir_base, "e1")
      gfile.MkDir(base_path)
      return base_path

    def _post_export_fn(unused_orig_path, unused_new_path):
      return tempfile.mkdtemp()

    base_export_strategy = export_strategy_lib.ExportStrategy(
        "Servo", _base_export_fn)

    final_export_strategy = saved_model_export_utils.extend_export_strategy(
        base_export_strategy, _post_export_fn)

    test_estimator = TestEstimator()
    tmpdir = tempfile.mkdtemp()
    with self.assertRaises(ValueError) as ve:
      final_export_strategy.export(test_estimator, tmpdir,
                                   os.path.join(tmpdir, "checkpoint"))

    self.assertTrue(
        "post_export_fn must return a sub-directory" in str(ve.exception))


def _create_test_export_dir(export_dir_base):
  export_dir = saved_model_export_utils.get_timestamped_export_dir(
      export_dir_base)
  gfile.MkDir(export_dir)
  time.sleep(2)
  return export_dir


if __name__ == "__main__":
  test.main()
