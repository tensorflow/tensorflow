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
"""Tests for export utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import time

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model.model_utils import export_output
from tensorflow.python.saved_model.model_utils import export_utils
from tensorflow.python.saved_model.model_utils.mode_keys import KerasModeKeys


class ExportTest(test_util.TensorFlowTestCase):

  @test_util.deprecated_graph_mode_only
  def test_build_all_signature_defs_without_receiver_alternatives(self):
    receiver_tensor = array_ops.placeholder(dtypes.string)
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

    signature_defs = export_utils.build_all_signature_defs(
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
                "input": receiver_tensor
            }, {"some_output_3": output_3})
    }

    self.assertDictEqual(expected_signature_defs, signature_defs)

  @test_util.deprecated_graph_mode_only
  def test_build_all_signature_defs_with_dict_alternatives(self):
    receiver_tensor = array_ops.placeholder(dtypes.string)
    receiver_tensors_alternative_1 = {
        "foo": array_ops.placeholder(dtypes.int64),
        "bar": array_ops.sparse_placeholder(dtypes.float32)}
    receiver_tensors_alternatives = {"other": receiver_tensors_alternative_1}
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

    signature_defs = export_utils.build_all_signature_defs(
        receiver_tensor, export_outputs, receiver_tensors_alternatives)

    expected_signature_defs = {
        "serving_default":
            signature_def_utils.regression_signature_def(
                receiver_tensor,
                output_1),
        "head-2":
            signature_def_utils.classification_signature_def(
                receiver_tensor,
                output_2, None),
        "head-3":
            signature_def_utils.predict_signature_def(
                {"input": receiver_tensor},
                {"some_output_3": output_3}),
        "other:head-3":
            signature_def_utils.predict_signature_def(
                receiver_tensors_alternative_1,
                {"some_output_3": output_3})

        # Note that the alternatives 'other:serving_default' and
        # 'other:head-2' are invalid, because regession and classification
        # signatures must take a single string input.  Here we verify that
        # these invalid signatures are not included in the export_utils.
    }

    self.assertDictEqual(expected_signature_defs, signature_defs)

  @test_util.deprecated_graph_mode_only
  def test_build_all_signature_defs_with_single_alternatives(self):
    receiver_tensor = array_ops.placeholder(dtypes.string)
    receiver_tensors_alternative_1 = array_ops.placeholder(dtypes.int64)
    receiver_tensors_alternative_2 = array_ops.sparse_placeholder(
        dtypes.float32)
    # Note we are passing single Tensors as values of
    # receiver_tensors_alternatives, where normally that is a dict.
    # In this case a dict will be created using the default receiver tensor
    # name "input".
    receiver_tensors_alternatives = {"other1": receiver_tensors_alternative_1,
                                     "other2": receiver_tensors_alternative_2}
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

    signature_defs = export_utils.build_all_signature_defs(
        receiver_tensor, export_outputs, receiver_tensors_alternatives)

    expected_signature_defs = {
        "serving_default":
            signature_def_utils.regression_signature_def(
                receiver_tensor,
                output_1),
        "head-2":
            signature_def_utils.classification_signature_def(
                receiver_tensor,
                output_2, None),
        "head-3":
            signature_def_utils.predict_signature_def(
                {"input": receiver_tensor},
                {"some_output_3": output_3}),
        "other1:head-3":
            signature_def_utils.predict_signature_def(
                {"input": receiver_tensors_alternative_1},
                {"some_output_3": output_3}),
        "other2:head-3":
            signature_def_utils.predict_signature_def(
                {"input": receiver_tensors_alternative_2},
                {"some_output_3": output_3})

        # Note that the alternatives 'other:serving_default' and 'other:head-2'
        # are invalid, because regession and classification signatures must take
        # a single string input.  Here we verify that these invalid signatures
        # are not included in the export_utils.
    }

    self.assertDictEqual(expected_signature_defs, signature_defs)

  def test_build_all_signature_defs_export_outputs_required(self):
    receiver_tensor = constant_op.constant(["11"])

    with self.assertRaises(ValueError) as e:
      export_utils.build_all_signature_defs(receiver_tensor, None)

    self.assertTrue(str(e.exception).startswith(
        "export_outputs must be a dict"))

  def test_get_timestamped_export_dir(self):
    export_dir_base = tempfile.mkdtemp() + "export/"
    export_dir_1 = export_utils.get_timestamped_export_dir(
        export_dir_base)
    time.sleep(2)
    export_dir_2 = export_utils.get_timestamped_export_dir(
        export_dir_base)
    time.sleep(2)
    export_dir_3 = export_utils.get_timestamped_export_dir(
        export_dir_base)

    # Export directories should be named using a timestamp that is seconds
    # since epoch.  Such a timestamp is 10 digits long.
    time_1 = os.path.basename(export_dir_1)
    self.assertEqual(10, len(time_1))
    time_2 = os.path.basename(export_dir_2)
    self.assertEqual(10, len(time_2))
    time_3 = os.path.basename(export_dir_3)
    self.assertEqual(10, len(time_3))

    self.assertLess(int(time_1), int(time_2))
    self.assertLess(int(time_2), int(time_3))

  def test_get_temp_export_dir(self):
    export_dir = "/tmp/export/1576013284"
    tmp_export_dir = export_utils.get_temp_export_dir(export_dir)
    self.assertEqual(tmp_export_dir, b"/tmp/export/temp-1576013284")

    export_dir = b"/tmp/export/1576013284"
    tmp_export_dir = export_utils.get_temp_export_dir(export_dir)
    self.assertEqual(tmp_export_dir, b"/tmp/export/temp-1576013284")

  @test_util.deprecated_graph_mode_only
  def test_build_all_signature_defs_serving_only(self):
    receiver_tensor = {"input": array_ops.placeholder(dtypes.string)}
    output_1 = constant_op.constant([1.])
    export_outputs = {
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            export_output.PredictOutput(outputs=output_1),
        "train": export_output.TrainOutput(loss=output_1),
    }

    signature_defs = export_utils.build_all_signature_defs(
        receiver_tensor, export_outputs)

    expected_signature_defs = {
        "serving_default": signature_def_utils.predict_signature_def(
            receiver_tensor, {"output": output_1})
    }

    self.assertDictEqual(expected_signature_defs, signature_defs)

    signature_defs = export_utils.build_all_signature_defs(
        receiver_tensor, export_outputs, serving_only=False)

    expected_signature_defs.update({
        "train": signature_def_utils.supervised_train_signature_def(
            receiver_tensor, loss={"loss": output_1})
    })

    self.assertDictEqual(expected_signature_defs, signature_defs)

  @test_util.deprecated_graph_mode_only
  def test_export_outputs_for_mode(self):
    predictions = {"predictions": constant_op.constant([1.])}
    loss = {"loss": constant_op.constant([2.])}
    metrics = {
        "metrics": (constant_op.constant([3.]), constant_op.constant([4.]))}
    expected_metrics = {
        "metrics/value": metrics["metrics"][0],
        "metrics/update_op": metrics["metrics"][1]
    }

    def _build_export_output(mode):
      return export_utils.export_outputs_for_mode(
          mode, None, predictions, loss, metrics)

    ret = _build_export_output(KerasModeKeys.TRAIN)
    self.assertIn(signature_constants.DEFAULT_TRAIN_SIGNATURE_DEF_KEY, ret)
    export_out = ret[signature_constants.DEFAULT_TRAIN_SIGNATURE_DEF_KEY]
    self.assertIsInstance(export_out, export_output.TrainOutput)
    self.assertEqual(export_out.predictions, predictions)
    self.assertEqual(export_out.loss, loss)
    self.assertEqual(export_out.metrics, expected_metrics)

    ret = _build_export_output(KerasModeKeys.TEST)
    self.assertIn(signature_constants.DEFAULT_EVAL_SIGNATURE_DEF_KEY, ret)
    export_out = ret[signature_constants.DEFAULT_EVAL_SIGNATURE_DEF_KEY]
    self.assertIsInstance(export_out, export_output.EvalOutput)
    self.assertEqual(export_out.predictions, predictions)
    self.assertEqual(export_out.loss, loss)
    self.assertEqual(export_out.metrics, expected_metrics)

    ret = _build_export_output(KerasModeKeys.PREDICT)
    self.assertIn(signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY, ret)
    export_out = ret[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    self.assertIsInstance(export_out, export_output.PredictOutput)
    self.assertEqual(export_out.outputs, predictions)

    classes = constant_op.constant(["class5"])
    ret = export_utils.export_outputs_for_mode(
        KerasModeKeys.PREDICT,
        {"classify": export_output.ClassificationOutput(
            classes=classes)})
    self.assertIn("classify", ret)
    export_out = ret["classify"]
    self.assertIsInstance(export_out, export_output.ClassificationOutput)
    self.assertEqual(export_out.classes, classes)


if __name__ == "__main__":
  test.main()
