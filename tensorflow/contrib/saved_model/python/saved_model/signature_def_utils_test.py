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
# ==============================================================================
"""Tests for SignatureDef utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils as signature_def_contrib_utils
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils


class SignatureDefUtilsTest(test.TestCase):

  def _add_to_signature_def_map(self, meta_graph_def, signature_def_map=None):
    if signature_def_map is not None:
      for key in signature_def_map:
        meta_graph_def.signature_def[key].CopyFrom(signature_def_map[key])

  def _check_tensor_info(self, tensor_info_map, map_key, expected_tensor_name):
    actual_tensor_info = tensor_info_map[map_key]
    self.assertEqual(expected_tensor_name, actual_tensor_info.name)

  def testGetSignatureDefByKey(self):
    x = array_ops.placeholder(dtypes.float32, 1, name="x")
    x_tensor_info = utils.build_tensor_info(x)

    y = array_ops.placeholder(dtypes.float32, name="y")
    y_tensor_info = utils.build_tensor_info(y)

    foo_signature_def = signature_def_utils.build_signature_def({
        "foo-input": x_tensor_info
    }, {"foo-output": y_tensor_info}, "foo-method-name")
    bar_signature_def = signature_def_utils.build_signature_def({
        "bar-input": x_tensor_info
    }, {"bar-output": y_tensor_info}, "bar-method-name")
    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    self._add_to_signature_def_map(
        meta_graph_def, {"foo": foo_signature_def,
                         "bar": bar_signature_def})

    # Look up a key that does not exist in the SignatureDefMap.
    missing_key = "missing-key"
    with self.assertRaisesRegexp(
        ValueError,
        "No SignatureDef with key '%s' found in MetaGraphDef" % missing_key):
      signature_def_contrib_utils.get_signature_def_by_key(
          meta_graph_def, missing_key)

    # Look up the key, `foo` which exists in the SignatureDefMap.
    foo_signature_def = signature_def_contrib_utils.get_signature_def_by_key(
        meta_graph_def, "foo")
    self.assertTrue("foo-method-name", foo_signature_def.method_name)

    # Check inputs in signature def.
    self.assertEqual(1, len(foo_signature_def.inputs))
    self._check_tensor_info(foo_signature_def.inputs, "foo-input", "x:0")

    # Check outputs in signature def.
    self.assertEqual(1, len(foo_signature_def.outputs))
    self._check_tensor_info(foo_signature_def.outputs, "foo-output", "y:0")

    # Look up the key, `bar` which exists in the SignatureDefMap.
    bar_signature_def = signature_def_contrib_utils.get_signature_def_by_key(
        meta_graph_def, "bar")
    self.assertTrue("bar-method-name", bar_signature_def.method_name)

    # Check inputs in signature def.
    self.assertEqual(1, len(bar_signature_def.inputs))
    self._check_tensor_info(bar_signature_def.inputs, "bar-input", "x:0")

    # Check outputs in signature def.
    self.assertEqual(1, len(bar_signature_def.outputs))
    self._check_tensor_info(bar_signature_def.outputs, "bar-output", "y:0")

  def testGetSignatureDefByKeyRegression(self):
    input1 = constant_op.constant("a", name="input-1")
    output1 = constant_op.constant("b", name="output-1")

    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    self._add_to_signature_def_map(meta_graph_def, {
        "my_regression":
            signature_def_utils.regression_signature_def(input1, output1)
    })

    # Look up the regression signature with the key used while saving.
    signature_def = signature_def_contrib_utils.get_signature_def_by_key(
        meta_graph_def, "my_regression")

    # Check the method name to match the constants regression method name.
    self.assertEqual(signature_constants.REGRESS_METHOD_NAME,
                     signature_def.method_name)

    # Check inputs in signature def.
    self.assertEqual(1, len(signature_def.inputs))
    self._check_tensor_info(signature_def.inputs,
                            signature_constants.REGRESS_INPUTS, "input-1:0")

    # Check outputs in signature def.
    self.assertEqual(1, len(signature_def.outputs))
    self._check_tensor_info(signature_def.outputs,
                            signature_constants.REGRESS_OUTPUTS, "output-1:0")

  def testGetSignatureDefByKeyClassification(self):
    input1 = constant_op.constant("a", name="input-1")
    output1 = constant_op.constant("b", name="output-1")
    output2 = constant_op.constant("c", name="output-2")

    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    self._add_to_signature_def_map(meta_graph_def, {
        "my_classification":
            signature_def_utils.classification_signature_def(
                input1, output1, output2)
    })

    # Look up the classification signature def with the key used while saving.
    signature_def = signature_def_contrib_utils.get_signature_def_by_key(
        meta_graph_def, "my_classification")

    # Check the method name to match the constants classification method name.
    self.assertEqual(signature_constants.CLASSIFY_METHOD_NAME,
                     signature_def.method_name)

    # Check inputs in signature def.
    self.assertEqual(1, len(signature_def.inputs))
    self._check_tensor_info(signature_def.inputs,
                            signature_constants.CLASSIFY_INPUTS, "input-1:0")

    # Check outputs in signature def.
    self.assertEqual(2, len(signature_def.outputs))
    self._check_tensor_info(signature_def.outputs,
                            signature_constants.CLASSIFY_OUTPUT_CLASSES,
                            "output-1:0")
    self._check_tensor_info(signature_def.outputs,
                            signature_constants.CLASSIFY_OUTPUT_SCORES,
                            "output-2:0")

  def testPredictionSignatureDef(self):
    input1 = constant_op.constant("a", name="input-1")
    input2 = constant_op.constant("b", name="input-2")
    output1 = constant_op.constant("c", name="output-1")
    output2 = constant_op.constant("d", name="output-2")

    meta_graph_def = meta_graph_pb2.MetaGraphDef()
    self._add_to_signature_def_map(meta_graph_def, {
        "my_prediction":
            signature_def_utils.predict_signature_def({
                "input-1": input1,
                "input-2": input2
            }, {"output-1": output1,
                "output-2": output2})
    })

    # Look up the prediction signature def with the key used while saving.
    signature_def = signature_def_contrib_utils.get_signature_def_by_key(
        meta_graph_def, "my_prediction")
    self.assertEqual(signature_constants.PREDICT_METHOD_NAME,
                     signature_def.method_name)

    # Check inputs in signature def.
    self.assertEqual(2, len(signature_def.inputs))
    self._check_tensor_info(signature_def.inputs, "input-1", "input-1:0")
    self._check_tensor_info(signature_def.inputs, "input-2", "input-2:0")

    # Check outputs in signature def.
    self.assertEqual(2, len(signature_def.outputs))
    self._check_tensor_info(signature_def.outputs, "output-1", "output-1:0")
    self._check_tensor_info(signature_def.outputs, "output-2", "output-2:0")


if __name__ == "__main__":
  test.main()
