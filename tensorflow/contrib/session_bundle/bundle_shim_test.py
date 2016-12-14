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
"""Tests for bundle_shim.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf

from tensorflow.contrib.session_bundle import bundle_shim
from tensorflow.contrib.session_bundle import constants
from tensorflow.contrib.session_bundle import manifest_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import meta_graph
from tensorflow.python.saved_model import constants as saved_model_constants
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat

SAVED_MODEL_PATH = ("cc/saved_model/testdata/half_plus_two/00000123")
SESSION_BUNDLE_PATH = "contrib/session_bundle/testdata/half_plus_two/00000123"


class BundleShimTest(tf.test.TestCase):

  def testBadPath(self):
    base_path = tf.test.test_src_dir_path("/no/such/a/dir")
    tf.reset_default_graph()
    with self.assertRaises(RuntimeError) as cm:
      _, _ = bundle_shim.load_session_bundle_or_saved_model_bundle_from_path(
          base_path)

  def testAddInputToSignatureDef(self):
    signature_def = meta_graph_pb2.SignatureDef()
    signature_def_compare = meta_graph_pb2.SignatureDef()

    # Add input to signature-def corresponding to `foo_key`.
    bundle_shim._add_input_to_signature_def("foo-name", "foo-key",
                                            signature_def)
    self.assertEqual(len(signature_def.inputs), 1)
    self.assertEqual(len(signature_def.outputs), 0)
    self.assertProtoEquals(
        signature_def.inputs["foo-key"],
        meta_graph_pb2.TensorInfo(name="foo-name"))

    # Attempt to add another input to the signature-def with the same tensor
    # name and key.
    bundle_shim._add_input_to_signature_def("foo-name", "foo-key",
                                            signature_def)
    self.assertEqual(len(signature_def.inputs), 1)
    self.assertEqual(len(signature_def.outputs), 0)
    self.assertProtoEquals(
        signature_def.inputs["foo-key"],
        meta_graph_pb2.TensorInfo(name="foo-name"))

    # Add another input to the signature-def corresponding to `bar-key`.
    bundle_shim._add_input_to_signature_def("bar-name", "bar-key",
                                            signature_def)
    self.assertEqual(len(signature_def.inputs), 2)
    self.assertEqual(len(signature_def.outputs), 0)
    self.assertProtoEquals(
        signature_def.inputs["bar-key"],
        meta_graph_pb2.TensorInfo(name="bar-name"))

    # Add an input to the signature-def corresponding to `foo-key` with an
    # updated tensor name.
    bundle_shim._add_input_to_signature_def("bar-name", "foo-key",
                                            signature_def)
    self.assertEqual(len(signature_def.inputs), 2)
    self.assertEqual(len(signature_def.outputs), 0)
    self.assertProtoEquals(
        signature_def.inputs["foo-key"],
        meta_graph_pb2.TensorInfo(name="bar-name"))

    # Test that there are no other side-effects.
    del signature_def.inputs["foo-key"]
    del signature_def.inputs["bar-key"]
    self.assertProtoEquals(signature_def, signature_def_compare)

  def testAddOutputToSignatureDef(self):
    signature_def = meta_graph_pb2.SignatureDef()
    signature_def_compare = meta_graph_pb2.SignatureDef()

    # Add output to signature-def corresponding to `foo_key`.
    bundle_shim._add_output_to_signature_def("foo-name", "foo-key",
                                             signature_def)
    self.assertEqual(len(signature_def.outputs), 1)
    self.assertEqual(len(signature_def.inputs), 0)
    self.assertProtoEquals(
        signature_def.outputs["foo-key"],
        meta_graph_pb2.TensorInfo(name="foo-name"))

    # Attempt to add another output to the signature-def with the same tensor
    # name and key.
    bundle_shim._add_output_to_signature_def("foo-name", "foo-key",
                                             signature_def)
    self.assertEqual(len(signature_def.outputs), 1)
    self.assertEqual(len(signature_def.inputs), 0)
    self.assertProtoEquals(
        signature_def.outputs["foo-key"],
        meta_graph_pb2.TensorInfo(name="foo-name"))

    # Add another output to the signature-def corresponding to `bar-key`.
    bundle_shim._add_output_to_signature_def("bar-name", "bar-key",
                                             signature_def)
    self.assertEqual(len(signature_def.outputs), 2)
    self.assertEqual(len(signature_def.inputs), 0)
    self.assertProtoEquals(
        signature_def.outputs["bar-key"],
        meta_graph_pb2.TensorInfo(name="bar-name"))

    # Add an output to the signature-def corresponding to `foo-key` with an
    # updated tensor name.
    bundle_shim._add_output_to_signature_def("bar-name", "foo-key",
                                             signature_def)
    self.assertEqual(len(signature_def.outputs), 2)
    self.assertEqual(len(signature_def.inputs), 0)
    self.assertProtoEquals(
        signature_def.outputs["foo-key"],
        meta_graph_pb2.TensorInfo(name="bar-name"))

    # Test that there are no other sideeffects.
    del signature_def.outputs["foo-key"]
    del signature_def.outputs["bar-key"]
    self.assertProtoEquals(signature_def, signature_def_compare)

  def testConvertDefaultSignatureBadTypeToSignatureDef(self):
    signatures_proto = manifest_pb2.Signatures()
    generic_signature = manifest_pb2.GenericSignature()
    signatures_proto.default_signature.generic_signature.CopyFrom(
        generic_signature)
    with self.assertRaises(RuntimeError) as cm:
      _ = bundle_shim._convert_default_signature_to_signature_def(
          signatures_proto)

  def testConvertDefaultSignatureRegressionToSignatureDef(self):
    signatures_proto = manifest_pb2.Signatures()
    regression_signature = manifest_pb2.RegressionSignature()
    regression_signature.input.CopyFrom(
        manifest_pb2.TensorBinding(
            tensor_name=signature_constants.REGRESS_INPUTS))
    regression_signature.output.CopyFrom(
        manifest_pb2.TensorBinding(
            tensor_name=signature_constants.REGRESS_OUTPUTS))
    signatures_proto.default_signature.regression_signature.CopyFrom(
        regression_signature)
    signature_def = bundle_shim._convert_default_signature_to_signature_def(
        signatures_proto)

    # Validate regression signature correctly copied over.
    self.assertEqual(signature_def.method_name,
                     signature_constants.REGRESS_METHOD_NAME)
    self.assertEqual(len(signature_def.inputs), 1)
    self.assertEqual(len(signature_def.outputs), 1)
    self.assertProtoEquals(
        signature_def.inputs[signature_constants.REGRESS_INPUTS],
        meta_graph_pb2.TensorInfo(name=signature_constants.REGRESS_INPUTS))
    self.assertProtoEquals(
        signature_def.outputs[signature_constants.REGRESS_OUTPUTS],
        meta_graph_pb2.TensorInfo(name=signature_constants.REGRESS_OUTPUTS))

  def testConvertDefaultSignatureClassificationToSignatureDef(self):
    signatures_proto = manifest_pb2.Signatures()
    classification_signature = manifest_pb2.ClassificationSignature()
    classification_signature.input.CopyFrom(
        manifest_pb2.TensorBinding(
            tensor_name=signature_constants.CLASSIFY_INPUTS))
    classification_signature.classes.CopyFrom(
        manifest_pb2.TensorBinding(
            tensor_name=signature_constants.CLASSIFY_OUTPUT_CLASSES))
    classification_signature.scores.CopyFrom(
        manifest_pb2.TensorBinding(
            tensor_name=signature_constants.CLASSIFY_OUTPUT_SCORES))
    signatures_proto.default_signature.classification_signature.CopyFrom(
        classification_signature)

    signatures_proto.default_signature.classification_signature.CopyFrom(
        classification_signature)
    signature_def = bundle_shim._convert_default_signature_to_signature_def(
        signatures_proto)

    # Validate classification signature correctly copied over.
    self.assertEqual(signature_def.method_name,
                     signature_constants.CLASSIFY_METHOD_NAME)
    self.assertEqual(len(signature_def.inputs), 1)
    self.assertEqual(len(signature_def.outputs), 2)
    self.assertProtoEquals(
        signature_def.inputs[signature_constants.CLASSIFY_INPUTS],
        meta_graph_pb2.TensorInfo(name=signature_constants.CLASSIFY_INPUTS))
    self.assertProtoEquals(
        signature_def.outputs[signature_constants.CLASSIFY_OUTPUT_SCORES],
        meta_graph_pb2.TensorInfo(
            name=signature_constants.CLASSIFY_OUTPUT_SCORES))
    self.assertProtoEquals(
        signature_def.outputs[signature_constants.CLASSIFY_OUTPUT_CLASSES],
        meta_graph_pb2.TensorInfo(
            name=signature_constants.CLASSIFY_OUTPUT_CLASSES))

  def testConvertNamedSignatureNonGenericToSignatureDef(self):
    signatures_proto = manifest_pb2.Signatures()
    regression_signature = manifest_pb2.RegressionSignature()
    signatures_proto.named_signatures[
        signature_constants.PREDICT_INPUTS].regression_signature.CopyFrom(
            regression_signature)
    with self.assertRaises(RuntimeError) as cm:
      _ = bundle_shim._convert_named_signatures_to_signature_def(
          signatures_proto)
    signatures_proto = manifest_pb2.Signatures()
    classification_signature = manifest_pb2.ClassificationSignature()
    signatures_proto.named_signatures[
        signature_constants.PREDICT_INPUTS].classification_signature.CopyFrom(
            classification_signature)
    with self.assertRaises(RuntimeError) as cm:
      _ = bundle_shim._convert_named_signatures_to_signature_def(
          signatures_proto)

  def testConvertNamedSignatureToSignatureDef(self):
    signatures_proto = manifest_pb2.Signatures()
    generic_signature = manifest_pb2.GenericSignature()
    generic_signature.map["input_key"].CopyFrom(
        manifest_pb2.TensorBinding(tensor_name="input"))
    signatures_proto.named_signatures[
        signature_constants.PREDICT_INPUTS].generic_signature.CopyFrom(
            generic_signature)

    generic_signature = manifest_pb2.GenericSignature()
    generic_signature.map["output_key"].CopyFrom(
        manifest_pb2.TensorBinding(tensor_name="output"))
    signatures_proto.named_signatures[
        signature_constants.PREDICT_OUTPUTS].generic_signature.CopyFrom(
            generic_signature)
    signature_def = bundle_shim._convert_named_signatures_to_signature_def(
        signatures_proto)
    self.assertEqual(signature_def.method_name,
                      signature_constants.PREDICT_METHOD_NAME)
    self.assertEqual(len(signature_def.inputs), 1)
    self.assertEqual(len(signature_def.outputs), 1)
    self.assertProtoEquals(
        signature_def.inputs["input_key"],
        meta_graph_pb2.TensorInfo(name="input"))
    self.assertProtoEquals(
        signature_def.outputs["output_key"],
        meta_graph_pb2.TensorInfo(name="output"))

  def testConvertSignaturesToSignatureDefs(self):
    base_path = tf.test.test_src_dir_path(SESSION_BUNDLE_PATH)
    meta_graph_filename = os.path.join(base_path,
                                       constants.META_GRAPH_DEF_FILENAME)
    metagraph_def = meta_graph.read_meta_graph_file(meta_graph_filename)
    default_signature_def, named_signature_def = (
        bundle_shim._convert_signatures_to_signature_defs(metagraph_def))
    self.assertEqual(default_signature_def.method_name,
                      signature_constants.REGRESS_METHOD_NAME)
    self.assertEqual(len(default_signature_def.inputs), 1)
    self.assertEqual(len(default_signature_def.outputs), 1)
    self.assertProtoEquals(
        default_signature_def.inputs[signature_constants.REGRESS_INPUTS],
        meta_graph_pb2.TensorInfo(name="tf_example:0"))
    self.assertProtoEquals(
        default_signature_def.outputs[signature_constants.REGRESS_OUTPUTS],
        meta_graph_pb2.TensorInfo(name="Identity:0"))
    self.assertEqual(named_signature_def.method_name,
                      signature_constants.PREDICT_METHOD_NAME)
    self.assertEqual(len(named_signature_def.inputs), 1)
    self.assertEqual(len(named_signature_def.outputs), 1)
    self.assertProtoEquals(
        named_signature_def.inputs["x"], meta_graph_pb2.TensorInfo(name="x:0"))
    self.assertProtoEquals(
        named_signature_def.outputs["y"], meta_graph_pb2.TensorInfo(name="y:0"))

    # Now try default signature only
    collection_def = metagraph_def.collection_def
    signatures_proto = manifest_pb2.Signatures()
    signatures = collection_def[constants.SIGNATURES_KEY].any_list.value[0]
    signatures.Unpack(signatures_proto)
    named_only_signatures_proto = manifest_pb2.Signatures()
    named_only_signatures_proto.CopyFrom(signatures_proto)

    default_only_signatures_proto = manifest_pb2.Signatures()
    default_only_signatures_proto.CopyFrom(signatures_proto)
    default_only_signatures_proto.named_signatures.clear()
    default_only_signatures_proto.ClearField("named_signatures")
    metagraph_def.collection_def[constants.SIGNATURES_KEY].any_list.value[
        0].Pack(default_only_signatures_proto)
    default_signature_def, named_signature_def = (
        bundle_shim._convert_signatures_to_signature_defs(metagraph_def))
    self.assertEqual(default_signature_def.method_name,
                      signature_constants.REGRESS_METHOD_NAME)
    self.assertEqual(named_signature_def, None)

    named_only_signatures_proto.ClearField("default_signature")
    metagraph_def.collection_def[constants.SIGNATURES_KEY].any_list.value[
        0].Pack(named_only_signatures_proto)
    default_signature_def, named_signature_def = (
        bundle_shim._convert_signatures_to_signature_defs(metagraph_def))
    self.assertEqual(named_signature_def.method_name,
                      signature_constants.PREDICT_METHOD_NAME)
    self.assertEqual(default_signature_def, None)

  def testLegacyBasic(self):
    base_path = tf.test.test_src_dir_path(SESSION_BUNDLE_PATH)
    tf.reset_default_graph()
    sess, meta_graph_def = (
        bundle_shim.load_session_bundle_or_saved_model_bundle_from_path(
            base_path,
            tags=[""],
            target="",
            config=tf.ConfigProto(device_count={"CPU": 2})))

    self.assertTrue(sess)
    asset_path = os.path.join(base_path, constants.ASSETS_DIRECTORY)
    with sess.as_default():
      path1, path2 = sess.run(["filename1:0", "filename2:0"])
      self.assertEqual(
          compat.as_bytes(os.path.join(asset_path, "hello1.txt")), path1)
      self.assertEqual(
          compat.as_bytes(os.path.join(asset_path, "hello2.txt")), path2)

      collection_def = meta_graph_def.collection_def

      signatures_any = collection_def[constants.SIGNATURES_KEY].any_list.value
      self.assertEqual(len(signatures_any), 1)

  def testSavedModelBasic(self):
    base_path = tf.test.test_src_dir_path(SAVED_MODEL_PATH)
    tf.reset_default_graph()
    sess, meta_graph_def = (
        bundle_shim.load_session_bundle_or_saved_model_bundle_from_path(
            base_path,
            tags=[tag_constants.SERVING],
            target="",
            config=tf.ConfigProto(device_count={"CPU": 2})))

    self.assertTrue(sess)

    # Check basic signature def property.
    signature_def = meta_graph_def.signature_def
    self.assertEqual(len(signature_def), 2)
    self.assertEqual(
        signature_def[signature_constants.REGRESS_METHOD_NAME].method_name,
        signature_constants.REGRESS_METHOD_NAME)
    signature = signature_def["tensorflow/serving/regress"]
    asset_path = os.path.join(base_path, saved_model_constants.ASSETS_DIRECTORY)
    with sess.as_default():
      output1 = sess.run(["filename_tensor:0"])
      self.assertEqual(["foo.txt"], output1)


if __name__ == "__main__":
  tf.test.main()
