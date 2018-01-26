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
"""Shim for systems that need to load both SessionBundle and SavedModel.

This is intended to be used during migration to SavedModel.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.contrib.session_bundle import constants as legacy_constants
from tensorflow.contrib.session_bundle import manifest_pb2
from tensorflow.contrib.session_bundle import session_bundle
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import meta_graph
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import signature_constants


def _add_input_to_signature_def(tensor_name, map_key, signature_def):
  """Add input tensor to signature_def.

  Args:
    tensor_name: string name of tensor to add to signature_def inputs
    map_key: string key to key into signature_def inputs map
    signature_def: object of type meta_graph_pb2.SignatureDef()

  Sideffect:
    adds a TensorInfo with tensor_name to signature_def inputs map keyed with
    map_key
  """
  tensor_info = meta_graph_pb2.TensorInfo(name=tensor_name)
  signature_def.inputs[map_key].CopyFrom(tensor_info)


def _add_output_to_signature_def(tensor_name, map_key, signature_def):
  """Add output tensor to signature_def.

  Args:
    tensor_name: string name of tensor to add to signature_def outputs
    map_key: string key to key into signature_def outputs map
    signature_def: object of type meta_graph_pb2.SignatureDef()

  Sideffect:
    adds a TensorInfo with tensor_name to signature_def outputs map keyed with
    map_key
  """

  tensor_info = meta_graph_pb2.TensorInfo(name=tensor_name)
  signature_def.outputs[map_key].CopyFrom(tensor_info)


def _convert_default_signature_to_signature_def(signatures):
  """Convert default signature to object of type SignatureDef.

  Args:
    signatures: object of type manifest_pb2.Signatures()

  Returns:
    object of type SignatureDef which contains a converted version of default
    signature from input signatures object

    Returns None if signature is of generic type because it cannot be converted
    to SignatureDef.

  """
  default_signature = signatures.default_signature
  signature_def = meta_graph_pb2.SignatureDef()
  if default_signature.WhichOneof("type") == legacy_constants.REGRESSION_SIGNATURE:
    regression_signature = default_signature.regression_signature
    signature_def.method_name = signature_constants.REGRESS_METHOD_NAME
    _add_input_to_signature_def(regression_signature.input.tensor_name,
                                signature_constants.REGRESS_INPUTS,
                                signature_def)
    _add_output_to_signature_def(regression_signature.output.tensor_name,
                                 signature_constants.REGRESS_OUTPUTS,
                                 signature_def)
  elif default_signature.WhichOneof("type") == legacy_constants.CLASSIFICATION_SIGNATURE:
    classification_signature = default_signature.classification_signature
    signature_def.method_name = signature_constants.CLASSIFY_METHOD_NAME
    _add_input_to_signature_def(classification_signature.input.tensor_name,
                                signature_constants.CLASSIFY_INPUTS,
                                signature_def)
    _add_output_to_signature_def(classification_signature.classes.tensor_name,
                                 signature_constants.CLASSIFY_OUTPUT_CLASSES,
                                 signature_def)
    _add_output_to_signature_def(classification_signature.scores.tensor_name,
                                 signature_constants.CLASSIFY_OUTPUT_SCORES,
                                 signature_def)
  else:
    logging.error("Only classification and regression default signatures "
                  "are supported for up-conversion. %s is not "
                  "supported" % default_signature.WhichOneof("type"))
    return None
  return signature_def


def _convert_named_signatures_to_signature_def(signatures):
  """Convert named signatures to object of type SignatureDef.

  Args:
    signatures: object of type manifest_pb2.Signatures()

  Returns:
    object of type SignatureDef which contains a converted version of named
    signatures from input signatures object

  Raises:
    RuntimeError: if input and output named signatures are not of type
    GenericSignature
  """
  signature_def = meta_graph_pb2.SignatureDef()
  input_signature = signatures.named_signatures[
      signature_constants.PREDICT_INPUTS]
  output_signature = signatures.named_signatures[
      signature_constants.PREDICT_OUTPUTS]
  # TODO(pdudnik): what if there are other signatures? Mimic cr/140900781 once
  # it is submitted.
  if (input_signature.WhichOneof("type") != legacy_constants.GENERIC_SIGNATURE or
      output_signature.WhichOneof("type") != legacy_constants.GENERIC_SIGNATURE):
    raise RuntimeError("Named input and output signatures can only be "
                       "up-converted if they are generic signature. "
                       "Input signature type is %s, output signature type is "
                       "%s" % (input_signature.WhichOneof("type"),
                               output_signature.WhichOneof("type")))

  signature_def.method_name = signature_constants.PREDICT_METHOD_NAME
  for key, val in input_signature.generic_signature.map.items():
    _add_input_to_signature_def(val.tensor_name, key, signature_def)
  for key, val in output_signature.generic_signature.map.items():
    _add_output_to_signature_def(val.tensor_name, key, signature_def)
  return signature_def


def _convert_signatures_to_signature_defs(metagraph_def):
  """Produce default and named upconverted SignatureDef objects from Signatures.

  Args:
    metagraph_def: object of type meta_graph_pb2.MetaGraphDef containing legacy
    format Session Bundle signatures

  Returns:
    default_signature_def: object of type SignatureDef which contains an
        upconverted version of default signatures in metagraph_def
    named_signature_def: object of type SignatureDef which contains an
        upconverted version of named signatures in metagraph_def
  """

  collection_def = metagraph_def.collection_def
  signatures_proto = manifest_pb2.Signatures()
  signatures = collection_def[legacy_constants.SIGNATURES_KEY].any_list.value[0]
  signatures.Unpack(signatures_proto)

  default_signature_def = None
  named_signature_def = None
  if signatures_proto.HasField("default_signature"):
    default_signature_def = _convert_default_signature_to_signature_def(
        signatures_proto)
  if len(signatures_proto.named_signatures) > 1:
    named_signature_def = _convert_named_signatures_to_signature_def(
        signatures_proto)
  return default_signature_def, named_signature_def


def _load_saved_model_from_session_bundle_path(export_dir, target, config):
  """Load legacy TF Exporter/SessionBundle checkpoint.

  Args:
    export_dir: the directory that contains files exported by exporter.
    target: The execution engine to connect to. See target in tf.Session()
    config: A ConfigProto proto with configuration options. See config in
    tf.Session()

  Returns:
    session: a tensorflow session created from the variable files.
    metagraph_def: The `MetaGraphDef` protocol buffer loaded in the provided
    session. This can be used to further extract signature-defs,
    collection-defs, etc.
    This model is up-converted to SavedModel format. Specifically, metagraph_def
    SignatureDef field is populated with Signatures converted from legacy
    signatures contained within CollectionDef

  Raises:
    RuntimeError: If metagraph already contains signature_def and cannot be
    up-converted.
  """

  meta_graph_filename = os.path.join(export_dir,
                                     legacy_constants.META_GRAPH_DEF_FILENAME)

  metagraph_def = meta_graph.read_meta_graph_file(meta_graph_filename)
  if metagraph_def.signature_def:
    raise RuntimeError("Legacy graph contains signature def, unable to "
                       "up-convert.")

  # Add SignatureDef to metagraph.
  default_signature_def, named_signature_def = (
      _convert_signatures_to_signature_defs(metagraph_def))
  if default_signature_def:
    metagraph_def.signature_def[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].CopyFrom(
            default_signature_def)
  if named_signature_def:
    signature_def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    if default_signature_def:
      signature_def_key += "_from_named"
    metagraph_def.signature_def[signature_def_key].CopyFrom(named_signature_def)

  # We cannot just output session we loaded with older metagraph_def and
  # up-converted metagraph definition because Session has an internal object of
  # type Graph which is populated from meta_graph_def. If we do not create
  # session with our new meta_graph_def, then Graph will be out of sync with
  # meta_graph_def.
  sess, metagraph_def = session_bundle.load_session_bundle_from_path(
      export_dir, target, config, meta_graph_def=metagraph_def)
  return sess, metagraph_def


def load_session_bundle_or_saved_model_bundle_from_path(export_dir,
                                                        tags=None,
                                                        target="",
                                                        config=None):
  """Load session bundle from the given path.

  The function reads input from the export_dir, constructs the graph data to the
  default graph and restores the parameters for the session created.

  Args:
    export_dir: the directory that contains files exported by exporter.
    tags: Set of string tags to identify the required MetaGraphDef when model is
          saved as SavedModel. These should correspond to the tags used when
          saving the variables using the SavedModel `save()` API.
    target: The execution engine to connect to. See target in tf.Session()
    config: A ConfigProto proto with configuration options. See config in
            tf.Session()

  Returns:
    session: a tensorflow session created from the variable files.
    meta_graph: a meta graph proto saved in the exporter directory.

  Raises:
    RuntimeError: if the required files are missing or contain unrecognizable
    fields, i.e. the exported model is invalid.
  """
  metagraph_def = None
  sess = None
  if loader.maybe_saved_model_directory(export_dir):
    sess = session.Session(target, graph=None, config=config)
    metagraph_def = loader.load(sess, tags, export_dir)
  elif session_bundle.maybe_session_bundle_dir(export_dir):
    sess, metagraph_def = _load_saved_model_from_session_bundle_path(export_dir,
                                                                     target,
                                                                     config)
  else:
    raise RuntimeError("SessionBundle or SavedModelBundle not found at "
                       "specified export location: %s" % export_dir)

  return sess, metagraph_def
