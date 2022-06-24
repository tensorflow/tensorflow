# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Defines TF Quantization API from SavedModel to SavedModel."""

import tempfile
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union
import uuid
import warnings

import numpy as np

# pylint: disable=invalid-import-order,g-bad-import-order
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import

from tensorflow.compiler.mlir.quantization.tensorflow.python import pywrap_quantize_model as quantize_model_wrapper
from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2 as quant_opts_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import loader_impl as saved_model_loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.load import load as saved_model_load
from tensorflow.python.trackable import autotrackable
from tensorflow.python.types import core

# The signature key of the saved model init op.
_INIT_OP_SIGNATURE_KEY = '__saved_model_init_op'

# Type aliases for quant_opts_pb2 messages.
_Method = quant_opts_pb2.QuantizationMethod.Method
_ExperimentalMethod = quant_opts_pb2.QuantizationMethod.ExperimentalMethod

# Types required for representative dataset. A representative dataset should
# be a callable that returns an iterable of representative samples:
# A representative sample should be either:
# 1. (signature_key, {input_name -> input_tensor}) tuple, or
# 2. {input_name -> input_tensor} mappings.
_RepresentativeSample = Union[Tuple[str, Mapping[str, core.Tensor]],
                              Mapping[str, core.Tensor]]
_RepresentativeDataset = Callable[[], Iterable[_RepresentativeSample]]


def _legalize_tensor_name(tensor_name: str) -> str:
  """Converts tensor name from 'name:index' to 'name__index' format."""
  return tensor_name.replace(':', '__')


def _is_qat_saved_model(saved_model_path: str):
  """Checks if the SavedModel is QAT-enabled by looking for 'FakeQuant' ops."""
  saved_model_proto = saved_model_loader.parse_saved_model(saved_model_path)
  for meta_graph in saved_model_proto.meta_graphs:
    if any(
        node.op.startswith('FakeQuant') for node in meta_graph.graph_def.node):
      return True
    for function in meta_graph.graph_def.library.function:
      if any(node.op.startswith('FakeQuant') for node in function.node_def):
        return True
  return False


def _get_signatures_from_saved_model(saved_model_path: str,
                                     signature_keys=None,
                                     tags=None):
  """Gets a map from signature keys to their SignatureDef from a saved model."""
  if tags is None:
    tags = set([tag_constants.SERVING])

  loader = saved_model_loader.SavedModelLoader(saved_model_path)
  try:
    meta_graphdef = loader.get_meta_graph_def_from_tags(tags)
  except RuntimeError as runtime_error:
    raise RuntimeError(
        f'Failed to retrieve MetaGraphDef with tags {tags}'
        f' from a SavedModel in {saved_model_path}.') from runtime_error

  signatures = {}
  for key, signature_def in meta_graphdef.signature_def.items():
    if key == _INIT_OP_SIGNATURE_KEY:
      continue
    if signature_keys is not None and key not in signature_keys:
      continue
    signatures[key] = signature_def

  return signatures


def _fix_tensor_names(signatures, exported_graph):
  """Tries fixing tensor names in the signatures to match the exported graph.

  The output tensor names in the original graph usually become names of the
  return nodes in the exported graph. This function tries to fix that and checks
  if the input tensor names are found in the exported graph.

  Args:
    signatures: the signatures of the original graph.
    exported_graph: The PTQ-exported GraphDef.

  Returns:
    Fixed signatures or None if it couldn't be fixed.
  """
  if signatures is None:
    return None

  # The InsertMainFunctionPass populates input and output nodes of the newly
  # inserted main function with "tf_saved_model.index_path" attributes. These
  # attributes can be used to identify outputs in the exported graph.
  output_index_path_map = {}
  for op in exported_graph.get_operations():
    if (op.type == '_Retval' and
        op.get_attr('tf_saved_model.index_path') is not None):
      index_path_name = op.get_attr('tf_saved_model.index_path')[0]
      index_path_name = index_path_name.decode('utf-8')
      output_index_path_map[index_path_name] = op.inputs[0].name

  for signature_def in signatures.values():
    for tensor_info in signature_def.inputs.values():
      try:
        exported_graph.get_tensor_by_name(tensor_info.name)
      except KeyError:
        # If input tensors are not found, the signatures can't be used for the
        # exported graph.
        warnings.warn('Cannot find the tensor with name %s in the graph.' %
                      tensor_info.name)
        return None

    for tensor_info in signature_def.outputs.values():
      try:
        if tensor_info.name in output_index_path_map:
          tensor_info.name = output_index_path_map[tensor_info.name]
        else:
          # Tries to find the return node with the given name and use its input
          # as the output tensor name.
          return_node = exported_graph.get_operation_by_name(
              _legalize_tensor_name(tensor_info.name))
          tensor_info.name = return_node.inputs[0].name
      except KeyError:
        warnings.warn(
            'Cannot find the tensor or node with name %s in the graph.' %
            tensor_info.name)
        return None

  return signatures


def _get_signature_key_and_input(
    representative_sample: _RepresentativeSample,
    signature_keys: List[str],
) -> Tuple[str, Mapping[str, core.Tensor]]:
  """Gets the signature key and input data from `representative_sample`.

  The `representative_sample` can be in two formats:

  1. A tuple of: (signature_key, {input_name -> input_tensor})
  2. A dict: {input_name -> input_tensor}.

  (2) assumes the signature_key to be the default signature key (first item in
  `signature_keys`).

  Args:
    representative_sample: A single sample from the representative dataset, used
      for calibration.
    signature_keys: A list of signature keys that identifies a function to run
      the data samples with. When the `representative_sample` is provided as a
      `dict`, it should have a single item.

  Returns:
    signature_key: Signature key that indicates the function to be used for the
      returned input data.
    input data: A input_name -> input_tensor mapping (dict).

  Raises:
    ValueError: When the format of `representative_sample` is invalid, or when
    the length of `signature_keys` not 1 when `representative_sample` is `dict`.
  """
  # TODO(b/214311251): Add a test case with multiple signatures.
  if isinstance(representative_sample, tuple):
    if (not isinstance(representative_sample[1], dict) or
        len(representative_sample) != 2):
      raise ValueError('You need to provide a dictionary with input '
                       'names and values in the second argument in the '
                       'tuple')
    return representative_sample
  elif isinstance(representative_sample, dict):
    if len(signature_keys) > 1:
      raise ValueError('When the model has multiple signatures, you need '
                       'to provide a tuple with signature key and a '
                       'dictionary with input names and values')
    return signature_keys[0], representative_sample
  else:
    raise ValueError('You need to provide either a dictionary with input '
                     'names and values or a tuple with signature key and a '
                     'dictionary with input names and values')


def _create_feed_dict_from_input_data(
    input_data: Mapping[str, core.Tensor],
    signature_def: meta_graph_pb2.SignatureDef) -> Dict[str, np.ndarray]:
  """Constructs a feed_dict from input data.

  Note: This function should only be used in graph mode.

  This is a helper function that converts an 'input key -> input tensor' mapping
  to a feed dict. A feed dict is an 'input tensor name -> input data' mapping
  and can be directly passed to the `feed_dict` argument of `sess.run()`.

  Args:
    input_data: Input key -> input tensor mapping. The input keys should match
      the input keys of `signature_def`.
    signature_def: A SignatureDef representing the function that `input_data` is
      an input to.

  Raises:
    KeyError: When the input key provided from `input_data` does not exist as
      one of `signature_def`'s input keys.

  Returns:
    Feed dict, which is intended to be used as input for `sess.run`. It is
    essentially a mapping: input tensor name -> tensor data.
  """
  feed_dict = {}
  for input_key, input_tensor in input_data.items():
    if input_key not in signature_def.inputs:
      raise KeyError(f"Invalid input key '{input_key}'. Available input keys"
                     f' are: {list(signature_def.inputs.keys())}.')

    input_tensor_name = signature_def.inputs[input_key].name
    feed_dict[input_tensor_name] = input_tensor.eval()

  return feed_dict


def _run_graph_for_calibration_graph_mode(
    model_dir: str, signature_keys: List[str], tags: Set[str],
    representative_dataset: _RepresentativeDataset) -> None:
  """Runs the graph for calibration in graph mode.

  This function assumes _graph mode_ (used when legacy TF1 is used or when eager
  mode is explicitly disabled) when running the graph. This step is used in
  order to collect the statistics in CustomAggregatorOp for quantization using
  the representative dataset for the actual data provided for inference.

  Args:
    model_dir: Path to SavedModel directory.
    signature_keys: A list of signature keys that identifies a function to run
      the data samples with.
    tags: Set of tags identifying the MetaGraphDef within the SavedModel.
    representative_dataset: Representative dataset used for calibration.

  Raises:
    ValueError: When the samples in representative dataset is invalid.
  """
  with session.Session() as sess:
    meta_graph: meta_graph_pb2.MetaGraphDef = saved_model_loader.load(
        sess, tags, export_dir=model_dir)

    for sample in representative_dataset():
      signature_key, input_data = _get_signature_key_and_input(
          sample, signature_keys)

      sig_def = meta_graph.signature_def[signature_key]
      output_tensor_names = [
          output_tensor_info.name
          for output_tensor_info in sig_def.outputs.values()
      ]

      # Create a mapping from input tensor name to the input tensor value.
      # ex) "Placeholder:0" -> [0, 1, 2]
      try:
        feed_dict = _create_feed_dict_from_input_data(input_data, sig_def)
      except KeyError as key_error:
        raise ValueError(f'Invalid input data for signature: {signature_key}.'
                        ) from key_error

      sess.run(output_tensor_names, feed_dict=feed_dict)


def _run_graph_for_calibration_eager_mode(
    model_dir: str, signature_keys: List[str], tags: Set[str],
    representative_dataset: _RepresentativeDataset) -> None:
  """Runs the graph for calibration in eager mode.

  This function assumes _eager mode_ (enabled in TF2 by default) when running
  the graph. This step is used in order to collect the statistics in
  CustomAggregatorOp for quantization using the representative dataset for the
  actual data provided for inference.

  Args:
    model_dir: Path to SavedModel directory.
    signature_keys: A list of signature keys that identifies a function to run
      the data samples with.
    tags: Set of tags identifying the MetaGraphDef within the SavedModel.
    representative_dataset: Representative dataset used for calibration.

  Raises:
    ValueError: When the samples in representative dataset is invalid.
  """
  root: autotrackable.AutoTrackable = saved_model_load(model_dir, tags)
  for sample in representative_dataset():
    signature_key, input_data = _get_signature_key_and_input(
        sample, signature_keys)

    func = root.signatures[signature_key]
    try:
      func(**input_data)
    except Exception as ex:
      raise ValueError(
          f'Failed to run the function with signature key: {signature_key}'
      ) from ex


def _static_range_quantize(
    saved_model_path: str,
    signature_keys: List[str],
    tags: Set[str],
    output_directory: str,
    representative_dataset: Optional[_RepresentativeDataset] = None) ->...:
  """Quantizes the given SavedModel via static range quantization.

  Args:
    saved_model_path: Path to the saved model. When representative_dataset is
      not provided, this should be a model trained with QAT.
    signature_keys: List of keys identifying SignatureDef containing inputs and
      outputs.
    tags: Set of tags identifying the MetaGraphDef within the SavedModel to
      analyze.
    output_directory: The path to save the output SavedModel (must be an empty
      directory).
    representative_dataset: a generator that returns a dictionary in
      {input_name: input_tensor} format or a tuple with signature key and a
      dictionary in {input_name: input_tensor} format that feeds calibration
      data for quantizing model. This should be provided when the model is not a
      QAT model.

  Returns:
    A SavedModel object with TF quantization applied.

  Raises:
    ValueError: when representative_dataset is not provided for non-QAT model.
    RuntimeError: When a MetaGraphDef could not be found associated with `tags`
      in the SavedModel.
  """
  is_qat_saved_model = _is_qat_saved_model(saved_model_path)
  signatures = _get_signatures_from_saved_model(saved_model_path,
                                                signature_keys, tags)

  # Checks if the model is from QAT
  if representative_dataset is None and not is_qat_saved_model:
    raise ValueError(
        'When `representative_dataset` is not provided, the model should be '
        'trained with quantization-aware training (QAT).')

  if is_qat_saved_model:
    # Handle QAT models are supported.
    graph_def_serialized = (
        quantize_model_wrapper.quantize_qat_model(saved_model_path,
                                                  ','.join(signature_keys),
                                                  ','.join(tags)))
  else:
    # Handle PTQ models are supported with mocking calibration.
    graph_def_serialized = (
        quantize_model_wrapper.quantize_ptq_model_pre_calibration(
            saved_model_path, ','.join(signature_keys), ','.join(tags)))

    graph_def = graph_pb2.GraphDef()
    graph_def.ParseFromString(graph_def_serialized)

    float_model_dir = tempfile.mkdtemp()
    v1_builder = builder.SavedModelBuilder(float_model_dir)

    with session.Session(graph=ops.Graph()) as sess:
      for function_def in graph_def.library.function:
        for node_def in function_def.node_def:
          if node_def.op == 'CustomAggregator':
            node_def.attr['id'].s = uuid.uuid4().hex.encode('ascii')

      importer.import_graph_def(graph_def, name='')
      working_graph = ops.get_default_graph()
      graph_def = working_graph.as_graph_def()

      signatures = _fix_tensor_names(signatures, working_graph)
      if signatures is None:
        raise ValueError(
            "The input SavedModel doesn't contain a valid signature")

      v1_builder.add_meta_graph_and_variables(
          sess, tags, signature_def_map=signatures)

    v1_builder.save()

    # Uses the representative dataset to collect statistics for calibration.
    # Handles the graph mode execution separately in case TF2 is disabled or
    # eager execution is disabled. The min & max values are stored separately
    # in a global CalibratorSingleton instance.
    try:
      if context.executing_eagerly():
        _run_graph_for_calibration_eager_mode(float_model_dir, signature_keys,
                                              tags, representative_dataset)
      else:
        _run_graph_for_calibration_graph_mode(float_model_dir, signature_keys,
                                              tags, representative_dataset)
    except Exception as ex:
      raise ValueError(
          'Failed to run graph for post-training quantization calibration.'
      ) from ex

    for function_def in graph_def.library.function:
      for node_def in function_def.node_def:
        if node_def.op == 'CustomAggregator':
          node_id = node_def.attr['id'].s
          try:
            min_val = quantize_model_wrapper.get_min_from_calibrator(node_id)
            max_val = quantize_model_wrapper.get_max_from_calibrator(node_id)
            quantize_model_wrapper.clear_data_from_calibrator(node_id)
            node_def.attr['min'].f = float(min_val)
            node_def.attr['max'].f = float(max_val)
          except ValueError:
            warnings.warn(
                f'CustomAggregator id "{node_id.decode("utf-8")}" from '
                f'FunctionDef "{function_def.signature.name}" does not have '
                'min or max values. This function may not be quantized.')

    calibrated_model_dir = tempfile.mkdtemp()
    v1_builder = builder.SavedModelBuilder(calibrated_model_dir)

    with session.Session(graph=ops.Graph()) as sess:
      importer.import_graph_def(graph_def, name='')
      working_graph = ops.get_default_graph()
      graph_def = working_graph.as_graph_def()

      v1_builder.add_meta_graph_and_variables(
          sess, tags, signature_def_map=signatures)

    v1_builder.save()
    signatures = _get_signatures_from_saved_model(calibrated_model_dir,
                                                  signature_keys, tags)

    graph_def_serialized = (
        quantize_model_wrapper.quantize_ptq_model_post_calibration(
            calibrated_model_dir,
            ','.join(signature_keys),
            ','.join(tags),
        ))

  graph_def = graph_pb2.GraphDef()
  graph_def.ParseFromString(graph_def_serialized)

  if output_directory is None:
    output_directory = tempfile.mkdtemp()
  v1_builder = builder.SavedModelBuilder(output_directory)

  with session.Session(graph=ops.Graph()) as sess:
    importer.import_graph_def(graph_def, name='')
    working_graph = ops.get_default_graph()

    signatures = _fix_tensor_names(signatures, working_graph)
    if signatures is None:
      raise ValueError("The input SavedModel doesn't contain a valid signature")

    v1_builder.add_meta_graph_and_variables(
        sess, tags, signature_def_map=signatures)

  v1_builder.save()

  return saved_model_load(output_directory)


def _dynamic_range_quantize(saved_model_path: str,
                            signature_keys: List[str],
                            tags: Set[str],
                            output_directory: str = ''):
  """Quantizes the given SavedModel via post-training dynamic range quantization.

  Args:
    saved_model_path: Path to the saved model.
    signature_keys: List of keys identifying SignatureDef containing inputs and
      outputs.
    tags: Set of tags identifying the MetaGraphDef within the SavedModel to
      analyze.
    output_directory: The path to save the output SavedModel (must be an empty
      directory).

  Returns:
    A SavedModel object with TF quantization applied.

  Raises:
    ValueError: when the model is QAT model.
  """
  is_qat_saved_model = _is_qat_saved_model(saved_model_path)
  signatures = _get_signatures_from_saved_model(saved_model_path,
                                                signature_keys, tags)

  # Checks if the model is from QAT.
  if is_qat_saved_model:
    raise ValueError(
        'The models trained with quantization-aware training (QAT) is not '
        'supported.')

  # Apply post-training dynamic range quantization to the model.
  graph_def_serialized = (
      quantize_model_wrapper.quantize_ptq_dynamic_range(
          saved_model_path, ','.join(signature_keys), ','.join(tags)))

  graph_def = graph_pb2.GraphDef()
  graph_def.ParseFromString(graph_def_serialized)

  if not output_directory:
    output_directory = tempfile.mkdtemp()
  v1_builder = builder.SavedModelBuilder(output_directory)

  with session.Session(graph=ops.Graph()) as sess:
    importer.import_graph_def(graph_def, name='')
    working_graph = ops.get_default_graph()

    signatures = _fix_tensor_names(signatures, working_graph)
    if signatures is None:
      raise ValueError("The input SavedModel doesn't contain a valid signature")

    v1_builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING], signature_def_map=signatures)

  v1_builder.save()

  return saved_model_load(output_directory)


def quantize(
    saved_model_path: str,
    signature_keys: Optional[List[str]] = None,
    tags: Optional[Iterable[str]] = None,
    output_directory: Optional[str] = None,
    quantization_options: Optional[quant_opts_pb2.QuantizationOptions] = None,
    representative_dataset: Optional[_RepresentativeDataset] = None) ->...:
  """Quantizes the given SavedModel.

  Args:
    saved_model_path: Path to the saved model. When representative_dataset is
      not provided, this should be a model trained with QAT.
    signature_keys: List of keys identifying SignatureDef containing inputs and
      outputs. If None, ["serving_default"] is used.
    tags: (TF1 SavedModel only) Set of tags identifying the MetaGraphDef within
      the SavedModel to analyze. If None, {"serve"} is used.
    output_directory: The path to save the output SavedModel (must be an empty
      directory).
    quantization_options: A set of options for quantization.
    representative_dataset: a generator that returns a dictionary in
      {input_name: input_tensor} format or a tuple with signature key and a
      dictionary in {input_name: input_tensor} format that feeds calibration
      data for quantizing model. This should be provided when the model is a PTQ
      model.

  Returns:
    A SavedModel object with TF quantization applied, or None if no quantization
    is performed.

  Raises:
    ValueError: When 1) representative_dataset is not provided for non QAT model
      for enabling static range quantization, or 2) invalid value is provided as
      a quantization method.
    NotImplementedError: When the specified quantization method is not yet
      implemented.
  """
  if tags is None:
    tags = {tag_constants.SERVING}
  if signature_keys is None:
    signature_keys = [signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

  if quantization_options is None:
    quantization_options = quant_opts_pb2.QuantizationOptions()

  method: quant_opts_pb2.QuantizationMethod = quantization_options.quantization_method
  if method.HasField('method'):
    raise ValueError(f'Invalid value for QuantizationMethod: {method.method}.')
  elif method.HasField('experimental_method'):
    if method.experimental_method == _ExperimentalMethod.STATIC_RANGE:
      return _static_range_quantize(saved_model_path, signature_keys, tags,
                                    output_directory, representative_dataset)
    elif method.experimental_method == _ExperimentalMethod.DYNAMIC_RANGE:
      return _dynamic_range_quantize(saved_model_path, signature_keys, tags,
                                     output_directory)
    else:
      raise NotImplementedError(
          'Experimental quantization method {method.experimental_method}'
          ' is not implemented.')
  else:
    logging.debug(
        'Neither "method" nor "experimental_method" for QuantizationMethod '
        'is specified. Static range quantization is used by default.')
    return _static_range_quantize(saved_model_path, signature_keys, tags,
                                  output_directory, representative_dataset)
