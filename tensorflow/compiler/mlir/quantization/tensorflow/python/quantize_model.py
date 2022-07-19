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

import collections.abc
import tempfile
from typing import Callable, Collection, Dict, Mapping, Optional, Sequence
import uuid
import warnings

import numpy as np

# pylint: disable=invalid-import-order,g-bad-import-order
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import

from tensorflow.compiler.mlir.quantization.tensorflow.python import pywrap_quantize_model as quantize_model_wrapper
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as repr_dataset
from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2 as quant_opts_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
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


def _create_sample_validator(
    expected_input_keys: Collection[str],
) -> Callable[[repr_dataset.RepresentativeSample],
              repr_dataset.RepresentativeSample]:
  """Creates a validator function for a representative sample.

  Args:
    expected_input_keys: Input keys (keyword argument names) that the function
      the sample will be used for is expecting to receive.

  Returns:
    A callable that validates a `RepresentativeSample`.
  """

  def validator(
      sample: repr_dataset.RepresentativeSample
  ) -> repr_dataset.RepresentativeSample:
    """Validates a single instance of representative sample.

    This provides a simple check for `sample` that this is a mapping of
    {input_key: input_value}.

    Args:
      sample: A `RepresentativeSample` to validate.

    Returns:
      `sample` iff it is valid.

    Raises:
      ValueError: iff the sample isn't an instance of `Mapping`.
      KeyError: iff the sample does not have the set of input keys that match
        the input keys of the function.
    """
    if not isinstance(sample, collections.abc.Mapping):
      raise ValueError('Invalid representative sample type. Provide a mapping '
                       '(usually a dict) of {input_key: input_value}. '
                       f'Got type: {type(sample)} instead.')

    if set(sample.keys()) != expected_input_keys:
      raise KeyError(
          'Invalid input keys for representative sample. The function expects '
          f'input keys of: {set(expected_input_keys)}. '
          f'Got: {set(sample.keys())}. Please provide correct input keys for '
          'representative samples.')

    return sample

  return validator


def _validate_representative_dataset(
    representative_dataset: repr_dataset.RepresentativeDatasetOrMapping,
    signature_keys: Collection[str]) -> None:
  """Validates the representative dataset, based on the signature keys.

  Representative dataset can be provided in two different forms: a single
  instance of `RepresentativeDataset` or a map of signature key to the
  corresponding `RepresentativeDataset`. These have a relationship with
  `signature_keys`.

  This function validates the following conditions:
  * If `len(signature_keys) > 1`, then `representative_dataset` should be a
    mapping where the keys exactly match the elements in `signature_keys`.
  * If `len(signature_keys) == 1`, then both a mapping and a single instance of
    `RepresentativeDataset` are allowed.
  * This function also assumes `len(signature_keys) > 0`.

  Args:
    representative_dataset: A `RepresentativeDataset` or a map of string to
      `RepresentativeDataset` to be validated.
    signature_keys: A collection of strings that contains the signature keys,
      each identifying a `SignatureDef`.

  Raises:
    ValueError: Iff `representative_dataset` does not satisfy the conditions
      above.
  """
  if isinstance(representative_dataset, collections.abc.Mapping):
    if set(signature_keys) != set(representative_dataset.keys()):
      raise ValueError(
          'The signature keys and the keys of representative dataset map '
          f'do not match. Signature keys: {set(signature_keys)}, '
          f'representative dataset map: {set(representative_dataset.keys())}.')
  else:
    if len(signature_keys) > 1:
      raise ValueError('Representative dataset is not a mapping '
                       f'(got: {type(representative_dataset)}), '
                       'but there is more than one signature key provided. '
                       'Please provide a map of {signature_key -> dataset} '
                       'with more than one signature key.')


def _convert_values_to_tf_tensors(
    sample: repr_dataset.RepresentativeSample) -> Mapping[str, core.Tensor]:
  """Converts TensorLike values of `sample` to Tensors.

  Creates a copy of `sample`, where each value is converted to Tensors
  unless it is already a Tensor.
  The values are not converted in-place (i.e. `sample` is not mutated).

  Args:
    sample: A representative sample, which is a map of {name -> tensorlike
      value}.

  Returns:
    Converted map of {name -> tensor}.
  """
  tensor_mapping = {}
  for name, tensorlike_value in sample.items():
    if isinstance(tensorlike_value, core.Tensor):
      tensor_value = tensorlike_value
    else:
      tensor_value = ops.convert_to_tensor_v2_with_dispatch(tensorlike_value)

    tensor_mapping[name] = tensor_value

  return tensor_mapping


def _create_feed_dict_from_input_data(
    input_data: repr_dataset.RepresentativeSample,
    signature_def: meta_graph_pb2.SignatureDef) -> Dict[str, np.ndarray]:
  """Constructs a feed_dict from input data.

  Note: This function should only be used in graph mode.

  This is a helper function that converts an 'input key -> input value' mapping
  to a feed dict. A feed dict is an 'input tensor name -> input value' mapping
  and can be directly passed to the `feed_dict` argument of `sess.run()`.

  Args:
    input_data: Input key -> input value mapping. The input keys should match
      the input keys of `signature_def`.
    signature_def: A SignatureDef representing the function that `input_data` is
      an input to.

  Returns:
    Feed dict, which is intended to be used as input for `sess.run`. It is
    essentially a mapping: input tensor name -> input value. Note that the input
    value in the feed dict is not a `Tensor`.
  """
  feed_dict = {}
  for input_key, input_value in input_data.items():
    input_tensor_name = signature_def.inputs[input_key].name

    value = input_value
    if isinstance(input_value, core.Tensor):
      # Take the data out of the tensor.
      value = input_value.eval()

    feed_dict[input_tensor_name] = value

  return feed_dict


def _run_function_for_calibration_graph_mode(
    sess: session.Session, signature_def: meta_graph_pb2.SignatureDef,
    representative_dataset: repr_dataset.RepresentativeDataset) -> None:
  """Runs the representative dataset through a function for calibration.

  NOTE: This is intended to be run in graph mode (TF1).

  The function is identified by the SignatureDef.

  Args:
    sess: The Session object to run the function in.
    signature_def: A SignatureDef that identifies a function by specifying the
      inputs and outputs.
    representative_dataset: The representative dataset to run through the
      function.
  """
  output_tensor_names = [
      output_tensor_info.name
      for output_tensor_info in signature_def.outputs.values()
  ]

  sample_validator = _create_sample_validator(
      expected_input_keys=signature_def.inputs.keys())
  for sample in map(sample_validator, representative_dataset):
    # Create a mapping from input tensor name to the input tensor value.
    # ex) "Placeholder:0" -> [0, 1, 2]
    feed_dict = _create_feed_dict_from_input_data(sample, signature_def)
    sess.run(output_tensor_names, feed_dict=feed_dict)


def _run_graph_for_calibration_graph_mode(
    model_dir: str,
    tags: Collection[str],
    representative_dataset_map: repr_dataset.RepresentativeDatasetMapping,
) -> None:
  """Runs the graph for calibration in graph mode.

  This function assumes _graph mode_ (used when legacy TF1 is used or when eager
  mode is explicitly disabled) when running the graph. This step is used in
  order to collect the statistics in CustomAggregatorOp for quantization using
  the representative dataset for the actual data provided for inference.

  Args:
    model_dir: Path to SavedModel directory.
    tags: Collection of tags identifying the MetaGraphDef within the SavedModel.
    representative_dataset_map: A map where signature keys are mapped to
      corresponding representative datasets.

  Raises:
    ValueError: When running the function with the representative dataset fails.
  """
  with session.Session() as sess:
    meta_graph: meta_graph_pb2.MetaGraphDef = saved_model_loader.load(
        sess, tags, export_dir=model_dir)

    for signature_key, repr_ds in representative_dataset_map.items():
      sig_def = meta_graph.signature_def[signature_key]

      try:
        _run_function_for_calibration_graph_mode(
            sess, signature_def=sig_def, representative_dataset=repr_ds)
      except Exception as ex:
        raise ValueError(
            'Failed to run representative dataset through the '
            f'function with the signature key: {signature_key}.') from ex


def _run_function_for_calibration_eager_mode(
    func: wrap_function.WrappedFunction,
    representative_dataset: repr_dataset.RepresentativeDataset) -> None:
  """Runs the representative dataset through a function for calibration.

  NOTE: This is intended to be run in eager mode (TF2).

  Args:
    func: The function to run the representative samples through.
    representative_dataset: Representative dataset used for calibration. The
      input keys and input values of the representative samples should match the
      keyword arguments of `func`.
  """
  _, keyword_args = func.structured_input_signature
  sample_validator = _create_sample_validator(
      expected_input_keys=keyword_args.keys())

  for sample in map(sample_validator, representative_dataset):
    # Convert any non-Tensor values from the sample to Tensors.
    # This conversion is required because the model saved in `model_dir` is
    # saved using TF1 SavedModelBuilder, which doesn't save the
    # SavedObjectGraph.
    # TODO(b/236795224): Remove the need for this conversion by keeping the
    # FunctionSpec (object graph) in the SavedModel. Related: b/213406917.
    func_kwargs = _convert_values_to_tf_tensors(sample)
    func(**func_kwargs)


def _run_graph_for_calibration_eager_mode(
    model_dir: str,
    tags: Collection[str],
    representative_dataset_map: repr_dataset.RepresentativeDatasetMapping,
) -> None:
  """Runs the graph for calibration in eager mode.

  This function assumes _eager mode_ (enabled in TF2 by default) when running
  the graph. This step is used in order to collect the statistics in
  CustomAggregatorOp for quantization using the representative dataset for the
  actual data provided for inference.

  Args:
    model_dir: Path to SavedModel directory.
    tags: Collection of tags identifying the MetaGraphDef within the SavedModel.
    representative_dataset_map: A map where signature keys are mapped to
      corresponding representative datasets.

  Raises:
    ValueError: When running the function with the representative dataset fails.
  """
  root: autotrackable.AutoTrackable = saved_model_load(model_dir, tags)
  for signature_key, repr_ds in representative_dataset_map.items():
    try:
      _run_function_for_calibration_eager_mode(
          func=root.signatures[signature_key], representative_dataset=repr_ds)
    except Exception as ex:
      raise ValueError(
          'Failed to run representative dataset through the '
          f'function with the signature key: {signature_key}.') from ex


def _run_graph_for_calibration(
    float_model_dir: str,
    signature_keys: Sequence[str],
    tags: Collection[str],
    representative_dataset: repr_dataset.RepresentativeDatasetOrMapping,
) -> None:
  """Runs the graph for calibration using representative datasets.

  Args:
    float_model_dir: Path to the model to calibrate.
    signature_keys: Sequence of keys identifying SignatureDef containing inputs
      and outputs.
    tags: Collection of tags identifying the MetaGraphDef within the SavedModel
      to analyze.
    representative_dataset: An iterator that returns a dictionary of {input_key:
      input_value} or a mapping from signature keys to such iterators. When
      `signature_keys` contains more than one signature key,
      `representative_datsaet` should be a mapping that maps each signature keys
      to the corresponding representative dataset.

  Raises:
    ValueError iff:
      * The representative dataset format is invalid.
      * It fails to run the functions using the representative datasets.
  """
  try:
    _validate_representative_dataset(representative_dataset, signature_keys)
  except Exception as ex:
    raise ValueError('Invalid representative dataset.') from ex

  # If `representative_dataset` is not a mapping, convert to a mapping for the
  # following functions to handle representative datasets more conveniently.
  representative_dataset_map = representative_dataset
  if not isinstance(representative_dataset, collections.abc.Mapping):
    # `signature_keys` is guaranteed to have only one element after the
    # validation.
    representative_dataset_map = {signature_keys[0]: representative_dataset}

  try:
    if context.executing_eagerly():
      _run_graph_for_calibration_eager_mode(float_model_dir, tags,
                                            representative_dataset_map)
    else:
      _run_graph_for_calibration_graph_mode(float_model_dir, tags,
                                            representative_dataset_map)
  except Exception as ex:
    raise ValueError(
        'Failed to run graph for post-training quantization calibration.'
    ) from ex


def _create_empty_output_dir(output_directory: str) -> None:
  """Creates the `output_directory`.

  If `output_directory` already exists, it recursively deletes all contents
  inside the directory.

  Also creates the parent & intermediate directories.

  Args:
    output_directory: Output directory.
  """
  if file_io.file_exists_v2(output_directory):
    logging.info('Deleting existing directory for quantized model output: %s .',
                 output_directory)
    file_io.delete_recursively_v2(output_directory)

  file_io.recursive_create_dir_v2(output_directory)


def _static_range_quantize(
    saved_model_path: str,
    signature_keys: Sequence[str],
    tags: Collection[str],
    output_directory: str,
    representative_dataset: Optional[
        repr_dataset.RepresentativeDatasetOrMapping] = None
) ->...:
  """Quantizes the given SavedModel via static range quantization.

  Args:
    saved_model_path: Path to the saved model. When representative_dataset is
      not provided, this should be a model trained with QAT.
    signature_keys: Sequence of keys identifying SignatureDef containing inputs
      and outputs.
    tags: Collection of tags identifying the MetaGraphDef within the SavedModel
      to analyze.
    output_directory: The path to save the output SavedModel. The directory will
      be overwritten if not empty.
    representative_dataset: a generator that returns a dictionary in {input_key:
      input_value} format or a tuple with signature key and a dictionary in
      {input_key: input_value} format that feeds calibration data for quantizing
      model. This should be provided when the model is not a QAT model.

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
    _run_graph_for_calibration(float_model_dir, signature_keys, tags,
                               representative_dataset)

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

  _create_empty_output_dir(output_directory)
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
                            signature_keys: Sequence[str],
                            tags: Collection[str], output_directory: str) ->...:
  """Quantizes the given SavedModel via post-training dynamic range quantization.

  Args:
    saved_model_path: Path to the saved model.
    signature_keys: Sequence of keys identifying SignatureDef containing inputs
      and outputs.
    tags: Collection of tags identifying the MetaGraphDef within the SavedModel
      to analyze.
    output_directory: The path to save the output SavedModel. The directory will
      be overwritten if not empty.

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

  _create_empty_output_dir(output_directory)
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


def _verify_output_dir(output_dir: Optional[str], overwrite: bool) -> None:
  """Verifies the output directory.

  Raises an error if `output_dir` is not suitable for writing the output saved
  model.

  Args:
    output_dir: Output directory.
    overwrite: An option allowing to overwrite the existing output directory if
      set to true. Does not actually create or modify the `output_dir` in this
      function.

  Raises:
    FileExistsError: Iff `output_dir` is not empty and `overwrite` is false.
  """
  dir_not_empty = (
      output_dir is not None and file_io.file_exists_v2(output_dir) and
      file_io.list_directory_v2(output_dir))

  if dir_not_empty and not overwrite:
    raise FileExistsError(f'Output directory already exists: {output_dir} . '
                          'Please set overwrite_output_directory to true to '
                          'overwrite the existing directory.')


def quantize(
    saved_model_path: str,
    signature_keys: Optional[Sequence[str]] = None,
    tags: Optional[Collection[str]] = None,
    output_directory: Optional[str] = None,
    quantization_options: Optional[quant_opts_pb2.QuantizationOptions] = None,
    representative_dataset: Optional[
        repr_dataset.RepresentativeDatasetOrMapping] = None,
    *,
    overwrite_output_directory: bool = False,
) ->...:
  """Quantizes the given SavedModel.

  Args:
    saved_model_path: Path to the saved model. When representative_dataset is
      not provided, this should be a model trained with QAT.
    signature_keys: Sequence of keys identifying SignatureDef containing inputs
      and outputs. If None, ["serving_default"] is used.
    tags: (TF1 SavedModel only) Collection of tags identifying the MetaGraphDef
      within the SavedModel to analyze. If None, {"serve"} is used.
    output_directory: The path to save the output SavedModel. Set
      `overwrite_output_directory` to `True` to overwrite any existing contents
      in the directory if not empty.
    quantization_options: A set of options for quantization.
    representative_dataset: an iterator that returns a dictionary of {input_key:
      input_value} or a tuple with signature key and a dictionary of {input_key:
      input_value} that feeds calibration data for quantizing model. This should
      be provided when the model is a PTQ model.
    overwrite_output_directory: If set to true, overwrites the output directory
      iff it isn't empty. The default value is false.

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
  _verify_output_dir(output_directory, overwrite_output_directory)
  if output_directory is None:
    output_directory = tempfile.mkdtemp()

  # Set default values for None arguments.
  if quantization_options is None:
    quantization_options = quant_opts_pb2.QuantizationOptions()
  if tags is None:
    tags = {tag_constants.SERVING}
  if signature_keys is None:
    signature_keys = [signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

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
