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
from typing import Callable, Collection, Dict, Mapping, Optional, Sequence, Tuple
import uuid
from absl import logging

import numpy as np

# pylint: disable=invalid-import-order,g-bad-import-order
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import

from tensorflow.compiler.mlir.quantization.tensorflow.python import pywrap_quantize_model as quantize_model_wrapper
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as repr_dataset
from tensorflow.compiler.mlir.quantization.tensorflow.python import save_model
from tensorflow.compiler.mlir.quantization.tensorflow import exported_model_pb2
from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2 as quant_opts_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import tf_logging as logging
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

# Mapping of signature def key -> SignatureDef.
_SignatureDefMap = Mapping[str, meta_graph_pb2.SignatureDef]

# Default minimum number of elements in the weights for them to be quantized
# during dynamic range quantization (DRQ).
_DYNAMIC_RANGE_DEFAULT_MIN_NUM_ELEMENTS_FOR_WEIGHTS = 1024


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


# TODO(b/249918070): Implement a progress bar.
def _log_sample_num_for_calibration(
    representative_dataset: repr_dataset.RepresentativeDataset,
) -> repr_dataset.RepresentativeDataset:
  """Logs the sample number for calibration.

  If in debug logging level, the "sample number / total num samples" is logged
  for every 5 iterations.

  This is often useful when tracking the progress of the calibration step which
  is often slow and may look stale if there's no logs being printed.

  Args:
    representative_dataset: The representative dataset.

  Yields:
    The representative samples from `representative_dataset` without any
    modification.
  """
  num_samples: Optional[int] = repr_dataset.get_num_samples(
      representative_dataset)
  if num_samples is None:
    total_num_samples = '?'
    logging.info('Representative dataset size unknown.')
  else:
    total_num_samples = str(num_samples)
    logging.info('Using representative dataset of size: %s', total_num_samples)

  sample_num = 0
  for sample in representative_dataset:
    sample_num += 1

    # Log the sample number for every 5 iterations.
    logging.log_every_n(
        logging.DEBUG, 'Running representative sample for calibration: %d / %s',
        5, sample_num, total_num_samples)
    yield sample

  logging.info('Running representative samples complete: %d / %s', sample_num,
               total_num_samples)


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

  for sample in map(sample_validator,
                    _log_sample_num_for_calibration(representative_dataset)):
    # Create a mapping from input tensor name to the input tensor value.
    # ex) "Placeholder:0" -> [0, 1, 2]
    feed_dict = _create_feed_dict_from_input_data(sample, signature_def)
    sess.run(output_tensor_names, feed_dict=feed_dict)


def _replace_tensors_by_numpy_ndarrays(
    repr_ds_map: repr_dataset.RepresentativeDatasetMapping) -> None:
  """Replaces tf.Tensors by their evaluated numpy arrays.

  This assumes that tf.Tensors in representative samples are created in the
  default Graph. It will raise an error if tensors are created in a different
  graph.

  Args:
    repr_ds_map: SignatureDef key -> RepresentativeDataset mapping.
  """
  with session.Session() as sess:
    for signature_def_key in repr_ds_map:
      # Replaces the dataset with a new dataset where tf.Tensors are replaced
      # by their evaluated values.
      ds = repr_ds_map[signature_def_key]
      repr_ds_map[signature_def_key] = (
          repr_dataset.replace_tensors_by_numpy_ndarrays(ds, sess))


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
  # Replace tf.Tensors by numpy ndarrays in order to reuse the samples in a
  # different graph when running the calibration.
  _replace_tensors_by_numpy_ndarrays(representative_dataset_map)

  # Run the calibration in a new graph to avoid name collision, which could
  # happen when the same model is loaded multiple times in the default graph.
  with ops.Graph().as_default(), session.Session() as sess:
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

  for sample in map(sample_validator,
                    _log_sample_num_for_calibration(representative_dataset)):
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

  logging.info('Calibration step complete.')


def _run_static_range_qat(
    saved_model_path: str, signature_def_keys: Sequence[str],
    tags: Collection[str], quant_opts: quant_opts_pb2.QuantizationOptions
) -> Tuple[graph_pb2.GraphDef, str]:
  """Runs static-range quantization for a Quantization-Aware Trained model.

  Runs the quantization for a model trained using QAT.

  Args:
    saved_model_path: Path to SavedModel.
    signature_def_keys: Keys of the signatures of the functions that are the
      target for quantization.
    tags: Tags identifying the MetaGraphDef.
    quant_opts: Quantization options.

  Returns:
    (graph, init_node_name), where graph is the static-range quantized graph and
    init_node_name is the name of the initializer op, which is fetched once
    during model load to initialize resources (e.g. hash tables).
  """
  logging.info('Running static-range quantization for QAT model.')
  exported_model_serialized = (
      quantize_model_wrapper.quantize_qat_model(saved_model_path,
                                                ','.join(signature_def_keys),
                                                ','.join(tags),
                                                quant_opts.SerializeToString()))

  exported_model = exported_model_pb2.ExportedModel.FromString(
      exported_model_serialized)

  return exported_model.graph_def, exported_model.init_node_name


def _add_calibration_statistics(graph_def: graph_pb2.GraphDef) -> None:
  """Adds calibration statistics to the graph def.

  This function must be run after running the graph with a representative
  dataset. Retrieves calibration statistics from the global calibrator and adds
  them to the corresponding nodes as attributes.

  Args:
    graph_def: GraphDef to add calibration statistics to.
  """
  for function_def in graph_def.library.function:
    for node_def in function_def.node_def:
      if node_def.op != 'CustomAggregator':
        continue

      node_id = node_def.attr['id'].s
      try:
        min_val = quantize_model_wrapper.get_min_from_calibrator(node_id)
        max_val = quantize_model_wrapper.get_max_from_calibrator(node_id)
        quantize_model_wrapper.clear_data_from_calibrator(node_id)
        node_def.attr['min'].f = float(min_val)
        node_def.attr['max'].f = float(max_val)
      except ValueError:
        logging.warn(
            'CustomAggregator id "%s" from FunctionDef "%s" does not have '
            'min or max values. Parts of this function are not quantized.',
            node_id.decode('utf-8'), function_def.signature.name)


def _run_static_range_ptq(
    saved_model_path: str,
    signature_def_keys: Sequence[str],
    tags: Collection[str],
    quant_opts: quant_opts_pb2.QuantizationOptions,
    representative_dataset: repr_dataset.RepresentativeDatasetOrMapping,
    signature_def_map: _SignatureDefMap,
) -> Tuple[graph_pb2.GraphDef, _SignatureDefMap, str]:
  """Runs static-range Post-Training Quantization.

  Runs static-range PTQ for the model. Runs the calibration step with
  `representative_dataset` to collect statistics required for quantization. This
  produces the quantized GraphDef along with the SignatureDefs which might have
  been modified according to the changes in the graph.

  Args:
    saved_model_path: Path to SavedModel.
    signature_def_keys: Keys of the signature defs of the functions that are the
      target for quantization.
    tags: Tags to identify the MetaGraphDef to be used for quantization.
    quant_opts: Quantization options.
    representative_dataset: Representative dataset used for the calibration
      step. Representative datasets should exist for each signature def key in
      `signature_def_keys`.
    signature_def_map: Signature def key -> SignatureDef mapping.

  Raises:
    ValueError if the graph doesn't contain a valid signature.

  Returns:
    (graph_def, signature_def_map, init_op_name) where graph_def is the
    quantized graph and
    the signature_def_map contains the SignatureDefs, possibly modified
    according to the quantized graph to match the original signature defs.
    init_op_name is the name of the initializer op, which is fetched once to
    initialize resources (e.g. hash tables) when a SavedModel is loaded.
  """
  logging.info('Running post-training quantization pre-calibration step.')
  exported_model_serialized = (
      quantize_model_wrapper.quantize_ptq_model_pre_calibration(
          saved_model_path, ','.join(signature_def_keys), ','.join(tags),
          quant_opts.SerializeToString()))

  exported_model = exported_model_pb2.ExportedModel.FromString(
      exported_model_serialized)

  graph_def = exported_model.graph_def
  for function_def in graph_def.library.function:
    for node_def in function_def.node_def:
      if node_def.op == 'CustomAggregator':
        node_def.attr['id'].s = uuid.uuid4().hex.encode('ascii')

  float_model_dir = tempfile.mkdtemp()
  save_model.save_model_v1(graph_def, float_model_dir, signature_def_map, tags,
                           exported_model.init_node_name)

  # Uses the representative dataset to collect statistics for calibration.
  # Handles the graph mode execution separately in case TF2 is disabled or
  # eager execution is disabled. The min & max values are stored separately
  # in a global CalibratorSingleton instance.
  _run_graph_for_calibration(float_model_dir, signature_def_keys, tags,
                             representative_dataset)
  _add_calibration_statistics(graph_def)

  calibrated_model_dir = tempfile.mkdtemp()
  save_model.save_model_v1(graph_def, calibrated_model_dir, signature_def_map,
                           tags, exported_model.init_node_name)

  logging.info('Running post-training quantization post-calibration step.')
  exported_model_serialized = (
      quantize_model_wrapper.quantize_ptq_model_post_calibration(
          calibrated_model_dir, ','.join(signature_def_keys), ','.join(tags),
          quant_opts.SerializeToString()))

  exported_model = exported_model_pb2.ExportedModel.FromString(
      exported_model_serialized)

  return (exported_model.graph_def, signature_def_map,
          exported_model.init_node_name)


def _static_range_quantize(
    saved_model_path: str,
    signature_keys: Sequence[str],
    tags: Collection[str],
    output_directory: str,
    quantization_options: quant_opts_pb2.QuantizationOptions,
    representative_dataset: Optional[
        repr_dataset.RepresentativeDatasetOrMapping] = None
) -> autotrackable.AutoTrackable:
  """Quantizes the given SavedModel via static range quantization.

  If the model is not trained with Quantization-Aware Training (QAT) technique,
  it requires `representative_dataset` to collect statistics required for
  quantization. If non-None `representative_dataset` is provided with a QAT
  model input, `representative_dataset` will be ignored.

  Args:
    saved_model_path: Path to the saved model. When representative_dataset is
      not provided, this should be a model trained with QAT.
    signature_keys: Sequence of keys identifying SignatureDef containing inputs
      and outputs.
    tags: Collection of tags identifying the MetaGraphDef within the SavedModel
      to analyze.
    output_directory: The path to save the output SavedModel. The directory will
      be overwritten if not empty.
    quantization_options: QuantizationOptions proto describing quantization
      related config.
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
  logging.info('Running static range quantization on model: %s',
               saved_model_path)
  logging.info('Using SignatureDef keys: %s', signature_keys)
  logging.info('Using tags: %s', tags)
  logging.info('QuantizationOptions: \n%s', quantization_options)

  is_qat_saved_model = _is_qat_saved_model(saved_model_path)
  signature_def_map = save_model.get_signatures_from_saved_model(
      saved_model_path, signature_keys, tags)

  # Checks if the model is from QAT
  if representative_dataset is None and not is_qat_saved_model:
    raise ValueError(
        'When `representative_dataset` is not provided, the model should be '
        'trained with quantization-aware training (QAT).')
  if quantization_options.min_num_elements_for_weights > 0:
    logging.warn(
        'min_num_elements_for_weights is set but is not supported for the '
        'Post-training static range quantization. '
        'The flag is ignored.')

  if is_qat_saved_model:
    graph_def, init_node_name = _run_static_range_qat(saved_model_path,
                                                      signature_keys, tags,
                                                      quantization_options)
  else:
    graph_def, signature_def_map, init_node_name = _run_static_range_ptq(
        saved_model_path, signature_keys, tags, quantization_options,
        representative_dataset, signature_def_map)

  save_model.save_model_v1(
      graph_def,
      output_directory,
      signature_def_map,
      tags,
      init_op_name=init_node_name)

  return saved_model_load(output_directory)


def _dynamic_range_quantize(
    saved_model_path: str,
    signature_keys: Sequence[str],
    tags: Collection[str],
    output_directory: str,
    quantization_options: quant_opts_pb2.QuantizationOptions,
) -> autotrackable.AutoTrackable:
  """Quantizes the given SavedModel via post-training dynamic range quantization.

  Weight-only quantization also uses this path.

  Args:
    saved_model_path: Path to the saved model.
    signature_keys: Sequence of keys identifying SignatureDef containing inputs
      and outputs.
    tags: Collection of tags identifying the MetaGraphDef within the SavedModel
      to analyze.
    output_directory: The path to save the output SavedModel. The directory will
      be overwritten if not empty.
    quantization_options: QuantizationOptions proto describing quantization
      related config.

  Returns:
    A SavedModel object with TF quantization applied.

  Raises:
    ValueError: when the model is QAT model.
  """
  if (quantization_options.quantization_method.experimental_method ==
      _ExperimentalMethod.WEIGHT_ONLY):
    mode_str = 'weight-only quantization'
  else:
    mode_str = 'dynamic-range quantization'
  if _is_qat_saved_model(saved_model_path):
    raise ValueError(
        'The models trained with quantization-aware training (QAT) is not '
        'supported for %s.' % mode_str)

  logging.info('Running post-training %s on model: %s', mode_str,
               saved_model_path)
  logging.info('Using SignatureDef keys: %s', signature_keys)
  logging.info('Using tags: %s', tags)
  logging.info('QuantizationOptions: \n%s', quantization_options)

  # Check default quantization option values for post-training dynamic range
  # quantization case.
  # TODO(b/242805842): Find good minimum_elements_for_weights number for server.
  # please also update default value in tflite converter:
  # tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.cc;l=201
  if quantization_options.min_num_elements_for_weights == 0:
    (quantization_options.min_num_elements_for_weights
    ) = _DYNAMIC_RANGE_DEFAULT_MIN_NUM_ELEMENTS_FOR_WEIGHTS
    logging.warn(
        'QuantizationOptions.min_num_elements_for_weights is not set (0). '
        'Setting to the default value: %s.',
        _DYNAMIC_RANGE_DEFAULT_MIN_NUM_ELEMENTS_FOR_WEIGHTS)

  # Apply post-training dynamic range quantization to the model.
  exported_model_serialized = (
      quantize_model_wrapper.quantize_ptq_dynamic_range(
          saved_model_path, ','.join(signature_keys), ','.join(tags),
          quantization_options.SerializeToString()))

  exported_model = exported_model_pb2.ExportedModel.FromString(
      exported_model_serialized)
  signature_def_map = save_model.get_signatures_from_saved_model(
      saved_model_path, signature_keys, tags)

  save_model.save_model_v1(
      exported_model.graph_def,
      output_directory,
      signature_def_map,
      tags=tags,
      init_op_name=exported_model.init_node_name)

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


def _populate_quantization_options_default_values(
    quantization_options: quant_opts_pb2.QuantizationOptions) -> None:
  """Populates default values for QuantizationOptions.

  Populates unspecified or unset fields of QuantizationOptions with the default
  values.

  * If `op_set` is unspecified, it defaults to `OpSet.TF`.
  * If `freeze_all_variables` is not set, it defaults to `True`.
  * Check if configurations are set correctly:
    - Per-channel quantization is supported for Uniform Quantized opset only.

  Args:
    quantization_options: An instance of QuantizationOptions.
  """
  if quantization_options.op_set == quant_opts_pb2.OpSet.OP_SET_UNSPECIFIED:
    quantization_options.op_set = quant_opts_pb2.OpSet.TF

  if not quantization_options.HasField('freeze_all_variables'):
    quantization_options.freeze_all_variables.enabled = True

  if quantization_options.enable_per_channel_quantization and (
      quantization_options.op_set != quant_opts_pb2.OpSet.UNIFORM_QUANTIZED):
    raise ValueError(
        'Currently, per-channel quantization is supported for Uniform '
        'Quantized opset only.')

  if (quantization_options.quantization_method.experimental_method
      == _ExperimentalMethod.WEIGHT_ONLY and
      quantization_options.op_set == quant_opts_pb2.OpSet.UNIFORM_QUANTIZED):
    raise ValueError('Uniform quantized opset does not support weight-only.')


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
) -> autotrackable.AutoTrackable:
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
    quantization_options: A set of options for quantization. If None, it uses
      post-training static range quantization with TF opset by default.
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

  # Set default values for None arguments.
  if output_directory is None:
    output_directory = tempfile.mkdtemp()

  if quantization_options is None:
    quantization_options = quant_opts_pb2.QuantizationOptions()

  _populate_quantization_options_default_values(quantization_options)

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
                                    output_directory, quantization_options,
                                    representative_dataset)
    elif (method.experimental_method == _ExperimentalMethod.DYNAMIC_RANGE or
          method.experimental_method == _ExperimentalMethod.WEIGHT_ONLY):
      return _dynamic_range_quantize(saved_model_path, signature_keys, tags,
                                     output_directory, quantization_options)
    else:
      raise NotImplementedError(
          'Experimental quantization method {method.experimental_method}'
          ' is not implemented.')
  else:
    logging.debug(
        'Neither "method" nor "experimental_method" for QuantizationMethod '
        'is specified. Static range quantization is used by default.')
    return _static_range_quantize(saved_model_path, signature_keys, tags,
                                  output_directory, quantization_options,
                                  representative_dataset)
