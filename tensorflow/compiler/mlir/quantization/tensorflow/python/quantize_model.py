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

from absl import logging
import numpy as np

from tensorflow.compiler.mlir.quantization.tensorflow import exported_model_pb2
from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2 as quant_opts_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import calibration_algorithm
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import calibration_statistics_pb2 as calib_stats_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.python import pywrap_quantize_model
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as repr_dataset
from tensorflow.compiler.mlir.quantization.tensorflow.python import save_model
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import load as saved_model_load
from tensorflow.python.saved_model import loader_impl as saved_model_loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.trackable import autotrackable
from tensorflow.python.types import core
from tensorflow.python.util import tf_export

# Type aliases for quant_opts_pb2 messages.
_QuantizationOptions = tf_export.tf_export(
    'quantization.experimental.QuantizationOptions'
)(quant_opts_pb2.QuantizationOptions)

_QuantizationMethod = tf_export.tf_export(
    'quantization.experimental.QuantizationMethod'
)(quant_opts_pb2.QuantizationMethod)

_QuantizationComponentSpec = tf_export.tf_export(
    'quantization.experimental.QuantizationComponentSpec'
)(quant_opts_pb2.QuantizationComponentSpec)

_UnitWiseQuantizationSpec = tf_export.tf_export(
    'quantization.experimental.UnitWiseQuantizationSpec'
)(quant_opts_pb2.UnitWiseQuantizationSpec)

_PresetMethod = _QuantizationMethod.PresetMethod
_CalibrationMethod = quant_opts_pb2.CalibrationOptions.CalibrationMethod

_QuantizationComponent = _QuantizationComponentSpec.QuantizationComponent
_TensorType = _QuantizationComponentSpec.TensorType

# Mapping of signature def key -> SignatureDef.
_SignatureDefMap = Mapping[str, meta_graph_pb2.SignatureDef]

# Default minimum number of elements in the weights for them to be quantized
# during dynamic range quantization (DRQ) and weight-only quantization.
_DYNAMIC_RANGE_DEFAULT_MIN_NUM_ELEMENTS_FOR_WEIGHTS = 1024

# Name of the saved model assets directory.
_ASSETS_DIR = 'assets'
_ASSETS_EXTRA_DIR = 'assets.extra'


def _is_qat_saved_model(saved_model_path: str):
  """Checks if the SavedModel is QAT-enabled by looking for 'FakeQuant' ops."""
  saved_model_proto = saved_model_loader.parse_saved_model(saved_model_path)
  for meta_graph in saved_model_proto.meta_graphs:
    if any(
        node.op.startswith('FakeQuant') for node in meta_graph.graph_def.node
    ):
      return True
    for function in meta_graph.graph_def.library.function:
      if any(node.op.startswith('FakeQuant') for node in function.node_def):
        return True
  return False


def _create_sample_validator(
    expected_input_keys: Collection[str],
) -> Callable[
    [repr_dataset.RepresentativeSample], repr_dataset.RepresentativeSample
]:
  """Creates a validator function for a representative sample.

  Args:
    expected_input_keys: Input keys (keyword argument names) that the function
      the sample will be used for is expecting to receive.

  Returns:
    A callable that validates a `RepresentativeSample`.
  """

  def validator(
      sample: repr_dataset.RepresentativeSample,
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
      raise ValueError(
          'Invalid representative sample type. Provide a mapping '
          '(usually a dict) of {input_key: input_value}. '
          f'Got type: {type(sample)} instead.'
      )

    if set(sample.keys()) != expected_input_keys:
      raise KeyError(
          'Invalid input keys for representative sample. The function expects '
          f'input keys of: {set(expected_input_keys)}. '
          f'Got: {set(sample.keys())}. Please provide correct input keys for '
          'representative samples.'
      )

    return sample

  return validator


def _validate_representative_dataset(
    representative_dataset: repr_dataset.RepresentativeDatasetOrMapping,
    signature_keys: Collection[str],
) -> None:
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
          f'representative dataset map: {set(representative_dataset.keys())}.'
      )
  else:
    if len(signature_keys) > 1:
      raise ValueError(
          'Representative dataset is not a mapping '
          f'(got: {type(representative_dataset)}), '
          'but there is more than one signature key provided. '
          'Please provide a map of {signature_key -> dataset} '
          'with more than one signature key.'
      )


def _convert_values_to_tf_tensors(
    sample: repr_dataset.RepresentativeSample,
) -> Mapping[str, core.Tensor]:
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
      tensor_value = tensor_conversion.convert_to_tensor_v2_with_dispatch(
          tensorlike_value
      )

    tensor_mapping[name] = tensor_value

  return tensor_mapping


def _create_feed_dict_from_input_data(
    input_data: repr_dataset.RepresentativeSample,
    signature_def: meta_graph_pb2.SignatureDef,
) -> Dict[str, np.ndarray]:
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
      representative_dataset
  )
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
        logging.DEBUG,
        'Running representative sample for calibration: %d / %s',
        5,
        sample_num,
        total_num_samples,
    )
    yield sample

  logging.info(
      'Running representative samples complete: %d / %s',
      sample_num,
      total_num_samples,
  )


def _run_function_for_calibration_graph_mode(
    sess: session.Session,
    signature_def: meta_graph_pb2.SignatureDef,
    representative_dataset: repr_dataset.RepresentativeDataset,
) -> None:
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
      expected_input_keys=signature_def.inputs.keys()
  )

  for sample in map(
      sample_validator, _log_sample_num_for_calibration(representative_dataset)
  ):
    # Create a mapping from input tensor name to the input tensor value.
    # ex) "Placeholder:0" -> [0, 1, 2]
    feed_dict = _create_feed_dict_from_input_data(sample, signature_def)
    sess.run(output_tensor_names, feed_dict=feed_dict)


def _replace_tensors_by_numpy_ndarrays(
    repr_ds_map: repr_dataset.RepresentativeDatasetMapping,
) -> None:
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
          repr_dataset.replace_tensors_by_numpy_ndarrays(ds, sess)
      )


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
        sess, tags, export_dir=model_dir
    )

    for signature_key, repr_ds in representative_dataset_map.items():
      sig_def = meta_graph.signature_def[signature_key]

      try:
        _run_function_for_calibration_graph_mode(
            sess, signature_def=sig_def, representative_dataset=repr_ds
        )
      except Exception as ex:
        raise ValueError(
            'Failed to run representative dataset through the '
            f'function with the signature key: {signature_key}.'
        ) from ex


def _run_function_for_calibration_eager_mode(
    func: wrap_function.WrappedFunction,
    representative_dataset: repr_dataset.RepresentativeDataset,
) -> None:
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
      expected_input_keys=keyword_args.keys()
  )

  for sample in map(
      sample_validator, _log_sample_num_for_calibration(representative_dataset)
  ):
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
  root: autotrackable.AutoTrackable = saved_model_load.load(model_dir, tags)
  for signature_key, repr_ds in representative_dataset_map.items():
    try:
      _run_function_for_calibration_eager_mode(
          func=root.signatures[signature_key], representative_dataset=repr_ds
      )
    except Exception as ex:
      raise ValueError(
          'Failed to run representative dataset through the '
          f'function with the signature key: {signature_key}.'
      ) from ex


def _run_graph_for_calibration(
    float_model_dir: str,
    signature_keys: Sequence[str],
    tags: Collection[str],
    representative_dataset: repr_dataset.RepresentativeDatasetOrMapping,
    force_graph_mode_calibration: bool,
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
    force_graph_mode_calibration: If set to true, it forces calibration in graph
      model instead of eager mode when the context is in eager mode.

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
    if context.executing_eagerly() and not force_graph_mode_calibration:
      logging.info('Calibration step is executed in eager mode.')
      _run_graph_for_calibration_eager_mode(
          float_model_dir, tags, representative_dataset_map
      )
    else:
      logging.info('Calibration step is executed in graph mode.')
      _run_graph_for_calibration_graph_mode(
          float_model_dir, tags, representative_dataset_map
      )
  except Exception as ex:
    raise ValueError(
        'Failed to run graph for post-training quantization calibration.'
    ) from ex

  logging.info('Calibration step complete.')


def _copy_assets(src_path: str, dst_path: str) -> None:
  """Copies the assets directory of the saved model.

  Clones the contents of the assets/ directory from the source saved model
  directory to the destination saved model directory. Nothing will be copied if
  there are no assets directory in the source directory.

  Args:
    src_path: Source saved model directory.
    dst_path: Destination saved model directory. This directory must exist.
  """
  for assets_dir_name in [_ASSETS_DIR, _ASSETS_EXTRA_DIR]:
    src_assets_path = file_io.join(src_path, assets_dir_name)
    if not file_io.file_exists_v2(src_assets_path):
      # Do nothing if the source assets path does not exist.
      continue

    dst_assets_path = file_io.join(dst_path, assets_dir_name)
    file_io.create_dir_v2(dst_assets_path)

    for curr_dir, _, files in file_io.walk_v2(src_assets_path):
      for asset_file_name in files:
        src_asset_file = file_io.join(curr_dir, asset_file_name)

        # Construct the destination assets file path.
        curr_dst_dir = curr_dir.replace(src_assets_path, dst_assets_path)
        dst_asset_file = file_io.join(curr_dst_dir, asset_file_name)

        file_io.copy_v2(src_asset_file, dst_asset_file)
        logging.info(
            'Copied asset file: %s -> %s', src_asset_file, dst_asset_file
        )


def _run_static_range_qat(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quant_opts: _QuantizationOptions,
    signature_def_map: _SignatureDefMap,
) -> None:
  """Runs static-range quantization for a Quantization-Aware Trained model.

  Runs the quantization for a model trained using QAT.

  Args:
    src_saved_model_path: Path to the source SavedModel directory.
    dst_saved_model_path: Path to the destination SavedModel directory.
    quant_opts: Quantization options.
    signature_def_map: Signature def key -> SignatureDef mapping.
  """
  logging.info('Running static-range quantization for QAT model.')

  loader = saved_model_loader.SavedModelLoader(src_saved_model_path)
  function_aliases = loader.get_meta_graph_def_from_tags(
      quant_opts.tags
  ).meta_info_def.function_aliases

  exported_model_serialized = pywrap_quantize_model.quantize_qat_model(
      src_saved_model_path,
      list(quant_opts.signature_keys),
      set(quant_opts.tags),
      quant_opts.SerializeToString(),
      dict(function_aliases),
  )

  exported_model = exported_model_pb2.ExportedModel.FromString(
      exported_model_serialized
  )

  save_model.save_model_v1(
      exported_model.graph_def,
      dst_saved_model_path,
      signature_def_map,
      quant_opts.tags,
      init_op_name=exported_model.init_node_name,
      saver_def=_get_saver_def_or_none(exported_model),
      checkpoint_dir=exported_model.checkpoint_dir,
      function_aliases=exported_model.function_aliases,
      asset_file_defs=exported_model.asset_file_defs,
  )

  _copy_assets(src_saved_model_path, dst_saved_model_path)


def _get_min_max_from_calibrator(
    node_id: bytes,
    calib_opts: quant_opts_pb2.CalibrationOptions,
) -> tuple[float, float]:
  """Calculate min and max from statistics using calibration options.

  Args:
    node_id: bytes of node id.
    calib_opts: Calibration options used for calculating min and max.

  Returns:
    (min_value, max_value): Min and max calculated using calib_opts.

  Raises:
    ValueError: Unsupported calibration method is given.
  """
  statistics: calib_stats_pb2.CalibrationStatistics = (
      pywrap_quantize_model.get_statistics_from_calibrator(node_id)
  )
  min_value, max_value = calibration_algorithm.get_min_max_value(
      statistics, calib_opts
  )
  return min_value, max_value


def _add_calibration_statistics(
    graph_def: graph_pb2.GraphDef,
    calib_opts: quant_opts_pb2.CalibrationOptions,
) -> None:
  """Adds calibration statistics to the graph def.

  This function must be run after running the graph with a representative
  dataset. Retrieves calibration statistics from the global calibrator and adds
  them to the corresponding nodes as attributes.

  Args:
    graph_def: GraphDef to add calibration statistics to.
    calib_opts: Calibration options to calculate min and max.
  """
  for function_def in graph_def.library.function:
    for node_def in function_def.node_def:
      if node_def.op != 'CustomAggregator':
        continue

      node_id = node_def.attr['id'].s
      try:
        min_value, max_value = _get_min_max_from_calibrator(node_id, calib_opts)
        pywrap_quantize_model.clear_data_from_calibrator(node_id)

        node_def.attr['min'].f = min_value
        node_def.attr['max'].f = max_value
      except ValueError:
        logging.warning(
            (
                'CustomAggregator id "%s" from FunctionDef "%s" does not have '
                'min or max values. Parts of this function are not quantized.'
            ),
            node_id.decode('utf-8'),
            function_def.signature.name,
        )


def _enable_dump_tensor(graph_def: graph_pb2.GraphDef) -> None:
  """Enable DumpTensor in the graph def.

  DumpTensor is disabled by default to avoid logging data during calibration.
  This function is called after calibration to enable DumpTensor.

  Args:
    graph_def: GraphDef to enable DumpTensor
  """
  for function_def in graph_def.library.function:
    for node_def in function_def.node_def:
      if node_def.op != 'DumpTensor':
        continue

      node_def.attr['enabled'].b = True


def _change_dump_tensor_file_name(graph_def: graph_pb2.GraphDef) -> None:
  """Change file_name used by DumpTensor to quantized_tensor_data.pb.

  In whole model verify, DumpTensor in unquantized model uses file_name
  unquantized_tensor_data.pb.
  After unquantized dump model is created, this function allows quantized dump
  model to use quantized_tensor_data.pb as file_name.

  Args:
    graph_def: GraphDef to change file_name of DumpTensor
  """
  for function_def in graph_def.library.function:
    for node_def in function_def.node_def:
      if node_def.op != 'DumpTensor':
        continue

      node_def.attr['file_name'].s = 'quantized_tensor_data.pb'.encode('utf-8')


def _get_saver_def_or_none(
    exported_model: exported_model_pb2.ExportedModel,
) -> Optional[saver_pb2.SaverDef]:
  """Returns the SaverDef from ExportedModel, None otherwise.

  Args:
    exported_model: ExportedModel to take the SaverDef from.

  Returns:
    SaverDef instance if the field `saver_def` is set. None otherwise.
  """
  if exported_model.HasField('saver_def'):
    return exported_model.saver_def
  return None


def _run_static_range_ptq(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quant_opts: _QuantizationOptions,
    representative_dataset: repr_dataset.RepresentativeDatasetOrMapping,
    signature_def_map: _SignatureDefMap,
) -> None:
  """Runs static-range Post-Training Quantization.

  Runs static-range PTQ for the model. Runs the calibration step with
  `representative_dataset` to collect statistics required for quantization. This
  produces the quantized GraphDef along with the SignatureDefs which might have
  been modified according to the changes in the graph.

  Args:
    src_saved_model_path: Path to the source SavedModel directory.
    dst_saved_model_path: Path to the destination SavedModel directory.
    quant_opts: Quantization options.
    representative_dataset: Representative dataset used for the calibration
      step. Representative datasets should exist for each signature def key in
      `signature_def_keys`.
    signature_def_map: Signature def key -> SignatureDef mapping.

  Raises:
    ValueError if the graph doesn't contain a valid signature.
  """
  logging.info('Running post-training quantization pre-calibration step.')

  loader = saved_model_loader.SavedModelLoader(src_saved_model_path)
  function_aliases = loader.get_meta_graph_def_from_tags(
      quant_opts.tags
  ).meta_info_def.function_aliases

  exported_model_serialized = (
      pywrap_quantize_model.quantize_ptq_model_pre_calibration(
          src_saved_model_path,
          list(quant_opts.signature_keys),
          set(quant_opts.tags),
          quant_opts.SerializeToString(),
          dict(function_aliases),
      )
  )

  exported_model = exported_model_pb2.ExportedModel.FromString(
      exported_model_serialized
  )

  graph_def = exported_model.graph_def
  for function_def in graph_def.library.function:
    for node_def in function_def.node_def:
      if node_def.op == 'CustomAggregator':
        node_def.attr['id'].s = uuid.uuid4().hex.encode('ascii')

  pre_calib_output_model_path = tempfile.mkdtemp()
  save_model.save_model_v1(
      graph_def,
      pre_calib_output_model_path,
      signature_def_map,
      quant_opts.tags,
      exported_model.init_node_name,
      _get_saver_def_or_none(exported_model),
      exported_model.checkpoint_dir,
      exported_model.function_aliases,
      asset_file_defs=exported_model.asset_file_defs,
  )

  _copy_assets(src_saved_model_path, pre_calib_output_model_path)

  # Uses the representative dataset to collect statistics for calibration.
  # Handles the graph mode execution separately in case TF2 is disabled or
  # eager execution is disabled. The min & max values are stored separately
  # in a global CalibratorSingleton instance.
  _run_graph_for_calibration(
      pre_calib_output_model_path,
      quant_opts.signature_keys,
      quant_opts.tags,
      representative_dataset,
      quant_opts.force_graph_mode_calibration,
  )

  _add_calibration_statistics(graph_def, quant_opts.calibration_options)

  if quant_opts.HasField('debugger_options'):
    # Since DumpTensor was disabled by default, we need to enable them.
    _enable_dump_tensor(graph_def)

    if (
        quant_opts.debugger_options.debugger_type
        == quant_opts_pb2.DebuggerOptions.DebuggerType.DEBUGGER_TYPE_WHOLE_MODEL
    ):
      # TODO: b/295139417 - Remove CustomAggregator op in unquantized dump model
      # TODO: b/296916287 - Create a separate function for saving unquantized
      # dump model
      save_model.save_model_v1(
          graph_def,
          quant_opts.debugger_options.unquantized_dump_model_path,
          signature_def_map,
          quant_opts.tags,
          exported_model.init_node_name,
          _get_saver_def_or_none(exported_model),
          exported_model.checkpoint_dir,
          exported_model.function_aliases,
          asset_file_defs=exported_model.asset_file_defs,
      )

      _copy_assets(
          src_saved_model_path,
          quant_opts.debugger_options.unquantized_dump_model_path,
      )

      _change_dump_tensor_file_name(graph_def)

  calibrated_model_path = tempfile.mkdtemp()
  save_model.save_model_v1(
      graph_def,
      calibrated_model_path,
      signature_def_map,
      quant_opts.tags,
      exported_model.init_node_name,
      _get_saver_def_or_none(exported_model),
      exported_model.checkpoint_dir,
      asset_file_defs=exported_model.asset_file_defs,
  )

  _copy_assets(pre_calib_output_model_path, calibrated_model_path)

  logging.info('Running post-training quantization post-calibration step.')
  exported_model_serialized = (
      pywrap_quantize_model.quantize_ptq_model_post_calibration(
          calibrated_model_path,
          list(quant_opts.signature_keys),
          set(quant_opts.tags),
          quant_opts.SerializeToString(),
          dict(exported_model.function_aliases),
      )
  )

  exported_model = exported_model_pb2.ExportedModel.FromString(
      exported_model_serialized
  )

  save_model.save_model_v1(
      exported_model.graph_def,
      dst_saved_model_path,
      signature_def_map,
      quant_opts.tags,
      init_op_name=exported_model.init_node_name,
      saver_def=_get_saver_def_or_none(exported_model),
      checkpoint_dir=exported_model.checkpoint_dir,
      function_aliases=exported_model.function_aliases,
      asset_file_defs=exported_model.asset_file_defs,
  )

  _copy_assets(calibrated_model_path, dst_saved_model_path)


def _static_range_quantize(
    saved_model_path: str,
    output_directory: str,
    quantization_options: _QuantizationOptions,
    representative_dataset: Optional[
        repr_dataset.RepresentativeDatasetOrMapping
    ] = None,
) -> autotrackable.AutoTrackable:
  """Quantizes the given SavedModel via static range quantization.

  If the model is not trained with Quantization-Aware Training (QAT) technique,
  it requires `representative_dataset` to collect statistics required for
  quantization. If non-None `representative_dataset` is provided with a QAT
  model input, `representative_dataset` will be ignored.

  Args:
    saved_model_path: Path to the saved model. When representative_dataset is
      not provided, this should be a model trained with QAT.
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
  logging.info(
      'Running static range quantization on model: %s', saved_model_path
  )
  logging.info('QuantizationOptions: \n%s', quantization_options)

  is_qat_saved_model_or_method_no_quantize = _is_qat_saved_model(
      saved_model_path
  ) or (
      quantization_options.quantization_method.preset_method
      == _QuantizationMethod.METHOD_NO_QUANTIZE
  )
  signature_def_map = save_model.get_signatures_from_saved_model(
      saved_model_path,
      quantization_options.signature_keys,
      set(quantization_options.tags),
  )

  # Checks if the model is from QAT or method is METHOD_NO_QUANTIZE.
  if (
      representative_dataset is None
      and not is_qat_saved_model_or_method_no_quantize
  ):
    raise ValueError(
        'When `representative_dataset` is not provided, the model should be '
        'trained with quantization-aware training (QAT).'
    )
  if quantization_options.min_num_elements_for_weights > 0:
    logging.warn(
        'min_num_elements_for_weights is set but is not supported for the '
        'Post-training static range quantization. '
        'The flag is ignored.'
    )

  if is_qat_saved_model_or_method_no_quantize:
    _run_static_range_qat(
        saved_model_path,
        output_directory,
        quantization_options,
        signature_def_map,
    )
  else:
    _run_static_range_ptq(
        saved_model_path,
        output_directory,
        quantization_options,
        representative_dataset,
        signature_def_map,
    )

  return saved_model_load.load(output_directory)


def _dynamic_range_quantize(
    saved_model_path: str,
    output_directory: str,
    quantization_options: _QuantizationOptions,
) -> autotrackable.AutoTrackable:
  """Quantizes the given SavedModel via post-training dynamic range quantization.

  Args:
    saved_model_path: Path to the saved model.
    output_directory: The path to save the output SavedModel. The directory will
      be overwritten if not empty.
    quantization_options: QuantizationOptions proto describing quantization
      related config.

  Returns:
    A SavedModel object with TF quantization applied.

  Raises:
    ValueError: when the model is QAT model.
  """
  mode_str = 'dynamic-range quantization'
  if _is_qat_saved_model(saved_model_path):
    raise ValueError(
        'The models trained with quantization-aware training (QAT) is not '
        'supported for %s.' % mode_str
    )

  logging.info(
      'Running post-training %s on model: %s', mode_str, saved_model_path
  )
  logging.info('QuantizationOptions: \n%s', quantization_options)

  loader = saved_model_loader.SavedModelLoader(saved_model_path)

  function_aliases = loader.get_meta_graph_def_from_tags(
      quantization_options.tags
  ).meta_info_def.function_aliases

  # Apply post-training dynamic range quantization to the model.
  exported_model_serialized = pywrap_quantize_model.quantize_ptq_dynamic_range(
      saved_model_path,
      list(quantization_options.signature_keys),
      set(quantization_options.tags),
      quantization_options.SerializeToString(),
      dict(function_aliases),
  )

  exported_model = exported_model_pb2.ExportedModel.FromString(
      exported_model_serialized
  )
  signature_def_map = save_model.get_signatures_from_saved_model(
      saved_model_path,
      quantization_options.signature_keys,
      quantization_options.tags,
  )

  save_model.save_model_v1(
      exported_model.graph_def,
      output_directory,
      signature_def_map,
      quantization_options.tags,
      init_op_name=exported_model.init_node_name,
      saver_def=_get_saver_def_or_none(exported_model),
      checkpoint_dir=exported_model.checkpoint_dir,
      function_aliases=exported_model.function_aliases,
      asset_file_defs=exported_model.asset_file_defs,
  )
  _copy_assets(saved_model_path, output_directory)

  return saved_model_load.load(output_directory)


def _weight_only_quantize(
    saved_model_path: str,
    output_directory: str,
    quantization_options: quant_opts_pb2.QuantizationOptions,
) -> autotrackable.AutoTrackable:
  """Quantizes the given SavedModel via weight-only quantization.

  Args:
    saved_model_path: Path to the saved model.
    output_directory: The path to save the output SavedModel. The directory will
      be overwritten if not empty.
    quantization_options: QuantizationOptions proto describing quantization
      related config.

  Returns:
    A SavedModel object with TF quantization applied.

  Raises:
    ValueError: when the model is QAT model.
  """
  mode_str = 'weight-only quantization'

  # QAT weight-only is not supported yet.
  if _is_qat_saved_model(saved_model_path):
    raise ValueError(
        'The models trained with quantization-aware training (QAT) is not '
        'supported for %s.' % mode_str
    )

  logging.info(
      'Running post-training %s on model: %s', mode_str, saved_model_path
  )
  logging.info('QuantizationOptions: \n%s', quantization_options)

  loader = saved_model_loader.SavedModelLoader(saved_model_path)

  function_aliases = loader.get_meta_graph_def_from_tags(
      quantization_options.tags
  ).meta_info_def.function_aliases

  exported_model_serialized = pywrap_quantize_model.quantize_weight_only(
      saved_model_path,
      quantization_options.SerializeToString(),
      dict(function_aliases),
  )

  exported_model = exported_model_pb2.ExportedModel.FromString(
      exported_model_serialized
  )
  signature_def_map = save_model.get_signatures_from_saved_model(
      saved_model_path,
      list(quantization_options.signature_keys),
      set(quantization_options.tags),
  )

  save_model.save_model_v1(
      exported_model.graph_def,
      output_directory,
      signature_def_map,
      quantization_options.tags,
      init_op_name=exported_model.init_node_name,
      saver_def=_get_saver_def_or_none(exported_model),
      checkpoint_dir=exported_model.checkpoint_dir,
      function_aliases=exported_model.function_aliases,
      asset_file_defs=exported_model.asset_file_defs,
  )
  _copy_assets(saved_model_path, output_directory)

  return saved_model_load.load(output_directory)


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
      output_dir is not None
      and file_io.file_exists_v2(output_dir)
      and file_io.list_directory_v2(output_dir)
  )

  if dir_not_empty and not overwrite:
    raise FileExistsError(
        f'Output directory already exists: {output_dir} . '
        'Please set overwrite_output_directory to true to '
        'overwrite the existing directory.'
    )


def _populate_quantization_component_spec(
    quant_method: _QuantizationMethod,
) -> None:
  """Populates default values for QuantizationComponentSpec.

  Args:
    quant_method: The quantization method to be updated.
  """
  # Make sure creating one spec per component.
  updated_component_spec = dict()

  # Populate default configuration.
  if (
      quant_method.preset_method == _PresetMethod.METHOD_STATIC_RANGE_INT8
      or quant_method.preset_method == _PresetMethod.METHOD_DYNAMIC_RANGE_INT8
  ):
    updated_component_spec[_QuantizationComponent.COMPONENT_ACTIVATION] = (
        _QuantizationComponentSpec(
            quantization_component=_QuantizationComponent.COMPONENT_ACTIVATION,
            tensor_type=_TensorType.TENSORTYPE_INT_8,
        )
    )
    updated_component_spec[_QuantizationComponent.COMPONENT_WEIGHT] = (
        _QuantizationComponentSpec(
            quantization_component=_QuantizationComponent.COMPONENT_WEIGHT,
            tensor_type=_TensorType.TENSORTYPE_INT_8,
        )
    )
    updated_component_spec[_QuantizationComponent.COMPONENT_BIAS] = (
        _QuantizationComponentSpec(
            quantization_component=_QuantizationComponent.COMPONENT_BIAS,
            tensor_type=_TensorType.TENSORTYPE_INT_32,
        )
    )
  elif (
      quant_method.preset_method
      == _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
  ):
    updated_component_spec[_QuantizationComponent.COMPONENT_WEIGHT] = (
        _QuantizationComponentSpec(
            quantization_component=_QuantizationComponent.COMPONENT_WEIGHT,
            tensor_type=_TensorType.TENSORTYPE_INT_8,
        )
    )

  # Override if quantization_component_spec is specified.
  if quant_method.quantization_component_specs:
    # Check if the component spec is supported configuration in TF-Quant.
    for component_spec in quant_method.quantization_component_specs:
      if component_spec.quantization_component in [
          _QuantizationComponent.COMPONENT_WEIGHT,
          _QuantizationComponent.COMPONENT_ACTIVATION,
      ]:
        if component_spec.tensor_type != _TensorType.TENSORTYPE_INT_8:
          raise ValueError(
              'Only int8 precision is supported for input operands.'
          )
      else:
        if component_spec.tensor_type != _TensorType.TENSORTYPE_INT_32:
          raise ValueError('Only int32 precision is supported for bias.')
      # Update with the custom spec.
      updated_component_spec[component_spec.quantization_component] = (
          component_spec
      )

  # Update the componet spec
  del quant_method.quantization_component_specs[:]
  quant_method.quantization_component_specs.extend(
      updated_component_spec.values()
  )

  if (
      quant_method.preset_method == _PresetMethod.METHOD_STATIC_RANGE_INT8
      or quant_method.preset_method == _PresetMethod.METHOD_DYNAMIC_RANGE_INT8
  ) and (len(quant_method.quantization_component_specs) != 3):
    raise ValueError('Only 3 components are needed for', quant_method)
  elif (
      quant_method.preset_method
      == _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
  ) and len(quant_method.quantization_component_specs) != 1:
    raise ValueError('At least one component spec needs to be specified.')


def _populate_unitwise_quantization_specs(
    quantization_options: _QuantizationOptions,
) -> None:
  """Verifies and pupulates unitwise quantization specs."""
  if not quantization_options.unit_wise_quantization_specs:
    return

  sorted_top_level_component_specs = sorted(
      quantization_options.quantization_method.quantization_component_specs,
      key=lambda x: x.quantization_component,
  )

  for unitwise_spec in quantization_options.unit_wise_quantization_specs:
    if not unitwise_spec.unit:
      raise ValueError(
          'UnitWiseQuantizationSpec must contain at least one unit.'
      )

    for unit in unitwise_spec.unit:
      if not unit.op_type and not unit.node_name:
        raise ValueError('Either `op_type` or `node_name` must be specified.')

    _populate_quantization_component_spec(unitwise_spec.quantization_method)

    component_specs = (
        unitwise_spec.quantization_method.quantization_component_specs
    )
    if component_specs and (
        sorted_top_level_component_specs
        != sorted(component_specs, key=lambda x: x.quantization_component)
    ):
      raise ValueError(
          'Currently unit-wise quantization spec only supports NO_QUANTIZE and'
          ' same quantization method as the top-level `quantization_method`'
      )


def _populate_calibration_options(
    quantization_options: quant_opts_pb2.QuantizationOptions,
):
  """Populates default values for CalibrationOptions.

  Args:
    quantization_options: An instance of QuantizationOptions with a field
      specifying CalibrationOptions
  """

  calib_opts = quantization_options.calibration_options
  if (
      calib_opts.calibration_method
      == _CalibrationMethod.CALIBRATION_METHOD_UNSPECIFIED
  ):
    calib_opts.calibration_method = (
        _CalibrationMethod.CALIBRATION_METHOD_MIN_MAX
    )
  elif (
      calib_opts.calibration_method
      == _CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_PERCENTILE
  ):
    if not calib_opts.calibration_parameters.initial_num_bins:
      calib_opts.calibration_parameters.initial_num_bins = 256
    if not calib_opts.calibration_parameters.min_percentile:
      calib_opts.calibration_parameters.min_percentile = 0.001
    if not calib_opts.calibration_parameters.max_percentile:
      calib_opts.calibration_parameters.max_percentile = 99.999
  # Check the activation_tensor_type of HISTOGRAM_MSE methods.
  elif calib_opts.calibration_method in [
      _CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE,
      _CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY,
      _CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC,
  ]:
    activation_tensor_type = (
        quantization_options.quantization_method.quantization_component_specs[
            _QuantizationComponent.COMPONENT_ACTIVATION
        ].tensor_type
    )
    # Unlike the HISTOGRAM_PERCENTILE method, the HISTOGRAM_MSE method uses
    # num_bits because it actually quantizes and dequantizes values.
    if activation_tensor_type != _TensorType.TENSORTYPE_INT_8:
      raise ValueError(
          'Only TENSORTYPE_INT_8 is supported for HISTOGRAM_MSE calibration'
          f' methods. calibration_method={calib_opts.calibration_method}'
      )

    if not calib_opts.calibration_parameters.initial_num_bins:
      calib_opts.calibration_parameters.initial_num_bins = 256


def _populate_quantization_options_default_values(
    quantization_options: _QuantizationOptions,
) -> None:
  """Populates default values for QuantizationOptions.

  Populates unspecified or unset fields of QuantizationOptions with the default
  values.

  * If `op_set` is unspecified, it defaults to `OpSet.XLA`.
  * If `freeze_all_variables` is not set, it defaults to `True`.
  * Check if configurations are set correctly:
    - Per-channel quantization is supported for Uniform Quantized opset only.

  Args:
    quantization_options: An instance of QuantizationOptions.
  """
  if quantization_options.op_set == quant_opts_pb2.OpSet.OP_SET_UNSPECIFIED:
    quantization_options.op_set = quant_opts_pb2.OpSet.XLA

  if not quantization_options.tags:
    quantization_options.tags.append(tag_constants.SERVING)

  if not quantization_options.signature_keys:
    quantization_options.signature_keys.append(
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    )

  if not quantization_options.HasField('freeze_all_variables'):
    quantization_options.freeze_all_variables = True

  if quantization_options.enable_legacy_weight_only:
    raise ValueError(
        'Legacy weight-only is deprecated. Use weight-only quantization method.'
    )

  # Check default quantization option values for weight-only quantization.
  # TODO(b/242805842): Find good minimum_elements_for_weights number for server.
  # please also update default value in tflite converter:
  # tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.cc;l=201
  if (
      quantization_options.quantization_method.preset_method
      == _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
  ) or (
      quantization_options.quantization_method.preset_method
      == _PresetMethod.METHOD_DYNAMIC_RANGE_INT8
  ):
    if quantization_options.min_num_elements_for_weights == 0:
      quantization_options.min_num_elements_for_weights = (
          _DYNAMIC_RANGE_DEFAULT_MIN_NUM_ELEMENTS_FOR_WEIGHTS
      )
      logging.warn(
          (
              'QuantizationOptions.min_num_elements_for_weights is not set (0).'
              ' Setting to the default value: %d.'
          ),
          _DYNAMIC_RANGE_DEFAULT_MIN_NUM_ELEMENTS_FOR_WEIGHTS,
      )

  # TODO(b/281595329): Implement static range quantization per-channel support
  if quantization_options.enable_per_channel_quantization and not (
      quantization_options.op_set == quant_opts_pb2.OpSet.UNIFORM_QUANTIZED
      or quantization_options.quantization_method.preset_method
      == _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
  ):
    raise ValueError(
        'Currently, per-channel quantization is supported for Uniform '
        'Quantized opset and Weight-only.'
    )

  if (
      quantization_options.quantization_method.preset_method
      == _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
      and (
          quantization_options.op_set == quant_opts_pb2.OpSet.UNIFORM_QUANTIZED
          or quantization_options.op_set == quant_opts_pb2.OpSet.TF
      )
  ):
    raise ValueError('TF/Uniform quantized opset does not support weight-only.')

  if (quantization_options.op_set == quant_opts_pb2.OpSet.STABLEHLO) and (
      quantization_options.quantization_method.preset_method
      != _PresetMethod.METHOD_STATIC_RANGE_INT8
  ):
    raise ValueError(
        'StableHLO quantized opset currently only supports static range'
        ' quantization via TF Quantizer.'
    )

  # Converter assumes options are specified. So set SRQ explicitly.
  if (
      quantization_options.quantization_method.preset_method
      == _PresetMethod.METHOD_UNSPECIFIED
  ):
    logging.debug(
        '"preset_method" for QuantizationMethod is not specified.'
        'Static range quantization is used by default.'
    )
    quantization_options.quantization_method.preset_method = (
        _PresetMethod.METHOD_STATIC_RANGE_INT8
    )

  if quantization_options.HasField('debugger_options'):
    if not quantization_options.debugger_options.log_dir_path:
      quantization_options.debugger_options.log_dir_path = '/tmp/dumps'

    if (
        quantization_options.debugger_options.debugger_type
        == quant_opts_pb2.DebuggerOptions.DebuggerType.DEBUGGER_TYPE_UNSPECIFIED
    ):
      raise ValueError(
          'Debugger is enabled but debugger type was not specified.'
      )

    if (
        quantization_options.debugger_options.debugger_type
        == quant_opts_pb2.DebuggerOptions.DebuggerType.DEBUGGER_TYPE_WHOLE_MODEL
        and not quantization_options.debugger_options.unquantized_dump_model_path
    ):
      raise ValueError(
          'Debugger type whole model verify was used but'
          ' unquantized_dump_model_path was not specified.'
      )

  # Check and populate quantization component spec.
  _populate_quantization_component_spec(
      quantization_options.quantization_method
  )
  # Verify and populate unit-wise quantization specs.
  _populate_unitwise_quantization_specs(quantization_options)

  if (
      quantization_options.quantization_method.preset_method
      == _PresetMethod.METHOD_STATIC_RANGE_INT8
  ):
    # Check and populate calibration options.
    _populate_calibration_options(quantization_options)


@tf_export.tf_export('quantization.experimental.quantize_saved_model')
def quantize(
    saved_model_path: str,
    output_directory: Optional[str] = None,
    quantization_options: Optional[_QuantizationOptions] = None,
    representative_dataset: Optional[
        repr_dataset.RepresentativeDatasetOrMapping
    ] = None,
    *,
    overwrite_output_directory: bool = False,
) -> autotrackable.AutoTrackable:
  """Quantizes the SavedModel with the given quantization options.

  Example usage:
  ```python
  # Quantizing a model trained with QAT.
  quantization_options = tf.quantization.experimental.QuantizationOptions(
      signature_keys=['your_signature_key'],
  )
  tf.quantization.experimental.quantize_saved_model(
      '/tmp/input_model',
      '/tmp/output_model',
      quantization_options=quantization_options,
  )

  # When quantizing a model trained without QAT (Post-Training Quantization),
  # a representative dataset is required.
  representative_dataset = [{"input": tf.random.uniform(shape=(3, 3))}
                        for _ in range(256)]
  tf.quantization.experimental.quantize_saved_model(
      '/tmp/input_model',
      '/tmp/output_model',
      quantization_options=quantization_options,
      representative_dataset={'your_signature_key': representative_dataset},
    )

  # In addition to preset quantization methods, fine-grained control of
  # quantization for each component is also supported.
  _QuantizationComponentSpec = (
      tf.quantization.experimental.QuantizationComponentSpec
  )
  quantization_options = tf.quantization.experimental.QuantizationOptions(
      signature_keys=['your_signature_key'],
      quantization_method=tf.quantization.experimental.QuantizationMethod(
          quantization_component_specs=[
              _QuantizationComponentSpec(
                  quantization_component=(
                      _QuantizationComponentSpec.COMPONENT_ACTIVATION
                  ),
                  tensor_type=_QuantizationComponentSpec.TENSORTYPE_INT_8,
              )
          ]
      )
  )
  tf.quantization.experimental.quantize_saved_model(
      '/tmp/input_model',
      '/tmp/output_model',
      quantization_options=quantization_options,
  )
  ```

  Args:
    saved_model_path: Path to the saved model. When representative_dataset is
      not provided, this should be a model trained with QAT.
    output_directory: The path to save the output SavedModel. Set
      `overwrite_output_directory` to `True` to overwrite any existing contents
      in the directory if not empty.
    quantization_options: A set of options for quantization. If None, it uses
      post-training static range quantization with XLA opset by default.
    representative_dataset: an iterator that returns a dictionary of {input_key:
      input_value} or a map from signature key to a dictionary of {input_key:
      input_value} that feeds calibration data for quantizing model. The
      representative should be provided when the model is a PTQ model. It can be
      provided either via this parameter or via the `representative_datasets`
      field in `QuantizationOptions`.
    overwrite_output_directory: If set to true, overwrites the output directory
      iff it isn't empty. The default value is false.

  Returns:
    A SavedModel object with TF quantization applied, or None if no quantization
    is performed.

  Raises:
    ValueError: When 1) representative_dataset is not provided for non QAT model
      for enabling static range quantization, 2) invalid value is provided as
      a quantization method, or 3) provide representative dataset via both
      argument and QuantizationOptions.
    ValueError: When the specified quantization method is not yet supported.
  """
  _verify_output_dir(output_directory, overwrite_output_directory)

  # Set default values for None arguments.
  if output_directory is None:
    output_directory = tempfile.mkdtemp()

  if quantization_options is None:
    quantization_options = _QuantizationOptions()

  _populate_quantization_options_default_values(quantization_options)

  if (
      representative_dataset is not None
      and quantization_options.representative_datasets
  ):
    raise ValueError(
        'Do not specify both the `representative_dataset` argument and'
        ' the `representative_datasets` field in `QuantizationOptions`.'
    )

  if quantization_options.representative_datasets:
    representative_dataset = repr_dataset.RepresentativeDatasetLoader(
        quantization_options.representative_datasets
    ).load()

  method: _QuantizationMethod = quantization_options.quantization_method
  if (
      method.preset_method == _PresetMethod.METHOD_STATIC_RANGE_INT8
      or method.preset_method == _PresetMethod.METHOD_NO_QUANTIZE
  ):
    return _static_range_quantize(
        saved_model_path,
        output_directory,
        quantization_options,
        representative_dataset,
    )
  elif method.preset_method == _PresetMethod.METHOD_DYNAMIC_RANGE_INT8:
    return _dynamic_range_quantize(
        saved_model_path,
        output_directory,
        quantization_options,
    )
  elif (
      method.preset_method == _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
  ):
    return _weight_only_quantize(
        saved_model_path,
        output_directory,
        quantization_options,
    )
  else:
    raise ValueError(
        'Quantization method {method.preset_method} is not supported.'
    )
