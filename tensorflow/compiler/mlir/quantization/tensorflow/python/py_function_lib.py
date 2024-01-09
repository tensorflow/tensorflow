# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Defines a wrapper class for overridden python method definitions."""
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Optional

from absl import logging

from tensorflow.compiler.mlir.quantization.tensorflow import exported_model_pb2
from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import calibration_algorithm
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import calibration_statistics_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import pywrap_calibration
from tensorflow.compiler.mlir.quantization.tensorflow.python import pywrap_function_lib
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as rd
from tensorflow.compiler.mlir.quantization.tensorflow.python import save_model
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.trackable import autotrackable
from tensorflow.python.types import core

# Name of the saved model assets directory.
_ASSETS_DIR = 'assets'
_ASSETS_EXTRA_DIR = 'assets.extra'


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


def _validate_representative_dataset(
    representative_dataset: rd.RepresentativeDatasetOrMapping,
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
  if isinstance(representative_dataset, Mapping):
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


def _replace_tensors_by_numpy_ndarrays(
    repr_ds_map: rd.RepresentativeDatasetMapping,
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
      repr_ds_map[signature_def_key] = rd.replace_tensors_by_numpy_ndarrays(
          ds, sess
      )


def _create_sample_validator(
    expected_input_keys: Collection[str],
) -> Callable[[rd.RepresentativeSample], rd.RepresentativeSample]:
  """Creates a validator function for a representative sample.

  Args:
    expected_input_keys: Input keys (keyword argument names) that the function
      the sample will be used for is expecting to receive.

  Returns:
    A callable that validates a `RepresentativeSample`.
  """

  def validator(
      sample: rd.RepresentativeSample,
  ) -> rd.RepresentativeSample:
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
    if not isinstance(sample, Mapping):
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


# TODO(b/249918070): Implement a progress bar.
def _log_sample_num_for_calibration(
    representative_dataset: rd.RepresentativeDataset,
) -> rd.RepresentativeDataset:
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
  num_samples: Optional[int] = rd.get_num_samples(representative_dataset)
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
    representative_dataset: rd.RepresentativeDataset,
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
    feed_dict = rd.create_feed_dict_from_input_data(sample, signature_def)
    sess.run(output_tensor_names, feed_dict=feed_dict)


def _run_graph_for_calibration_graph_mode(
    model_dir: str,
    tags: Collection[str],
    representative_dataset_map: rd.RepresentativeDatasetMapping,
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
    meta_graph: meta_graph_pb2.MetaGraphDef = loader_impl.load(
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


def _convert_values_to_tf_tensors(
    sample: rd.RepresentativeSample,
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


def _run_function_for_calibration_eager_mode(
    func: wrap_function.WrappedFunction,
    representative_dataset: rd.RepresentativeDataset,
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
    representative_dataset_map: rd.RepresentativeDatasetMapping,
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
  root: autotrackable.AutoTrackable = load.load(model_dir, tags)
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
    representative_dataset: rd.RepresentativeDatasetOrMapping,
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
  if not isinstance(representative_dataset, Mapping):
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


def _get_min_max_from_calibrator(
    node_id: bytes,
    calib_opts: quantization_options_pb2.CalibrationOptions,
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
  statistics: calibration_statistics_pb2.CalibrationStatistics = (
      pywrap_calibration.get_statistics_from_calibrator(node_id)
  )
  min_value, max_value = calibration_algorithm.get_min_max_value(
      statistics, calib_opts
  )
  return min_value, max_value


class PyFunctionLibrary(pywrap_function_lib.PyFunctionLibrary):
  """Wrapper class for overridden python method definitions.

  This class contains python methods that overrides C++ virtual functions
  declared in `pywrap_function_lib.PyFunctionLibrary`.
  """

  # LINT.IfChange(save_exported_model)
  def save_exported_model(
      self,
      dst_saved_model_path: str,
      exported_model_serialized: bytes,
      src_saved_model_path: str,
      tags: set[str],
      serialized_signature_def_map: dict[str, bytes],
  ) -> None:
    # LINT.ThenChange(py_function_lib.h:save_exported_model)
    """Saves `ExportedModel` to `dst_saved_model_path` as a SavedModel.

    Args:
      dst_saved_model_path: Destination path to save the exported model.
      exported_model_serialized: Exported model to export as SavedModel.
      src_saved_model_path: Path to the source SavedModel. This will be used to
        copy the asset files to `dst_saved_model_path`.
      tags: Tags to attach to the saved MetaGraphDef.
      serialized_signature_def_map: Signature key -> serialized SignatureDef.
    """
    exported_model = exported_model_pb2.ExportedModel.FromString(
        exported_model_serialized
    )

    # Deserialize values in signature_def_map.
    signature_def_map = {}
    for key, serialized_signature_def in serialized_signature_def_map.items():
      signature_def_map[key] = meta_graph_pb2.SignatureDef.FromString(
          serialized_signature_def
      )

    save_model.save_model_v1(
        exported_model.graph_def,
        dst_saved_model_path,
        signature_def_map,
        tags,
        init_op_name=exported_model.init_node_name,
        saver_def=_get_saver_def_or_none(exported_model),
        checkpoint_dir=exported_model.checkpoint_dir,
        function_aliases=exported_model.function_aliases,
        asset_file_defs=exported_model.asset_file_defs,
    )

    _copy_assets(src_saved_model_path, dst_saved_model_path)

  # TODO: b/311097139 - Extract calibration related functions into a separate
  # file.
  # LINT.IfChange(run_calibration)
  def run_calibration(
      self,
      saved_model_path: str,
      signature_keys: list[str],
      tags: set[str],
      calibration_options_serialized: bytes,
      force_graph_mode_calibration: bool,
      representative_dataset_file_map_serialized: dict[str, bytes],
  ) -> None:
    # LINT.ThenChange(py_function_lib.h:run_calibration)
    """Runs calibration and adds calibration statistics to exported model.

    Args:
      saved_model_path: Path to the SavedModel to run calibration.
      signature_keys: List of signature keys corresponding to SignatureDefs to
        run calibration on.
      tags: A set of tags that identify the MetaGraphDef.
      calibration_options_serialized: Serialized `CalibrationOptions`.
      force_graph_mode_calibration: If True, runs the calibration in graph mode.
      representative_dataset_file_map_serialized: Signature key ->
        `RepresentativeDatasetFile` mapping for running the calibration step.
        Each dataset file stores the representative dataset for the function
        matching the signature key.

    Returns:
      Updated exported model (serialized) where the collected calibration
      statistics are added to `CustomerAggregator` nodes at the `min` and `max`
      attributes.
    """
    dataset_file_map = {}
    for (
        signature_key,
        dataset_file_serialized,
    ) in representative_dataset_file_map_serialized.items():
      dataset_file_map[signature_key] = (
          quantization_options_pb2.RepresentativeDatasetFile.FromString(
              dataset_file_serialized
          )
      )

    repr_dataset_map = rd.TfRecordRepresentativeDatasetLoader(
        dataset_file_map=dataset_file_map
    ).load()

    # Uses the representative dataset to collect statistics for calibration.
    # After this operation, min & max values are stored separately in a global
    # CalibratorSingleton instance.
    _run_graph_for_calibration(
        saved_model_path,
        signature_keys,
        tags,
        repr_dataset_map,
        force_graph_mode_calibration,
    )

  # LINT.IfChange(get_calibration_min_max_value)
  def get_calibration_min_max_value(
      self,
      calibration_statistics_serialized: bytes,
      calibration_options_serialized: bytes,
  ) -> tuple[float, float]:
    """Calculates min and max values from statistics.

    Args:
      calibration_statistics_serialized: Serialized `CalibrationStatistics`.
        This will be the source to calculate min and max values from.
      calibration_options_serialized: Serialized `CalibrationOptions`. Specifies
        how the min / max should be calculated.

    Returns:
      (min_value, max_value): Min and max calculated using calib_opts.

    Raises:
      ValueError: Unsupported calibration method is given.
    """
    # LINT.ThenChange(py_function_lib.h:get_calibration_min_max_value)
    return calibration_algorithm.get_min_max_value(
        calibration_statistics_pb2.CalibrationStatistics.FromString(
            calibration_statistics_serialized
        ),
        quantization_options_pb2.CalibrationOptions.FromString(
            calibration_options_serialized
        ),
    )
