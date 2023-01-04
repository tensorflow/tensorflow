# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Python TF-Lite QuantizationDebugger."""
import collections
import csv
import re
from typing import (Any, Callable, Dict, IO, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

import numpy as np

from tensorflow.lite.python import convert
from tensorflow.lite.python import interpreter as _interpreter
from tensorflow.lite.python.metrics import metrics as metrics_stub  # type: ignore
from tensorflow.python.util import tf_export


# TODO(b/198099651): move converter implementation out of lite.py
TFLiteConverter = Any  # importing tf.lite creates circular dependency

# Returns metrics based on difference of values for quantized/float ops.
_DEFAULT_LAYER_DEBUG_METRICS = {
    'num_elements': lambda diffs: diffs.size,
    'stddev': np.std,
    'mean_error': np.average,
    'max_abs_error': lambda diffs: np.max(np.abs(diffs)),
    'mean_squared_error': lambda diffs: np.average(diffs**2),
}

_NUMERIC_VERIFY_OP_NAME = 'NumericVerify'


def _get_quant_params(
    tensor_detail: Mapping[str, Any]) -> Optional[Tuple[float, int]]:
  """Returns first scale and zero point from tensor detail, if present."""
  quant_params = tensor_detail['quantization_parameters']
  if not quant_params:
    return None
  if quant_params['scales'] and quant_params['zero_points']:
    return (quant_params['scales'][0], quant_params['zero_points'][0])
  return None


@tf_export.tf_export('lite.experimental.QuantizationDebugOptions')
class QuantizationDebugOptions:
  """Debug options to set up a given QuantizationDebugger."""

  def __init__(self,
               layer_debug_metrics: Optional[Mapping[str,
                                                     Callable[[np.ndarray],
                                                              float]]] = None,
               model_debug_metrics: Optional[Mapping[
                   str, Callable[[Sequence[np.ndarray], Sequence[np.ndarray]],
                                 float]]] = None,
               layer_direct_compare_metrics: Optional[Mapping[str, Callable[
                   [Sequence[np.ndarray], Sequence[np.ndarray], float, int],
                   float]]] = None,
               denylisted_ops: Optional[List[str]] = None,
               denylisted_nodes: Optional[List[str]] = None,
               fully_quantize: bool = False) -> None:
    """Initializes debugger options.

    Args:
      layer_debug_metrics: a dict to specify layer debug functions
        {function_name_str: function} where the function accepts result of
          NumericVerify Op, which is value difference between float and
          dequantized op results. The function returns single scalar value.
      model_debug_metrics: a dict to specify model debug functions
        {function_name_str: function} where the function accepts outputs from
          two models, and returns single scalar value for a metric. (e.g.
          accuracy, IoU)
      layer_direct_compare_metrics: a dict to specify layer debug functions
        {function_name_str: function}. The signature is different from that of
          `layer_debug_metrics`, and this one gets passed (original float value,
          original quantized value, scale, zero point). The function's
          implementation is responsible for correctly dequantize the quantized
          value to compare. Use this one when comparing diff is not enough.
          (Note) quantized value is passed as int8, so cast to int32 is needed.
      denylisted_ops: a list of op names which is expected to be removed from
        quantization.
      denylisted_nodes: a list of op's output tensor names to be removed from
        quantization.
      fully_quantize: Bool indicating whether to fully quantize the model.
        Besides model body, the input/output will be quantized as well.
        Corresponding to mlir_quantize's fully_quantize parameter.

    Raises:
      ValueError: when there are duplicate keys
    """
    self.layer_debug_metrics = layer_debug_metrics
    self.model_debug_metrics = model_debug_metrics
    self.layer_direct_compare_metrics = layer_direct_compare_metrics

    keys = []
    for metrics in [
        layer_debug_metrics, model_debug_metrics, layer_direct_compare_metrics
    ]:
      if metrics is not None:
        keys.extend(metrics.keys())
    if len(keys) != len(set(keys)):
      raise ValueError('Provided metrics have duplicate keys.')

    self.denylisted_ops = denylisted_ops
    self.denylisted_nodes = denylisted_nodes
    self.fully_quantize = fully_quantize


@tf_export.tf_export('lite.experimental.QuantizationDebugger')
class QuantizationDebugger:
  """Debugger for Quantized TensorFlow Lite debug mode models.

  This can run the TensorFlow Lite converted models equipped with debug ops and
  collect debug information. This debugger calculates statistics from
  user-defined post-processing functions as well as default ones.
  """

  def __init__(self,
               quant_debug_model_path: Optional[str] = None,
               quant_debug_model_content: Optional[bytes] = None,
               float_model_path: Optional[str] = None,
               float_model_content: Optional[bytes] = None,
               debug_dataset: Optional[Callable[
                   [], Iterable[Sequence[np.ndarray]]]] = None,
               debug_options: Optional[QuantizationDebugOptions] = None,
               converter: Optional[TFLiteConverter] = None) -> None:
    """Runs the TFLite debugging model with given debug options.

    Args:
      quant_debug_model_path: Path to the quantized debug TFLite model file.
      quant_debug_model_content: Content of the quantized debug TFLite model.
      float_model_path: Path to float TFLite model file.
      float_model_content: Content of the float TFLite model.
      debug_dataset: a factory function that returns dataset generator which is
        used to generate input samples (list of np.ndarray) for the model. The
        generated elements must have same types and shape as inputs to the
        model.
      debug_options: Debug options to debug the given model.
      converter: Optional, use converter instead of quantized model.

    Raises:
      ValueError: If the debugger was unable to be created.

    Attributes:
      layer_statistics: results of error metrics for each NumericVerify op
        results. in {layer_name: {metric_name: metric}} format.
      model_statistics: results of error metrics for difference between float
        and quantized models. in {metric_name: metric} format.
    """
    self._data_gen = debug_dataset
    self._debug_options = debug_options or QuantizationDebugOptions()
    self.converter = None
    self.calibrated_model = None
    self.float_model = None
    self._float_interpreter = None
    if converter is not None:
      if self._debug_options.model_debug_metrics:
        old_optimizations = converter.optimizations
        self.converter = self._set_converter_options_for_float(converter)
        self.float_model = self.converter.convert()
        converter.optimizations = old_optimizations

      self.converter = self._set_converter_options_for_calibration(converter)
      self.calibrated_model = self.converter.convert()
      # Converter should be already set up with all options
      self._init_from_converter(
          self._debug_options,
          self.converter,
          self.calibrated_model,
          float_model=self.float_model)
    else:
      self._quant_interpreter = _interpreter.Interpreter(
          quant_debug_model_path,
          quant_debug_model_content,
          experimental_preserve_all_tensors=(
              self._debug_options.layer_direct_compare_metrics is not None))
      if self._debug_options.model_debug_metrics:
        self._float_interpreter = _interpreter.Interpreter(
            float_model_path, float_model_content)
    self._initialize_stats()

  @property
  def options(self) -> QuantizationDebugOptions:
    return self._debug_options

  @options.setter
  def options(self, options: QuantizationDebugOptions) -> None:
    self._debug_options = options
    if not self.converter or not self.calibrated_model:
      return
    self._init_from_converter(
        self._debug_options,
        self.converter,
        self.calibrated_model,
        float_model=self.float_model)
    self._initialize_stats()

  def _initialize_stats(self):
    """Helper function initializes stats."""
    # TODO(b/177749613) : Fix the dependency on tf.lite._get_ops_details()
    # Following code is needed to get op's name from the output tensor index,
    # since NumericVerify op only provides its quantized input tensor index.
    self._defining_op = dict()
    for op_info in self._quant_interpreter._get_ops_details():  # pylint: disable=protected-access
      self._defining_op.update(
          {tensor_idx: op_info['index'] for tensor_idx in op_info['outputs']})

    self._numeric_verify_tensor_details = None
    self._numeric_verify_op_details = None
    if not self._get_numeric_verify_tensor_details():
      raise ValueError('Please check if the quantized model is in debug mode')

    self._layer_debug_metrics = _DEFAULT_LAYER_DEBUG_METRICS.copy()
    if self._debug_options.layer_debug_metrics:
      self._layer_debug_metrics.update(self._debug_options.layer_debug_metrics)

    self.layer_statistics = None
    self.model_statistics = None

    self._metrics = metrics_stub.TFLiteMetrics()
    self._metrics.increase_counter_debugger_creation()

  def _get_quantized_model(self, is_debug: bool) -> bytes:
    if not self.converter:
      raise ValueError('No converter found, use this function with the '
                       'converter option in the constructor.')

    return convert.mlir_quantize(
        self.calibrated_model,
        disable_per_channel=self.converter._experimental_disable_per_channel,  # pylint: disable=protected-access
        fully_quantize=self._debug_options.fully_quantize,
        enable_numeric_verify=is_debug,
        denylisted_ops=self._debug_options.denylisted_ops,
        denylisted_nodes=self._debug_options.denylisted_nodes)

  def get_nondebug_quantized_model(self) -> bytes:
    """Returns a non-instrumented quantized model.

    Convert the quantized model with the initialized converter and
    return bytes for nondebug model. The model will not be instrumented with
    numeric verification operations.

    Returns:
      Model bytes corresponding to the model.
    Raises:
      ValueError: if converter is not passed to the debugger.
    """
    return self._get_quantized_model(is_debug=False)

  def get_debug_quantized_model(self) -> bytes:
    """Returns an instrumented quantized model.

    Convert the quantized model with the initialized converter and
    return bytes for model. The model will be instrumented with numeric
    verification operations and should only be used for debugging.

    Returns:
      Model bytes corresponding to the model.
    Raises:
      ValueError: if converter is not passed to the debugger.
    """
    return self._get_quantized_model(is_debug=True)

  def _init_from_converter(self,
                           options: QuantizationDebugOptions,
                           converter: TFLiteConverter,
                           calibrated_model: Optional[bytes] = None,
                           float_model: Optional[bytes] = None) -> None:
    """Convert the model and apply options.

    Converts the quantized model and initializes a quantized model interpreter
    with the quantized model. Returns a float model interpreter if float model
    is provided.

    Args:
      options: a QuantizationDebugOptions object.
      converter: an initialized tf.lite.TFLiteConverter.
      calibrated_model: Calibrated model bytes.
      float_model: Float model bytes.
    """
    self.quant_model = convert.mlir_quantize(
        calibrated_model,
        disable_per_channel=converter._experimental_disable_per_channel,  # pylint: disable=protected-access
        fully_quantize=options.fully_quantize,
        enable_numeric_verify=True,
        denylisted_ops=options.denylisted_ops,
        denylisted_nodes=options.denylisted_nodes)
    self._quant_interpreter = _interpreter.Interpreter(
        model_content=self.quant_model)
    self._float_interpreter = None
    if float_model is not None:
      self._float_interpreter = _interpreter.Interpreter(
          model_content=float_model)

  def _set_converter_options_for_float(
      self, converter: TFLiteConverter) -> TFLiteConverter:
    """Verify converter options and set required experimental options."""
    if converter.optimizations:
      converter.optimizations = []
    return converter

  def _set_converter_options_for_calibration(
      self, converter: TFLiteConverter) -> TFLiteConverter:
    """Verify converter options and set required experimental options."""
    if not converter.optimizations:
      raise ValueError(
          'converter object must set optimizations to lite.Optimize.DEFAULT')
    if not converter.representative_dataset:
      raise ValueError('converter object must set representative_dataset')

    converter.experimental_mlir_quantizer = True
    converter._experimental_calibrate_only = True  # pylint: disable=protected-access
    return converter

  def run(self) -> None:
    """Runs models and gets metrics."""
    self.layer_statistics = self._collect_layer_statistics()
    if self._debug_options.model_debug_metrics:
      self.model_statistics = self._collect_model_statistics()

  def _collect_layer_statistics(self) -> Dict[str, Dict[str, float]]:
    """Collects layer statistics by applying layer debug metrics.

    For all data from the given RepresentativeDataset, collect statistics per
    example by getting the NumericVerify op results in _quant_interpreter
    and calculating layer debug metrics on the results.

    Returns:
      aggregated per-layer statistics of NumericVerify results.
      {layer_name: {metric_name: metric}}
    """
    layer_statistics = collections.defaultdict(
        lambda: collections.defaultdict(list))

    initialize = True
    for tensor_data in self._data_gen():
      self._set_input_tensors(self._quant_interpreter, tensor_data, initialize)
      initialize = False

      # Run the model.
      self._quant_interpreter.invoke()

      # Collect the statistics of this invoke result.
      for tensor_detail in self._get_numeric_verify_tensor_details():
        tensor_name = tensor_detail['name']  # pytype: disable=unsupported-operands  # dynamic-method-lookup
        diffs = self._quant_interpreter.get_tensor(tensor_detail['index'])  # pytype: disable=unsupported-operands  # dynamic-method-lookup
        for metric_name, metric_fn in self._layer_debug_metrics.items():
          layer_statistics[tensor_name][metric_name].append(metric_fn(diffs))

      if self._debug_options.layer_direct_compare_metrics is not None:
        for tensor_detail in self._get_numeric_verify_tensor_details():
          tensor_name = tensor_detail['name']  # pytype: disable=unsupported-operands  # dynamic-method-lookup
          op_idx = self._defining_op[tensor_detail['index']]  # pytype: disable=unsupported-operands  # dynamic-method-lookup
          op_detail = self._quant_interpreter._get_op_details(op_idx)  # pylint: disable=protected-access
          q_idx, f_idx = op_detail['inputs']
          quant_input_detail = self._quant_interpreter._get_tensor_details(  # pylint: disable=protected-access
              q_idx, subgraph_index=0)
          for (metric_name, metric_fn
              ) in self._debug_options.layer_direct_compare_metrics.items():
            layer_statistics[tensor_name][metric_name].append(
                metric_fn(
                    self._quant_interpreter.get_tensor(f_idx),
                    self._quant_interpreter.get_tensor(q_idx),
                    quant_input_detail['quantization_parameters']['scales'][0],
                    quant_input_detail['quantization_parameters']['zero_points']
                    [0]))

    # Calculate final aggregated metrics for each layer.
    for metrics in layer_statistics.values():
      for metric_name in metrics:
        metrics[metric_name] = np.nanmean(metrics[metric_name])

    return layer_statistics

  def _collect_model_statistics(self) -> Dict[str, float]:
    """Collects model output metrics.

    For all data from the given RepresentativeDataset, collect all model output
    results from float model & quantized debug model, and calculate metrics
    by using model output functions. As a result, self.model_results is filled,

    where self.model_results[model_output_function_name] = `aggregated model
    output function value` (a scalar).

    Returns:
      aggregated per-model output discrepancy metrics.
      {metric_name: aggregated_metric}
    """

    model_statistics = collections.defaultdict(list)

    initialize = True
    for tensor_data in self._data_gen():
      self._set_input_tensors(self._quant_interpreter, tensor_data, initialize)
      self._set_input_tensors(self._float_interpreter, tensor_data, initialize)
      initialize = False

      # Run the models.
      self._quant_interpreter.invoke()
      self._float_interpreter.invoke()

      # Collect the output results from both models.
      float_tensor_data = self._get_output_tensors(self._float_interpreter)
      quant_tensor_data = self._get_output_tensors(self._quant_interpreter)

      # Calculate the metrics.
      for (metric_name,
           metric_fn) in self._debug_options.model_debug_metrics.items():
        model_statistics[metric_name].append(
            metric_fn(float_tensor_data, quant_tensor_data))

    # Calculate final aggregated metrics for each outputs.
    return {
        metric_name: np.mean(metric)
        for metric_name, metric in model_statistics.items()
    }

  def _set_input_tensors(self, interpreter: _interpreter.Interpreter,
                         tensor_data: Sequence[np.ndarray],
                         initialize: bool) -> None:
    """Sets input tensors into TFLite model Interpreter.

    Args:
      interpreter: a tf.lite.Interpreter object with allocated tensors.
      tensor_data: a list of Numpy array data.
      initialize: set to true when input is first set for the interpreter, to
        set input shapes and allocate tensors.

    Raises:
      ValueError: when inputs can't be set, or size of provided inputs does not
      match size of model inputs.
    """
    input_details = interpreter.get_input_details()
    if len(input_details) != len(tensor_data):
      raise ValueError(
          'Number of inputs provided ({}) does not match number of inputs to '
          'the model ({})'.format(len(tensor_data), len(input_details)))

    if initialize:
      for input_detail, tensor in zip(input_details, tensor_data):
        interpreter.resize_tensor_input(input_detail['index'], tensor.shape)
      interpreter.allocate_tensors()

    for input_detail, tensor in zip(input_details, tensor_data):
      if tensor.dtype == np.float32 and input_detail['dtype'] == np.int8:
        quant_params = _get_quant_params(input_detail)
        if quant_params:
          scale, zero_point = quant_params
          tensor = np.round((tensor / scale) + zero_point).astype(np.int8)
      interpreter.set_tensor(input_detail['index'], tensor)

  def _get_output_tensors(
      self, interpreter: _interpreter.Interpreter) -> List[np.ndarray]:
    """Returns output tensors of given TFLite model Interpreter.

    Args:
      interpreter: a tf.lite.Interpreter object with allocated tensors.

    Returns:
      a list of numpy arrays representing output tensor results.
    """

    outputs = []
    for output_detail in interpreter.get_output_details():
      tensor = interpreter.get_tensor(output_detail['index'])
      if output_detail['dtype'] == np.int8:
        quant_params = _get_quant_params(output_detail)
        if quant_params:
          scale, zero_point = quant_params
          tensor = ((tensor.astype(np.float32) - zero_point) * scale).astype(
              np.float32)
      outputs.append(tensor)

    return outputs

  def _get_numeric_verify_tensor_details(self) -> List[str]:
    """Returns all names of all tensors from NumericVerify op."""
    # pylint: disable=protected-access
    if not self._numeric_verify_tensor_details:
      self._numeric_verify_tensor_details = []
      self._numeric_verify_op_details = {}
      for op_info in self._quant_interpreter._get_ops_details():
        if op_info['op_name'] == _NUMERIC_VERIFY_OP_NAME:
          self._numeric_verify_tensor_details.append(
              self._quant_interpreter._get_tensor_details(
                  op_info['outputs'][0], subgraph_index=0))
          tensor_name = self._numeric_verify_tensor_details[-1]['name']
          self._numeric_verify_op_details[tensor_name] = op_info
    # pylint: enable=protected-access
    return self._numeric_verify_tensor_details

  def _get_operand_name_and_index(self,
                                  numeric_verify_name: str) -> Tuple[str, int]:
    """Gets the index and name of NumericVerify Op's quantized input tensor.

    Args:
      numeric_verify_name: name of the NumericVerify op's output tensor. It has
        format of `NumericVerify/{quantized_tensor_name}:{quantized_tensor_idx}`

    Returns:
      Tuple of (tensor_name, tensor_idx) for quantized op's output tensor.
    """
    tensor_name, tensor_idx = numeric_verify_name.rsplit(':', 1)
    float_tensor_name = tensor_name[len(_NUMERIC_VERIFY_OP_NAME) + 1:]
    if re.match(r'\d', float_tensor_name[-1]):
      float_tensor_name = float_tensor_name[:-1]

    return (float_tensor_name, int(tensor_idx))

  def layer_statistics_dump(self, file: IO[str]) -> None:
    """Dumps layer statistics into file, in csv format.

    Args:
      file: file, or file-like object to write.
    """
    # order of `fields` is the order of fields in csv.
    fields = ['op_name', 'tensor_idx'] + list(self._layer_debug_metrics.keys())
    if self._debug_options.layer_direct_compare_metrics is not None:
      fields += list(self._debug_options.layer_direct_compare_metrics.keys())
    fields += ['scale', 'zero_point', 'tensor_name']
    writer = csv.DictWriter(file, fields)
    writer.writeheader()
    for name, metrics in self.layer_statistics.items():
      data = metrics.copy()
      (data['tensor_name'], _) = self._get_operand_name_and_index(name)
      data['tensor_idx'] = self._numeric_verify_op_details[name]['inputs'][0]
      data['op_name'] = self._quant_interpreter._get_op_details(  # pylint: disable=protected-access
          self._defining_op[data['tensor_idx']])['op_name']
      details = self._quant_interpreter._get_tensor_details(  # pylint: disable=protected-access
          data['tensor_idx'], subgraph_index=0)
      data['scale'], data['zero_point'] = (
          details['quantization_parameters']['scales'][0],
          details['quantization_parameters']['zero_points'][0])
      writer.writerow(data)
