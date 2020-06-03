# Lint as: python2, python3
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
"""TensorFlow Lite tooling helper functionality."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
import shutil
import tempfile
import warnings

from absl import logging
import six
from six import PY2

from google.protobuf import text_format as _text_format
from google.protobuf.message import DecodeError
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.lite.experimental.examples.lstm.rnn import dynamic_rnn  # pylint: disable=unused-import
from tensorflow.lite.experimental.examples.lstm.rnn_cell import TFLiteLSTMCell  # pylint: disable=unused-import
from tensorflow.lite.experimental.examples.lstm.rnn_cell import TfLiteRNNCell  # pylint: disable=unused-import
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op  # pylint: disable=unused-import
from tensorflow.lite.experimental.tensorboard.ops_util import get_potentially_supported_ops  # pylint: disable=unused-import
from tensorflow.lite.python import lite_constants as constants
from tensorflow.lite.python.convert import build_toco_convert_protos  # pylint: disable=unused-import
from tensorflow.lite.python.convert import ConverterError  # pylint: disable=unused-import
from tensorflow.lite.python.convert import mlir_quantize as _mlir_quantize
from tensorflow.lite.python.convert import mlir_sparsify as _mlir_sparsify
from tensorflow.lite.python.convert import OpsSet
from tensorflow.lite.python.convert import toco_convert  # pylint: disable=unused-import
from tensorflow.lite.python.convert import toco_convert_graph_def as _toco_convert_graph_def
from tensorflow.lite.python.convert import toco_convert_impl as _toco_convert_impl
from tensorflow.lite.python.convert import toco_convert_protos  # pylint: disable=unused-import
from tensorflow.lite.python.convert_saved_model import freeze_saved_model as _freeze_saved_model
from tensorflow.lite.python.interpreter import Interpreter  # pylint: disable=unused-import
from tensorflow.lite.python.interpreter import load_delegate  # pylint: disable=unused-import
from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs  # pylint: disable=unused-import
from tensorflow.lite.python.op_hint import is_ophint_converted as _is_ophint_converted
from tensorflow.lite.python.op_hint import OpHint  # pylint: disable=unused-import
from tensorflow.lite.python.optimize import calibrator as _calibrator
from tensorflow.lite.python.util import build_debug_info_func as _build_debug_info_func
from tensorflow.lite.python.util import convert_debug_info_func as _convert_debug_info_func
from tensorflow.lite.python.util import freeze_graph as _freeze_graph
from tensorflow.lite.python.util import get_debug_info as _get_debug_info
from tensorflow.lite.python.util import get_grappler_config as _get_grappler_config
from tensorflow.lite.python.util import get_tensor_name as _get_tensor_name
from tensorflow.lite.python.util import get_tensors_from_tensor_names as _get_tensors_from_tensor_names
from tensorflow.lite.python.util import is_frozen_graph as _is_frozen_graph
from tensorflow.lite.python.util import run_graph_optimizations as _run_graph_optimizations
from tensorflow.lite.python.util import set_tensor_shapes as _set_tensor_shapes
from tensorflow.python import keras as _keras
from tensorflow.python.client import session as _session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function as _def_function
from tensorflow.python.eager import function as _function
from tensorflow.python.framework import convert_to_constants as _convert_to_constants
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework.errors_impl import NotFoundError as _NotFoundError
from tensorflow.python.framework.importer import import_graph_def as _import_graph_def
from tensorflow.python.keras.saving import saving_utils as _saving_utils
from tensorflow.python.lib.io import file_io as _file_io
from tensorflow.python.saved_model import loader_impl as _loader_impl
from tensorflow.python.saved_model import signature_constants as _signature_constants
from tensorflow.python.saved_model import tag_constants as _tag_constants
from tensorflow.python.saved_model.load import load as _load
from tensorflow.python.saved_model.loader_impl import parse_saved_model_with_debug_info as _parse_saved_model_with_debug_info
from tensorflow.python.util import deprecation as _deprecation
from tensorflow.python.util.tf_export import tf_export as _tf_export


# The default value of `experimental_new_converter`.
_USE_EXPERIMENTAL_NEW_CONVERTER = True


@_tf_export("lite.Optimize")
class Optimize(enum.Enum):
  """Enum defining the optimizations to apply when generating tflite graphs.

  Some optimizations may come at the cost of accuracy.

  DEFAULT
      Default optimization strategy.

      Converter will do its best to improve size and latency based on the
      information provided.
      Enhanced optimizations are gained by providing a representative_dataset.
      This is recommended, and is currently equivalent to the modes below.
      Currently, weights will be quantized and if representative_dataset is
      provided, activations for quantizable operations will also be quantized.

  OPTIMIZE_FOR_SIZE
      Deprecated. Does the same as DEFAULT.

  OPTIMIZE_FOR_LATENCY
      Deprecated. Does the same as DEFAULT.
  """

  # Default optimization strategy.
  #
  # Converter will do its best to improve size and latency based on the
  # information provided.
  # Enhanced optimizations can be gained by providing a representative_dataset.
  # This is recommended, and is currently equivalent to the modes below.
  # Currently, weights will be quantized and if representative_dataset is
  # provided, activations for quantizable operations will also be quantized.
  DEFAULT = "DEFAULT"

  # Deprecated. Does the same as DEFAULT.
  OPTIMIZE_FOR_SIZE = "OPTIMIZE_FOR_SIZE"

  # Deprecated. Does the same as DEFAULT.
  OPTIMIZE_FOR_LATENCY = "OPTIMIZE_FOR_LATENCY"

  def __str__(self):
    return self.value


@_tf_export("lite.RepresentativeDataset")
class RepresentativeDataset(object):
  """Representative dataset to evaluate optimizations.

  A representative dataset that can be used to evaluate optimizations by the
  converter. E.g. converter can use these examples to estimate (min, max) ranges
  by calibrating the model on inputs. This can allow converter to quantize a
  converted floating point model.
  """

  def __init__(self, input_gen):
    """Creates a representative dataset.

    Args:
      input_gen: an input generator that can be used to generate input samples
        for the model. This must be a callable object that returns an object
        that supports the `iter()` protocol (e.g. a generator function). The
        elements generated must have same type and shape as inputs to the model.
    """
    self.input_gen = input_gen


@_tf_export("lite.TargetSpec")
class TargetSpec(object):
  """Specification of target device.

  Details about target device. Converter optimizes the generated model for
  specific device.

  Attributes:
    supported_ops: Experimental flag, subject to change. Set of OpsSet options
      supported by the device. (default set([OpsSet.TFLITE_BUILTINS]))
    supported_types: List of types for constant values on the target device.
      Supported values are types exported by lite.constants. Frequently, an
      optimization choice is driven by the most compact (i.e. smallest) type in
      this list (default [constants.FLOAT])
  """

  def __init__(self, supported_ops=None, supported_types=None):
    if supported_ops is None:
      supported_ops = set([OpsSet.TFLITE_BUILTINS])
    self.supported_ops = supported_ops
    if supported_types is None:
      supported_types = []
    self.supported_types = supported_types


class QuantizationMode(object):
  """QuantizationMode determines the quantized conversion from user options."""

  def __init__(self, optimizations, target_spec, representative_dataset,
               graph_def):
    self._optimizations = optimizations
    self._target_spec = target_spec
    self._representative_dataset = representative_dataset
    self._graph_def = graph_def

    self._validate_int8_required()

  def post_training_int8_no_float(self):
    """Post training int8 quantize, disallow float fallback."""
    return (self._is_int8_target_required() and
            not self._is_int16x8_target_required() and
            self._representative_dataset is not None)

  def post_training_int8_allow_float(self):
    """Post training int8 quantize, allow float fallback."""
    return (self._any_optimization_enabled() and
            not self._is_int16x8_target_required() and
            self._representative_dataset is not None and
            self._smallest_supported_type() == constants.INT8)

  def training_time_int8_allow_float(self):
    """Training-time int8 quantize, allow float fallback."""
    return (self._any_optimization_enabled() and
            not self.post_training_dynamic_range_int8() and
            not self.post_training_fp16())

  def post_training_int16x8_no_float(self):
    """Post training int16x8 quantize, disallow float fallback."""
    return (not self._is_int8_target_required() and
            self._is_int16x8_target_required() and
            not self._is_allow_float() and
            self._representative_dataset is not None)

  def post_training_int16x8_allow_float(self):
    """Post training int16x8 quantize, allow float fallback."""
    return (self._is_int16x8_target_required() and
            self._is_allow_float())

  def post_training_dynamic_range_int8(self):
    """Post training int8 const, on-the-fly int8 quantize of dynamic tensors."""
    # Post-training dynamic range quantization is only enabled if post-training
    # int8 quantization and training time quantization was not done.
    return (self._any_optimization_enabled() and
            self._representative_dataset is None and
            not self.contains_training_quant_op() and
            self._smallest_supported_type() == constants.INT8)

  def post_training_fp16(self):
    """Post training fp16 quantize."""
    return (self._any_optimization_enabled() and
            self._smallest_supported_type() == constants.FLOAT16)

  def fp32_execution(self):
    """If none of the above are true."""
    return not (self.post_training_int8_no_float() or
                self.post_training_int8_allow_float() or
                self.training_time_int8_allow_float() or
                self.post_training_int16x8_no_float() or
                self.post_training_int16x8_allow_float() or
                self.post_training_dynamic_range_int8() or
                self.post_training_fp16())

  def activations_type(self):
    return constants.INT16 if self._is_int16x8_target_required() else constants.INT8

  def converter_flags(self, inference_ty=None, inference_input_ty=None):
    """Flags to the converter."""
    if (self.post_training_int8_no_float() or
        self.post_training_int8_allow_float()):
      # The inference_input_type is for the quantizer, then we need to keep the
      # converter inference_iput_type to float.
      inference_input_ty = constants.FLOAT

    if self.training_time_int8_allow_float():
      return {
          "inference_type": inference_ty if inference_ty else self.activations_type(),
          "inference_input_type":
              inference_input_ty if inference_input_ty else constants.FLOAT,
          "post_training_quantize": False,  # disable dynamic range quantization
          "quantize_to_float16": False  # disable float16 quantization
      }
    elif self.post_training_dynamic_range_int8():
      return {
          "inference_type": constants.FLOAT,
          "inference_input_type": constants.FLOAT,
          "post_training_quantize": True,  # enable dynamic range quantization
          "quantize_to_float16": False  # disable float16 quantization
      }
    elif self.post_training_fp16():
      return {
          "inference_type": constants.FLOAT,
          "inference_input_type": constants.FLOAT,
          "post_training_quantize": True,
          "quantize_to_float16": True  # enable float16 quantization
      }
    else:
      # Note this might still trigger (uint8) quantization to be compatible with
      # TOCO.
      return {
          "inference_type": inference_ty if inference_ty else constants.FLOAT,
          "inference_input_type": inference_input_ty,
          "post_training_quantize": False,  # enable dynamic range quantization
          "quantize_to_float16": False  # disable float16 quantization
      }

  def quantizer_flags(self, input_ty=None, output_ty=None):
    """Default flags to the TFMOT quantizer."""

    inference_input_type = input_ty if input_ty else constants.FLOAT
    inference_output_type = output_ty if output_ty else constants.FLOAT
    if self.post_training_int8_no_float():
      return True, {
          "inference_input_type": inference_input_type,
          "inference_output_type": inference_output_type,
          "activations_type": constants.INT8,
          "allow_float": False
      }
    elif self.post_training_int8_allow_float():
      return True, {
          "inference_input_type": inference_input_type,
          "inference_output_type": inference_output_type,
          "activations_type": constants.INT8,
          "allow_float": True
      }
    elif self.post_training_int16x8_no_float():
      return True, {
          "inference_input_type": inference_input_type,
          "inference_output_type": inference_output_type,
          "activations_type": constants.INT16,
          "allow_float": False
      }
    elif self.post_training_int16x8_allow_float():
      return True, {
          "inference_input_type": inference_input_type,
          "inference_output_type": inference_output_type,
          "activations_type": constants.INT16,
          "allow_float": True
      }
    else:
      return False, None

  # Below are helpers for the above functions.

  def _validate_int8_required(self):
    """Int8 mode requires certain parameters to exist and be compatible."""
    if not self._is_int8_target_required():
      return

    if self._target_spec.supported_types and (self._smallest_supported_type() !=
                                              constants.INT8):
      raise ValueError("TFLITE_BUILTINS_INT8 requires smallest supported "
                       "type to be INT8.")

    if self._representative_dataset:
      if not isinstance(self._representative_dataset, RepresentativeDataset):
        self._representative_dataset = RepresentativeDataset(
            self._representative_dataset)
      if self._representative_dataset.input_gen is None:
        raise ValueError(
            "Provide an input generator for representative_dataset")
    else:
      # TODO(b/150661651): Relax this check for QAT.
      raise ValueError("representative_dataset is required when specifying "
                       "TFLITE_BUILTINS_INT8 or INT8 supported types.")

  def _is_int8_target_required(self):
    return (set([OpsSet.TFLITE_BUILTINS_INT8]) == set(
        self._target_spec.supported_ops) or
            set(self._target_spec.supported_types) == set([constants.INT8]))

  def _is_int16x8_target_required(self):
    return bool(
          set(self._target_spec.supported_ops).intersection([
            OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ]))

  def _is_allow_float(self):
    return bool(
          set(self._target_spec.supported_ops).intersection([
            OpsSet.TFLITE_BUILTINS
        ]))

  def _any_optimization_enabled(self):
    return bool(
        set(self._optimizations).intersection([
            Optimize.OPTIMIZE_FOR_LATENCY, Optimize.OPTIMIZE_FOR_SIZE,
            Optimize.DEFAULT
        ]))

  def _smallest_supported_type(self):
    if self._target_spec.supported_types:
      return min(self._target_spec.supported_types, key=lambda x: x.size)
    else:
      # The default smallest supported type is INT8.
      return constants.INT8

  def contains_training_quant_op(self):
    """Checks if the graph contains any training-time quantization ops."""
    training_quant_ops = frozenset({
        "FakeQuantWithMinMaxVars", "FakeQuantWithMinMaxVarsPerChannel",
        "QuantizeAndDequantizeV2", "QuantizeAndDequantizeV3"
    })

    for node_def in self._graph_def.node:
      if any(op in node_def.name for op in training_quant_ops):
        return True
    return False


class TFLiteConverterBase(object):
  """Converter subclass to share functionality between V1 and V2 converters."""

  def __init__(self):
    self.allow_custom_ops = False
    self.target_spec = TargetSpec()
    self.optimizations = []
    self.representative_dataset = None
    self.experimental_new_converter = _USE_EXPERIMENTAL_NEW_CONVERTER
    self._experimental_new_quantizer = False
    self._experimental_calibrate_only = False
    # The 'GraphDebugInfo'  contains the stack traces of all the original nodes
    # in the `GraphDef` to the converter.
    self._debug_info = None
    self.saved_model_dir = None
    self._saved_model_tags = None
    self._saved_model_version = 0
    self._saved_model_exported_names = []
    self._experimental_sparsify_model = False

  def _grappler_config(self, optimizers=None):
    """Creates a tf.compat.v1.ConfigProto for configuring Grappler.

    Args:
      optimizers: List of strings that represents the list of optimizers.

    Returns:
      tf.ConfigProto.
    """
    if not optimizers:
      optimizers = []
    # MLIR converter will take care of constant folding instead of grappler.
    if not self.experimental_new_converter:
      optimizers.append("constfold")

    is_only_flex_enabled = (
        set([OpsSet.SELECT_TF_OPS]) == set(self.target_spec.supported_ops))
    if is_only_flex_enabled:
      # The layout optimizer turns NHCW to NCHW. This provides performance
      # optimizations when Flex mode is enabled. However, this is not compatible
      # with builtin ops.
      optimizers.append("layout")
    return _get_grappler_config(optimizers)

  def _calibrate_quantize_model(self, result, inference_input_type,
                                inference_output_type, activations_type, allow_float):
    """Calibrate and quantize the model."""
    if not isinstance(self.representative_dataset, RepresentativeDataset):
      self.representative_dataset = RepresentativeDataset(
          self.representative_dataset)

    calibrate_quantize = _calibrator.Calibrator(result)
    if self._experimental_calibrate_only or self._experimental_new_quantizer:
      calibrated = calibrate_quantize.calibrate(
          self.representative_dataset.input_gen)

    if self._experimental_calibrate_only:
      return calibrated
    elif self._experimental_new_quantizer:
      return _mlir_quantize(calibrated)
    else:
      return calibrate_quantize.calibrate_and_quantize(
        self.representative_dataset.input_gen, inference_input_type,
        inference_output_type, allow_float, activations_type)

  def _is_unknown_shapes_allowed(self):
    # Unknown dimensions are only allowed with the new converter.
    return self.experimental_new_converter

  def _get_base_converter_args(self):
    """Returns the base converter args.

    Returns:
      {key str: val}
    """
    args = {
        "input_format": constants.TENSORFLOW_GRAPHDEF,
        "allow_custom_ops": self.allow_custom_ops,
        "debug_info": self._debug_info,
        "target_ops": self.target_spec.supported_ops,
        "enable_mlir_converter": self.experimental_new_converter,
    }

    if self.saved_model_dir:
      args.update({
          "saved_model_dir": self.saved_model_dir,
          "saved_model_version": self._saved_model_version,
          "saved_model_tags": self._saved_model_tags,
          "saved_model_exported_names": self._saved_model_exported_names,
      })

    return args

  def _contains_function_with_implements_attr(self, saved_model_proto):
    meta_graph = saved_model_proto.meta_graphs[0]
    for function in meta_graph.graph_def.library.function:
      if function.attr.get("_implements", None) or function.attr.get(
          "api_implements", None):
        return True
    return False

  def _parse_saved_model_args(self):
    """Parses SavedModel arguments from the given Keras/RNN SavedModel."""
    if not self.experimental_new_converter:
      self.saved_model_dir = None
      return
    if self.saved_model_dir:
      try:
        saved_model_proto, _ = (
            _parse_saved_model_with_debug_info(self.saved_model_dir))
      except OSError:
        # If it fails to read the given saved model, it will fall back to the
        # frozen graph def path.
        self.saved_model_dir = None
        return
      if not self._contains_function_with_implements_attr(saved_model_proto):
        self.saved_model_dir = None
      else:
        if not self._saved_model_exported_names:
          self._saved_model_exported_names = []
        self._saved_model_version = saved_model_proto.saved_model_schema_version
        if self._saved_model_version not in [1, 2]:
          raise ValueError(
              "SavedModel file format({0}) is not supported".format(
                  self._saved_model_version))


class TFLiteConverterBaseV2(TFLiteConverterBase):
  """Converter subclass to share functionality between V2 converters."""

  def convert(self, graph_def, input_tensors, output_tensors):
    """Converts a TensorFlow GraphDef based on instance variables.

    Args:
      graph_def: Frozen TensorFlow GraphDef.
      input_tensors: List of input tensors. Type and shape are computed using
        `foo.shape` and `foo.dtype`.
      output_tensors: List of output tensors (only .name is used from this).

    Returns:
      The converted data in serialized format.

    Raises:
      ValueError:
        No concrete functions is specified.
        Multiple concrete functions are specified.
        Input shape is not specified.
        Invalid quantization parameters.
    """
    quant_mode = QuantizationMode(self.optimizations, self.target_spec,
                                  self.representative_dataset, graph_def)

    if not self._is_unknown_shapes_allowed():
      # Checks dimensions in input tensor.
      for tensor in input_tensors:
        # Note that shape_list might be empty for scalar shapes.
        shape_list = tensor.shape.as_list()
        if None in shape_list[1:]:
          raise ValueError(
              "None is only supported in the 1st dimension. Tensor '{0}' has "
              "invalid shape '{1}'.".format(
                  _get_tensor_name(tensor), shape_list))
        elif shape_list and shape_list[0] is None:
          # Set the batch size to 1 if undefined.
          shape = tensor.shape.as_list()
          shape[0] = 1
          tensor.set_shape(shape)

    if self._trackable_obj is None:
      self._debug_info = _get_debug_info(
          _build_debug_info_func(self._funcs[0].graph), graph_def)
    else:
      self._debug_info = _get_debug_info(
          _convert_debug_info_func(self._trackable_obj.graph_debug_info),
          graph_def)

    converter_kwargs = self._get_base_converter_args()
    converter_kwargs.update(quant_mode.converter_flags())

    if not self.experimental_new_converter:
      logging.warning(
          "Please consider switching to use new converter by setting "
          "experimental_new_converter to true. "
          "Old converter (TOCO) is deprecated and flow will be switched on "
          "by default to use new converter soon.")
    else:
      logging.info("Using experimental converter: If you encountered a problem "
                   "please file a bug. You can opt-out "
                   "by setting experimental_new_converter=False")

    # Converts model.
    result = _toco_convert_impl(
        input_data=graph_def,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        **converter_kwargs)

    calibrate_and_quantize, flags = quant_mode.quantizer_flags()
    if calibrate_and_quantize:
      result = self._calibrate_quantize_model(result, **flags)

    if self._experimental_sparsify_model:
      result = _mlir_sparsify(result)

    return result


class TFLiteSavedModelConverterV2(TFLiteConverterBaseV2):
  """Converts the given SavedModel into TensorFlow Lite model.

  Attributes:
      saved_model_dir: Directory of the SavedModel.
  """

  def __init__(self,
               saved_model_dir,
               saved_model_tags=None,
               saved_model_exported_names=None,
               trackable_obj=None):
    """Constructor for TFLiteConverter.

    Args:
      saved_model_dir: Directory of the SavedModel.
      saved_model_tags: Set of tags identifying the MetaGraphDef within the
        SavedModel to analyze. All tags in the tag set must be present. (default
        set(SERVING)).
      saved_model_exported_names: Names to be exported (default: export all)
        when the saved model import path is on.
      trackable_obj: tf.AutoTrackable object associated with `funcs`. A
        reference to this object needs to be maintained so that Variables do not
        get garbage collected since functions have a weak reference to
        Variables. This is only required when the tf.AutoTrackable object is not
        maintained by the user (e.g. `from_saved_model`).
    """
    super(TFLiteSavedModelConverterV2, self).__init__()
    self.saved_model_dir = saved_model_dir
    self._saved_model_tags = saved_model_tags
    self._saved_model_exported_names = saved_model_exported_names
    self._trackable_obj = trackable_obj
    self._parse_saved_model_args()

  def convert(self):
    """Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format.

    Raises:
      ValueError:
        No concrete functions is specified.
        Multiple concrete functions are specified.
        Input shape is not specified.
        Invalid quantization parameters.
    """
    graph = _ops.Graph()
    saved_model = _loader_impl.SavedModelLoader(self.saved_model_dir)
    saved_model.load_graph(graph, tags=self._saved_model_tags)
    meta_graph = saved_model.get_meta_graph_def_from_tags(
        self._saved_model_tags)
    signature_def = meta_graph.signature_def[
        _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    input_tensors = [
        graph.get_tensor_by_name(signature_def.inputs[key].name)
        for key in signature_def.inputs
    ]
    output_tensors = [
        graph.get_tensor_by_name(signature_def.outputs[key].name)
        for key in signature_def.outputs
    ]
    return super(TFLiteSavedModelConverterV2,
                 self).convert(meta_graph.graph_def, input_tensors,
                               output_tensors)


class TFLiteKerasModelConverterV2(TFLiteConverterBaseV2):
  """Converts the given Keras model into TensorFlow Lite model."""

  def __init__(self, keras_model, trackable_obj=None):
    """Constructor for TFLiteConverter.

    Args:
      keras_model: tf.Keras.Model.
      trackable_obj: tf.AutoTrackable object associated with `funcs`. A
        reference to this object needs to be maintained so that Variables do not
        get garbage collected since functions have a weak reference to
        Variables. This is only required when the tf.AutoTrackable object is not
        maintained by the user (e.g. `from_saved_model`).
    """
    super(TFLiteKerasModelConverterV2, self).__init__()
    self._keras_model = keras_model
    self._trackable_obj = trackable_obj

  def convert(self):
    """Converts a keras model based on instance variables.

    Returns:
      The converted data in serialized format.

    Raises:
      ValueError:
        Multiple concrete functions are specified.
        Input shape is not specified.
        Invalid quantization parameters.
    """
    temp_dir = tempfile.mkdtemp()
    try:
      self._keras_model.save(temp_dir, save_format="tf")
      self.saved_model_dir = temp_dir
      self._saved_model_tags = set([_tag_constants.SERVING])
      self._saved_model_exported_names = [
          _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
      ]
      self._parse_saved_model_args()
      if self.saved_model_dir:
        graph = _ops.Graph()
        saved_model = _loader_impl.SavedModelLoader(self.saved_model_dir)
        saved_model.load_graph(graph, tags=self._saved_model_tags)
        meta_graph = saved_model.get_meta_graph_def_from_tags(
            self._saved_model_tags)
        signature_def = meta_graph.signature_def[
            _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        input_tensors = [
            graph.get_tensor_by_name(signature_def.inputs[key].name)
            for key in signature_def.inputs
        ]
        output_tensors = [
            graph.get_tensor_by_name(signature_def.outputs[key].name)
            for key in signature_def.outputs
        ]
        self._trackable_obj = _load(self.saved_model_dir,
                                    self._saved_model_tags)
        return super(TFLiteKerasModelConverterV2,
                     self).convert(meta_graph.graph_def, input_tensors,
                                   output_tensors)
    finally:
      shutil.rmtree(temp_dir, True)

    input_signature = None
    # If the model's call is not a `tf.function`, then we need to first get its
    # input signature from `model_input_signature` method. We can't directly
    # call `trace_model_call` because otherwise the batch dimension is set
    # to None.
    # Once we have better support for dynamic shapes, we can remove this.
    if not isinstance(self._keras_model.call, _def_function.Function):
      # Pass `keep_original_batch_size=True` will ensure that we get an input
      # signature including the batch dimension specified by the user.
      input_signature = _saving_utils.model_input_signature(
          self._keras_model, keep_original_batch_size=True)

    func = _saving_utils.trace_model_call(self._keras_model, input_signature)
    concrete_func = func.get_concrete_function()
    self._funcs = [concrete_func]

    frozen_func, graph_def = (
        _convert_to_constants.convert_variables_to_constants_v2_as_graph(
            self._funcs[0], lower_control_flow=False))

    input_tensors = [
        tensor for tensor in frozen_func.inputs
        if tensor.dtype != _dtypes.resource
    ]
    output_tensors = frozen_func.outputs

    # Run a Grappler pass.
    grappler_config = self._grappler_config()
    # Skip running grappler when there are no optimizers to run. If not,
    # grappler will run with the default optimizer set and it will lead to
    # causing an unexpected behavior.
    if grappler_config.graph_options.rewrite_options.optimizers:
      graph_def = _run_graph_optimizations(
          graph_def,
          input_tensors,
          output_tensors,
          config=grappler_config,
          graph=frozen_func.graph)

    return super(TFLiteKerasModelConverterV2,
                 self).convert(graph_def, input_tensors, output_tensors)


class TFLiteFrozenGraphConverterV2(TFLiteConverterBaseV2):
  """Converts the given frozen graph into TensorFlow Lite model."""

  def __init__(self, funcs, trackable_obj=None):
    """Constructor for TFLiteConverter.

    Args:
      funcs: List of TensorFlow ConcreteFunctions. The list should not contain
        duplicate elements.
      trackable_obj: tf.AutoTrackable object associated with `funcs`. A
        reference to this object needs to be maintained so that Variables do not
        get garbage collected since functions have a weak reference to
        Variables. This is only required when the tf.AutoTrackable object is not
        maintained by the user (e.g. `from_saved_model`).
    """
    super(TFLiteFrozenGraphConverterV2, self).__init__()
    self._funcs = funcs
    self._trackable_obj = trackable_obj

  def convert(self):
    """Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format.

    Raises:
      ValueError:
        No concrete functions is specified.
        Multiple concrete functions are specified.
        Input shape is not specified.
        Invalid quantization parameters.
    """
    # TODO(b/130297984): Add support for converting multiple function.

    if len(self._funcs) == 0:
      raise ValueError("No ConcreteFunction is specified.")

    if len(self._funcs) > 1:
      raise ValueError("This converter can only convert a single "
                       "ConcreteFunction. Converting multiple functions is "
                       "under development.")

    frozen_func, graph_def = (
        _convert_to_constants.convert_variables_to_constants_v2_as_graph(
            self._funcs[0], lower_control_flow=False))

    input_tensors = [
        tensor for tensor in frozen_func.inputs
        if tensor.dtype != _dtypes.resource
    ]
    output_tensors = frozen_func.outputs

    # Run a Grappler pass.
    grappler_config = self._grappler_config()
    # Skip running grappler when there are no optimizers to run. If not,
    # grappler will run with the default optimizer set and it will lead to
    # causing an unexpected behavior.
    if grappler_config.graph_options.rewrite_options.optimizers:
      graph_def = _run_graph_optimizations(
          graph_def,
          input_tensors,
          output_tensors,
          config=grappler_config,
          graph=frozen_func.graph)

    return super(TFLiteFrozenGraphConverterV2,
                 self).convert(graph_def, input_tensors, output_tensors)


@_tf_export("lite.TFLiteConverter", v1=[])
class TFLiteConverterV2(TFLiteFrozenGraphConverterV2):
  """Converts a TensorFlow model into TensorFlow Lite model.

  Attributes:
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      When false any unknown operation is an error. When true, custom ops are
      created for any op that is unknown. The developer will need to provide
      these to the TensorFlow Lite runtime with a custom resolver.
      (default False)
    target_spec: Experimental flag, subject to change. Specification of target
      device.
    optimizations: Experimental flag, subject to change. A list of optimizations
      to apply when converting the model. E.g. `[Optimize.DEFAULT]`
    representative_dataset: A representative dataset that can be used to
      generate input and output samples for the model. The converter can use the
      dataset to evaluate different optimizations. Note that this is an optional
      attribute but it is necessary if INT8 is the only support builtin ops in
      target ops.
    experimental_new_converter: Experimental flag, subject to change.
      Enables MLIR-based conversion instead of TOCO conversion.
  Example usage:

    ```python
    # Converting a SavedModel to a TensorFlow Lite model.
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    # Converting a tf.Keras model to a TensorFlow Lite model.
    converter = lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Converting ConcreteFunctions to a TensorFlow Lite model.
    converter = lite.TFLiteConverter.from_concrete_functions([func])
    tflite_model = converter.convert()
    ```
  """

  # pylint: disable=useless-super-delegation
  def __init__(self, funcs, trackable_obj=None):
    """Constructor for TFLiteConverter.

    Args:
      funcs: List of TensorFlow ConcreteFunctions. The list should not contain
        duplicate elements.
      trackable_obj: tf.AutoTrackable object associated with `funcs`. A
        reference to this object needs to be maintained so that Variables do not
        get garbage collected since functions have a weak reference to
        Variables. This is only required when the tf.AutoTrackable object is not
        maintained by the user (e.g. `from_saved_model`).
    """
    super(TFLiteConverterV2, self).__init__(funcs, trackable_obj)

  @classmethod
  def from_concrete_functions(cls, funcs):
    """Creates a TFLiteConverter object from ConcreteFunctions.

    Args:
      funcs: List of TensorFlow ConcreteFunctions. The list should not contain
        duplicate elements. Currently converter can only convert a single
        ConcreteFunction. Converting multiple functions is under development.

    Returns:
      TFLiteConverter object.

    Raises:
      Invalid input type.
    """
    for func in funcs:
      if not isinstance(func, _function.ConcreteFunction):
        message = "This function takes in a list of ConcreteFunction."
        if isinstance(func, _def_function.Function):
          message += (" To get the ConcreteFunction from a Function,"
                      " call get_concrete_function.")
        raise ValueError(message)
    return cls(funcs)

  @classmethod
  def from_saved_model(cls, saved_model_dir, signature_keys=None, tags=None):
    """Creates a TFLiteConverter object from a SavedModel directory.

    Args:
      saved_model_dir: SavedModel directory to convert.
      signature_keys: List of keys identifying SignatureDef containing inputs
        and outputs. Elements should not be duplicated. By default the
        `signatures` attribute of the MetaGraphdef is used. (default
        saved_model.signatures)
      tags: Set of tags identifying the MetaGraphDef within the SavedModel to
        analyze. All tags in the tag set must be present. (default set(SERVING))

    Returns:
      TFLiteConverter object.

    Raises:
      Invalid signature keys.
    """
    # When run without eager enabled, this will return the legacy
    # TFLiteConverter.
    if not context.executing_eagerly():
      signature_key = None
      if signature_keys:
        if len(signature_keys) != 1:
          raise ValueError("Only support a single signature key.")
        else:
          signature_key = signature_keys[0]
      logging.warning("Invoking the TF1 implementation of TFLiteConverter "
                      "because eager is disabled. Consider enabling eager.")
      return TFLiteConverter.from_saved_model(saved_model_dir,
                                              signature_key=signature_key,
                                              tag_set=tags)

    # Ensures any graphs created in Eager mode are able to run. This is required
    # in order to create a tf.estimator.Exporter that exports a TFLite model.
    if tags is None:
      tags = set([_tag_constants.SERVING])

    with context.eager_mode():
      saved_model = _load(saved_model_dir, tags)
    if not signature_keys:
      signature_keys = saved_model.signatures

    if len(signature_keys) != 1:
      raise ValueError("Only support a single signature key.")

    funcs = []
    for key in signature_keys:
      if key not in saved_model.signatures:
        raise ValueError("Invalid signature key '{}' found. Valid keys are "
                         "'{}'.".format(key, ",".join(saved_model.signatures)))
      funcs.append(saved_model.signatures[key])

    saved_model_converter = TFLiteSavedModelConverterV2(saved_model_dir, tags,
                                                        signature_keys,
                                                        saved_model)
    if saved_model_converter.saved_model_dir:
      return saved_model_converter

    return cls(funcs, saved_model)

  @classmethod
  def from_keras_model(cls, model):
    """Creates a TFLiteConverter object from a Keras model.

    Args:
      model: tf.Keras.Model

    Returns:
      TFLiteConverter object.
    """
    return TFLiteKerasModelConverterV2(model)

  # pylint: disable=useless-super-delegation
  def convert(self):
    """Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format.

    Raises:
      ValueError:
        No concrete functions is specified.
        Multiple concrete functions are specified.
        Input shape is not specified.
        Invalid quantization parameters.
    """
    return super(TFLiteConverterV2, self).convert()


class TFLiteConverterBaseV1(TFLiteConverterBase):
  """Converter subclass to share functionality between V1 converters.

  Attributes:
    inference_type: Target data type of real-number arrays in the output file.
      Must be `{tf.float32, tf.uint8}`. If `optimzations` are provided, this
      parameter is ignored. (default tf.float32)
    inference_input_type: Target data type of real-number input arrays. Allows
      for a different type for input arrays. If an integer type is provided and
      `optimizations` are not used, `quantized_inputs_stats` must be provided.
      If `inference_type` is tf.uint8, signaling conversion to a fully quantized
      model from a quantization-aware trained input model, then
      `inference_input_type` defaults to tf.uint8. In all other cases,
      `inference_input_type` defaults to tf.float32. Must be `{tf.float32,
      tf.uint8, tf.int8}`
    inference_output_type: Target data type of real-number output arrays. Allows
      for a different type for output arrays. If `inference_type` is tf.uint8,
      signaling conversion to a fully quantized model from a quantization-aware
      trained output model, then `inference_output_type` defaults to tf.uint8.
      In all other cases, `inference_output_type` must be tf.float32, an error
      will be thrown otherwise. Must be `{tf.float32, tf.uint8, tf.int8}`
    output_format: Output file format. Currently must be `{TFLITE,
      GRAPHVIZ_DOT}`. (default TFLITE)
    quantized_input_stats: Dict of strings representing input tensor names
      mapped to tuple of floats representing the mean and standard deviation
      of the training data (e.g., {"foo" : (0., 1.)}). Only need if
        `inference_input_type` is `QUANTIZED_UINT8`. real_input_value =
        (quantized_input_value - mean_value) / std_dev_value. (default {})
    default_ranges_stats: Tuple of integers representing (min, max) range values
      for all arrays without a specified range. Intended for experimenting with
      quantization via "dummy quantization". (default None)
    drop_control_dependency: Boolean indicating whether to drop control
      dependencies silently. This is due to TFLite not supporting control
      dependencies. (default True)
    reorder_across_fake_quant: Boolean indicating whether to reorder FakeQuant
      nodes in unexpected locations. Used when the location of the FakeQuant
      nodes is preventing graph transformations necessary to convert the graph.
      Results in a graph that differs from the quantized training graph,
      potentially causing differing arithmetic behavior. (default False)
    change_concat_input_ranges: Boolean to change behavior of min/max ranges for
      inputs and outputs of the concat operator for quantized models. Changes
      the ranges of concat operator overlap when true. (default False)
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      When false any unknown operation is an error. When true, custom ops are
      created for any op that is unknown. The developer will need to provide
      these to the TensorFlow Lite runtime with a custom resolver. (default
      False)
    post_training_quantize: Deprecated. Please specify `[Optimize.DEFAULT]` for
      `optimizations` instead. Boolean indicating whether to quantize the
      weights of the converted float model.  Model size will be reduced and
      there will be latency improvements (at the cost of accuracy). (default
      False)
    dump_graphviz_dir: Full filepath of folder to dump the graphs at various
      stages of processing GraphViz .dot files. Preferred over
      --output_format=GRAPHVIZ_DOT in order to keep the requirements of the
      output file. (default None)
    dump_graphviz_video: Boolean indicating whether to dump the graph after
      every graph transformation. (default False)
    conversion_summary_dir: A string indicating the path to the generated
      conversion logs.
    target_ops: Deprecated. Please specify `target_spec.supported_ops` instead.
      Set of OpsSet options indicating which converter to use. (default
      set([OpsSet.TFLITE_BUILTINS]))
    target_spec: Experimental flag, subject to change. Specification of target
      device.
    optimizations: Experimental flag, subject to change. A list of optimizations
      to apply when converting the model. E.g. `[Optimize.DEFAULT]`
    representative_dataset: A representative dataset that can be used to
      generate input and output samples for the model. The converter can use the
      dataset to evaluate different optimizations.
    experimental_new_converter: Experimental flag, subject to change. Enables
      MLIR-based conversion instead of TOCO conversion.
  """

  def __init__(self, experimental_debug_info_func):
    """Constructor for TFLiteConverter.

    Args:
      experimental_debug_info_func: An experimental function to retrieve the
        graph debug info for a set of nodes from the `graph_def`.
    """
    super(TFLiteConverterBaseV1, self).__init__()
    self.inference_type = constants.FLOAT
    self.inference_input_type = None
    self.inference_output_type = None
    self.output_format = constants.TFLITE
    self.quantized_input_stats = {}
    self.default_ranges_stats = None
    self.drop_control_dependency = True
    self.reorder_across_fake_quant = False
    self.change_concat_input_ranges = False
    self.dump_graphviz_dir = None
    self.dump_graphviz_video = False
    self.conversion_summary_dir = None
    self._debug_info_func = experimental_debug_info_func
    self._custom_opdefs = None

  def __setattr__(self, name, value):
    if name == "post_training_quantize":
      warnings.warn("Property %s is deprecated, "
                    "please use optimizations=[Optimize.DEFAULT]"
                    " instead." % name)
      if value:
        self.optimizations = [Optimize.DEFAULT]
      else:
        self.optimizations = []
      return
    if name == "target_ops":
      warnings.warn("Property %s is deprecated, please use "
                    "target_spec.supported_ops instead." % name)
      self.target_spec.supported_ops = value
      return
    object.__setattr__(self, name, value)

  def __getattribute__(self, name):
    if name == "post_training_quantize":
      warnings.warn("Property %s is deprecated, "
                    "please use optimizations=[Optimize.DEFAULT]"
                    " instead." % name)
      return Optimize.DEFAULT in set(self.optimizations)
    if name == "target_ops":
      warnings.warn("Property %s is deprecated, please use "
                    "target_spec.supported_ops instead." % name)
      return self.target_spec.supported_ops
    return object.__getattribute__(self, name)

  def _validate_quantized_input_stats(self, converter_kwargs, calibrate):
    """Ensure quantized_input_stats provided if required."""

    quantized_types = frozenset({constants.INT8, constants.QUANTIZED_UINT8})

    requires_quantized_input_stats = (
        (converter_kwargs["inference_type"] in quantized_types or
         converter_kwargs["inference_input_type"] in quantized_types) and
        not calibrate)

    if (requires_quantized_input_stats and
        not converter_kwargs["quantized_input_stats"]):
      raise ValueError("std_dev and mean must be defined when inference_type "
                       "or inference_input_type is QUANTIZED_UINT8 or INT8.")

  def convert(self):
    """Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format. Either a TFLite Flatbuffer or a
      Graphviz graph depending on value in `output_format`.

    Raises:
      ValueError:
        Input shape is not specified.
        None value for dimension in input_tensor.
    """
    quant_mode = QuantizationMode(self.optimizations, self.target_spec,
                                  self.representative_dataset, self._graph_def)

    if (not self._is_unknown_shapes_allowed() and self._has_valid_tensors()):
      # Checks dimensions in input tensor.
      for tensor in self._input_tensors:
        shape = tensor.shape
        if not shape:
          raise ValueError("Provide an input shape for input array "
                           "'{0}'.".format(_get_tensor_name(tensor)))
        # Note that shape_list might be empty for scalar shapes.
        shape_list = shape.as_list()
        if None in shape_list[1:]:
          raise ValueError(
              "None is only supported in the 1st dimension. Tensor '{0}' has "
              "invalid shape '{1}'.".format(
                  _get_tensor_name(tensor), shape_list))
        elif shape_list and shape_list[0] is None:
          self._set_batch_size(batch_size=1)

    # Get quantization stats. Ensures there is one stat per name if the stats
    # are specified.
    if self.quantized_input_stats:
      quantized_stats = []
      invalid_stats = []
      for name in self.get_input_arrays():
        if name in self.quantized_input_stats:
          quantized_stats.append(self.quantized_input_stats[name])
        else:
          invalid_stats.append(name)

      if invalid_stats:
        raise ValueError("Quantization input stats are not available for input "
                         "tensors '{0}'.".format(",".join(invalid_stats)))
    else:
      quantized_stats = None

    optimized_graph = self._graph_def
    if not self.saved_model_dir:
      # Disable grappler constant folding if there are training quant ops.
      if not quant_mode.contains_training_quant_op():
        try:
          # TODO(b/150163103): Merge `disabling lower using switch merge' calls.
          # Grappler will also try to lower while loop into switch merge
          # representation which is undesired for Ophints, so we simply remove
          # those attributes to prevent Grappler from doing so.
          graph_def = _convert_to_constants.disable_lower_using_switch_merge(
              optimized_graph)
          # Run function inlining optimization to ensure any models generated
          # through the from_frozen_graph path have been inlined.
          optimized_graph = _run_graph_optimizations(
              graph_def,
              self._input_tensors,
              self._output_tensors,
              config=self._grappler_config(["function"]))
        except Exception:  # pylint: disable=broad-except
          optimized_graph = self._graph_def

    self._debug_info = _get_debug_info(self._debug_info_func, optimized_graph)

    converter_kwargs = self._get_base_converter_args()
    converter_kwargs.update(
        quant_mode.converter_flags(self.inference_type,
                                   self.inference_input_type))
    converter_kwargs.update({
        "output_format": self.output_format,
        "quantized_input_stats": quantized_stats,
        "default_ranges_stats": self.default_ranges_stats,
        "drop_control_dependency": self.drop_control_dependency,
        "reorder_across_fake_quant": self.reorder_across_fake_quant,
        "change_concat_input_ranges": self.change_concat_input_ranges,
        "dump_graphviz_dir": self.dump_graphviz_dir,
        "dump_graphviz_video": self.dump_graphviz_video,
        "conversion_summary_dir": self.conversion_summary_dir,
        "custom_opdefs": self._custom_opdefs,
    })

    if not self.experimental_new_converter:
      logging.warning(
          "Please consider switching to use new converter by setting "
          "experimental_new_converter to true. "
          "Old converter (TOCO) is deprecated and flow will be switched on "
          "by default to use new converter soon.")
    else:
      logging.info("Using experimental converter: If you encountered a problem "
                   "please file a bug. You can opt-out "
                   "by setting experimental_new_converter=False")

    calibrate_quantize, flags = quant_mode.quantizer_flags(
        self.inference_input_type, self.inference_output_type)

    self._validate_quantized_input_stats(converter_kwargs, calibrate_quantize)

    # Converts model.
    if self._has_valid_tensors():
      result = _toco_convert_impl(
          input_data=optimized_graph,
          input_tensors=self._input_tensors,
          output_tensors=self._output_tensors,
          **converter_kwargs)
    else:
      result = _toco_convert_graph_def(
          input_data=optimized_graph,
          input_arrays_with_shape=self._input_arrays_with_shape,
          output_arrays=self._output_arrays,
          **converter_kwargs)

    if calibrate_quantize:
      result = self._calibrate_quantize_model(result, **flags)

    if self._experimental_sparsify_model:
      result = _mlir_sparsify(result)
      
    return result

  def get_input_arrays(self):
    """Returns a list of the names of the input tensors.

    Returns:
      List of strings.
    """
    if self._has_valid_tensors():
      return [_get_tensor_name(tensor) for tensor in self._input_tensors]
    else:
      return [name for name, _ in self._input_arrays_with_shape]

  def _has_valid_tensors(self):
    """Checks if the input and output tensors have been initialized.

    Returns:
      Bool.
    """
    return self._input_tensors and self._output_tensors

  def _set_batch_size(self, batch_size):
    """Sets the first dimension of the input tensor to `batch_size`.

    Args:
      batch_size: Batch size for the model. Replaces the first dimension of an
        input size array if undefined. (default 1)

    Raises:
      ValueError: input_tensor is not defined.
    """
    if not self._has_valid_tensors():
      raise ValueError("The batch size cannot be set for this model. Please "
                       "use input_shapes parameter.")

    for tensor in self._input_tensors:
      shape = tensor.shape.as_list()
      if shape[0] is None:
        shape[0] = batch_size
        tensor.set_shape(shape)

  def _is_unknown_shapes_allowed(self):
    # Ophint Converted nodes will need the shapes to be known.
    if _is_ophint_converted(self._graph_def):
      return False

    if not super(TFLiteConverterBaseV1, self)._is_unknown_shapes_allowed():
      return False

    # `conversion_summary_dir` calls TOCO. Unknown shapes are only supported by
    # the MLIR converter.
    if self.conversion_summary_dir:
      logging.warning(
          "`conversion_summary_dir` does not work with unknown shapes. "
          "Graphs with unknown shapes might be different than when this flag "
          "is disabled.")
      return False
    return True


class TFLiteSavedModelConverter(TFLiteConverterBaseV1):
  """Converts the given SavedModel into TensorFlow Lite model.

  Attributes:
      saved_model_dir: Directory of the SavedModel.
  """

  def __init__(self,
               saved_model_dir,
               saved_model_tags,
               saved_model_exported_names,
               experimental_debug_info_func=None):
    """Constructor for TFLiteConverter.

    Args:
      saved_model_dir: Directory of the SavedModel.
      saved_model_tags: Set of tags identifying the MetaGraphDef within the
        SavedModel to analyze. All tags in the tag set must be present. (default
        set(SERVING)).
      saved_model_exported_names: Names to be exported (default: export all)
        when the saved model import path is on.
      experimental_debug_info_func: An experimental function to retrieve the
        graph debug info for a set of nodes from the `graph_def`.

    Raises:
      ValueError: Invalid arguments.
    """
    super(TFLiteSavedModelConverter,
          self).__init__(experimental_debug_info_func)
    self.saved_model_dir = saved_model_dir
    self._saved_model_tags = saved_model_tags
    self._saved_model_exported_names = saved_model_exported_names

    signature_key = _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    if len(self._saved_model_exported_names) != 1:
      raise ValueError("Only support a single signature key.")

    signature_key = self._saved_model_exported_names[0]

    result = _freeze_saved_model(self.saved_model_dir, None, None, None,
                                 self._saved_model_tags, signature_key)
    self._graph_def = result[0]
    self._input_tensors = result[1]
    self._output_tensors = result[2]
    self._parse_saved_model_args()


class TFLiteKerasModelConverter(TFLiteConverterBaseV1):
  """Converts the given SavedModel into TensorFlow Lite model."""

  def __init__(self,
               model_file,
               input_arrays=None,
               input_shapes=None,
               output_arrays=None,
               custom_objects=None):
    """Constructor for TFLiteConverter.

    Args:
      model_file: Full filepath of HDF5 file containing the tf.keras model.
      input_arrays: List of input tensors to freeze graph with. Uses input
        arrays from SignatureDef when none are provided. (default None)
      input_shapes: Dict of strings representing input tensor names to list of
        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
        Automatically determined when input shapes is None (e.g., {"foo" :
          None}). (default None)
      output_arrays: List of output tensors to freeze graph with. Uses output
        arrays from SignatureDef when none are provided. (default None)
      custom_objects: Dict mapping names (strings) to custom classes or
        functions to be considered during model deserialization. (default None)

    Raises:
      ValueError: Invalid arguments.
    """
    super(TFLiteKerasModelConverter,
          self).__init__(experimental_debug_info_func=None)
    # Handles Keras when Eager mode is enabled.
    if context.executing_eagerly():
      if input_arrays or output_arrays:
        raise ValueError("`input_arrays` and `output_arrays` are unsupported "
                         "with Eager mode. If your model requires any of these "
                         "parameters, please use disable_eager_execution().")

      _keras.backend.set_learning_phase(False)
      keras_model = _keras.models.load_model(model_file, custom_objects)

      function = _saving_utils.trace_model_call(keras_model)
      concrete_func = function.get_concrete_function()

      frozen_func = _convert_to_constants.convert_variables_to_constants_v2(
          concrete_func, lower_control_flow=False)
      _set_tensor_shapes(frozen_func.inputs, input_shapes)
      self._keras_model = keras_model
      self._graph_def = frozen_func.graph.as_graph_def()
      self._input_tensors = frozen_func.inputs
      self._output_tensors = frozen_func.outputs
      self._debug_info_func = _build_debug_info_func(frozen_func.graph)
      return

    # Handles Keras when Eager mode is disabled.
    _keras.backend.clear_session()
    _keras.backend.set_learning_phase(False)
    keras_model = _keras.models.load_model(model_file, custom_objects)
    sess = _keras.backend.get_session()

    # Get input and output tensors.
    if input_arrays:
      input_tensors = _get_tensors_from_tensor_names(sess.graph, input_arrays)
    else:
      input_tensors = keras_model.inputs

    if output_arrays:
      output_tensors = _get_tensors_from_tensor_names(sess.graph, output_arrays)
    else:
      output_tensors = keras_model.outputs
    _set_tensor_shapes(input_tensors, input_shapes)

    graph_def = _freeze_graph(sess, input_tensors, output_tensors)
    self._keras_model = keras_model
    self._graph_def = graph_def
    self._input_tensors = input_tensors
    self._output_tensors = output_tensors
    self._debug_info_func = _build_debug_info_func(sess.graph)

  def convert(self):
    """Converts a Keras model based on instance variables.

    Returns:
      The converted data in serialized format. Either a TFLite Flatbuffer or a
      Graphviz graph depending on value in `output_format`.

    Raises:
      ValueError:
        Input shape is not specified.
        None value for dimension in input_tensor.
    """
    temp_dir = tempfile.mkdtemp()
    try:
      self._keras_model.save(temp_dir, save_format="tf")
      tag_set = set([_tag_constants.SERVING])
      signature_key = _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
      result = _freeze_saved_model(temp_dir, None, None, None, tag_set,
                                   signature_key)

      self.saved_model_dir = temp_dir
      self._saved_model_tags = tag_set
      self._saved_model_exported_names = [signature_key]
      self._parse_saved_model_args()
      if self.saved_model_dir:
        self._graph_def = result[0]
        self._input_tensors = result[1]
        self._output_tensors = result[2]
        self._debug_info_func = _build_debug_info_func(result[3])
        return super(TFLiteKerasModelConverter, self).convert()
    finally:
      shutil.rmtree(temp_dir, True)

    return super(TFLiteKerasModelConverter, self).convert()


class TFLiteFrozenGraphConverter(TFLiteConverterBaseV1):
  """Converts the given frozen graph def into TensorFlow Lite model."""

  def __init__(self,
               graph_def,
               input_tensors,
               output_tensors,
               input_arrays_with_shape=None,
               output_arrays=None,
               experimental_debug_info_func=None):
    """Constructor for TFLiteConverter.

    Args:
      graph_def: Frozen TensorFlow GraphDef.
      input_tensors: List of input tensors. Type and shape are computed using
        `foo.shape` and `foo.dtype`.
      output_tensors: List of output tensors (only .name is used from this).
      input_arrays_with_shape: Tuple of strings representing input tensor names
        and list of integers representing input shapes
        (e.g., [("foo" : [1, 16, 16, 3])]). Use only when graph cannot be loaded
          into TensorFlow and when `input_tensors` and `output_tensors` are
          None. (default None)
      output_arrays: List of output tensors to freeze graph with. Use only when
        graph cannot be loaded into TensorFlow and when `input_tensors` and
        `output_tensors` are None. (default None)
      experimental_debug_info_func: An experimental function to retrieve the
        graph debug info for a set of nodes from the `graph_def`.

    Raises:
      ValueError: Invalid arguments.
    """
    super(TFLiteFrozenGraphConverter,
          self).__init__(experimental_debug_info_func)
    self._graph_def = graph_def
    self._input_tensors = input_tensors
    self._output_tensors = output_tensors

    # Attributes are used by models that cannot be loaded into TensorFlow.
    if not self._has_valid_tensors():
      if not input_arrays_with_shape or not output_arrays:
        raise ValueError(
            "If input_tensors and output_tensors are None, both "
            "input_arrays_with_shape and output_arrays must be defined.")
      self._input_arrays_with_shape = input_arrays_with_shape
      self._output_arrays = output_arrays


@_tf_export(v1=["lite.TFLiteConverter"])
class TFLiteConverter(TFLiteFrozenGraphConverter):
  """Convert a TensorFlow model into `output_format`.

  This is used to convert from a TensorFlow GraphDef, SavedModel or tf.keras
  model into either a TFLite FlatBuffer or graph visualization.

  Attributes:
    inference_type: Target data type of real-number arrays in the output file.
      Must be `{tf.float32, tf.uint8}`. If `optimzations` are provided, this
      parameter is ignored. (default tf.float32)
    inference_input_type: Target data type of real-number input arrays. Allows
      for a different type for input arrays.
      If an integer type is provided and `optimizations` are not used,
      `quantized_inputs_stats` must be provided.
      If `inference_type` is tf.uint8, signaling conversion to a fully quantized
      model from a quantization-aware trained input model, then
      `inference_input_type` defaults to tf.uint8.
      In all other cases, `inference_input_type` defaults to tf.float32.
      Must be `{tf.float32, tf.uint8, tf.int8}`
    inference_output_type: Target data type of real-number output arrays. Allows
      for a different type for output arrays.
      If `inference_type` is tf.uint8, signaling conversion to a fully quantized
      model from a quantization-aware trained output model, then
      `inference_output_type` defaults to tf.uint8.
      In all other cases, `inference_output_type` must be tf.float32, an error
      will be thrown otherwise.
      Must be `{tf.float32, tf.uint8, tf.int8}`
    output_format: Output file format. Currently must be `{TFLITE,
      GRAPHVIZ_DOT}`. (default TFLITE)
    quantized_input_stats: Dict of strings representing input tensor names
      mapped to tuple of floats representing the mean and standard deviation
      of the training data (e.g., {"foo" : (0., 1.)}). Only need if
      `inference_input_type` is `QUANTIZED_UINT8`.
      real_input_value = (quantized_input_value - mean_value) / std_dev_value.
      (default {})
    default_ranges_stats: Tuple of integers representing (min, max) range values
      for all arrays without a specified range. Intended for experimenting with
      quantization via "dummy quantization". (default None)
    drop_control_dependency: Boolean indicating whether to drop control
      dependencies silently. This is due to TFLite not supporting control
      dependencies. (default True)
    reorder_across_fake_quant: Boolean indicating whether to reorder FakeQuant
      nodes in unexpected locations. Used when the location of the FakeQuant
      nodes is preventing graph transformations necessary to convert the graph.
      Results in a graph that differs from the quantized training graph,
      potentially causing differing arithmetic behavior. (default False)
    change_concat_input_ranges: Boolean to change behavior of min/max ranges for
      inputs and outputs of the concat operator for quantized models. Changes
      the ranges of concat operator overlap when true. (default False)
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      When false any unknown operation is an error. When true, custom ops are
      created for any op that is unknown. The developer will need to provide
      these to the TensorFlow Lite runtime with a custom resolver.
      (default False)
    post_training_quantize: Deprecated. Please specify `[Optimize.DEFAULT]` for
      `optimizations` instead. Boolean indicating whether to quantize the
      weights of the converted float model.  Model size will be reduced and
      there will be latency improvements (at the cost of accuracy).
      (default False)
    dump_graphviz_dir: Full filepath of folder to dump the graphs at various
      stages of processing GraphViz .dot files. Preferred over
      --output_format=GRAPHVIZ_DOT in order to keep the requirements of the
      output file. (default None)
    dump_graphviz_video: Boolean indicating whether to dump the graph after
      every graph transformation. (default False)
    conversion_summary_dir: A string indicating the path to the generated
      conversion logs.
    target_ops: Deprecated. Please specify `target_spec.supported_ops` instead.
      Set of OpsSet options indicating which converter to use.
      (default set([OpsSet.TFLITE_BUILTINS]))
    target_spec: Experimental flag, subject to change. Specification of target
      device.
    optimizations: Experimental flag, subject to change. A list of optimizations
      to apply when converting the model. E.g. `[Optimize.DEFAULT]`
    representative_dataset: A representative dataset that can be used to
      generate input and output samples for the model. The converter can use
      the dataset to evaluate different optimizations.
    experimental_new_converter: Experimental flag, subject to change.
      Enables MLIR-based conversion instead of TOCO conversion.

  Example usage:

    ```python
    # Converting a GraphDef from session.
    converter = lite.TFLiteConverter.from_session(sess, in_tensors, out_tensors)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

    # Converting a GraphDef from file.
    converter = lite.TFLiteConverter.from_frozen_graph(
      graph_def_file, input_arrays, output_arrays)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

    # Converting a SavedModel.
    converter = lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

    # Converting a tf.keras model.
    converter = lite.TFLiteConverter.from_keras_model_file(keras_model)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)
    ```
  """

  # pylint: disable=useless-super-delegation
  def __init__(self,
               graph_def,
               input_tensors,
               output_tensors,
               input_arrays_with_shape=None,
               output_arrays=None,
               experimental_debug_info_func=None):
    """Constructor for TFLiteConverter.

    Args:
      graph_def: Frozen TensorFlow GraphDef.
      input_tensors: List of input tensors. Type and shape are computed using
        `foo.shape` and `foo.dtype`.
      output_tensors: List of output tensors (only .name is used from this).
      input_arrays_with_shape: Tuple of strings representing input tensor names
        and list of integers representing input shapes
        (e.g., [("foo" : [1, 16, 16, 3])]). Use only when graph cannot be loaded
          into TensorFlow and when `input_tensors` and `output_tensors` are
          None. (default None)
      output_arrays: List of output tensors to freeze graph with. Use only when
        graph cannot be loaded into TensorFlow and when `input_tensors` and
        `output_tensors` are None. (default None)
      experimental_debug_info_func: An experimental function to retrieve the
        graph debug info for a set of nodes from the `graph_def`.

    Raises:
      ValueError: Invalid arguments.
    """
    super(TFLiteConverter,
          self).__init__(graph_def, input_tensors, output_tensors,
                         input_arrays_with_shape, output_arrays,
                         experimental_debug_info_func)

  @classmethod
  def from_session(cls, sess, input_tensors, output_tensors):
    """Creates a TFLiteConverter class from a TensorFlow Session.

    Args:
      sess: TensorFlow Session.
      input_tensors: List of input tensors. Type and shape are computed using
        `foo.shape` and `foo.dtype`.
      output_tensors: List of output tensors (only .name is used from this).

    Returns:
      TFLiteConverter class.
    """
    graph_def = _freeze_graph(sess, input_tensors, output_tensors)
    return cls(
        graph_def,
        input_tensors,
        output_tensors,
        experimental_debug_info_func=_build_debug_info_func(sess.graph))

  @classmethod
  def from_frozen_graph(cls,
                        graph_def_file,
                        input_arrays,
                        output_arrays,
                        input_shapes=None):
    """Creates a TFLiteConverter class from a file containing a frozen GraphDef.

    Args:
      graph_def_file: Full filepath of file containing frozen GraphDef.
      input_arrays: List of input tensors to freeze graph with.
      output_arrays: List of output tensors to freeze graph with.
      input_shapes: Dict of strings representing input tensor names to list of
        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
        Automatically determined when input shapes is None (e.g., {"foo" :
          None}). (default None)

    Returns:
      TFLiteConverter class.

    Raises:
      IOError:
        File not found.
        Unable to parse input file.
      ValueError:
        The graph is not frozen.
        input_arrays or output_arrays contains an invalid tensor name.
        input_shapes is not correctly defined when required
    """
    with _ops.Graph().as_default():
      with _session.Session() as sess:
        # Read GraphDef from file.
        if not _file_io.file_exists(graph_def_file):
          raise IOError("File '{0}' does not exist.".format(graph_def_file))
        with _file_io.FileIO(graph_def_file, "rb") as f:
          file_content = f.read()

        try:
          graph_def = _graph_pb2.GraphDef()
          graph_def.ParseFromString(file_content)
        except (_text_format.ParseError, DecodeError):
          try:
            print("Ignore 'tcmalloc: large alloc' warnings.")

            if not isinstance(file_content, str):
              if PY2:
                file_content = six.ensure_binary(file_content, "utf-8")
              else:
                file_content = six.ensure_text(file_content, "utf-8")
            graph_def = _graph_pb2.GraphDef()
            _text_format.Merge(file_content, graph_def)
          except (_text_format.ParseError, DecodeError):
            raise IOError(
                "Unable to parse input file '{}'.".format(graph_def_file))

        # Handles models with custom TFLite ops that cannot be resolved in
        # TensorFlow.
        load_model_in_session = True
        try:
          _import_graph_def(graph_def, name="")
        except _NotFoundError:
          load_model_in_session = False

        if load_model_in_session:
          # Check if graph is frozen.
          if not _is_frozen_graph(sess):
            raise ValueError("Please freeze the graph using freeze_graph.py.")

          # Get input and output tensors.
          input_tensors = _get_tensors_from_tensor_names(
              sess.graph, input_arrays)
          output_tensors = _get_tensors_from_tensor_names(
              sess.graph, output_arrays)
          _set_tensor_shapes(input_tensors, input_shapes)

          return cls(sess.graph_def, input_tensors, output_tensors)
        else:
          if not input_shapes:
            raise ValueError("input_shapes must be defined for this model.")
          if set(input_arrays) != set(input_shapes.keys()):
            raise ValueError("input_shapes must contain a value for each item "
                             "in input_array.")

          input_arrays_with_shape = [
              (name, input_shapes[name]) for name in input_arrays
          ]
          return cls(
              graph_def,
              input_tensors=None,
              output_tensors=None,
              input_arrays_with_shape=input_arrays_with_shape,
              output_arrays=output_arrays)

  @classmethod
  def from_saved_model(cls,
                       saved_model_dir,
                       input_arrays=None,
                       input_shapes=None,
                       output_arrays=None,
                       tag_set=None,
                       signature_key=None):
    """Creates a TFLiteConverter class from a SavedModel.

    Args:
      saved_model_dir: SavedModel directory to convert.
      input_arrays: List of input tensors to freeze graph with. Uses input
        arrays from SignatureDef when none are provided. (default None)
      input_shapes: Dict of strings representing input tensor names to list of
        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
        Automatically determined when input shapes is None (e.g., {"foo" :
          None}). (default None)
      output_arrays: List of output tensors to freeze graph with. Uses output
        arrays from SignatureDef when none are provided. (default None)
      tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to
        analyze. All tags in the tag set must be present. (default set("serve"))
      signature_key: Key identifying SignatureDef containing inputs and outputs.
        (default DEFAULT_SERVING_SIGNATURE_DEF_KEY)

    Returns:
      TFLiteConverter class.
    """
    if tag_set is None:
      tag_set = set([_tag_constants.SERVING])
    if signature_key is None:
      signature_key = _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    saved_model_converter = TFLiteSavedModelConverter(saved_model_dir, tag_set,
                                                      [signature_key])
    if saved_model_converter.saved_model_dir:
      return saved_model_converter

    result = _freeze_saved_model(saved_model_dir, input_arrays, input_shapes,
                                 output_arrays, tag_set, signature_key)

    return cls(
        graph_def=result[0],
        input_tensors=result[1],
        output_tensors=result[2],
        experimental_debug_info_func=_build_debug_info_func(result[3]))

  @classmethod
  def from_keras_model_file(cls,
                            model_file,
                            input_arrays=None,
                            input_shapes=None,
                            output_arrays=None,
                            custom_objects=None):
    """Creates a TFLiteConverter class from a tf.keras model file.

    Args:
      model_file: Full filepath of HDF5 file containing the tf.keras model.
      input_arrays: List of input tensors to freeze graph with. Uses input
        arrays from SignatureDef when none are provided. (default None)
      input_shapes: Dict of strings representing input tensor names to list of
        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
        Automatically determined when input shapes is None (e.g., {"foo" :
          None}). (default None)
      output_arrays: List of output tensors to freeze graph with. Uses output
        arrays from SignatureDef when none are provided. (default None)
      custom_objects: Dict mapping names (strings) to custom classes or
        functions to be considered during model deserialization. (default None)

    Returns:
      TFLiteConverter class.
    """
    return TFLiteKerasModelConverter(model_file, input_arrays, input_shapes,
                                     output_arrays, custom_objects)

  # pylint: disable=useless-super-delegation
  def convert(self):
    """Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format. Either a TFLite Flatbuffer or a
      Graphviz graph depending on value in `output_format`.

    Raises:
      ValueError:
        Input shape is not specified.
        None value for dimension in input_tensor.
    """
    return super(TFLiteConverter, self).convert()

@_tf_export(v1=["lite.TocoConverter"])
class TocoConverter(object):
  """Convert a TensorFlow model into `output_format` using TOCO.

  This class has been deprecated. Please use `lite.TFLiteConverter` instead.
  """

  @classmethod
  @_deprecation.deprecated(None,
                           "Use `lite.TFLiteConverter.from_session` instead.")
  def from_session(cls, sess, input_tensors, output_tensors):
    """Creates a TocoConverter class from a TensorFlow Session."""
    return TFLiteConverter.from_session(sess, input_tensors, output_tensors)

  @classmethod
  @_deprecation.deprecated(
      None, "Use `lite.TFLiteConverter.from_frozen_graph` instead.")
  def from_frozen_graph(cls,
                        graph_def_file,
                        input_arrays,
                        output_arrays,
                        input_shapes=None):
    """Creates a TocoConverter class from a file containing a frozen graph."""
    return TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays,
                                             output_arrays, input_shapes)

  @classmethod
  @_deprecation.deprecated(
      None, "Use `lite.TFLiteConverter.from_saved_model` instead.")
  def from_saved_model(cls,
                       saved_model_dir,
                       input_arrays=None,
                       input_shapes=None,
                       output_arrays=None,
                       tag_set=None,
                       signature_key=None):
    """Creates a TocoConverter class from a SavedModel."""
    return TFLiteConverter.from_saved_model(saved_model_dir, input_arrays,
                                            input_shapes, output_arrays,
                                            tag_set, signature_key)

  @classmethod
  @_deprecation.deprecated(
    None, "Use `lite.TFLiteConverter.from_keras_model_file` instead.")
  def from_keras_model_file(cls,
                            model_file,
                            input_arrays=None,
                            input_shapes=None,
                            output_arrays=None):
    """Creates a TocoConverter class from a tf.keras model file."""
    return TFLiteConverter.from_keras_model_file(model_file, input_arrays,
                                                 input_shapes, output_arrays)
