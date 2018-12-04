# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Converts a frozen graph into a TFLite FlatBuffer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum  # pylint: disable=g-bad-import-order

import os as _os
import platform as _platform
import subprocess as _subprocess
import tempfile as _tempfile

from tensorflow.lite.python import lite_constants
from tensorflow.lite.toco import model_flags_pb2 as _model_flags_pb2
from tensorflow.lite.toco import toco_flags_pb2 as _toco_flags_pb2
from tensorflow.lite.toco import types_pb2 as _types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import resource_loader as _resource_loader
from tensorflow.python.util import deprecation
from tensorflow.python.util.lazy_loader import LazyLoader
from tensorflow.python.util.tf_export import tf_export as _tf_export

# Lazy load since some of the performance benchmark skylark rules
# break dependencies.
_toco_python = LazyLoader(
    "tensorflow_wrap_toco", globals(),
    "tensorflow.lite.toco.python."
    "tensorflow_wrap_toco")
del LazyLoader

# Find the toco_from_protos binary using the resource loader if using from
# bazel, otherwise we are in a pip where console_scripts already has
# the toco_from_protos tool.
if lite_constants.EXPERIMENTAL_USE_TOCO_API_DIRECTLY:
  _toco_from_proto_bin = ""
else:
  _toco_from_proto_bin = _resource_loader.get_path_to_datafile(
      "../toco/python/toco_from_protos")

if _toco_from_proto_bin and not _os.path.exists(_toco_from_proto_bin):
  _toco_from_proto_bin = "toco_from_protos"


# Map of tf.dtypes to TFLite types_flag_pb2.
_MAP_TF_TO_TFLITE_TYPES = {
    dtypes.float32: _types_pb2.FLOAT,
    dtypes.int32: _types_pb2.INT32,
    dtypes.int64: _types_pb2.INT64,
    dtypes.string: _types_pb2.STRING,
    dtypes.uint8: _types_pb2.QUANTIZED_UINT8,
    dtypes.complex64: _types_pb2.COMPLEX64
}


def _try_convert_to_unicode(output):
  if output is None:
    return u""

  if isinstance(output, bytes):
    try:
      return output.decode()
    except UnicodeDecodeError:
      pass
  return output


def convert_dtype_to_tflite_type(tf_dtype):
  """Converts tf.dtype to TFLite proto type.

  Args:
    tf_dtype: tf.dtype

  Raises:
    ValueError: Unsupported tf.dtype.

  Returns:
    types_flag_pb2.
  """
  result = _MAP_TF_TO_TFLITE_TYPES.get(tf_dtype)
  if result is None:
    raise ValueError("Unsupported tf.dtype {0}".format(tf_dtype))
  return result


class OpsSet(enum.Enum):
  """Enum class defining the sets of ops available to generate TFLite models.

  WARNING: Experimental interface, subject to change.
  """
  # Convert model using TensorFlow Lite builtin ops.
  TFLITE_BUILTINS = "TFLITE_BUILTINS"

  # Convert model using TensorFlow ops. Not all TensorFlow ops are available.
  # WARNING: Experimental interface, subject to change.
  SELECT_TF_OPS = "SELECT_TF_OPS"

  def __str__(self):
    return self.value

  @staticmethod
  def get_options():
    """Returns a list of OpsSet options as a list of strings."""
    return [str(option) for option in list(OpsSet)]


class ConverterError(Exception):
  """Raised when an error occurs during model conversion."""
  pass


# Don't expose these for now.
#  @_tf_export("lite.toco_convert_protos")
def toco_convert_protos(model_flags_str, toco_flags_str, input_data_str):
  """Convert `input_data_str` according to model and toco parameters.

  Unless you know what you are doing consider using
  the more friendly `tf.lite.toco_convert`.

  Args:
    model_flags_str: Serialized proto describing model properties, see
      `toco/model_flags.proto`.
    toco_flags_str: Serialized proto describing conversion properties, see
      `toco/toco_flags.proto`.
    input_data_str: Input data in serialized form (e.g. a graphdef is common)
  Returns:
    Converted model in serialized form (e.g. a TFLITE model is common).
  Raises:
    ConverterError: When conversion fails in TFLiteConverter, usually due to
      ops not being supported.
    RuntimeError: When conversion fails, an exception is raised with the error
      message embedded.
  """
  # TODO(aselle): When toco does not use fatal errors for failure, we can
  # switch this on.
  if not _toco_from_proto_bin:
    try:
      model_str = _toco_python.TocoConvert(model_flags_str, toco_flags_str,
                                           input_data_str)
      return model_str
    except Exception as e:
      raise ConverterError("TOCO failed: %s" % e)

  # Windows and TemporaryFile are not that useful together,
  # since you cannot have two readers/writers. So we have to
  # make the temporaries and close and delete them explicitly.
  toco_filename, model_filename, input_filename, output_filename = (
      None, None, None, None)
  try:
    # Build all input files
    with _tempfile.NamedTemporaryFile(delete=False) as fp_toco, \
             _tempfile.NamedTemporaryFile(delete=False) as fp_model, \
             _tempfile.NamedTemporaryFile(delete=False) as fp_input:
      toco_filename = fp_toco.name
      input_filename = fp_input.name
      model_filename = fp_model.name
      fp_model.write(model_flags_str)
      fp_toco.write(toco_flags_str)
      fp_input.write(input_data_str)
      fp_model.flush()
      fp_toco.flush()
      fp_input.flush()

    # Reserve an output file
    with _tempfile.NamedTemporaryFile(delete=False) as fp:
      output_filename = fp.name

    # Run
    cmd = [
        _toco_from_proto_bin, model_filename, toco_filename, input_filename,
        output_filename
    ]
    cmdline = " ".join(cmd)
    is_windows = _platform.system() == "Windows"
    proc = _subprocess.Popen(
        cmdline,
        shell=True,
        stdout=_subprocess.PIPE,
        stderr=_subprocess.STDOUT,
        close_fds=not is_windows)
    stdout, stderr = proc.communicate()
    exitcode = proc.returncode
    if exitcode == 0:
      with open(output_filename, "rb") as fp:
        return fp.read()
    else:
      stdout = _try_convert_to_unicode(stdout)
      stderr = _try_convert_to_unicode(stderr)
      raise ConverterError(
          "TOCO failed. See console for info.\n%s\n%s\n" % (stdout, stderr))
  finally:
    # Must manually cleanup files.
    for filename in [
        toco_filename, input_filename, model_filename, output_filename]:
      try:
        _os.unlink(filename)
      except (OSError, TypeError):
        pass


def tensor_name(x):
  return x.name.split(":")[0]


# Don't expose these for now.
# @_tf_export("lite.build_toco_convert_protos")
def build_toco_convert_protos(input_tensors,
                              output_tensors,
                              inference_type=lite_constants.FLOAT,
                              inference_input_type=None,
                              input_format=lite_constants.TENSORFLOW_GRAPHDEF,
                              input_shapes=None,
                              output_format=lite_constants.TFLITE,
                              quantized_input_stats=None,
                              default_ranges_stats=None,
                              drop_control_dependency=True,
                              reorder_across_fake_quant=False,
                              allow_custom_ops=False,
                              change_concat_input_ranges=False,
                              post_training_quantize=False,
                              dump_graphviz_dir=None,
                              dump_graphviz_video=False,
                              target_ops=None,
                              allow_nonexistent_arrays=False):
  """Builds protocol buffers describing a conversion of a model using TOCO.

  Typically this is to convert from TensorFlow GraphDef to TFLite, in which
  case the default `input_format` and `output_format` are sufficient.

  Args:
    input_tensors: List of input tensors. Type and shape are computed using
      `foo.get_shape()` and `foo.dtype`.
    output_tensors: List of output tensors (only .name is used from this).
    inference_type: Target data type of real-number arrays in the output file.
      Must be `{tf.float32, tf.uint8}`.  (default tf.float32)
    inference_input_type: Target data type of real-number input arrays. Allows
      for a different type for input arrays in the case of quantization.
      Must be `{tf.float32, tf.uint8}`. (default `inference_type`)
    input_format: Type of data to read Currently must be
      `{TENSORFLOW_GRAPHDEF}`. (default TENSORFLOW_GRAPHDEF)
    input_shapes: Input array shape. It needs to be a list of the same length
      as `input_tensors`, or None. (default None)
    output_format: Output file format. Currently must be `{TFLITE,
      GRAPHVIZ_DOT}`. (default TFLITE)
    quantized_input_stats: List of tuples of floats representing the mean and
      standard deviation. Each tuple maps to the corresponding input tensor.
      Only need if `inference_input_type` is `QUANTIZED_UINT8`.
      real_input_value = (quantized_input_value - mean_value) / std_dev_value.
      (default None)
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
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      When false any unknown operation is an error. When true, custom ops are
      created for any op that is unknown. The developer will need to provide
      these to the TensorFlow Lite runtime with a custom resolver.
      (default False)
    change_concat_input_ranges: Boolean to change behavior of min/max ranges for
      inputs and outputs of the concat operator for quantized models. Changes
      the ranges of concat operator overlap when true. (default False)
    post_training_quantize: Boolean indicating whether to quantize the weights
      of the converted float model. Model size will be reduced and there will be
      latency improvements (at the cost of accuracy).
      (default False)
    dump_graphviz_dir: Full filepath of folder to dump the graphs at various
      stages of processing GraphViz .dot files. Preferred over
      --output_format=GRAPHVIZ_DOT in order to keep the requirements of the
      output file. (default None)
    dump_graphviz_video: Boolean indicating whether to dump the graph after
      every graph transformation. (default False)
    target_ops: Experimental flag, subject to change. Set of OpsSet
      options indicating which converter to use.
      (default set([OpsSet.TFLITE_BUILTINS]))
    allow_nonexistent_arrays: Allow specifying array names that don't exist
      or are unused in the final graph. (default False)

  Returns:
    model_flags, toco_flags: two protocol buffers describing the conversion
    process.

  Raises:
    ValueError: If the input tensor type is unknown
    RuntimeError: If TOCO fails to convert (in which case the runtime error's
      error text will contain the TOCO error log)
  """
  toco = _toco_flags_pb2.TocoFlags()
  toco.input_format = input_format
  toco.output_format = output_format
  toco.inference_type = convert_dtype_to_tflite_type(inference_type)
  if inference_input_type:
    toco.inference_input_type = convert_dtype_to_tflite_type(
        inference_input_type)
  else:
    toco.inference_input_type = toco.inference_type
  toco.drop_control_dependency = drop_control_dependency
  toco.reorder_across_fake_quant = reorder_across_fake_quant
  toco.allow_custom_ops = allow_custom_ops
  toco.post_training_quantize = post_training_quantize
  if default_ranges_stats:
    toco.default_ranges_min = default_ranges_stats[0]
    toco.default_ranges_max = default_ranges_stats[1]
  if dump_graphviz_dir:
    toco.dump_graphviz_dir = dump_graphviz_dir
  toco.dump_graphviz_include_video = dump_graphviz_video
  if target_ops:
    if set(target_ops) == set([OpsSet.TFLITE_BUILTINS, OpsSet.SELECT_TF_OPS]):
      toco.enable_select_tf_ops = True
    elif set(target_ops) == set([OpsSet.SELECT_TF_OPS]):
      toco.enable_select_tf_ops = True
      toco.force_select_tf_ops = True

  model = _model_flags_pb2.ModelFlags()
  model.change_concat_input_ranges = change_concat_input_ranges
  for idx, input_tensor in enumerate(input_tensors):
    input_array = model.input_arrays.add()
    input_array.name = tensor_name(input_tensor)
    input_array.data_type = convert_dtype_to_tflite_type(input_tensor.dtype)

    if toco.inference_input_type == _types_pb2.QUANTIZED_UINT8:
      input_array.mean_value, input_array.std_value = quantized_input_stats[idx]
    if input_shapes is None:
      shape = input_tensor.get_shape()
    else:
      shape = input_shapes[idx]
    input_array.shape.dims.extend(map(int, shape))

  for output_tensor in output_tensors:
    model.output_arrays.append(tensor_name(output_tensor))

  model.allow_nonexistent_arrays = allow_nonexistent_arrays

  return model, toco


def toco_convert_graph_def(input_data, input_arrays_with_shape, output_arrays,
                           *args, **kwargs):
  """"Convert a model using TOCO.

  This function is used to convert GraphDefs that cannot be loaded into
  TensorFlow to TFLite. Conversion can be customized by providing arguments
  that are forwarded to `build_toco_convert_protos` (see documentation for
  details).

  Args:
    input_data: Input data (i.e. often `sess.graph_def`),
    input_arrays_with_shape: Tuple of strings representing input tensor names
      and list of integers representing input shapes
      (e.g., [("foo" : [1, 16, 16, 3])]). Use only when graph cannot be loaded
      into TensorFlow and when `input_tensors` is None. (default None)
    output_arrays: List of output tensors to freeze graph with. Use only when
      graph cannot be loaded into TensorFlow and when `output_tensors` is None.
      (default None)
    *args: See `build_toco_convert_protos`,
    **kwargs: See `build_toco_convert_protos`.

  Returns:
    The converted data. For example if TFLite was the destination, then
    this will be a tflite flatbuffer in a bytes array.

  Raises:
    Defined in `build_toco_convert_protos`.
  """
  model_flags, toco_flags = build_toco_convert_protos(
      input_tensors=[], output_tensors=[], *args, **kwargs)

  for idx, (name, shape) in enumerate(input_arrays_with_shape):
    input_array = model_flags.input_arrays.add()
    if kwargs["inference_type"] == lite_constants.QUANTIZED_UINT8:
      input_array.mean_value, input_array.std_value = kwargs[
          "quantized_input_stats"][idx]
    input_array.name = name
    input_array.shape.dims.extend(map(int, shape))

  for name in output_arrays:
    model_flags.output_arrays.append(name)

  data = toco_convert_protos(model_flags.SerializeToString(),
                             toco_flags.SerializeToString(),
                             input_data.SerializeToString())
  return data


def toco_convert_impl(input_data, input_tensors, output_tensors, *args,
                      **kwargs):
  """"Convert a model using TOCO.

  Typically this function is used to convert from TensorFlow GraphDef to TFLite.
  Conversion can be customized by providing arguments that are forwarded to
  `build_toco_convert_protos` (see documentation for details).

  Args:
    input_data: Input data (i.e. often `sess.graph_def`),
    input_tensors: List of input tensors. Type and shape are computed using
      `foo.get_shape()` and `foo.dtype`.
    output_tensors: List of output tensors (only .name is used from this).
    *args: See `build_toco_convert_protos`,
    **kwargs: See `build_toco_convert_protos`.

  Returns:
    The converted data. For example if TFLite was the destination, then
    this will be a tflite flatbuffer in a bytes array.

  Raises:
    Defined in `build_toco_convert_protos`.
  """
  model_flags, toco_flags = build_toco_convert_protos(
      input_tensors, output_tensors, *args, **kwargs)
  data = toco_convert_protos(model_flags.SerializeToString(),
                             toco_flags.SerializeToString(),
                             input_data.SerializeToString())
  return data


@_tf_export("lite.toco_convert")
@deprecation.deprecated(None, "Use `lite.TFLiteConverter` instead.")
def toco_convert(input_data, input_tensors, output_tensors, *args, **kwargs):
  """Convert a model using TOCO.

  Typically this function is used to convert from TensorFlow GraphDef to TFLite.
  Conversion can be customized by providing arguments that are forwarded to
  `build_toco_convert_protos` (see documentation for details). This function has
  been deprecated. Please use `lite.TFLiteConverter` instead.

  Args:
    input_data: Input data (i.e. often `sess.graph_def`),
    input_tensors: List of input tensors. Type and shape are computed using
      `foo.get_shape()` and `foo.dtype`.
    output_tensors: List of output tensors (only .name is used from this).
    *args: See `build_toco_convert_protos`,
    **kwargs: See `build_toco_convert_protos`.

  Returns:
    The converted data. For example if TFLite was the destination, then
    this will be a tflite flatbuffer in a bytes array.

  Raises:
    Defined in `build_toco_convert_protos`.
  """
  return toco_convert_impl(input_data, input_tensors, output_tensors, *args,
                           **kwargs)
