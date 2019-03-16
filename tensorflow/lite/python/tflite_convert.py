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
"""Python command line interface for running TOCO."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from tensorflow.lite.python import lite
from tensorflow.lite.python import lite_constants
from tensorflow.lite.toco import toco_flags_pb2 as _toco_flags_pb2
from tensorflow.python import tf2
from tensorflow.python.platform import app


def _parse_array(values, type_fn=str):
  if values is not None:
    return [type_fn(val) for val in values.split(",") if val]
  return None


def _parse_set(values):
  if values is not None:
    return set([item for item in values.split(",") if item])
  return None


def _parse_inference_type(value, flag):
  """Converts the inference type to the value of the constant.

  Args:
    value: str representing the inference type.
    flag: str representing the flag name.

  Returns:
    tf.dtype.

  Raises:
    ValueError: Unsupported value.
  """
  if value == "FLOAT":
    return lite_constants.FLOAT
  if value == "QUANTIZED_UINT8":
    return lite_constants.QUANTIZED_UINT8
  raise ValueError("Unsupported value for --{0}. Only FLOAT and "
                   "QUANTIZED_UINT8 are supported.".format(flag))


def _get_toco_converter(flags):
  """Makes a TFLiteConverter object based on the flags provided.

  Args:
    flags: argparse.Namespace object containing TFLite flags.

  Returns:
    TFLiteConverter object.

  Raises:
    ValueError: Invalid flags.
  """
  # Parse input and output arrays.
  input_arrays = _parse_array(flags.input_arrays)
  input_shapes = None
  if flags.input_shapes:
    input_shapes_list = [
        _parse_array(shape, type_fn=int)
        for shape in flags.input_shapes.split(":")
    ]
    input_shapes = dict(zip(input_arrays, input_shapes_list))
  output_arrays = _parse_array(flags.output_arrays)

  converter_kwargs = {
      "input_arrays": input_arrays,
      "input_shapes": input_shapes,
      "output_arrays": output_arrays
  }

  # Create TFLiteConverter.
  if flags.graph_def_file:
    converter_fn = lite.TFLiteConverter.from_frozen_graph
    converter_kwargs["graph_def_file"] = flags.graph_def_file
  elif flags.saved_model_dir:
    converter_fn = lite.TFLiteConverter.from_saved_model
    converter_kwargs["saved_model_dir"] = flags.saved_model_dir
    converter_kwargs["tag_set"] = _parse_set(flags.saved_model_tag_set)
    converter_kwargs["signature_key"] = flags.saved_model_signature_key
  elif flags.keras_model_file:
    converter_fn = lite.TFLiteConverter.from_keras_model_file
    converter_kwargs["model_file"] = flags.keras_model_file
  else:
    raise ValueError("--graph_def_file, --saved_model_dir, or "
                     "--keras_model_file must be specified.")

  return converter_fn(**converter_kwargs)


def _convert_model(flags):
  """Calls function to convert the TensorFlow model into a TFLite model.

  Args:
    flags: argparse.Namespace object.

  Raises:
    ValueError: Invalid flags.
  """
  # Create converter.
  converter = _get_toco_converter(flags)
  if flags.inference_type:
    converter.inference_type = _parse_inference_type(flags.inference_type,
                                                     "inference_type")
  if flags.inference_input_type:
    converter.inference_input_type = _parse_inference_type(
        flags.inference_input_type, "inference_input_type")
  if flags.output_format:
    converter.output_format = _toco_flags_pb2.FileFormat.Value(
        flags.output_format)

  if flags.mean_values and flags.std_dev_values:
    input_arrays = converter.get_input_arrays()
    std_dev_values = _parse_array(flags.std_dev_values, type_fn=float)

    # In quantized inference, mean_value has to be integer so that the real
    # value 0.0 is exactly representable.
    if converter.inference_type == lite_constants.QUANTIZED_UINT8:
      mean_values = _parse_array(flags.mean_values, type_fn=int)
    else:
      mean_values = _parse_array(flags.mean_values, type_fn=float)
    quant_stats = list(zip(mean_values, std_dev_values))
    if ((not flags.input_arrays and len(input_arrays) > 1) or
        (len(input_arrays) != len(quant_stats))):
      raise ValueError("Mismatching --input_arrays, --std_dev_values, and "
                       "--mean_values. The flags must have the same number of "
                       "items. The current input arrays are '{0}'. "
                       "--input_arrays must be present when specifying "
                       "--std_dev_values and --mean_values with multiple input "
                       "tensors in order to map between names and "
                       "values.".format(",".join(input_arrays)))
    converter.quantized_input_stats = dict(zip(input_arrays, quant_stats))
  if (flags.default_ranges_min is not None) and (flags.default_ranges_max is
                                                 not None):
    converter.default_ranges_stats = (flags.default_ranges_min,
                                      flags.default_ranges_max)

  if flags.drop_control_dependency:
    converter.drop_control_dependency = flags.drop_control_dependency
  if flags.reorder_across_fake_quant:
    converter.reorder_across_fake_quant = flags.reorder_across_fake_quant
  if flags.change_concat_input_ranges:
    converter.change_concat_input_ranges = (
        flags.change_concat_input_ranges == "TRUE")

  if flags.allow_custom_ops:
    converter.allow_custom_ops = flags.allow_custom_ops
  if flags.target_ops:
    ops_set_options = lite.OpsSet.get_options()
    converter.target_ops = set()
    for option in flags.target_ops.split(","):
      if option not in ops_set_options:
        raise ValueError("Invalid value for --target_ops. Options: "
                         "{0}".format(",".join(ops_set_options)))
      converter.target_ops.add(lite.OpsSet(option))

  if flags.post_training_quantize:
    converter.post_training_quantize = flags.post_training_quantize
    if converter.inference_type == lite_constants.QUANTIZED_UINT8:
      print("--post_training_quantize quantizes a graph of inference_type "
            "FLOAT. Overriding inference type QUANTIZED_UINT8 to FLOAT.")
      converter.inference_type = lite_constants.FLOAT

  if flags.dump_graphviz_dir:
    converter.dump_graphviz_dir = flags.dump_graphviz_dir
  if flags.dump_graphviz_video:
    converter.dump_graphviz_vode = flags.dump_graphviz_video

  # Convert model.
  output_data = converter.convert()
  with open(flags.output_file, "wb") as f:
    f.write(output_data)


def _check_flags(flags, unparsed):
  """Checks the parsed and unparsed flags to ensure they are valid.

  Raises an error if previously support unparsed flags are found. Raises an
  error for parsed flags that don't meet the required conditions.

  Args:
    flags: argparse.Namespace object containing TFLite flags.
    unparsed: List of unparsed flags.

  Raises:
    ValueError: Invalid flags.
  """

  # Check unparsed flags for common mistakes based on previous TOCO.
  def _get_message_unparsed(flag, orig_flag, new_flag):
    if flag.startswith(orig_flag):
      return "\n  Use {0} instead of {1}".format(new_flag, orig_flag)
    return ""

  if unparsed:
    output = ""
    for flag in unparsed:
      output += _get_message_unparsed(flag, "--input_file", "--graph_def_file")
      output += _get_message_unparsed(flag, "--savedmodel_directory",
                                      "--saved_model_dir")
      output += _get_message_unparsed(flag, "--std_value", "--std_dev_values")
      output += _get_message_unparsed(flag, "--batch_size", "--input_shapes")
      output += _get_message_unparsed(flag, "--dump_graphviz",
                                      "--dump_graphviz_dir")
    if output:
      raise ValueError(output)

  # Check that flags are valid.
  if flags.graph_def_file and (not flags.input_arrays or
                               not flags.output_arrays):
    raise ValueError("--input_arrays and --output_arrays are required with "
                     "--graph_def_file")

  if flags.input_shapes:
    if not flags.input_arrays:
      raise ValueError("--input_shapes must be used with --input_arrays")
    if flags.input_shapes.count(":") != flags.input_arrays.count(","):
      raise ValueError("--input_shapes and --input_arrays must have the same "
                       "number of items")

  if flags.std_dev_values or flags.mean_values:
    if bool(flags.std_dev_values) != bool(flags.mean_values):
      raise ValueError("--std_dev_values and --mean_values must be used "
                       "together")
    if flags.std_dev_values.count(",") != flags.mean_values.count(","):
      raise ValueError("--std_dev_values, --mean_values must have the same "
                       "number of items")

  if (flags.default_ranges_min is None) != (flags.default_ranges_max is None):
    raise ValueError("--default_ranges_min and --default_ranges_max must be "
                     "used together")

  if flags.dump_graphviz_video and not flags.dump_graphviz_dir:
    raise ValueError("--dump_graphviz_video must be used with "
                     "--dump_graphviz_dir")


def run_main(_):
  """Main in toco_convert.py."""
  if tf2.enabled():
    raise ValueError("tflite_convert is currently unsupported in 2.0. "
                     "Please use the Python API "
                     "tf.lite.TFLiteConverter.from_concrete_function().")

  parser = argparse.ArgumentParser(
      description=("Command line tool to run TensorFlow Lite Optimizing "
                   "Converter (TOCO)."))

  # Output file flag.
  parser.add_argument(
      "--output_file",
      type=str,
      help="Full filepath of the output file.",
      required=True)

  # Input file flags.
  input_file_group = parser.add_mutually_exclusive_group(required=True)
  input_file_group.add_argument(
      "--graph_def_file",
      type=str,
      help="Full filepath of file containing frozen TensorFlow GraphDef.")
  input_file_group.add_argument(
      "--saved_model_dir",
      type=str,
      help="Full filepath of directory containing the SavedModel.")
  input_file_group.add_argument(
      "--keras_model_file",
      type=str,
      help="Full filepath of HDF5 file containing tf.Keras model.")

  # Model format flags.
  parser.add_argument(
      "--output_format",
      type=str.upper,
      choices=["TFLITE", "GRAPHVIZ_DOT"],
      help="Output file format.")
  parser.add_argument(
      "--inference_type",
      type=str.upper,
      choices=["FLOAT", "QUANTIZED_UINT8"],
      help="Target data type of real-number arrays in the output file.")
  parser.add_argument(
      "--inference_input_type",
      type=str.upper,
      choices=["FLOAT", "QUANTIZED_UINT8"],
      help=("Target data type of real-number input arrays. Allows for a "
            "different type for input arrays in the case of quantization."))

  # Input and output arrays flags.
  parser.add_argument(
      "--input_arrays",
      type=str,
      help="Names of the input arrays, comma-separated.")
  parser.add_argument(
      "--input_shapes",
      type=str,
      help="Shapes corresponding to --input_arrays, colon-separated.")
  parser.add_argument(
      "--output_arrays",
      type=str,
      help="Names of the output arrays, comma-separated.")

  # SavedModel related flags.
  parser.add_argument(
      "--saved_model_tag_set",
      type=str,
      help=("Comma-separated set of tags identifying the MetaGraphDef within "
            "the SavedModel to analyze. All tags must be present. In order to "
            "pass in an empty tag set, pass in \"\". (default \"serve\")"))
  parser.add_argument(
      "--saved_model_signature_key",
      type=str,
      help=("Key identifying the SignatureDef containing inputs and outputs. "
            "(default DEFAULT_SERVING_SIGNATURE_DEF_KEY)"))

  # Quantization flags.
  parser.add_argument(
      "--std_dev_values",
      type=str,
      help=("Standard deviation of training data for each input tensor, "
            "comma-separated floats. Used for quantized input tensors. "
            "(default None)"))
  parser.add_argument(
      "--mean_values",
      type=str,
      help=("Mean of training data for each input tensor, comma-separated "
            "floats. Used for quantized input tensors. (default None)"))
  parser.add_argument(
      "--default_ranges_min",
      type=float,
      help=("Default value for min bound of min/max range values used for all "
            "arrays without a specified range, Intended for experimenting with "
            "quantization via \"dummy quantization\". (default None)"))
  parser.add_argument(
      "--default_ranges_max",
      type=float,
      help=("Default value for max bound of min/max range values used for all "
            "arrays without a specified range, Intended for experimenting with "
            "quantization via \"dummy quantization\". (default None)"))
  # quantize_weights is DEPRECATED.
  parser.add_argument(
      "--quantize_weights",
      dest="post_training_quantize",
      action="store_true",
      help=argparse.SUPPRESS)
  parser.add_argument(
      "--post_training_quantize",
      dest="post_training_quantize",
      action="store_true",
      help=(
          "Boolean indicating whether to quantize the weights of the "
          "converted float model. Model size will be reduced and there will "
          "be latency improvements (at the cost of accuracy). (default False)"))

  # Graph manipulation flags.
  parser.add_argument(
      "--drop_control_dependency",
      action="store_true",
      help=("Boolean indicating whether to drop control dependencies silently. "
            "This is due to TensorFlow not supporting control dependencies. "
            "(default True)"))
  parser.add_argument(
      "--reorder_across_fake_quant",
      action="store_true",
      help=("Boolean indicating whether to reorder FakeQuant nodes in "
            "unexpected locations. Used when the location of the FakeQuant "
            "nodes is preventing graph transformations necessary to convert "
            "the graph. Results in a graph that differs from the quantized "
            "training graph, potentially causing differing arithmetic "
            "behavior. (default False)"))
  # Usage for this flag is --change_concat_input_ranges=true or
  # --change_concat_input_ranges=false in order to make it clear what the flag
  # is set to. This keeps the usage consistent with other usages of the flag
  # where the default is different. The default value here is False.
  parser.add_argument(
      "--change_concat_input_ranges",
      type=str.upper,
      choices=["TRUE", "FALSE"],
      help=("Boolean to change behavior of min/max ranges for inputs and "
            "outputs of the concat operator for quantized models. Changes the "
            "ranges of concat operator overlap when true. (default False)"))

  # Permitted ops flags.
  parser.add_argument(
      "--allow_custom_ops",
      action="store_true",
      help=("Boolean indicating whether to allow custom operations. When false "
            "any unknown operation is an error. When true, custom ops are "
            "created for any op that is unknown. The developer will need to "
            "provide these to the TensorFlow Lite runtime with a custom "
            "resolver. (default False)"))
  parser.add_argument(
      "--target_ops",
      type=str,
      help=("Experimental flag, subject to change. Set of OpsSet options "
            "indicating which converter to use. Options: {0}. One or more "
            "option may be specified. (default set([OpsSet.TFLITE_BUILTINS]))"
            "".format(",".join(lite.OpsSet.get_options()))))

  # Logging flags.
  parser.add_argument(
      "--dump_graphviz_dir",
      type=str,
      help=("Full filepath of folder to dump the graphs at various stages of "
            "processing GraphViz .dot files. Preferred over --output_format="
            "GRAPHVIZ_DOT in order to keep the requirements of the output "
            "file."))
  parser.add_argument(
      "--dump_graphviz_video",
      action="store_true",
      help=("Boolean indicating whether to dump the graph after every graph "
            "transformation"))

  tflite_flags, unparsed = parser.parse_known_args(args=sys.argv[1:])
  try:
    _check_flags(tflite_flags, unparsed)
  except ValueError as e:
    parser.print_usage()
    file_name = os.path.basename(sys.argv[0])
    sys.stderr.write("{0}: error: {1}\n".format(file_name, str(e)))
    sys.exit(1)
  _convert_model(tflite_flags)


def main():
  app.run(main=run_main, argv=sys.argv[:1])


if __name__ == "__main__":
  main()
