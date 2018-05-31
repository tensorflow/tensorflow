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

from tensorflow.contrib.lite.python import lite
from tensorflow.contrib.lite.toco import toco_flags_pb2 as _toco_flags_pb2
from tensorflow.contrib.lite.toco import types_pb2 as _types_pb2
from tensorflow.python.platform import app


def _parse_array(values):
  if values:
    return values.split(",")


def _parse_int_array(values):
  if values:
    return [int(val) for val in values.split(",")]


def _parse_set(values):
  if values:
    return set(values.split(","))


def _get_toco_converter(flags):
  """Makes a TocoConverter object based on the flags provided.

  Args:
    flags: argparse.Namespace object containing TFLite flags.

  Returns:
    TocoConverter object.
  """
  # Parse input and output arrays.
  input_arrays = _parse_array(flags.input_arrays)
  input_shapes = None
  if flags.input_shapes:
    input_shapes_list = [
        _parse_int_array(shape) for shape in flags.input_shapes.split(":")
    ]
    input_shapes = dict(zip(input_arrays, input_shapes_list))
  output_arrays = _parse_array(flags.output_arrays)

  converter_kwargs = {
      "input_arrays": input_arrays,
      "input_shapes": input_shapes,
      "output_arrays": output_arrays
  }

  # Create TocoConverter.
  if flags.graph_def_file:
    converter_fn = lite.TocoConverter.from_flatbuffer_file
    converter_kwargs["graph_def_file"] = flags.graph_def_file
  elif flags.saved_model_dir:
    converter_fn = lite.TocoConverter.from_saved_model
    converter_kwargs["saved_model_dir"] = flags.saved_model_dir
    converter_kwargs["tag_set"] = _parse_set(flags.saved_model_tag_set)
    converter_kwargs["signature_key"] = flags.saved_model_signature_key

  return converter_fn(**converter_kwargs)


def _convert_model(flags):
  """Calls function to convert the TensorFlow model into a TFLite model.

  Args:
    flags: argparse.Namespace object.
  """
  # Create converter.
  converter = _get_toco_converter(flags)
  if flags.inference_type:
    converter.inference_type = _types_pb2.IODataType.Value(flags.inference_type)
  if flags.output_format:
    converter.output_format = _toco_flags_pb2.FileFormat.Value(
        flags.output_format)

  if flags.mean_values and flags.std_dev_values:
    input_arrays = _parse_array(flags.input_arrays)
    std_dev_values = _parse_int_array(flags.std_dev_values)
    mean_values = _parse_int_array(flags.mean_values)
    quant_stats = zip(mean_values, std_dev_values)
    converter.quantized_input_stats = dict(zip(input_arrays, quant_stats))

  if flags.drop_control_dependency:
    converter.drop_control_dependency = flags.drop_control_dependency
  if flags.allow_custom_ops:
    converter.allow_custom_ops = flags.allow_custom_ops

  # Convert model.
  output_data = converter.convert()
  with open(flags.output_file, "wb") as f:
    f.write(output_data)


def _check_flags(flags, unparsed):
  """Checks the parsed and unparsed flags to ensure they are valid.

  Displays warnings for unparsed flags. Raises an error for parsed flags that
  don't meet the required conditions.

  Args:
    flags: argparse.Namespace object containing TFLite flags.
    unparsed: List of unparsed flags.

  Raises:
    ValueError: Invalid flags.
  """
  # Check unparsed flags for common mistakes based on previous TOCO.
  if unparsed:
    print("tflite_convert: warning: Unable to parse following flags "
          "'{}'".format(",".join(unparsed)))
    for flag in unparsed:
      if "--input_file=" in flag:
        print("tflite_convert: warning: Use --graph_def_file instead of "
              "--input_file")
      if "--std_values=" in flag:
        print("tflite_convert: warning: Use --std_dev_values instead of "
              "--std_values")

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
    if not flags.input_arrays:
      raise ValueError("--std_dev_values and --mean_values must be used with "
                       "--input_arrays")
    if (flags.std_dev_values.count(",") != flags.mean_values.count(",") or
        flags.std_dev_values.count(",") != flags.input_arrays.count(",")):
      raise ValueError("--std_dev_values, --mean_values, and --input_arrays "
                       "must have the same number of items")


def run_main(_):
  """Main in toco_convert.py."""
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
      help="Full filepath of file containing TensorFlow GraphDef.")
  input_file_group.add_argument(
      "--saved_model_dir",
      type=str,
      help="Full filepath of directory containing the SavedModel.")

  # Model format flags.
  parser.add_argument(
      "--output_format",
      type=str,
      choices=["TFLITE", "GRAPHVIZ_DOT"],
      help="Output file format.")
  parser.add_argument(
      "--inference_type",
      type=str,
      choices=["FLOAT", "QUANTIZED_UINT8"],
      help="Target data type of arrays in the output file.")

  # Input and output arrays flags.
  parser.add_argument(
      "--input_arrays",
      type=str,
      help="Names of the output arrays, comma-separated.")
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
      help=("Set of tags identifying the MetaGraphDef within the SavedModel "
            "to analyze. All tags must be present. (default \"serve\")"))
  parser.add_argument(
      "--saved_model_signature_key",
      type=str,
      help=("Key identifying SignatureDef containing inputs and outputs. "
            "(default DEFAULT_SERVING_SIGNATURE_DEF_KEY)"))

  # Quantization flags.
  parser.add_argument(
      "--std_dev_values",
      type=str,
      help=("Standard deviation of training data for each input tensor, "
            "comma-separated. Used for quantization. (default None)"))
  parser.add_argument(
      "--mean_values",
      type=str,
      help=("Mean of training data for each input tensor, comma-separated. "
            "Used for quantization. (default None)"))

  # Graph manipulation flags.
  parser.add_argument(
      "--drop_control_dependency",
      type=bool,
      help=("Boolean indicating whether to drop control dependencies silently. "
            "This is due to TensorFlow Lite not supporting control "
            "dependencies. (default True)"))
  parser.add_argument(
      "--allow_custom_ops",
      type=bool,
      help=("Boolean indicating whether to allow custom operations. When false "
            "any unknown operation is an error. When true, custom ops are "
            "created for any op that is unknown. The developer will need to "
            "provide these to the TensorFlow Lite runtime with a custom "
            "resolver. (default False)"))

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
