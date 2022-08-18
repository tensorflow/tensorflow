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
"""Converts a TFLite model to a TFLite Micro model (C++ Source)."""

from absl import app
from absl import flags

from tensorflow.lite.python import util

FLAGS = flags.FLAGS

flags.DEFINE_string("input_tflite_file", None,
                    "Full path name to the input TFLite model file.")
flags.DEFINE_string(
    "output_source_file", None,
    "Full path name to the output TFLite Micro model (C++ Source) file).")
flags.DEFINE_string("output_header_file", None,
                    "Full filepath of the output C header file.")
flags.DEFINE_string("array_variable_name", None,
                    "Name to use for the C data array variable.")
flags.DEFINE_integer("line_width", 80, "Width to use for formatting.")
flags.DEFINE_string("include_guard", None,
                    "Name to use for the C header include guard.")
flags.DEFINE_string("include_path", None,
                    "Optional path to include in generated source file.")
flags.DEFINE_boolean(
    "use_tensorflow_license", False,
    "Whether to prefix the generated files with the TF Apache2 license.")

flags.mark_flag_as_required("input_tflite_file")
flags.mark_flag_as_required("output_source_file")
flags.mark_flag_as_required("output_header_file")
flags.mark_flag_as_required("array_variable_name")


def main(_):
  with open(FLAGS.input_tflite_file, "rb") as input_handle:
    input_data = input_handle.read()

  source, header = util.convert_bytes_to_c_source(
      data=input_data,
      array_name=FLAGS.array_variable_name,
      max_line_width=FLAGS.line_width,
      include_guard=FLAGS.include_guard,
      include_path=FLAGS.include_path,
      use_tensorflow_license=FLAGS.use_tensorflow_license)

  with open(FLAGS.output_source_file, "w") as source_handle:
    source_handle.write(source)

  with open(FLAGS.output_header_file, "w") as header_handle:
    header_handle.write(header)


if __name__ == "__main__":
  app.run(main)
