# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""Generate a series of TensorFlow graphs that become tflite test cases.

Usage:

generate_examples <output directory>

bazel run //tensorflow/lite/testing:generate_examples

To more easily debug failures use (or override) the --save_graphdefs flag to
place text proto graphdefs into the generated zip files.
"""

import argparse
import os
import sys

import tensorflow.compat.v1 as tf

from tensorflow.lite.testing import generate_examples_lib
from tensorflow.lite.testing import mlir_convert

MLIR_CONVERTER_KNOWN_BUGS = {
    # We need to support dynamic_rnn case.
    r"unidirectional_sequence_rnn.*is_dynamic_rnn=True": "128997102",
    r"unidirectional_sequence_lstm.*is_dynamic_rnn=True": "128997102",
    # TODO(b/124314620): Test cases work with tf_tfl_translate binary
    # but not TFLiteConverter interface.
    # Concat & SpaceToDepth with uint8 doesn't work.
    r"concat.*type=tf\.uint8": "124314620",
    r"space_to_depth.*type=tf\.uint8": "124314620",
    r"l2norm.*fully_quantize=True": "134594898",
    # Below are not really a converter bug, but our kernels doesn't support
    # int64.
    r"div.*dtype=tf\.int64": "119126484",
    r"floor_div.*dtype=tf\.int64": "119126484",
    r"relu.*dtype=tf\.int64": "119126484",
    r"squared_difference.*dtype=tf\.int64": "119126484",
    # Post-training quantization support missing for below op in mlir.
    r"prelu.*fully_quantize=True": "156112683",
    # ResizeBilinear op kernel supports only float32 and quantized 8-bit
    # integers.
    r"resize_bilinear.*dtype=tf\.int32": "156569626",
}

# Disable GPU for now since we are just testing in TF against CPU reference
# value and creating non-device-specific graphs to export.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description="Script to generate TFLite tests.")
parser.add_argument(
    "output_path", help="Directory where the outputs will be go.")
parser.add_argument(
    "--zip_to_output",
    type=str,
    help="Particular zip to output.",
    required=True)
parser.add_argument(
    "--known_bugs_are_errors",
    action="store_true",
    help=("If a particular model is affected by a known bug,"
          " count it as a converter error."))
parser.add_argument(
    "--ignore_converter_errors",
    action="store_true",
    help="Raise an exception if any converter error is encountered.")
parser.add_argument(
    "--save_graphdefs",
    action="store_true",
    help="Include intermediate graphdefs in the output zip files.")
parser.add_argument(
    "--run_with_flex",
    action="store_true",
    help="Whether the TFLite Flex converter is being used.")
parser.add_argument(
    "--make_edgetpu_tests",
    action="store_true",
    help="Whether to generate test cases for edgetpu.")
parser.add_argument(
    "--make_tf_ptq_tests",
    action="store_true",
    help="Whether to generate test cases for TF post-training quantization.")
parser.add_argument(
    "--hlo_aware_conversion",
    action="store_true",
    help="For TF Quantization only: whether conversion for HLO target.")
parser.add_argument(
    "--make_forward_compat_test",
    action="store_true",
    help="Make tests by setting TF forward compatibility horizon to the future")
parser.add_argument(
    "--no_tests_limit",
    action="store_true",
    help="Remove the limit of the number of tests.")
parser.add_argument(
    "--test_sets",
    type=str,
    help=("Comma-separated list of test set names to generate. "
          "If not specified, a test set is selected by parsing the name of "
          "'zip_to_output' file."))
parser.add_argument(
    "--mlir_quantizer",
    action="store_true",
    help=("Whether the new MLIR quantizer is being used."))
parser.add_argument(
    "--skip_high_dimension_inputs",
    action="store_true",
    help=("Whether to skip generating tests with high dimension input shape."))


def main(unused_args):
  options = generate_examples_lib.Options()

  options.output_path = FLAGS.output_path
  options.zip_to_output = FLAGS.zip_to_output
  options.known_bugs_are_errors = FLAGS.known_bugs_are_errors
  options.ignore_converter_errors = FLAGS.ignore_converter_errors
  options.save_graphdefs = FLAGS.save_graphdefs
  options.run_with_flex = FLAGS.run_with_flex
  options.make_edgetpu_tests = FLAGS.make_edgetpu_tests
  options.make_tf_ptq_tests = FLAGS.make_tf_ptq_tests
  options.tflite_convert_function = mlir_convert.mlir_convert
  options.known_bugs = MLIR_CONVERTER_KNOWN_BUGS
  options.make_forward_compat_test = FLAGS.make_forward_compat_test
  options.no_tests_limit = FLAGS.no_tests_limit
  options.mlir_quantizer = FLAGS.mlir_quantizer
  options.skip_high_dimension_inputs = FLAGS.skip_high_dimension_inputs

  if FLAGS.test_sets:
    test_sets = FLAGS.test_sets.split(",")
    generate_examples_lib.generate_multi_set_examples(options, test_sets)
  else:
    generate_examples_lib.generate_examples(options)


if __name__ == "__main__":
  FLAGS, unparsed = parser.parse_known_args()

  if unparsed:
    print("\nGot the following unparsed args, %r please fix.\n" % unparsed +
          "Usage: %s <path out> <zip file to generate>")
    exit(1)
  else:
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
