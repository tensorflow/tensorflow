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
"""Generates a series of test cases using MLIR-based conversion."""

# This is forked from `tensorflow/lite/testing/generate_examples.py`.
# TODO(b/136499575): Merge this back to TFLite codebase when open sourcing.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

import tensorflow.compat.v1 as tf

from tensorflow.lite.experimental.mlir.testing import mlir_convert
# pylint: disable=unused-import
from tensorflow.lite.experimental.mlir.testing.op_tests.batchmatmul import make_batchmatmul_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.broadcast_gradient_args import make_broadcast_gradient_args_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.broadcast_to import make_broadcast_to_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.complex_abs import make_complex_abs_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.cond import make_cond_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.control_dep import make_control_dep_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.conv3d import make_conv3d_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.conv_bias_activation import make_conv_bias_relu6_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.cumsum import make_cumsum_tests
# Placeholder for make_dense_image_warp_tests import
from tensorflow.lite.experimental.mlir.testing.op_tests.dynamic_rnn import make_dynamic_rnn_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.einsum import make_einsum_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.identify_dilated_conv import make_identify_dilated_conv_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.identify_dilated_conv1d import make_identify_dilated_conv1d_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.imag import make_imag_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.irfft2d import make_irfft2d_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.is_finite import make_is_finite_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.max_pool_with_argmax import make_max_pool_with_argmax_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.parse_example import make_parse_example_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.real import make_real_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.reciprocal import make_reciprocal_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.rfft import make_rfft_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.rfft2d import make_rfft2d_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.segment_sum import make_segment_sum_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.shape_to_strided_slice import make_shape_to_strided_slice_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.softplus import make_softplus_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.static_hashtable import make_static_hashtable_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.static_rnn_with_control_flow_v2 import make_static_rnn_with_control_flow_v2_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.stft import make_stft_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.tensor_list_concat import make_tensor_list_concat_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.tensor_list_dynamic_shape import make_tensor_list_dynamic_shape_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.tensor_list_get_item import make_tensor_list_get_item_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.tensor_list_length import make_tensor_list_length_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.tensor_list_resize import make_tensor_list_resize_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.tensor_list_set_item import make_tensor_list_set_item_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.where_v2 import make_where_v2_tests
from tensorflow.lite.experimental.mlir.testing.op_tests.while_loop import make_while_tests

from tensorflow.lite.testing import generate_examples_lib


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
    r"mul.*dtype=tf\.int64": "119126484",
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
parser.add_argument("output_path",
                    help="Directory where the outputs will be go.")
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
    "--make_forward_compat_test",
    action="store_true",
    help="Make tests by setting TF forward compatibility horizon to the future")
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


def main(unused_args):
  options = generate_examples_lib.Options()

  options.output_path = FLAGS.output_path
  options.zip_to_output = FLAGS.zip_to_output
  options.known_bugs_are_errors = FLAGS.known_bugs_are_errors
  options.ignore_converter_errors = FLAGS.ignore_converter_errors
  options.save_graphdefs = FLAGS.save_graphdefs
  options.run_with_flex = FLAGS.run_with_flex
  options.make_edgetpu_tests = FLAGS.make_edgetpu_tests
  options.tflite_convert_function = mlir_convert.mlir_convert
  options.known_bugs = MLIR_CONVERTER_KNOWN_BUGS
  options.make_forward_compat_test = FLAGS.make_forward_compat_test
  options.use_experimental_converter = True
  options.mlir_quantizer = FLAGS.mlir_quantizer

  if FLAGS.test_sets:
    test_sets = FLAGS.test_sets.split(",")
    generate_examples_lib.generate_multi_set_examples(options, test_sets)
  else:
    generate_examples_lib.generate_examples(options)


if __name__ == "__main__":
  FLAGS, unparsed = parser.parse_known_args()

  if unparsed:
    print("Usage: %s <path out> <zip file to generate>")
    exit(1)
  else:
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
