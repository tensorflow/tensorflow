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
"""Python console command for generating frozen models from SavedModels.

This exists to add SavedModel compatibility to TOCO.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from tensorflow.contrib.lite.python.convert_saved_model import saved_model_to_frozen_graphdef
from tensorflow.python.platform import app

FLAGS = None


def execute(unused_args):
  """Calls function to convert the SavedModel to a frozen graph."""
  # Error handling.
  if FLAGS.input_shapes and not FLAGS.input_arrays:
    raise ValueError("Input shapes requires input arrays to be specified.")

  # Calls saved_model_to_frozen_graphdef function to generate frozen graph.
  input_arrays = (FLAGS.input_arrays.split(",") if FLAGS.input_arrays else None)
  input_shapes = None
  if FLAGS.input_shapes:
    input_shapes = {
        input_arrays[idx]: shape.split(",")
        for idx, shape in enumerate(FLAGS.input_shapes.split(":"))
    }
  output_arrays = (
      FLAGS.output_arrays.split(",") if FLAGS.output_arrays else None)
  tag_set = set(FLAGS.tag_set.split(",")) if FLAGS.tag_set else None

  saved_model_to_frozen_graphdef(
      saved_model_dir=FLAGS.saved_model_directory,
      output_file_model=FLAGS.output_file_model,
      output_file_flags=FLAGS.output_file_flags,
      input_arrays=input_arrays,
      input_shapes=input_shapes,
      output_arrays=output_arrays,
      tag_set=tag_set,
      signature_key=FLAGS.signature_key,
      batch_size=FLAGS.batch_size)


def main():
  global FLAGS
  # Parses flags.
  parser = argparse.ArgumentParser(
      description="Invoke SavedModel to frozen model converter.")
  parser.add_argument(
      "saved_model_directory",
      type=str,
      help="Full path to directory containing the SavedModel.")
  parser.add_argument(
      "output_file_model",
      type=str,
      help="Full file path to save frozen graph.")
  parser.add_argument(
      "output_file_flags", type=str, help="Full file path to save ModelFlags.")
  parser.add_argument(
      "--input_arrays",
      type=str,
      help="Name of the input arrays, comma-separated.")
  parser.add_argument(
      "--input_shapes",
      type=str,
      help="Shapes corresponding to --input_arrays, colon-separated.")
  parser.add_argument(
      "--output_arrays",
      type=str,
      help="Name of the output arrays, comma-separated.")
  parser.add_argument(
      "--tag_set", type=str, help="Name of output arrays, comma-separated.")
  parser.add_argument(
      "--signature_key",
      type=str,
      help="Key identifying SignatureDef containing inputs and outputs.")
  parser.add_argument(
      "--batch_size",
      type=int,
      help="Batch size for the model. Replaces the first dimension of an "
      "input size array if undefined.")

  FLAGS, unparsed = parser.parse_known_args()

  app.run(main=execute, argv=[sys.argv[0]] + unparsed)


if __name__ == "__main__":
  main()
