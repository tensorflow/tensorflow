## Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Exports an example linear regression inference graph.

Exports a TensorFlow graph to /tmp/saved_model/half_plus_two/ based on the
SavedModel format.

This graph calculates,
  y = a*x + b
and/or, independently,
  y2 = a*x2 + c
where a, b and c are variables with a=0.5 and b=2 and c=3.

Output from this program is typically used to exercise SavedModel load and
execution code.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat

tf.app.flags.DEFINE_string("output_dir", "/tmp/saved_model_half_plus_two",
                           "Directory where to ouput SavedModel.")
tf.app.flags.DEFINE_string("output_dir_pbtxt",
                           "/tmp/saved_model_half_plus_two_pbtxt",
                           "Directory where to ouput the text format of "
                           "SavedModel.")
FLAGS = tf.flags.FLAGS


def _write_assets(assets_directory, assets_filename):
  """Writes asset files to be used with SavedModel for half plus two.

  Args:
    assets_directory: The directory to which the assets should be written.
    assets_filename: Name of the file to which the asset contents should be
        written.

  Returns:
    The path to which the assets file was written.
  """
  if not file_io.file_exists(assets_directory):
    file_io.recursive_create_dir(assets_directory)

  path = os.path.join(
      compat.as_bytes(assets_directory), compat.as_bytes(assets_filename))
  file_io.write_string_to_file(path, "asset-file-contents")
  return path


def _generate_saved_model_for_half_plus_two(export_dir, as_text=False):
  """Generates SavedModel for half plus two.

  Args:
    export_dir: The directory to which the SavedModel should be written.
    as_text: Writes the SavedModel protocol buffer in text format to disk.
  """
  builder = saved_model_builder.SavedModelBuilder(export_dir)

  with tf.Session(graph=tf.Graph()) as sess:
    # Set up the model parameters as variables to exercise variable loading
    # functionality upon restore.
    a = tf.Variable(0.5, name="a")
    b = tf.Variable(2.0, name="b")
    c = tf.Variable(3.0, name="c")

    # Create a placeholder for serialized tensorflow.Example messages to be fed.
    serialized_tf_example = tf.placeholder(tf.string, name="tf_example")

    # Parse the tensorflow.Example looking for a feature named "x" with a single
    # floating point value.
    feature_configs = {"x": tf.FixedLenFeature([1], dtype=tf.float32)}
    tf_example = tf.parse_example(serialized_tf_example, feature_configs)
    # Use tf.identity() to assign name
    x = tf.identity(tf_example["x"], name="x")
    y = tf.add(tf.mul(a, x), b, name="y")

    x2 = tf.placeholder(tf.float32, name="x2")
    tf.add(tf.mul(a, x2), c, name="y2")

    # Create an assets file that can be saved and restored as part of the
    # SavedModel.
    original_assets_directory = "/tmp/original/export/assets"
    original_assets_filename = "foo.txt"
    original_assets_filepath = _write_assets(original_assets_directory,
                                             original_assets_filename)

    # Set up the assets collection.
    assets_filepath = tf.constant(original_assets_filepath)
    tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, assets_filepath)
    filename_tensor = tf.Variable(
        original_assets_filename,
        name="filename_tensor",
        trainable=False,
        collections=[])
    assign_filename_op = filename_tensor.assign(original_assets_filename)

    # Set up the signature for regression with input and output tensor
    # specification.
    input_tensor = meta_graph_pb2.TensorInfo()
    input_tensor.name = serialized_tf_example.name
    signature_inputs = {signature_constants.REGRESS_INPUTS: input_tensor}

    output_tensor = meta_graph_pb2.TensorInfo()
    output_tensor.name = tf.identity(y).name
    signature_outputs = {signature_constants.REGRESS_OUTPUTS: output_tensor}
    signature_def = signature_def_utils.build_signature_def(
        signature_inputs, signature_outputs,
        signature_constants.REGRESS_METHOD_NAME)

    # Set up the signature for Predict with input and output tensor
    # specification.
    predict_input_tensor = meta_graph_pb2.TensorInfo()
    predict_input_tensor.name = x.name
    predict_signature_inputs = {
        "x": predict_input_tensor
    }

    predict_output_tensor = meta_graph_pb2.TensorInfo()
    predict_output_tensor.name = y.name
    predict_signature_outputs = {
        "y": predict_output_tensor
    }
    predict_signature_def = signature_def_utils.build_signature_def(
        predict_signature_inputs, predict_signature_outputs,
        signature_constants.PREDICT_METHOD_NAME)

    # Initialize all variables and then save the SavedModel.
    sess.run(tf.global_variables_initializer())
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map={
            signature_constants.REGRESS_METHOD_NAME:
                signature_def,
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                predict_signature_def
        },
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
        legacy_init_op=tf.group(assign_filename_op))
    builder.save(as_text)


def main(_):
  _generate_saved_model_for_half_plus_two(FLAGS.output_dir)
  print("SavedModel generated at: %s" % FLAGS.output_dir)

  _generate_saved_model_for_half_plus_two(FLAGS.output_dir_pbtxt, as_text=True)
  print("SavedModel generated at: %s" % FLAGS.output_dir_pbtxt)


if __name__ == "__main__":
  tf.app.run()
