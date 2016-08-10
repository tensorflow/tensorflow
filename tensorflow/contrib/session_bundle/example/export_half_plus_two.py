# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Exports a toy linear regression inference graph.

Exports a TensorFlow graph to /tmp/half_plus_two/ based on the Exporter
format, go/tf-exporter.

This graph calculates,
  y = a*x + b
where a and b are variables with a=0.5 and b=2.

Output from this program is typically used to exercise Session
loading and execution code.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter


def Export():
  export_path = "/tmp/half_plus_two"
  with tf.Session() as sess:
    # Make model parameters a&b variables instead of constants to
    # exercise the variable reloading mechanisms.
    a = tf.Variable(0.5, name="a")
    b = tf.Variable(2.0, name="b")

    # Calculate, y = a*x + b
    # here we use a placeholder 'x' which is fed at inference time.
    x = tf.placeholder(tf.float32, name="x")
    y = tf.add(tf.mul(a, x), b, name="y")

    # Setup a standard Saver for our variables.
    save = tf.train.Saver({"a": a, "b": b}, sharded=True)

    # asset_path contains the base directory of assets used in training (e.g.
    # vocabulary files).
    original_asset_path = tf.constant("/tmp/original/export/assets")
    # Ops reading asset files should reference the asset_path tensor
    # which stores the original asset path at training time and the
    # overridden assets directory at restore time.
    asset_path = tf.Variable(original_asset_path,
                             name="asset_path",
                             trainable=False,
                             collections=[])
    assign_asset_path = asset_path.assign(original_asset_path)

    # Use a fixed global step number.
    global_step_tensor = tf.Variable(123, name="global_step")

    # Create a RegressionSignature for our input and output.
    signature = exporter.regression_signature(input_tensor=x, output_tensor=y)

    # Create two filename assets and corresponding tensors.
    # TODO(b/26254158) Consider adding validation of file existance as well as
    # hashes (e.g. sha1) for consistency.
    original_filename1 = tf.constant("hello1.txt")
    tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, original_filename1)
    filename1 = tf.Variable(original_filename1,
                            name="filename1",
                            trainable=False,
                            collections=[])
    assign_filename1 = filename1.assign(original_filename1)
    original_filename2 = tf.constant("hello2.txt")
    tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, original_filename2)
    filename2 = tf.Variable(original_filename2,
                            name="filename2",
                            trainable=False,
                            collections=[])
    assign_filename2 = filename2.assign(original_filename2)

    # Init op contains a group of all variables that we assign.
    init_op = tf.group(assign_asset_path, assign_filename1, assign_filename2)

    # CopyAssets is used as a callback during export to copy files to the
    # given export directory.
    def CopyAssets(filepaths, export_path):
      print("copying asset files to: %s" % export_path)
      for filepath in filepaths:
        print("copying asset file: %s" % filepath)

    # Run an export.
    tf.initialize_all_variables().run()
    export = exporter.Exporter(save)
    export.init(
        sess.graph.as_graph_def(),
        init_op=init_op,
        default_graph_signature=signature,
        assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
        assets_callback=CopyAssets)
    export.export(export_path, global_step_tensor, sess)


def main(_):
  Export()


if __name__ == "__main__":
  tf.app.run()
