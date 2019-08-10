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
# ================================
"""Imports a protobuf model as a graph in Tensorboard."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.summary import summary

# Try importing TensorRT ops if available
# TODO(aaroey): ideally we should import everything from contrib, but currently
# tensorrt module would cause build errors when being imported in
# tensorflow/contrib/__init__.py. Fix it.
# pylint: disable=unused-import,g-import-not-at-top,wildcard-import
try:
  from tensorflow.contrib.tensorrt.ops.gen_trt_engine_op import *
except ImportError:
  pass
# pylint: enable=unused-import,g-import-not-at-top,wildcard-import

def import_to_tensorboard(graph_def, log_dir):
  """View an imported protobuf model (`.pb` file) as a graph in Tensorboard.
  Args:
    graph_def: a GraphDef to visualize
    log_dir: The location for the Tensorboard log to begin visualization from.
  Usage:
    Call this function with GraphDef and desired log directory.
    Launch Tensorboard by pointing it to the log directory.
    View your imported `.pb` model as a graph.
  """
  with session.Session(graph=ops.Graph()) as sess:
    importer.import_graph_def(graph_def)

    pb_visual_writer = summary.FileWriter(log_dir)
    pb_visual_writer.add_graph(sess.graph)
    print("Model Imported. Visualize by running: "
          "tensorboard --logdir={}".format(log_dir))


def main(unused_args):
  if not FLAGS.input_graph_file and not FLAGS.input_meta_graph_file \
     and not FLAGS.input_saved_model_dir:
    print("Please specify one of input_graph_file, input_meta_graph_file"
          " and input_saved_model_dir")
    return -1

  if FLAGS.input_graph_file:
    with gfile.GFile(FLAGS.input_graph_file, 'rb') as f:
      graph_def = graph_pb2.GraphDef()
      graph_def.ParseFromString(f.read())
      return import_to_tensorboard(graph_def, FLAGS.log_dir)

  if FLAGS.input_meta_graph_file:
    with gfile.GFile(FLAGS.input_meta_graph_file, 'rb') as f:
      input_meta_graph_def = MetaGraphDef()
      input_meta_graph_def.ParseFromString(f.read())
      graph_def = input_meta_graph_def.graph_def
      return import_to_tensorboard(graph_def, FLAGS.log_dir)

  if FLAGS.input_saved_model_dir:
      graph_def = saved_model_utils.get_meta_graph_def(
        FLAGS.input_saved_model_dir, FLAGS.saved_model_tags).graph_def
      return import_to_tensorboard(graph_def, FLAGS.log_dir)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--input_graph_file",
      type=str,
      default="",
      help="Path to the TensorFlow \'GraphDef\' pb file to load.")
  parser.add_argument(
      "--input_meta_graph_file",
      type=str,
      default="",
      help="Path to the TensorFlow \'MetaGraphDef\' pb file to load.")
  parser.add_argument(
      "--input_saved_model_dir",
      type=str,
      default="",
      help="Path to the dir with TensorFlow \'SavedModel\' file and variables.")
  parser.add_argument(
      "--saved_model_tags",
      type=str,
      default="serve",
      help="""\
      Group of tag(s) of the MetaGraphDef to load, in string format,\
      separated by \',\'. For tag-set contains multiple tags, all tags \
      must be passed in. Default to 'serve'.\
      """)
  parser.add_argument(
      "--log_dir",
      type=str,
      default="",
      required=True,
      help="The location for the Tensorboard log to begin visualization from.")
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
