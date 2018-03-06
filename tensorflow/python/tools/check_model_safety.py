# Copyright 2018 Blade Team of Tencent. All Rights Reserved.
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

r"""Check whether a TensorFlow model contains security sensitive ops, Output related info.

This script is designed to take a checkpoint or SavedModel model file,
and detect whether it contains security ops, and output the check result.

Traditionally, a TensorFlow model is considered as data files, this
leads people to ignore the fact that the TensorFlow model/graph
is actual program/code which contains kinds of ops and runs under the
TensorFlow runtime. Therefore, it is possibile to use legitimate ops
to do some malicious things in a model file. This script just checks
some predefined sensitive ops, do not rely on its result completely.

An example of command-line usage is:
bazel build tensorflow/python/tools:check_model_safety && \
bazel-bin/tensorflow/python/tools/check_model_safety \
--input_checkpoint=model.ckpt-8361242 \
--verbose

"""
import argparse
import sys

from google.protobuf import text_format

from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.training import saver as saver_lib

FLAGS = None

# now we only have 2 sensitive ops
global_sensitive_ops = {
    "WriteFile": "Operation to write contents to a file,\n \
              maybe used to write malicious contents to system sensitive files.",
    "ReadFile": "Operation to read a file,\n \
              maybe used to read system sensitive files."
}

def _parse_input_meta_graph_proto(input_graph, input_binary):
  """Parser input tensorflow graph into MetaGraphDef proto."""
  if not gfile.Exists(input_graph):
    print("Input meta graph file '" + input_graph + "' does not exist!")
    return -1
  input_meta_graph_def = MetaGraphDef()
  mode = "rb" if input_binary else "r"
  with gfile.FastGFile(input_graph, mode) as f:
    if input_binary:
      input_meta_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), input_meta_graph_def)
  #print("Loaded meta graph file '" + input_graph)
  return input_meta_graph_def

def print_detect_result(ops):
  """output the sensitive ops info in graph."""
  if not ops:
    print("\nThe model does not contain security sensitive ops. Good Luck!")
    return
  print("\nThe model contains %d security sensitive ops.\
Please check carefully!\n" % len(ops))
  for node in ops:
    if FLAGS.verbose:
      print("Sensitive op: %s\nSecurity tips: %s\nNode info:\n%s" %\
            (node.op, global_sensitive_ops[node.op], node))
    else:
      print("Sensitive op: %s\nSecurity tips: %s\n" %\
            (node.op, global_sensitive_ops[node.op]))

def is_sensitive_op(node):
  """if an op is a sensitive op."""
  return node.op in global_sensitive_ops

def detect_ops(input_graph_def=None):
  """detect security sensitive ops in a graph."""
  if not input_graph_def:
    print("Invalid input graph")
    return -1
  sensitive_ops = []
  for node in input_graph_def.node:
    if is_sensitive_op(node):
      sensitive_ops.append(node)
  print_detect_result(sensitive_ops)
  return 0

def scan_graph(input_checkpoint=None,
               input_saved_model_dir=None,
               saved_model_tags=tag_constants.SERVING):
  """extract the graph to scan from a model file."""

  if (not input_saved_model_dir and not input_checkpoint):
    print("Please specify a checkpoint or \'SavedModel\' file!")
    return -1
  if (input_saved_model_dir and input_checkpoint):
    print("Please specify only *One* model file type: \
checkpoint or \'SavedModel\'!")
    return -1

  input_graph_def = None
  if input_checkpoint:
    # now we doesn't use the variables file, but still check it for completeness
    if not saver_lib.checkpoint_exists(input_checkpoint):
      print("Input checkpoint '" + input_checkpoint + "' doesn't exist!")
      return -1
    # Build meta file path for a checkpoint
    meta_file = input_checkpoint + ".meta"
    if not gfile.Exists(meta_file):
      print("Input checkpoint meta file '" + meta_file + "' doesn't exist!")
      return -1
    try:
      input_graph_def = _parse_input_meta_graph_proto(meta_file, True).graph_def
    except:
      exctype, value = sys.exc_info()[:2]
      print("Parse checkpoint meta-graph file '%s' failed: %s(%s)" %\
            (meta_file, exctype, value))
      return -1
  if input_saved_model_dir:
    try:
      input_graph_def = saved_model_utils.get_meta_graph_def(
          input_saved_model_dir, saved_model_tags).graph_def
    except:
      exctype, value = sys.exc_info()[:2]
      print("Parse SaveModel '%s' meta-graph file failed: %s(%s)" %\
            (input_saved_model_dir, exctype, value))
      return -1

  return detect_ops(input_graph_def)

def main(unused_args):
  scan_graph(FLAGS.input_checkpoint,\
             FLAGS.input_saved_model_dir,\
             FLAGS.saved_model_tags)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=\
           'Check whether a TensorFlow model contains security sensitive ops.')
  parser.add_argument(
      "-c", "--input_checkpoint",
      type=str,
      default="",
      help="Prefix of TensorFlow checkpoint variables and meta file.")
  parser.add_argument(
      "-s", "--input_saved_model_dir",
      type=str,
      default="",
      help="Path to the dir with TensorFlow \'SavedModel\' file and variables.")
  parser.add_argument(
      "-t", "--saved_model_tags",
      type=str,
      default="serve",
      help="""\
      Group of tag(s) of the MetaGraphDef to load, in string format,\
      separated by \',\'. For tag-set contains multiple tags, all tags \
      must be passed in.\
      """)
  parser.add_argument(
      "-v", "--verbose",
      action="store_true",
      help="Print verbose info about sensitive op.")
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
