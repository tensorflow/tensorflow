# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Updates generated docs from Python doc comments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys

import tensorflow as tf

from tensorflow.python.client import client_lib
from tensorflow.python.framework import docs
from tensorflow.python.framework import framework_lib


tf.flags.DEFINE_string("out_dir", None,
                       "Directory to which docs should be written.")
tf.flags.DEFINE_boolean("print_hidden_regex", False,
                        "Dump a regular expression matching any hidden symbol")
FLAGS = tf.flags.FLAGS


PREFIX_TEXT = """
Note: Functions taking `Tensor` arguments can also take anything accepted by
[`tf.convert_to_tensor`](framework.md#convert_to_tensor).
"""


def get_module_to_name():
  return {
      tf: "tf",
      tf.errors: "tf.errors",
      tf.image: "tf.image",
      tf.nn: "tf.nn",
      tf.train: "tf.train",
      tf.python_io: "tf.python_io",
      tf.test: "tf.test",
      tf.contrib.distributions: "tf.contrib.distributions",
      tf.contrib.layers: "tf.contrib.layers",
      tf.contrib.learn: "tf.contrib.learn",
      tf.contrib.metrics: "tf.contrib.metrics",
      tf.contrib.util: "tf.contrib.util",
      tf.contrib.copy_graph: "tf.contrib.copy_graph",
  }


def all_libraries(module_to_name, members, documented):
  """Make a list of the individual files that we want to create.

  Args:
    module_to_name: Dictionary mapping modules to short names.
    members: Dictionary mapping member name to (fullname, member).
    documented: Set of documented names to update.

  Returns:
    List of (filename, docs.Library) pairs.
  """
  def library(name, title, module=None, **args):
    if module is None:
      module = sys.modules["tensorflow.python.ops" +
                           ("" if name == "ops" else "." + name)]
    return (name + ".md", docs.Library(title=title,
                                       module_to_name=module_to_name,
                                       members=members,
                                       documented=documented,
                                       module=module,
                                       **args))
  return [
      # Splits of module 'tf'.
      library("framework", "Building Graphs", framework_lib),
      library("check_ops", "Asserts and boolean checks."),
      library("constant_op", "Constants, Sequences, and Random Values",
              prefix=PREFIX_TEXT),
      library("state_ops", "Variables",
              exclude_symbols=["create_partitioned_variables"],
              prefix=PREFIX_TEXT),
      library("array_ops", "Tensor Transformations",
              exclude_symbols=["list_diff"], prefix=PREFIX_TEXT),
      library("math_ops", "Math",
              exclude_symbols=["sparse_matmul", "arg_min", "arg_max",
                               "lin_space", "sparse_segment_mean_grad"],
              prefix=PREFIX_TEXT),
      library("string_ops", "Strings", prefix=PREFIX_TEXT),
      library("histogram_ops", "Histograms"),
      library("control_flow_ops", "Control Flow", prefix=PREFIX_TEXT),
      library("functional_ops", "Higher Order Functions", prefix=PREFIX_TEXT),
      library("session_ops", "Tensor Handle Operations", prefix=PREFIX_TEXT),
      library("image", "Images", tf.image, exclude_symbols=["ResizeMethod"],
              prefix=PREFIX_TEXT),
      library("sparse_ops", "Sparse Tensors",
              exclude_symbols=["serialize_sparse", "serialize_many_sparse",
                               "deserialize_many_sparse"],
              prefix=PREFIX_TEXT),
      library("io_ops", "Inputs and Readers",
              exclude_symbols=["LookupTableBase", "HashTable",
                               "PaddingFIFOQueue",
                               "initialize_all_tables",
                               "parse_single_sequence_example",
                               "string_to_hash_bucket"],
              prefix=PREFIX_TEXT),
      library("python_io", "Data IO (Python functions)", tf.python_io),
      library("nn", "Neural Network", tf.nn,
              exclude_symbols=["conv2d_backprop_input",
                               "conv2d_backprop_filter", "avg_pool_grad",
                               "max_pool_grad", "max_pool_grad_with_argmax",
                               "batch_norm_with_global_normalization_grad",
                               "lrn_grad", "relu6_grad", "softplus_grad",
                               "softsign_grad", "xw_plus_b", "relu_layer",
                               "lrn", "batch_norm_with_global_normalization",
                               "batch_norm_with_global_normalization_grad",
                               "all_candidate_sampler",
                               "rnn", "state_saving_rnn", "bidirectional_rnn",
                               "dynamic_rnn", "seq2seq", "rnn_cell"],
              prefix=PREFIX_TEXT),
      library("client", "Running Graphs", client_lib),
      library("train", "Training", tf.train,
              exclude_symbols=["Feature", "Features", "BytesList", "FloatList",
                               "Int64List", "Example", "InferenceExample",
                               "FeatureList", "FeatureLists",
                               "RankingExample", "SequenceExample"]),
      library("script_ops", "Wraps python functions", prefix=PREFIX_TEXT),
      library("test", "Testing", tf.test),
      library("contrib.distributions", "Statistical distributions (contrib)",
              tf.contrib.distributions),
      library("contrib.layers", "Layers (contrib)", tf.contrib.layers),
      library("contrib.learn", "Learn (contrib)", tf.contrib.learn),
      library("contrib.metrics", "Metrics (contrib)", tf.contrib.metrics),
      library("contrib.util", "Utilities (contrib)", tf.contrib.util),
      library("contrib.copy_graph", "Copying Graph Elements (contrib)",
              tf.contrib.copy_graph),
  ]

_hidden_symbols = ["Event", "LogMessage", "Summary", "SessionLog", "xrange",
                   "HistogramProto", "ConfigProto", "NodeDef", "GraphDef",
                   "GPUOptions", "GraphOptions", "RunOptions", "RunMetadata",
                   "SessionInterface", "BaseSession", "NameAttrList",
                   "AttrValue", "TensorArray", "OptimizerOptions",
                   "CollectionDef", "MetaGraphDef", "QueueRunnerDef",
                   "SaverDef", "VariableDef", "TestCase", "GrpcServer",
                   "ClusterDef", "JobDef", "ServerDef"]


def main(unused_argv):
  if not FLAGS.out_dir:
    tf.logging.error("out_dir not specified")
    return -1

  # Document libraries
  documented = set()
  module_to_name = get_module_to_name()
  members = docs.collect_members(module_to_name)
  libraries = all_libraries(module_to_name, members, documented)

  # Define catch_all library before calling write_libraries to avoid complaining
  # about generically hidden symbols.
  catch_all = docs.Library(title="Catch All", module=None,
                           exclude_symbols=_hidden_symbols,
                           module_to_name=module_to_name, members=members,
                           documented=documented)

  # Write docs to files
  docs.write_libraries(FLAGS.out_dir, libraries)

  # Make it easy to search for hidden symbols
  if FLAGS.print_hidden_regex:
    hidden = set(_hidden_symbols)
    for _, lib in libraries:
      hidden.update(lib.exclude_symbols)
    print(r"hidden symbols regex = r'\b(%s)\b'" % "|".join(sorted(hidden)))

  # Verify that all symbols are mentioned in some library doc.
  catch_all.assert_no_leftovers()

  # Generate index
  with open(os.path.join(FLAGS.out_dir, "index.md"), "w") as f:
    docs.Index(module_to_name, members, libraries,
               "../../api_docs/python/").write_markdown_to_file(f)


if __name__ == "__main__":
  tf.app.run()
