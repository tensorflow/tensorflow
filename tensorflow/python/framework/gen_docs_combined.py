"""Updates generated docs from Python doc comments."""

import os.path

import tensorflow.python.platform
import sys
import tensorflow as tf

from tensorflow.python.framework import docs
from tensorflow.python.framework import framework_lib
from tensorflow.python.client import client_lib


tf.flags.DEFINE_string("out_dir", None,
                       "Directory to which docs should be written.")
tf.flags.DEFINE_boolean("print_hidden_regex", False,
                        "Dump a regular expression matching any hidden symbol")
FLAGS = tf.flags.FLAGS


# TODO(josh11b,wicke): Remove the ../../api_docs/python/ once the
# website can handle it.
PREFIX_TEXT = """
Note: Functions taking `Tensor` arguments can also take anything accepted by
[`tf.convert_to_tensor`](../../api_docs/python/framework.md#convert_to_tensor).
"""


def get_module_to_name():
  return {tf: 'tf',
          tf.errors: 'tf.errors',
          tf.image: 'tf.image',
          tf.nn: 'tf.nn',
          tf.train: 'tf.train',
          tf.python_io: 'tf.python_io'}

def all_libraries(module_to_name, members, documented):
  # A list of (filename, docs.Library) pairs representing the individual files
  # that we want to create.
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
      library("constant_op", "Constants, Sequences, and Random Values",
              prefix=PREFIX_TEXT),
      library("state_ops", "Variables", prefix=PREFIX_TEXT),
      library("array_ops", "Tensor Transformations",
              exclude_symbols=["list_diff"], prefix=PREFIX_TEXT),
      library("math_ops", "Math",
              exclude_symbols=["sparse_matmul", "arg_min", "arg_max",
                               "lin_space", "sparse_segment_mean_grad"],
              prefix=PREFIX_TEXT),
      library("control_flow_ops", "Control Flow", prefix=PREFIX_TEXT),
      library("image", "Images", tf.image, exclude_symbols=["ResizeMethod"],
              prefix=PREFIX_TEXT),
      library("sparse_ops", "Sparse Tensors", prefix=PREFIX_TEXT),
      library("io_ops", "Inputs and Readers",
              exclude_symbols=["LookupTableBase", "HashTable",
                               "initialize_all_tables",
                               "string_to_hash_bucket"],
              prefix=PREFIX_TEXT),
      library("python_io", "Data IO (Python functions)", tf.python_io),
      library("nn", "Neural Network", tf.nn,
              exclude_symbols=["deconv2d", "conv2d_backprop_input",
                               "conv2d_backprop_filter", "avg_pool_grad",
                               "max_pool_grad", "max_pool_grad_with_argmax",
                               "batch_norm_with_global_normalization_grad",
                               "lrn_grad", "relu6_grad", "softplus_grad",
                               "xw_plus_b", "relu_layer", "lrn",
                               "batch_norm_with_global_normalization",
                               "batch_norm_with_global_normalization_grad",
                               "all_candidate_sampler",
                               "embedding_lookup_sparse"],
              prefix=PREFIX_TEXT),
      library('client', "Running Graphs", client_lib),
      library("train", "Training", tf.train,
              exclude_symbols=["Feature", "Features", "BytesList", "FloatList",
                               "Int64List", "Example", "InferenceExample",
                               "RankingExample", "SequenceExample"]),
  ]

_hidden_symbols = ["Event", "Summary",
                   "HistogramProto", "ConfigProto", "NodeDef", "GraphDef",
                   "GPUOptions", "SessionInterface", "BaseSession"]

def main(unused_argv):
  if not FLAGS.out_dir:
    tf.logging.error("out_dir not specified")
    return -1

  # Document libraries
  documented = set()
  module_to_name = get_module_to_name()
  members = docs.collect_members(module_to_name)
  libraries = all_libraries(module_to_name, members, documented)
  docs.write_libraries(FLAGS.out_dir, libraries)

  # Make it easy to search for hidden symbols
  if FLAGS.print_hidden_regex:
    hidden = set(_hidden_symbols)
    for _, lib in libraries:
      hidden.update(lib.exclude_symbols)
    print r"hidden symbols regex = r'\b(%s)\b'" % "|".join(sorted(hidden))

  # Verify that all symbols are mentioned in some library doc.
  catch_all = docs.Library(title="Catch All", module=None,
                           exclude_symbols=_hidden_symbols,
                           module_to_name=module_to_name, members=members,
                           documented=documented)
  catch_all.assert_no_leftovers()

  # Generate index
  with open(os.path.join(FLAGS.out_dir, "index.md"), "w") as f:
    docs.Index(module_to_name, members, libraries).write_markdown_to_file(f)


if __name__ == "__main__":
  tf.app.run()
