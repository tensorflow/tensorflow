#  Copyright 2015-present Scikit Flow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Updates generated docs from Python doc comments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys

import tensorflow as tf

import docs

import skflow

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
    skflow: "skflow",
  }

def all_libraries(module_to_name, members, documented):
  # A list of (filename, docs.Library) pairs representing the individual files
  # that we want to create.
  def library(name, title, module=None, **args):
    if module is None:
      module = sys.modules["skflow" +
                           ("" if name == "ops" else "." + name)]
    return (name + ".md", docs.Library(title=title,
                                       module_to_name=module_to_name,
                                       members=members,
                                       documented=documented,
                                       module=module,
                                       **args))
  return [
      # Splits of module 'skflow'.
     library("estimators", "Estimators"),
     library("io", "IO"),
     library("preprocessing", "Preprocessing"),
     library("trainer", "Trainer"),
     library("models", "Models"),
     library("ops", "Tensor Transformations",
              exclude_symbols=["list_diff"], prefix=PREFIX_TEXT),
 ]

_hidden_symbols = ["Event", "LogMessage", "Summary", "SessionLog", "xrange",
                   "HistogramProto", "ConfigProto", "NodeDef", "GraphDef",
                   "GPUOptions", "GraphOptions", "SessionInterface",
                   "BaseSession", "NameAttrList", "AttrValue",
                   "TensorArray", "OptimizerOptions",
                   "CollectionDef", "MetaGraphDef", "QueueRunnerDef",
                   "SaverDef", "VariableDef", "TestCase",
                  ]

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
