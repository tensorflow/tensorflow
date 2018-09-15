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
r"""Removes the auxiliary variables and ops added by the pruning library.

Usage:

bazel build tensorflow/contrib/model_pruning:strip_pruning_vars && \
bazel-bin/tensorflow/contrib/model_pruning/strip_pruning_vars \
--checkpoint_dir=/tmp/model_ckpts \
--output_node_names=softmax \
--output_dir=/tmp \
--filename=pruning_stripped.pb
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from tensorflow.contrib.model_pruning.python import strip_pruning_vars_lib
from tensorflow.python.framework import graph_io
from tensorflow.python.platform import app
from tensorflow.python.platform import tf_logging as logging

FLAGS = None


def strip_pruning_vars(checkpoint_dir, output_node_names, output_dir, filename):
  """Remove pruning-related auxiliary variables and ops from the graph.

  Accepts training checkpoints and produces a GraphDef in which the pruning vars
  and ops have been removed.

  Args:
    checkpoint_dir: Path to the checkpoints.
    output_node_names: The name of the output nodes, comma separated.
    output_dir: Directory where to write the graph.
    filename: Output GraphDef file name.

  Returns:
    None

  Raises:
    ValueError: if output_nodes_names are not provided.
  """
  if not output_node_names:
    raise ValueError(
        'Need to specify atleast 1 output node through output_node_names flag')
  output_node_names = output_node_names.replace(' ', '').split(',')

  initial_graph_def = strip_pruning_vars_lib.graph_def_from_checkpoint(
      checkpoint_dir, output_node_names)

  final_graph_def = strip_pruning_vars_lib.strip_pruning_vars_fn(
      initial_graph_def, output_node_names)
  graph_io.write_graph(final_graph_def, output_dir, filename, as_text=False)
  logging.info('\nFinal graph written to %s', os.path.join(
      output_dir, filename))


def main(unused_args):
  return strip_pruning_vars(FLAGS.checkpoint_dir, FLAGS.output_node_names,
                            FLAGS.output_dir, FLAGS.filename)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--checkpoint_dir', type=str, default='', help='Path to the checkpoints.')
  parser.add_argument(
      '--output_node_names',
      type=str,
      default='',
      help='The name of the output nodes, comma separated.')
  parser.add_argument(
      '--output_dir',
      type=str,
      default='/tmp',
      help='Directory where to write the graph.')
  parser.add_argument(
      '--filename',
      type=str,
      default='pruning_stripped.pb',
      help='Output \'GraphDef\' file name.')

  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
