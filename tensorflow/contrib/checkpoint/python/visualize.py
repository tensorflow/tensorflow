"""Utilities for visualizing dependency graphs."""
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.training import checkpointable
from tensorflow.python.training import checkpointable_utils


def dot_graph_from_checkpoint(save_path):
  r"""Visualizes an object-based checkpoint (from `tf.train.Checkpoint`).

  Useful for inspecting checkpoints and debugging loading issues.

  Example usage from Python (requires pydot):
  ```python
  import tensorflow as tf
  import pydot

  dot_string = tf.contrib.checkpoint.dot_graph_from_checkpoint('/path/to/ckpt')
  parsed, = pydot.graph_from_dot_data(dot_string)
  parsed.write_svg('/tmp/tensorflow/visualized_checkpoint.svg')
  ```

  Example command line usage:
  ```sh
  python -c "import tensorflow as tf;\
    print(tf.contrib.checkpoint.dot_graph_from_checkpoint('/path/to/ckpt'))"\
    | dot -Tsvg > /tmp/tensorflow/checkpoint_viz.svg
  ```

  Args:
    save_path: The checkpoint prefix, as returned by `tf.train.Checkpoint.save`
      or `tf.train.latest_checkpoint`.
  Returns:
    A graph in DOT format as a string.
  """
  reader = pywrap_tensorflow.NewCheckpointReader(save_path)
  object_graph = checkpointable_utils.object_metadata(save_path)
  shape_map = reader.get_variable_to_shape_map()
  dtype_map = reader.get_variable_to_dtype_map()
  graph = 'digraph {\n'
  def _escape(name):
    return name.replace('"', '\\"')
  slot_ids = set()
  for node in object_graph.nodes:
    for slot_reference in node.slot_variables:
      slot_ids.add(slot_reference.slot_variable_node_id)
  for node_id, node in enumerate(object_graph.nodes):
    if (len(node.attributes) == 1
        and node.attributes[0].name == checkpointable.VARIABLE_VALUE_KEY):
      if node_id in slot_ids:
        color = 'orange'
        tooltip_prefix = 'Slot variable'
      else:
        color = 'blue'
        tooltip_prefix = 'Variable'
      attribute = node.attributes[0]
      graph += ('N_%d [shape=point label="" color=%s width=.25'
                ' tooltip="%s %s shape=%s %s"]\n') % (
                    node_id,
                    color,
                    tooltip_prefix,
                    _escape(attribute.full_name),
                    shape_map[attribute.checkpoint_key],
                    dtype_map[attribute.checkpoint_key].name)
    elif node.slot_variables:
      graph += ('N_%d [shape=point label="" width=.25 color=red,'
                'tooltip="Optimizer"]\n') % node_id
    else:
      graph += 'N_%d [shape=point label="" width=.25]\n' % node_id
    for reference in node.children:
      graph += 'N_%d -> N_%d [label="%s"]\n' % (
          node_id, reference.node_id, _escape(reference.local_name))
    for slot_reference in node.slot_variables:
      graph += 'N_%d -> N_%d [label="%s" style=dotted]\n' % (
          node_id,
          slot_reference.slot_variable_node_id,
          _escape(slot_reference.slot_name))
      graph += 'N_%d -> N_%d [style=dotted]\n' % (
          slot_reference.original_variable_node_id,
          slot_reference.slot_variable_node_id)
  graph += '}\n'
  return graph
