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
"""Controller Class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict


class Controller(object):
  """Controller class."""

  def __init__(self, item, cluster):
    """Controller class initializer.

    Args:
      item: The metagraph to place wrapped in a cluster.
      cluster: A cluster of devices on which to place the item.
    """
    self.item = item

    self._node = {}
    for node in item.metagraph.graph_def.node:
      self._node[node.name] = node

    self._fanout = defaultdict(lambda: [])
    for node in item.metagraph.graph_def.node:
      for fanin in self._get_node_fanin(node):
        self._fanout[fanin.name].append(node)

    important_op_names = item.IdentifyImportantOps(sort_topologically=True)

    # List of important ops (these are the ops to place) sorted in topological
    # order. The order of this collection is deterministic.
    self.important_ops = []
    for name in important_op_names:
      self.important_ops.append(self._node[name])

    self.node_properties = item.GetOpProperties()

    self.cluster = cluster
    self.devices = cluster.ListDevices()

    self.colocation_constraints = item.GetColocationGroups()

    self.placement_constraints = cluster.GetSupportedDevices(item)
    for node_name, dev in self.placement_constraints.items():
      if len(dev) == 1:
        # Place the node on the supported device
        node = self._node[node_name]
        node.device = dev[0]
        fanout = self.get_node_fanout(node)
        # Update the fanout of the fanin to bypass the node
        for fanin in self._get_node_fanin(node):
          fanout_of_fanin = self.get_node_fanout(fanin)
          fanout_of_fanin += fanout
          fanout_of_fanin.remove(node)
        # Remove node from the list of important ops since we don't need to
        # place the node.
        if node in self.important_ops:
          self.important_ops.remove(node)
          important_op_names.remove(node.name)

    # List of important op names, in non deterministic order.
    self.important_op_names = frozenset(important_op_names)

  @property
  def input_graph_def(self):
    return self.item.metagraph.graph_def

  @property
  def num_devices(self):
    return len(self.devices)

  def get_node_by_name(self, node_name):
    return self._node[node_name]

  def get_node_fanout(self, node):
    return self._fanout[node.name]

  def get_placements(self, *args, **kwargs):
    """Returns: Two TF ops.

    Args:
      *args: "".
      **kwargs: "".

    Returns:
      y_preds: tensor of size [batch_size, num_ops]
      log_probs: python dict of at least two fields: "sample", "target" each
      containing a tensor of size [batch_size], corresponding to the log_probs.
    """
    raise NotImplementedError

  def eval_placement(self, sess, *args, **kwargs):
    """At this time, this method evaluates ONLY ONE placement.

    Args:
      sess: a tf.compat.v1.Session() object used to retrieve cached assignment
        info.
      *args: "".
      **kwargs: "".

    Returns:
      run_time: scalar
    """
    raise NotImplementedError

  def export_placement(self, metagraph):
    """Annotate the placement onto the specified metagraph.

    Args:
      metagraph: the metagraph to annotate with the placement.
    """
    for node in metagraph.graph_def.node:
      if node.name in self.important_op_names:
        node.device = self.get_node_by_name(node.name).device

  # Get the nodes in the immediate fanin of node.
  # Beware: this doesn't take into account the nodes that may be skipped
  # since placement constraints force their placement.
  def _get_node_fanin(self, node):
    input_ops = []
    for fanin_name in node.input:
      if fanin_name[0] == "^":
        fanin_name = fanin_name[1:]
      fanin_name = fanin_name.split(":")[0]
      input_ops.append(self.get_node_by_name(fanin_name))
    return input_ops
