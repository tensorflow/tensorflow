"""Utilities for saving/loading Checkpointable objects."""
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
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import weakref

from tensorflow.python.framework import ops
from tensorflow.python.training import checkpointable
from tensorflow.python.training import saver as saver_lib


class _Checkpoint(object):
  """Holds the status of an object-based checkpoint load."""

  def __init__(self, object_graph_proto, save_path, dtype_map=None):
    """Specify the checkpoint being loaded.

    Args:
      object_graph_proto: The CheckpointableObjectGraph protocol buffer
        associated with this checkpoint.
      save_path: A string `Tensor`. The path to the checkpoint, as returned by
        `tf.train.latest_checkpoint`.
      dtype_map: When executing eagerly, specifies dtypes for creating slot
        variables. None when graph building.
    """
    self.builder = saver_lib.BulkSaverBuilder()
    self.object_graph_proto = object_graph_proto
    self.restore_uid = ops.uid()
    # Maps from objects to lists of attributes which were in the checkpoint but
    # not loaded into any object, for error checking.
    self.unused_attributes = weakref.WeakKeyDictionary()
    # Dictionary mapping from an id in the protocol buffer flat array to
    # Checkpointable Python objects. This mapping may be deferred if a
    # checkpoint is restored before all dependencies have been tracked. Uses
    # weak references so that partial restorations don't create reference cycles
    # (as objects with deferred dependencies will generally have references to
    # this object).
    self.object_by_proto_id = weakref.WeakValueDictionary()
    self.save_path = save_path
    self.dtype_map = dtype_map
    # When graph building, contains a list of ops to run to restore objects from
    # this checkpoint.
    self.restore_ops = []
    self.restore_ops_by_name = {}
    # A mapping from optimizer proto ids to lists of slot variables to be
    # restored when the optimizer is tracked. Only includes slot variables whose
    # regular variables have already been created, and only for optimizer
    # objects which have not yet been created/tracked.
    self.deferred_slot_restorations = {}
    # A mapping from variable proto ids to lists of slot variables to be
    # restored when the variable is created/tracked. These get shifted over to
    # deferred_slot_restorations if the optimizer hasn't been created when that
    # happens.
    self.slot_restorations = {}
    for node_index, node in enumerate(self.object_graph_proto.nodes):
      for slot_reference in node.slot_variables:
        # `node` refers to an `Optimizer`, since only these have slot variables.
        self.slot_restorations.setdefault(
            slot_reference.original_variable_node_id, []).append(
                checkpointable._SlotVariableRestoration(  # pylint: disable=protected-access
                    optimizer_id=node_index,
                    slot_variable_id=slot_reference.slot_variable_node_id,
                    slot_name=slot_reference.slot_name))
