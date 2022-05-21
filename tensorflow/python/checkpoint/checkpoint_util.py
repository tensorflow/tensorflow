# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for checkpointing with Trackables.

Note: This module is currently imported from the Trackable base class.
See `util.py` the other checkpointing utils.
"""

import collections


_DeferredSlotVariableRestoration = collections.namedtuple(
    "_DeferredSlotVariableRestoration", [
        "original_variable",
        "slot_variable_id",
        "slot_name",
    ])


def queue_slot_variables(checkpoint_position, visit_queue):
  """Queues slot variables for restoration."""
  trackable = checkpoint_position.trackable
  checkpoint = checkpoint_position.checkpoint
  for deferred_slot_restoration in (
      checkpoint.deferred_slot_restorations.pop(checkpoint_position.proto_id,
                                                ())):
    slot_variable_position, slot_variable = (
        checkpoint_position.create_slot_variable_position(
            trackable, deferred_slot_restoration.original_variable,
            deferred_slot_restoration.slot_variable_id,
            deferred_slot_restoration.slot_name))
    if slot_variable_position is not None:
      visit_queue.append((slot_variable_position, slot_variable))
  for slot_restoration in checkpoint.slot_restorations.pop(
      checkpoint_position.proto_id, ()):
    optimizer_object = checkpoint.object_by_proto_id.get(
        slot_restoration.optimizer_id, None)
    if optimizer_object is None:
      # The optimizer has not yet been created or tracked. Record in the
      # checkpoint that the slot variables need to be restored when it is.
      checkpoint.deferred_slot_restorations.setdefault(
          slot_restoration.optimizer_id, []).append(
              _DeferredSlotVariableRestoration(
                  original_variable=trackable,
                  slot_variable_id=slot_restoration.slot_variable_id,
                  slot_name=slot_restoration.slot_name))

    # `optimizer_object` can be a `Checkpoint` when user only needs the
    # attributes the optimizer holds, such as `iterations`. In those cases,
    # it would not have the optimizer's `_create_or_restore_slot_variable`
    # method.
    elif hasattr(optimizer_object, "_create_or_restore_slot_variable"):
      slot_variable_position, slot_variable = (
          checkpoint_position.create_slot_variable_position(
              optimizer_object, trackable, slot_restoration.slot_variable_id,
              slot_restoration.slot_name))
      if slot_variable_position is not None:
        visit_queue.append((slot_variable_position, slot_variable))

