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
"""Constants used in Trackable for checkpointing and serialization."""

import enum


# Key where the object graph proto is saved in a TensorBundle
OBJECT_GRAPH_PROTO_KEY = "_CHECKPOINTABLE_OBJECT_GRAPH"

# A key indicating a variable's value in an object's checkpointed Tensors
# (Trackable._gather_saveables_for_checkpoint). If this is the only key and
# the object has no dependencies, then its value may be restored on object
# creation (avoiding double assignment when executing eagerly).
VARIABLE_VALUE_KEY = "VARIABLE_VALUE"
OBJECT_CONFIG_JSON_KEY = "OBJECT_CONFIG_JSON"


@enum.unique
class SaveType(str, enum.Enum):
  SAVEDMODEL = "savedmodel"
  CHECKPOINT = "checkpoint"
