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
"""Tools for working with object-based checkpoints.

Visualization and inspection:
@@dot_graph_from_checkpoint
@@list_objects
@@object_metadata

Managing dependencies:
@@capture_dependencies
@@Checkpointable
@@CheckpointableBase
@@CheckpointableObjectGraph
@@NoDependency
@@split_dependency

Trackable data structures:
@@List
@@Mapping
@@UniqueNameTracker

Checkpoint management:
@@CheckpointManager

Saving and restoring Python state:
@@NumpyState
@@PythonStateWrapper
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.checkpoint.python.containers import UniqueNameTracker
from tensorflow.contrib.checkpoint.python.python_state import NumpyState
from tensorflow.contrib.checkpoint.python.split_dependency import split_dependency
from tensorflow.contrib.checkpoint.python.visualize import dot_graph_from_checkpoint
from tensorflow.core.protobuf.trackable_object_graph_pb2 import TrackableObjectGraph as CheckpointableObjectGraph
from tensorflow.python.training.checkpoint_management import CheckpointManager
from tensorflow.python.training.tracking.base import Trackable as CheckpointableBase
from tensorflow.python.training.tracking.data_structures import List
from tensorflow.python.training.tracking.data_structures import Mapping
from tensorflow.python.training.tracking.data_structures import NoDependency
from tensorflow.python.training.tracking.python_state import PythonState as PythonStateWrapper
from tensorflow.python.training.tracking.tracking import AutoTrackable as Checkpointable
from tensorflow.python.training.tracking.util import capture_dependencies
from tensorflow.python.training.tracking.util import list_objects
from tensorflow.python.training.tracking.util import object_metadata
from tensorflow.python.util.all_util import remove_undocumented

remove_undocumented(module_name=__name__)

