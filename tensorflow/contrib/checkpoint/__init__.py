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

Checkpointable data structures:
@@List
@@Mapping
@@UniqueNameTracker
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.checkpoint.python.containers import UniqueNameTracker
from tensorflow.contrib.checkpoint.python.split_dependency import split_dependency
from tensorflow.contrib.checkpoint.python.visualize import dot_graph_from_checkpoint
from tensorflow.core.protobuf.checkpointable_object_graph_pb2 import CheckpointableObjectGraph
from tensorflow.python.training.checkpointable.base import Checkpointable
from tensorflow.python.training.checkpointable.base import CheckpointableBase
from tensorflow.python.training.checkpointable.base import NoDependency
from tensorflow.python.training.checkpointable.data_structures import List
from tensorflow.python.training.checkpointable.data_structures import Mapping
from tensorflow.python.training.checkpointable.util import capture_dependencies
from tensorflow.python.training.checkpointable.util import list_objects
from tensorflow.python.training.checkpointable.util import object_metadata

from tensorflow.python.util.all_util import remove_undocumented

remove_undocumented(module_name=__name__)
