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
"""Prototype of a distributed computation library for TF."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import,wildcard-import
from tensorflow.contrib.distribute.python.collective_all_reduce_strategy import CollectiveAllReduceStrategy
from tensorflow.contrib.distribute.python.cross_tower_ops import *
from tensorflow.contrib.distribute.python.mirrored_strategy import MirroredStrategy
from tensorflow.contrib.distribute.python.multi_worker_strategy import MultiWorkerMirroredStrategy
from tensorflow.contrib.distribute.python.monitor import Monitor
from tensorflow.contrib.distribute.python.one_device_strategy import OneDeviceStrategy
from tensorflow.contrib.distribute.python.parameter_server_strategy import ParameterServerStrategy
from tensorflow.contrib.distribute.python.step_fn import *
from tensorflow.contrib.distribute.python.tpu_strategy import TPUStrategy
from tensorflow.python.training.distribute import *

from tensorflow.python.util.all_util import remove_undocumented


_allowed_symbols = [
    'AllReduceCrossTowerOps',
    'CollectiveAllReduceStrategy',
    'CrossTowerOps',
    'DistributionStrategy',
    'MirroredStrategy',
    'MultiWorkerMirroredStrategy',
    'Monitor',
    'OneDeviceStrategy',
    'ParameterServerStrategy',
    'ReductionToOneDeviceCrossTowerOps',
    'Step',
    'StandardInputStep',
    'StandardSingleLossStep',
    'TowerContext',
    'TPUStrategy',
    'get_cross_tower_context',
    'get_distribution_strategy',
    'get_loss_reduction',
    'get_tower_context',
    'has_distribution_strategy',
    'require_tower_context',
]

remove_undocumented(__name__, _allowed_symbols)
