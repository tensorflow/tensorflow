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
# =============================================================================

"""Ops related to Tensor Processing Units.

@@cross_replica_sum
@@infeed_dequeue
@@infeed_dequeue_tuple
@@outfeed_enqueue
@@outfeed_enqueue_tuple

@@initialize_system
@@shutdown_system
@@device_assignment
@@core
@@replicate
@@shard
@@batch_parallel
@@rewrite

@@CrossShardOptimizer

@@InfeedQueue

@@DeviceAssignment
@@Topology

@@while_loop
@@repeat

@@TPUEstimator
@@TPUEstimatorSpec
@@export_estimator_savedmodel
@@RunConfig
@@InputPipelineConfig
@@TPUConfig
@@bfloat16_scope
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=wildcard-import,unused-import
from tensorflow.contrib.tpu.python import profiler
from tensorflow.contrib.tpu.python.ops.tpu_ops import *
from tensorflow.contrib.tpu.python.tpu.bfloat16 import *
from tensorflow.contrib.tpu.python.tpu.device_assignment import *
from tensorflow.contrib.tpu.python.tpu.topology import *
from tensorflow.contrib.tpu.python.tpu.tpu import *
from tensorflow.contrib.tpu.python.tpu.tpu_config import *
from tensorflow.contrib.tpu.python.tpu.tpu_estimator import *
from tensorflow.contrib.tpu.python.tpu.tpu_feed import *
from tensorflow.contrib.tpu.python.tpu.tpu_optimizer import *
from tensorflow.contrib.tpu.python.tpu.training_loop import *
# pylint: enable=wildcard-import,unused-import

from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = ['profiler']

remove_undocumented(__name__, _allowed_symbols)
