# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Support for training models.  See the @{$python/train} guide.

@@Optimizer
@@GradientDescentOptimizer
@@AdadeltaOptimizer
@@AdagradOptimizer
@@AdagradDAOptimizer
@@MomentumOptimizer
@@AdamOptimizer
@@FtrlOptimizer
@@ProximalGradientDescentOptimizer
@@ProximalAdagradOptimizer
@@RMSPropOptimizer
@@gradients
@@AggregationMethod
@@stop_gradient
@@hessians
@@clip_by_value
@@clip_by_norm
@@clip_by_average_norm
@@clip_by_global_norm
@@global_norm
@@exponential_decay
@@inverse_time_decay
@@natural_exp_decay
@@piecewise_constant
@@polynomial_decay
@@ExponentialMovingAverage
@@Coordinator
@@QueueRunner
@@LooperThread
@@add_queue_runner
@@start_queue_runners
@@Server
@@Supervisor
@@SessionManager
@@ClusterSpec
@@replica_device_setter
@@MonitoredTrainingSession
@@MonitoredSession
@@SingularMonitoredSession
@@Scaffold
@@SessionCreator
@@ChiefSessionCreator
@@WorkerSessionCreator
@@summary_iterator
@@SessionRunHook
@@SessionRunArgs
@@SessionRunContext
@@SessionRunValues
@@LoggingTensorHook
@@StopAtStepHook
@@CheckpointSaverHook
@@NewCheckpointReader
@@StepCounterHook
@@NanLossDuringTrainingError
@@NanTensorHook
@@SummarySaverHook
@@GlobalStepWaiterHook
@@FinalOpsHook
@@FeedFnHook
@@global_step
@@basic_train_loop
@@get_global_step
@@get_or_create_global_step
@@create_global_step
@@assert_global_step
@@write_graph
"""

# Optimizers.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys as _sys

from tensorflow.python.ops import io_ops as _io_ops
from tensorflow.python.ops import state_ops as _state_ops
from tensorflow.python.util.all_util import remove_undocumented

# pylint: disable=g-bad-import-order,unused-import
from tensorflow.python.training.adadelta import AdadeltaOptimizer
from tensorflow.python.training.adagrad import AdagradOptimizer
from tensorflow.python.training.adagrad_da import AdagradDAOptimizer
from tensorflow.python.training.proximal_adagrad import ProximalAdagradOptimizer
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.momentum import MomentumOptimizer
from tensorflow.python.training.moving_averages import ExponentialMovingAverage
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.python.training.proximal_gradient_descent import ProximalGradientDescentOptimizer
from tensorflow.python.training.sync_replicas_optimizer import SyncReplicasOptimizer

# Utility classes for training.
from tensorflow.python.training.coordinator import Coordinator
from tensorflow.python.training.coordinator import LooperThread
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.training.queue_runner import *

# For the module level doc.
from tensorflow.python.training import input as _input
from tensorflow.python.training.input import *
# pylint: enable=wildcard-import

from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.training.basic_session_run_hooks import LoggingTensorHook
from tensorflow.python.training.basic_session_run_hooks import StopAtStepHook
from tensorflow.python.training.basic_session_run_hooks import CheckpointSaverHook
from tensorflow.python.training.basic_session_run_hooks import StepCounterHook
from tensorflow.python.training.basic_session_run_hooks import NanLossDuringTrainingError
from tensorflow.python.training.basic_session_run_hooks import NanTensorHook
from tensorflow.python.training.basic_session_run_hooks import SummarySaverHook
from tensorflow.python.training.basic_session_run_hooks import GlobalStepWaiterHook
from tensorflow.python.training.basic_session_run_hooks import FinalOpsHook
from tensorflow.python.training.basic_session_run_hooks import FeedFnHook
from tensorflow.python.training.basic_loops import basic_train_loop
from tensorflow.python.training.device_setter import replica_device_setter
from tensorflow.python.training.monitored_session import Scaffold
from tensorflow.python.training.monitored_session import MonitoredTrainingSession
from tensorflow.python.training.monitored_session import SessionCreator
from tensorflow.python.training.monitored_session import ChiefSessionCreator
from tensorflow.python.training.monitored_session import WorkerSessionCreator
from tensorflow.python.training.monitored_session import MonitoredSession
from tensorflow.python.training.monitored_session import SingularMonitoredSession
from tensorflow.python.training.saver import Saver
from tensorflow.python.training.saver import checkpoint_exists
from tensorflow.python.training.saver import generate_checkpoint_state_proto
from tensorflow.python.training.saver import get_checkpoint_mtimes
from tensorflow.python.training.saver import get_checkpoint_state
from tensorflow.python.training.saver import latest_checkpoint
from tensorflow.python.training.saver import update_checkpoint_state
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.training.saver import import_meta_graph
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.session_run_hook import SessionRunContext
from tensorflow.python.training.session_run_hook import SessionRunValues
from tensorflow.python.training.session_manager import SessionManager
from tensorflow.python.training.summary_io import summary_iterator
from tensorflow.python.training.supervisor import Supervisor
from tensorflow.python.training.training_util import write_graph
from tensorflow.python.training.training_util import global_step
from tensorflow.python.training.training_util import get_global_step
from tensorflow.python.training.training_util import assert_global_step
from tensorflow.python.training.training_util import create_global_step
from tensorflow.python.training.training_util import get_or_create_global_step
from tensorflow.python.pywrap_tensorflow import do_quantize_training_on_graphdef
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader

# pylint: disable=wildcard-import
# Training data protos.
from tensorflow.core.example.example_pb2 import *
from tensorflow.core.example.feature_pb2 import *
from tensorflow.core.protobuf.saver_pb2 import *

# Utility op.  Open Source. TODO(touts): move to nn?
from tensorflow.python.training.learning_rate_decay import *
# pylint: enable=wildcard-import

# Distributed computing support.
from tensorflow.core.protobuf.tensorflow_server_pb2 import ClusterDef
from tensorflow.core.protobuf.tensorflow_server_pb2 import JobDef
from tensorflow.core.protobuf.tensorflow_server_pb2 import ServerDef
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.training.server_lib import Server

# Symbols whitelisted for export without documentation.
_allowed_symbols = [
    # TODO(cwhipkey): review these and move to contrib or expose through
    # documentation.
    "generate_checkpoint_state_proto",   # Used internally by saver.
    "checkpoint_exists",  # Only used in test?
    "get_checkpoint_mtimes",  # Only used in test?

    # Legacy: remove.
    "do_quantize_training_on_graphdef",  # At least use grah_def, not graphdef.
                                         # No uses within tensorflow.
    "queue_runner",  # Use tf.train.start_queue_runner etc directly.
                     # This is also imported internally.

    # TODO(drpng): document these. The reference in howtos/distributed does
    # not link.
    "SyncReplicasOptimizer",
    # Protobufs:
    "BytesList",          # from example_pb2.
    "ClusterDef",
    "Example",            # from example_pb2
    "Feature",            # from example_pb2
    "Features",           # from example_pb2
    "FeatureList",        # from example_pb2
    "FeatureLists",       # from example_pb2
    "FloatList",          # from example_pb2.
    "Int64List",          # from example_pb2.
    "JobDef",
    "SaverDef",           # From saver_pb2.
    "SequenceExample",    # from example_pb2.
    "ServerDef",
]
# Include extra modules for docstrings because:
# * Input methods in tf.train are documented in io_ops.
# * Saver methods in tf.train are documented in state_ops.
remove_undocumented(__name__, _allowed_symbols,
                    [_sys.modules[__name__], _io_ops, _state_ops])
