# Copyright 2015 Google Inc. All Rights Reserved.
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

# pylint: disable=line-too-long
"""This library provides a set of classes and functions that helps train models.

## Optimizers

The Optimizer base class provides methods to compute gradients for a loss and
apply gradients to variables.  A collection of subclasses implement classic
optimization algorithms such as GradientDescent and Adagrad.

You never instantiate the Optimizer class itself, but instead instantiate one
of the subclasses.

@@Optimizer

@@GradientDescentOptimizer
@@AdadeltaOptimizer
@@AdagradOptimizer
@@MomentumOptimizer
@@AdamOptimizer
@@FtrlOptimizer
@@RMSPropOptimizer

## Gradient Computation

TensorFlow provides functions to compute the derivatives for a given
TensorFlow computation graph, adding operations to the graph. The
optimizer classes automatically compute derivatives on your graph, but
creators of new Optimizers or expert users can call the lower-level
functions below.

@@gradients
@@AggregationMethod

@@stop_gradient


## Gradient Clipping

TensorFlow provides several operations that you can use to add clipping
functions to your graph. You can use these functions to perform general data
clipping, but they're particularly useful for handling exploding or vanishing
gradients.

@@clip_by_value
@@clip_by_norm
@@clip_by_average_norm
@@clip_by_global_norm
@@global_norm

## Decaying the learning rate
@@exponential_decay

## Moving Averages

Some training algorithms, such as GradientDescent and Momentum often benefit
from maintaining a moving average of variables during optimization.  Using the
moving averages for evaluations often improve results significantly.

@@ExponentialMovingAverage

## Coordinator and QueueRunner

See [Threading and Queues](../../how_tos/threading_and_queues/index.md)
for how to use threads and queues.  For documentation on the Queue API,
see [Queues](../../api_docs/python/io_ops.md#queues).

@@Coordinator
@@QueueRunner
@@add_queue_runner
@@start_queue_runners

## Distributed execution

See [Distributed TensorFlow](../../how_tos/distributed/index.md) for
more information about how to configure a distributed TensorFlow program.

@@Server
@@Supervisor
@@SessionManager
@@ClusterSpec
@@replica_device_setter

## Summary Operations

The following ops output
[`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
protocol buffers as serialized string tensors.

You can fetch the output of a summary op in a session, and pass it to
a [SummaryWriter](../../api_docs/python/train.md#SummaryWriter) to append it
to an event file.  Event files contain
[`Event`](https://www.tensorflow.org/code/tensorflow/core/util/event.proto)
protos that can contain `Summary` protos along with the timestamp and
step.  You can then use TensorBoard to visualize the contents of the
event files.  See [TensorBoard and
Summaries](../../how_tos/summaries_and_tensorboard/index.md) for more
details.

@@scalar_summary
@@image_summary
@@histogram_summary
@@zero_fraction

@@merge_summary
@@merge_all_summaries

## Adding Summaries to Event Files

See [Summaries and
TensorBoard](../../how_tos/summaries_and_tensorboard/index.md) for an
overview of summaries, event files, and visualization in TensorBoard.

@@SummaryWriter
@@summary_iterator

## Training utilities

@@global_step
@@write_graph

"""
# pylint: enable=line-too-long

# Optimizers.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# pylint: disable=g-bad-import-order,unused-import
from tensorflow.python.ops import gradients
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import state_ops

from tensorflow.python.training.adadelta import AdadeltaOptimizer
from tensorflow.python.training.adagrad import AdagradOptimizer
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.momentum import MomentumOptimizer
from tensorflow.python.training.moving_averages import ExponentialMovingAverage
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
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

from tensorflow.python.training.device_setter import replica_device_setter
from tensorflow.python.training.saver import generate_checkpoint_state_proto
from tensorflow.python.training.saver import get_checkpoint_state
from tensorflow.python.training.saver import latest_checkpoint
from tensorflow.python.training.saver import Saver
from tensorflow.python.training.saver import update_checkpoint_state
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.training.saver import import_meta_graph
from tensorflow.python.training.session_manager import SessionManager
from tensorflow.python.training.summary_io import summary_iterator
from tensorflow.python.training.summary_io import SummaryWriter
from tensorflow.python.training.supervisor import Supervisor
from tensorflow.python.training.training_util import write_graph
from tensorflow.python.training.training_util import global_step
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader


# Training data protos.
from tensorflow.core.example.example_pb2 import *
from tensorflow.core.example.feature_pb2 import *
from tensorflow.core.protobuf.saver_pb2 import *

# Utility op.  Open Source. TODO(touts): move to nn?
from tensorflow.python.training.learning_rate_decay import exponential_decay


# Distributed computing support
from tensorflow.core.protobuf.tensorflow_server_pb2 import ClusterDef
from tensorflow.core.protobuf.tensorflow_server_pb2 import JobDef
from tensorflow.core.protobuf.tensorflow_server_pb2 import ServerDef
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.training.server_lib import Server


from tensorflow.python.util.all_util import make_all

# Include extra modules for docstrings because:
# * Input methods in tf.train are documented in io_ops.
# * Saver methods in tf.train are documented in state_ops.
__all__ = make_all(__name__, [sys.modules[__name__], io_ops, state_ops])

# Symbols whitelisted for export without documentation.
# TODO(cwhipkey): review these and move to contrib or expose through
# documentation.
__all__.extend([
    "BytesList",
    "Example",
    "Feature",
    "FeatureList",
    "FeatureLists",
    "Features",
    "FloatList",
    "InferenceExample",
    "Int64List",
    "LooperThread",
    "SaverDef",
    "SequenceExample",
    "export_meta_graph",
    "generate_checkpoint_state_proto",
    "import_meta_graph",
    "queue_runner",
])
