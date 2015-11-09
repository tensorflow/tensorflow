# pylint: disable=wildcard-import,unused-import,g-bad-import-order,line-too-long
"""This library provides a set of classes and functions that helps train models.

## Optimizers

The Optimizer base class provides methods to compute gradients for a loss and
apply gradients to variables.  A collection of subclasses implement classic
optimization algorithms such as GradientDescent and Adagrad.

You never instantiate the Optimizer class itself, but instead instantiate one
of the subclasses.

@@Optimizer

@@GradientDescentOptimizer
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

## Summary Operations

The following ops output
[`Summary`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/summary.proto)
protocol buffers as serialized string tensors.

You can fetch the output of a summary op in a session, and pass it to
a [SummaryWriter](../../api_docs/python/train.md#SummaryWriter) to append it
to an event file.  Event files contain
[`Event`](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/util/event.proto)
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

# Optimizers.
from tensorflow.python.training.adagrad import AdagradOptimizer
from tensorflow.python.training.adam import AdamOptimizer
from tensorflow.python.training.ftrl import FtrlOptimizer
from tensorflow.python.training.momentum import MomentumOptimizer
from tensorflow.python.training.moving_averages import ExponentialMovingAverage
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.training.rmsprop import RMSPropOptimizer
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer

# Utility classes for training.
from tensorflow.python.training.coordinator import Coordinator
from tensorflow.python.training.queue_runner import *

# For the module level doc.
from tensorflow.python.training import input as _input
from tensorflow.python.training.input import *

from tensorflow.python.training.saver import get_checkpoint_state
from tensorflow.python.training.saver import latest_checkpoint
from tensorflow.python.training.saver import Saver
from tensorflow.python.training.saver import update_checkpoint_state
from tensorflow.python.training.summary_io import summary_iterator
from tensorflow.python.training.summary_io import SummaryWriter
from tensorflow.python.training.training_util import write_graph
from tensorflow.python.training.training_util import global_step

# Training data protos.
from tensorflow.core.example.example_pb2 import *
from tensorflow.core.example.feature_pb2 import *

# Utility op.  Open Source. TODO(touts): move to nn?
from tensorflow.python.training.learning_rate_decay import exponential_decay
