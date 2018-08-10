# Training
[TOC]

`tf.train` provides a set of classes and functions that help train models.

## Optimizers

The Optimizer base class provides methods to compute gradients for a loss and
apply gradients to variables.  A collection of subclasses implement classic
optimization algorithms such as GradientDescent and Adagrad.

You never instantiate the Optimizer class itself, but instead instantiate one
of the subclasses.

*   `tf.train.Optimizer`
*   `tf.train.GradientDescentOptimizer`
*   `tf.train.AdadeltaOptimizer`
*   `tf.train.AdagradOptimizer`
*   `tf.train.AdagradDAOptimizer`
*   `tf.train.MomentumOptimizer`
*   `tf.train.AdamOptimizer`
*   `tf.train.FtrlOptimizer`
*   `tf.train.ProximalGradientDescentOptimizer`
*   `tf.train.ProximalAdagradOptimizer`
*   `tf.train.RMSPropOptimizer`

See `tf.contrib.opt` for more optimizers.

## Gradient Computation

TensorFlow provides functions to compute the derivatives for a given
TensorFlow computation graph, adding operations to the graph. The
optimizer classes automatically compute derivatives on your graph, but
creators of new Optimizers or expert users can call the lower-level
functions below.

*   `tf.gradients`
*   `tf.AggregationMethod`
*   `tf.stop_gradient`
*   `tf.hessians`


## Gradient Clipping

TensorFlow provides several operations that you can use to add clipping
functions to your graph. You can use these functions to perform general data
clipping, but they're particularly useful for handling exploding or vanishing
gradients.

*   `tf.clip_by_value`
*   `tf.clip_by_norm`
*   `tf.clip_by_average_norm`
*   `tf.clip_by_global_norm`
*   `tf.global_norm`

## Decaying the learning rate

*   `tf.train.exponential_decay`
*   `tf.train.inverse_time_decay`
*   `tf.train.natural_exp_decay`
*   `tf.train.piecewise_constant`
*   `tf.train.polynomial_decay`
*   `tf.train.cosine_decay`
*   `tf.train.linear_cosine_decay`
*   `tf.train.noisy_linear_cosine_decay`

## Moving Averages

Some training algorithms, such as GradientDescent and Momentum often benefit
from maintaining a moving average of variables during optimization.  Using the
moving averages for evaluations often improve results significantly.

*   `tf.train.ExponentialMovingAverage`

## Coordinator and QueueRunner

See @{$threading_and_queues$Threading and Queues}
for how to use threads and queues.  For documentation on the Queue API,
see @{$python/io_ops#queues$Queues}.


*   `tf.train.Coordinator`
*   `tf.train.QueueRunner`
*   `tf.train.LooperThread`
*   `tf.train.add_queue_runner`
*   `tf.train.start_queue_runners`

## Distributed execution

See @{$distributed$Distributed TensorFlow} for
more information about how to configure a distributed TensorFlow program.

*   `tf.train.Server`
*   `tf.train.Supervisor`
*   `tf.train.SessionManager`
*   `tf.train.ClusterSpec`
*   `tf.train.replica_device_setter`
*   `tf.train.MonitoredTrainingSession`
*   `tf.train.MonitoredSession`
*   `tf.train.SingularMonitoredSession`
*   `tf.train.Scaffold`
*   `tf.train.SessionCreator`
*   `tf.train.ChiefSessionCreator`
*   `tf.train.WorkerSessionCreator`

## Reading Summaries from Event Files

See @{$summaries_and_tensorboard$Summaries and TensorBoard} for an
overview of summaries, event files, and visualization in TensorBoard.

*   `tf.train.summary_iterator`

## Training Hooks

Hooks are tools that run in the process of training/evaluation of the model.

*   `tf.train.SessionRunHook`
*   `tf.train.SessionRunArgs`
*   `tf.train.SessionRunContext`
*   `tf.train.SessionRunValues`
*   `tf.train.LoggingTensorHook`
*   `tf.train.StopAtStepHook`
*   `tf.train.CheckpointSaverHook`
*   `tf.train.NewCheckpointReader`
*   `tf.train.StepCounterHook`
*   `tf.train.NanLossDuringTrainingError`
*   `tf.train.NanTensorHook`
*   `tf.train.SummarySaverHook`
*   `tf.train.GlobalStepWaiterHook`
*   `tf.train.FinalOpsHook`
*   `tf.train.FeedFnHook`

## Training Utilities

*   `tf.train.global_step`
*   `tf.train.basic_train_loop`
*   `tf.train.get_global_step`
*   `tf.train.assert_global_step`
*   `tf.train.write_graph`
