---
---
<!-- This file is machine generated: DO NOT EDIT! -->

# Trainer
[TOC]

Generic trainer for TensorFlow models.

## Other Functions and Classes
- - -

### `class skflow.TensorFlowTrainer` {#TensorFlowTrainer}

General trainer class.

Attributes:
  model: Model object.
  gradients: Gradients tensor.
- - -

#### `skflow.TensorFlowTrainer.__init__(loss, global_step, optimizer, learning_rate, clip_gradients=5.0)` {#TensorFlowTrainer.__init__}

Build a trainer part of graph.

##### Args:


*  <b>`loss`</b>: Tensor that evaluates to model's loss.
*  <b>`global_step`</b>: Tensor with global step of the model.
*  <b>`optimizer`</b>: Name of the optimizer class (SGD, Adam, Adagrad) or class.
*  <b>`learning_rate`</b>: If this is constant float value, no decay function is used.
                 Instead, a customized decay function can be passed that accepts
                 global_step as parameter and returns a Tensor.
                 e.g. exponential decay function:
                 def exp_decay(global_step):
                    return tf.train.exponential_decay(
                        learning_rate=0.1, global_step=global_step,
                        decay_steps=2, decay_rate=0.001)

##### Raises:


*  <b>`ValueError`</b>: if learning_rate is not a float or a callable.


- - -

#### `skflow.TensorFlowTrainer.initialize(sess)` {#TensorFlowTrainer.initialize}

Initalizes all variables.

##### Args:


*  <b>`sess`</b>: Session object.

##### Returns:

    Values of initializers.


- - -

#### `skflow.TensorFlowTrainer.train(sess, feed_dict_fn, steps, monitor, summary_writer=None, summaries=None, feed_params_fn=None)` {#TensorFlowTrainer.train}

Trains a model for given number of steps, given feed_dict function.

##### Args:


*  <b>`sess`</b>: Session object.
*  <b>`feed_dict_fn`</b>: Function that will return a feed dictionary.
*  <b>`summary_writer`</b>: SummaryWriter object to use for writing summaries.
*  <b>`steps`</b>: Number of steps to run.
*  <b>`monitor`</b>: Monitor object to track training progress and induce early stopping
*  <b>`summaries`</b>: Joined object of all summaries that should be ran.

##### Returns:

    List of losses for each step.



