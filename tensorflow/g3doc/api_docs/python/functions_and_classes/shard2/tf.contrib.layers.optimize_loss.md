### `tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, optimizer, gradient_noise_scale=None, gradient_multipliers=None, clip_gradients=None, moving_average_decay=None, learning_rate_decay_fn=None, update_ops=None, variables=None, name=None, summaries=None)` {#optimize_loss}

Given loss and parameters for optimizer, returns a training op.

##### Args:


*  <b>`loss`</b>: Tensor, 0 dimensional.
*  <b>`global_step`</b>: Tensor, step counter for each update.
*  <b>`learning_rate`</b>: float or Tensor, magnitude of update per each training step.
*  <b>`optimizer`</b>: string, class or optimizer instance, used as trainer.
             string should be name of optimizer, like 'SGD',
               'Adam', 'Adagrad'. Full list in OPTIMIZER_CLS_NAMES constant.
             class should be sub-class of tf.Optimizer that implements
               `compute_gradients` and `apply_gradients` functions.
             optimizer instance should be instantion of tf.Optimizer sub-class
               and have `compute_gradients` and `apply_gradients` functions.
*  <b>`gradient_noise_scale`</b>: float or None, adds 0-mean normal noise scaled by this
                        value.
*  <b>`gradient_multipliers`</b>: dict of variables or variable names to floats.
                        If present, gradients for specified
                        variables will be multiplied by given constant.
*  <b>`clip_gradients`</b>: float or `None`, clips gradients by this value.
*  <b>`moving_average_decay`</b>: Deprecated. float or None, takes into account previous
                        loss to make learning smoother due to outliers.
*  <b>`learning_rate_decay_fn`</b>: function, takes `learning_rate` and `global_step`
                          `Tensor`s, returns `Tensor`.
                          Can be used to implement any learning rate decay
                          functions.
                          For example: tf.train.exponential_decay.
*  <b>`update_ops`</b>: list of update `Operation`s to execute at each step. If `None`,
              uses elements of UPDATE_OPS collection.
*  <b>`variables`</b>: list of variables to optimize or
             `None` to use all trainable variables.
*  <b>`name`</b>: The name for this operation is used to scope operations and summaries.
*  <b>`summaries`</b>: List of internal quantities to visualize on tensorboard. If not
             set only the loss and the learning rate will be reported. The
             complete list is in OPTIMIZER_SUMMARIES.

##### Returns:

  Training op.

##### Raises:


*  <b>`ValueError`</b>: if optimizer is wrong type.

