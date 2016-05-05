### `tf.contrib.layers.optimize_loss(loss, global_step, learning_rate, optimizer, gradient_noise_scale=None, clip_gradients=None, moving_average_decay=0.9, learning_rate_decay_fn=None, variables=None)` {#optimize_loss}

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
*  <b>`clip_gradients`</b>: float or None, clips gradients by this value.
*  <b>`moving_average_decay`</b>: float or None, takes into account previous loss
                        to make learning smoother due to outliers.
*  <b>`learning_rate_decay_fn`</b>: function, takes learning_rate and global_step
                          Tensors, returns Tensor. Can be used to implement
                          any learning rate decay functions.
                          For example: tf.train.exponential_decay.
*  <b>`variables`</b>: list of variables to optimizer or none.

##### Returns:

  Training op.

##### Raises:


*  <b>`ValueError`</b>: if optimizer is wrong type.

