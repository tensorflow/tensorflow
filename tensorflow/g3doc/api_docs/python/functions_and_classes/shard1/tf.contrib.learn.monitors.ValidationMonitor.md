Runs evaluation of a given estimator, at most every N steps.

Note that the evaluation is done based on the saved checkpoint, which will
usually be older than the current step.

Can do early stopping on validation metrics if `early_stopping_rounds` is
provided.
- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.__init__(x=None, y=None, input_fn=None, batch_size=None, eval_steps=None, every_n_steps=100, metrics=None, hooks=None, early_stopping_rounds=None, early_stopping_metric='loss', early_stopping_metric_minimize=True, name=None)` {#ValidationMonitor.__init__}

Initializes a ValidationMonitor.

##### Args:


*  <b>`x`</b>: See `BaseEstimator.evaluate`.
*  <b>`y`</b>: See `BaseEstimator.evaluate`.
*  <b>`input_fn`</b>: See `BaseEstimator.evaluate`.
*  <b>`batch_size`</b>: See `BaseEstimator.evaluate`.
*  <b>`eval_steps`</b>: See `BaseEstimator.evaluate`.
*  <b>`every_n_steps`</b>: Check for new checkpoints to evaluate every N steps. If a
      new checkpoint is found, it is evaluated. See `EveryN`.
*  <b>`metrics`</b>: See `BaseEstimator.evaluate`.
*  <b>`hooks`</b>: A list of `SessionRunHook` hooks to pass to the
    `Estimator`'s `evaluate` function.
*  <b>`early_stopping_rounds`</b>: `int`. If the metric indicated by
      `early_stopping_metric` does not change according to
      `early_stopping_metric_minimize` for this many steps, then training
      will be stopped.
*  <b>`early_stopping_metric`</b>: `string`, name of the metric to check for early
      stopping.
*  <b>`early_stopping_metric_minimize`</b>: `bool`, True if `early_stopping_metric` is
      expected to decrease (thus early stopping occurs when this metric
      stops decreasing), False if `early_stopping_metric` is expected to
      increase. Typically, `early_stopping_metric_minimize` is True for
      loss metrics like mean squared error, and False for performance
      metrics like accuracy.
*  <b>`name`</b>: See `BaseEstimator.evaluate`.

##### Raises:


*  <b>`ValueError`</b>: If both x and input_fn are provided.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.begin(max_steps=None)` {#ValidationMonitor.begin}

Called at the beginning of training.

When called, the default graph is the one we are executing.

##### Args:


*  <b>`max_steps`</b>: `int`, the maximum global step this training will run until.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun a run.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.best_step` {#ValidationMonitor.best_step}

Returns the step at which the best early stopping metric was found.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.best_value` {#ValidationMonitor.best_value}

Returns the best early stopping metric value found so far.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.early_stopped` {#ValidationMonitor.early_stopped}

Returns True if this monitor caused an early stop.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.end(session=None)` {#ValidationMonitor.end}




- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.epoch_begin(epoch)` {#ValidationMonitor.epoch_begin}

Begin epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've already begun an epoch, or `epoch` < 0.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.epoch_end(epoch)` {#ValidationMonitor.epoch_end}

End epoch.

##### Args:


*  <b>`epoch`</b>: `int`, the epoch number.

##### Raises:


*  <b>`ValueError`</b>: if we've not begun an epoch, or `epoch` number does not match.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.every_n_post_step(step, session)` {#ValidationMonitor.every_n_post_step}

Callback after a step is finished or `end()` is called.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.
*  <b>`session`</b>: `Session` object.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.every_n_step_begin(step)` {#ValidationMonitor.every_n_step_begin}

Callback before every n'th step begins.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.

##### Returns:

  A `list` of tensors that will be evaluated at this step.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.every_n_step_end(step, outputs)` {#ValidationMonitor.every_n_step_end}




- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.post_step(step, session)` {#ValidationMonitor.post_step}




- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.run_on_all_workers` {#ValidationMonitor.run_on_all_workers}




- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.set_estimator(estimator)` {#ValidationMonitor.set_estimator}

A setter called automatically by the target estimator.

If the estimator is locked, this method does nothing.

##### Args:


*  <b>`estimator`</b>: the estimator that this monitor monitors.

##### Raises:


*  <b>`ValueError`</b>: if the estimator is None.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.step_begin(step)` {#ValidationMonitor.step_begin}

Overrides `BaseMonitor.step_begin`.

When overriding this method, you must call the super implementation.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.

##### Returns:

  A `list`, the result of every_n_step_begin, if that was called this step,
  or an empty list otherwise.

##### Raises:


*  <b>`ValueError`</b>: if called more than once during a step.


- - -

#### `tf.contrib.learn.monitors.ValidationMonitor.step_end(step, output)` {#ValidationMonitor.step_end}

Overrides `BaseMonitor.step_end`.

When overriding this method, you must call the super implementation.

##### Args:


*  <b>`step`</b>: `int`, the current value of the global step.
*  <b>`output`</b>: `dict` mapping `string` values representing tensor names to
    the value resulted from running these tensors. Values may be either
    scalars, for scalar tensors, or Numpy `array`, for non-scalar tensors.

##### Returns:

  `bool`, the result of every_n_step_end, if that was called this step,
  or `False` otherwise.


