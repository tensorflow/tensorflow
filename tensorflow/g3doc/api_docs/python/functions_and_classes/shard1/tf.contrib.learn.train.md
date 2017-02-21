### `tf.contrib.learn.train(*args, **kwargs)` {#train}

Train a model. (deprecated)

THIS FUNCTION IS DEPRECATED. It will be removed after 2017-02-15.
Instructions for updating:
graph_actions.py will be deleted. Use tf.train.* utilities instead. You can use learn/estimators/estimator.py as an example.

Given `graph`, a directory to write outputs to (`output_dir`), and some ops,
run a training loop. The given `train_op` performs one step of training on the
model. The `loss_op` represents the objective function of the training. It is
expected to increment the `global_step_tensor`, a scalar integer tensor
counting training steps. This function uses `Supervisor` to initialize the
graph (from a checkpoint if one is available in `output_dir`), write summaries
defined in the graph, and write regular checkpoints as defined by
`supervisor_save_model_secs`.

Training continues until `global_step_tensor` evaluates to `max_steps`, or, if
`fail_on_nan_loss`, until `loss_op` evaluates to `NaN`. In that case the
program is terminated with exit code 1.

##### Args:


*  <b>`graph`</b>: A graph to train. It is expected that this graph is not in use
    elsewhere.
*  <b>`output_dir`</b>: A directory to write outputs to.
*  <b>`train_op`</b>: An op that performs one training step when run.
*  <b>`loss_op`</b>: A scalar loss tensor.
*  <b>`global_step_tensor`</b>: A tensor representing the global step. If none is given,
    one is extracted from the graph using the same logic as in `Supervisor`.
*  <b>`init_op`</b>: An op that initializes the graph. If `None`, use `Supervisor`'s
    default.
*  <b>`init_feed_dict`</b>: A dictionary that maps `Tensor` objects to feed values.
    This feed dictionary will be used when `init_op` is evaluated.
*  <b>`init_fn`</b>: Optional callable passed to Supervisor to initialize the model.
*  <b>`log_every_steps`</b>: Output logs regularly. The logs contain timing data and the
    current loss.
*  <b>`supervisor_is_chief`</b>: Whether the current process is the chief supervisor in
    charge of restoring the model and running standard services.
*  <b>`supervisor_master`</b>: The master string to use when preparing the session.
*  <b>`supervisor_save_model_secs`</b>: Save a checkpoint every
    `supervisor_save_model_secs` seconds when training.
*  <b>`keep_checkpoint_max`</b>: The maximum number of recent checkpoint files to
    keep. As new files are created, older files are deleted. If None or 0,
    all checkpoint files are kept. This is simply passed as the max_to_keep
    arg to tf.Saver constructor.
*  <b>`supervisor_save_summaries_steps`</b>: Save summaries every
    `supervisor_save_summaries_steps` seconds when training.
*  <b>`feed_fn`</b>: A function that is called every iteration to produce a `feed_dict`
    passed to `session.run` calls. Optional.
*  <b>`steps`</b>: Trains for this many steps (e.g. current global step + `steps`).
*  <b>`fail_on_nan_loss`</b>: If true, raise `NanLossDuringTrainingError` if `loss_op`
    evaluates to `NaN`. If false, continue training as if nothing happened.
*  <b>`monitors`</b>: List of `BaseMonitor` subclass instances. Used for callbacks
    inside the training loop.
*  <b>`max_steps`</b>: Number of total steps for which to train model. If `None`,
    train forever. Two calls fit(steps=100) means 200 training iterations.
    On the other hand two calls of fit(max_steps=100) means, second call
    will not do any iteration since first call did all 100 steps.

##### Returns:

  The final loss value.

##### Raises:


*  <b>`ValueError`</b>: If `output_dir`, `train_op`, `loss_op`, or `global_step_tensor`
    is not provided. See `tf.contrib.framework.get_global_step` for how we
    look up the latter if not provided explicitly.
*  <b>`NanLossDuringTrainingError`</b>: If `fail_on_nan_loss` is `True`, and loss ever
    evaluates to `NaN`.
*  <b>`ValueError`</b>: If both `steps` and `max_steps` are not `None`.

