### `tf.contrib.learn.evaluate(graph, output_dir, checkpoint_path, eval_dict, global_step_tensor=None, init_op=None, supervisor_master='', log_every_steps=10, max_steps=None, max_global_step=None, tuner=None, tuner_metric=None)` {#evaluate}

Evaluate a model loaded from a checkpoint.

Given `graph`, a directory to write summaries to (`output_dir`), a checkpoint
to restore variables from, and a `dict` of `Tensor`s to evaluate, run an eval
loop for `max_steps` steps.

In each step of evaluation, all tensors in the `eval_dict` are evaluated, and
every `log_every_steps` steps, they are logged. At the very end of evaluation,
a summary is evaluated (finding the summary ops using `Supervisor`'s logic)
and written to `output_dir`.

##### Args:


*  <b>`graph`</b>: A `Graph` to train. It is expected that this graph is not in use
    elsewhere.
*  <b>`output_dir`</b>: A string containing the directory to write a summary to.
*  <b>`checkpoint_path`</b>: A string containing the path to a checkpoint to restore.
    Can be `None` if the graph doesn't require loading any variables.
*  <b>`eval_dict`</b>: A `dict` mapping string names to tensors to evaluate for in every
    eval step.
*  <b>`global_step_tensor`</b>: A `Variable` containing the global step. If `None`,
    one is extracted from the graph using the same logic as in `Supervisor`.
    Used to place eval summaries on training curves.
*  <b>`init_op`</b>: An op that initializes the graph. If `None`, use `Supervisor`'s
    default.
*  <b>`supervisor_master`</b>: The master string to use when preparing the session.
*  <b>`log_every_steps`</b>: Integer. Output logs every `log_every_steps` evaluation
    steps. The logs contain the `eval_dict` and timing information.
*  <b>`max_steps`</b>: Integer. Evaluate `eval_dict` this many times.
*  <b>`max_global_step`</b>: Integer.  If the global_step is larger than this, skip
    the eval and return None.
*  <b>`tuner`</b>: A `Tuner` that will be notified of eval completion and updated
    with objective metrics.
*  <b>`tuner_metric`</b>: A `string` that specifies the eval metric to report to
    `tuner`.

##### Returns:

  A tuple `(eval_results, should_stop)`:

*  <b>`eval_results`</b>: A `dict` mapping `string` to numeric values (`int`, `float`)
    that are the eval results from the last step of the eval.  None if no
    eval steps were run.
  should stop: A `bool`, indicating whether it was detected that eval should
    stop.

##### Raises:


*  <b>`ValueError`</b>: if the caller specifies max_global_step without providing
    a global_step.

