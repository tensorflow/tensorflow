### `tf.contrib.learn.evaluate(graph, output_dir, checkpoint_path, eval_dict, update_op=None, global_step_tensor=None, supervisor_master='', log_every_steps=10, feed_fn=None, max_steps=None)` {#evaluate}

Evaluate a model loaded from a checkpoint.

Given `graph`, a directory to write summaries to (`output_dir`), a checkpoint
to restore variables from, and a `dict` of `Tensor`s to evaluate, run an eval
loop for `max_steps` steps, or until an exception (generally, an
end-of-input signal from a reader operation) is raised from running
`eval_dict`.

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
*  <b>`eval_dict`</b>: A `dict` mapping string names to tensors to evaluate. It is
    evaluated in every logging step. The result of the final evaluation is
    returned. If `update_op` is None, then it's evaluated in every step. If
    `max_steps` is `None`, this should depend on a reader that will raise an
    end-of-inupt exception when the inputs are exhausted.
*  <b>`update_op`</b>: A `Tensor` which is run in every step.
*  <b>`global_step_tensor`</b>: A `Variable` containing the global step. If `None`,
    one is extracted from the graph using the same logic as in `Supervisor`.
    Used to place eval summaries on training curves.
*  <b>`supervisor_master`</b>: The master string to use when preparing the session.
*  <b>`log_every_steps`</b>: Integer. Output logs every `log_every_steps` evaluation
    steps. The logs contain the `eval_dict` and timing information.
*  <b>`feed_fn`</b>: A function that is called every iteration to produce a `feed_dict`
    passed to `session.run` calls. Optional.
*  <b>`max_steps`</b>: Integer. Evaluate `eval_dict` this many times.

##### Returns:

  A tuple `(eval_results, global_step)`:

*  <b>`eval_results`</b>: A `dict` mapping `string` to numeric values (`int`, `float`)
    that are the result of running eval_dict in the last step. `None` if no
    eval steps were run.
*  <b>`global_step`</b>: The global step this evaluation corresponds to.

##### Raises:


*  <b>`ValueError`</b>: if `output_dir` is empty.

