Saves summaries every N steps.
- - -

#### `tf.train.SummarySaverHook.__init__(save_steps=None, save_secs=None, output_dir=None, summary_writer=None, scaffold=None, summary_op=None)` {#SummarySaverHook.__init__}

Initializes a `SummarySaver` monitor.

##### Args:


*  <b>`save_steps`</b>: `int`, save summaries every N steps. Exactly one of
      `save_secs` and `save_steps` should be set.
*  <b>`save_secs`</b>: `int`, save summaries every N seconds.
*  <b>`output_dir`</b>: `string`, the directory to save the summaries to. Only used
      if no `summary_writer` is supplied.
*  <b>`summary_writer`</b>: `SummaryWriter`. If `None` and an `output_dir` was passed,
      one will be created accordingly.
*  <b>`scaffold`</b>: `Scaffold` to get summary_op if it's not provided.
*  <b>`summary_op`</b>: `Tensor` of type `string` containing the serialized `Summary`
      protocol buffer or a list of `Tensor`. They are most likely an output
      by TF summary methods like `tf.summary.scalar` or
      `tf.summary.merge_all`. It can be passed in as one tensor; if more
      than one, they must be passed in as a list.

##### Raises:


*  <b>`ValueError`</b>: Exactly one of scaffold or summary_op should be set.


- - -

#### `tf.train.SummarySaverHook.after_create_session(session)` {#SummarySaverHook.after_create_session}

Called when new TensorFlow session is created.

This is called to signal the hooks that a new session has been created. This
has two essential differences with the situation in which `begin` is called:

* When this is called, the graph is finalized and ops can no longer be added
    to the graph.
* This method will be called as a result of recovering a wrapped session,
    instead of at the beginning of the overall session.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that has been created.


- - -

#### `tf.train.SummarySaverHook.after_run(run_context, run_values)` {#SummarySaverHook.after_run}




- - -

#### `tf.train.SummarySaverHook.before_run(run_context)` {#SummarySaverHook.before_run}




- - -

#### `tf.train.SummarySaverHook.begin()` {#SummarySaverHook.begin}




- - -

#### `tf.train.SummarySaverHook.end(session=None)` {#SummarySaverHook.end}




