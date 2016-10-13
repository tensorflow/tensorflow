Saves summaries every N steps.
- - -

#### `tf.train.SummarySaverHook.__init__(save_steps=100, output_dir=None, summary_writer=None, scaffold=None, summary_op=None)` {#SummarySaverHook.__init__}

Initializes a `SummarySaver` monitor.

##### Args:


*  <b>`save_steps`</b>: `int`, save summaries every N steps. See `EveryN`.
*  <b>`output_dir`</b>: `string`, the directory to save the summaries to. Only used
      if no `summary_writer` is supplied.
*  <b>`summary_writer`</b>: `SummaryWriter`. If `None` and an `output_dir` was passed,
      one will be created accordingly.
*  <b>`scaffold`</b>: `Scaffold` to get summary_op if it's not provided.
*  <b>`summary_op`</b>: `Tensor` of type `string`. A serialized `Summary` protocol
      buffer, as output by TF summary methods like `scalar_summary` or
      `merge_all_summaries`.


- - -

#### `tf.train.SummarySaverHook.after_run(run_context, run_values)` {#SummarySaverHook.after_run}




- - -

#### `tf.train.SummarySaverHook.before_run(run_context)` {#SummarySaverHook.before_run}




- - -

#### `tf.train.SummarySaverHook.begin()` {#SummarySaverHook.begin}




- - -

#### `tf.train.SummarySaverHook.end(session=None)` {#SummarySaverHook.end}




