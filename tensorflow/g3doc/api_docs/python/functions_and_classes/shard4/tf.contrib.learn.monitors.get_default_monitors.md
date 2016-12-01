### `tf.contrib.learn.monitors.get_default_monitors(loss_op=None, summary_op=None, save_summary_steps=100, output_dir=None, summary_writer=None)` {#get_default_monitors}

Returns a default set of typically-used monitors.

##### Args:


*  <b>`loss_op`</b>: `Tensor`, the loss tensor. This will be printed using `PrintTensor`
      at the default interval.
*  <b>`summary_op`</b>: See `SummarySaver`.
*  <b>`save_summary_steps`</b>: See `SummarySaver`.
*  <b>`output_dir`</b>: See `SummarySaver`.
*  <b>`summary_writer`</b>: See `SummarySaver`.

##### Returns:

  `list` of monitors.

