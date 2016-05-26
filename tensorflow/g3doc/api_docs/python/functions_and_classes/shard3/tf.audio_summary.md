### `tf.audio_summary(tag, tensor, sample_rate, max_outputs=3, collections=None, name=None)` {#audio_summary}

Outputs a `Summary` protocol buffer with audio.

The summary has up to `max_outputs` summary values containing audio. The
audio is built from `tensor` which must be 3-D with shape `[batch_size,
frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
assumed to be in the range of `[-1.0, 1.0]` with a sample rate of
`sample_rate`.

The `tag` argument is a scalar `Tensor` of type `string`.  It is used to
build the `tag` of the summary values:

*  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
*  If `max_outputs` is greater than 1, the summary value tags are
   generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.

##### Args:


*  <b>`tag`</b>: A scalar `Tensor` of type `string`. Used to build the `tag`
    of the summary values.
*  <b>`tensor`</b>: A 3-D `float32` `Tensor` of shape `[batch_size, frames, channels]`
    or a 2-D `float32` `Tensor` of shape `[batch_size, frames]`.
*  <b>`sample_rate`</b>: The sample rate of the signal in hertz.
*  <b>`max_outputs`</b>: Max number of batch elements to generate audio for.
*  <b>`collections`</b>: Optional list of ops.GraphKeys.  The collections to add the
    summary to.  Defaults to [ops.GraphKeys.SUMMARIES]
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A scalar `Tensor` of type `string`. The serialized `Summary` protocol
  buffer.

