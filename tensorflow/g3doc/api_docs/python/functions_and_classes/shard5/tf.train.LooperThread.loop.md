#### `tf.train.LooperThread.loop(coord, timer_interval_secs, target, args=None, kwargs=None)` {#LooperThread.loop}

Start a LooperThread that calls a function periodically.

If `timer_interval_secs` is None the thread calls `target(args)`
repeatedly.  Otherwise `target(args)` is called every `timer_interval_secs`
seconds.  The thread terminates when a stop of the coordinator is
requested.

##### Args:


*  <b>`coord`</b>: A Coordinator.
*  <b>`timer_interval_secs`</b>: Number. Time boundaries at which to call `target`.
*  <b>`target`</b>: A callable object.
*  <b>`args`</b>: Optional arguments to pass to `target` when calling it.
*  <b>`kwargs`</b>: Optional keyword arguments to pass to `target` when calling it.

##### Returns:

  The started thread.

