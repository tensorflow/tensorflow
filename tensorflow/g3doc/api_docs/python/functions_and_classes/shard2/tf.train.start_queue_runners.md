### `tf.train.start_queue_runners(sess=None, coord=None, daemon=True, start=True, collection='queue_runners')` {#start_queue_runners}

Starts all queue runners collected in the graph.

This is a companion method to `add_queue_runner()`.  It just starts
threads for all queue runners collected in the graph.  It returns
the list of all threads.

##### Args:


*  <b>`sess`</b>: `Session` used to run the queue ops.  Defaults to the
    default session.
*  <b>`coord`</b>: Optional `Coordinator` for coordinating the started threads.
*  <b>`daemon`</b>: Whether the threads should be marked as `daemons`, meaning
    they don't block program exit.
*  <b>`start`</b>: Set to `False` to only create the threads, not start them.
*  <b>`collection`</b>: A `GraphKey` specifying the graph collection to
    get the queue runners from.  Defaults to `GraphKeys.QUEUE_RUNNERS`.

##### Returns:

  A list of threads.

