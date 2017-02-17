### `tf.train.add_queue_runner(qr, collection='queue_runners')` {#add_queue_runner}

Adds a `QueueRunner` to a collection in the graph.

When building a complex model that uses many queues it is often difficult to
gather all the queue runners that need to be run.  This convenience function
allows you to add a queue runner to a well known collection in the graph.

The companion method `start_queue_runners()` can be used to start threads for
all the collected queue runners.

##### Args:


*  <b>`qr`</b>: A `QueueRunner`.
*  <b>`collection`</b>: A `GraphKey` specifying the graph collection to add
    the queue runner to.  Defaults to `GraphKeys.QUEUE_RUNNERS`.

