`tf.contrib.data` API
=====================

NOTE: The `tf.contrib.data` module has been deprecated. Use `tf.data` instead,
or `tf.data.experimental` for the experimental transformations previously hosted
in this module. We are continuing to support existing code using the
`tf.contrib.data` APIs in the current version of TensorFlow, but will eventually
remove support. The non-experimental `tf.data` APIs are subject to backwards
compatibility guarantees.

Porting your code to `tf.data`
------------------------------

The `tf.contrib.data.Dataset` class has been renamed to `tf.data.Dataset`, and
the `tf.contrib.data.Iterator` class has been renamed to `tf.data.Iterator`.
Most code can be ported by removing `.contrib` from the names of the classes.
However, there are some small differences, which are outlined below.

The arguments accepted by the `Dataset.map()` transformation have changed:

* `dataset.map(..., num_threads=T)` is now `dataset.map(num_parallel_calls=T)`.
* `dataset.map(..., output_buffer_size=B)` is now
  `dataset.map(...).prefetch(B)`.

Some transformations have been removed from `tf.data.Dataset`, and you must
instead apply them using `Dataset.apply()` transformation. The full list of
changes is as follows:

* `dataset.dense_to_sparse_batch(...)` is now
  `dataset.apply(tf.data.experimental.dense_to_sparse_batch(...)`.
* `dataset.enumerate(...)` is now
  `dataset.apply(tf.data.experimental.enumerate_dataset(...))`.
* `dataset.group_by_window(...)` is now
  `dataset.apply(tf.data.experimental.group_by_window(...))`.
* `dataset.ignore_errors()` is now
  `dataset.apply(tf.data.experimental.ignore_errors())`.
* `dataset.unbatch()` is now `dataset.apply(tf.contrib.data.unbatch())`.

The `Dataset.make_dataset_resource()` and `Iterator.dispose_op()` methods have
been removed from the API. Please open a GitHub issue if you have a need for
either of these.
