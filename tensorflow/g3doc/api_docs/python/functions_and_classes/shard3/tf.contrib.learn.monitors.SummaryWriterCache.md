Cache for file writers.

This class caches file writers, one per directory.
- - -

#### `tf.contrib.learn.monitors.SummaryWriterCache.clear()` {#SummaryWriterCache.clear}

Clear cached summary writers. Currently only used for unit tests.


- - -

#### `tf.contrib.learn.monitors.SummaryWriterCache.get(logdir)` {#SummaryWriterCache.get}

Returns the FileWriter for the specified directory.

##### Args:


*  <b>`logdir`</b>: str, name of the directory.

##### Returns:

  A `FileWriter`.


