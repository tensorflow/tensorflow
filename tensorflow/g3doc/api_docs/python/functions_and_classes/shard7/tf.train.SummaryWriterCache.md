Cache for summary writers.

This class caches summary writers, one per directory.
- - -

#### `tf.train.SummaryWriterCache.clear()` {#SummaryWriterCache.clear}

Clear cached summary writers. Currently only used for unit tests.


- - -

#### `tf.train.SummaryWriterCache.get(logdir)` {#SummaryWriterCache.get}

Returns the SummaryWriter for the specified directory.

##### Args:


*  <b>`logdir`</b>: str, name of the directory.

##### Returns:

  A `SummaryWriter`.


