Cache for file writers.

This class caches file writers, one per directory.
- - -

#### `tf.summary.FileWriterCache.clear()` {#FileWriterCache.clear}

Clear cached summary writers. Currently only used for unit tests.


- - -

#### `tf.summary.FileWriterCache.get(logdir)` {#FileWriterCache.get}

Returns the FileWriter for the specified directory.

##### Args:


*  <b>`logdir`</b>: str, name of the directory.

##### Returns:

  A `FileWriter`.


