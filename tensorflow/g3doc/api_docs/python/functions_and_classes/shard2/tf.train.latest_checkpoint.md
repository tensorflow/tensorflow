### `tf.train.latest_checkpoint(checkpoint_dir, latest_filename=None)` {#latest_checkpoint}

Finds the filename of latest saved checkpoint file.

##### Args:


*  <b>`checkpoint_dir`</b>: Directory where the variables were saved.
*  <b>`latest_filename`</b>: Optional name for the protocol buffer file that
    contains the list of most recent checkpoint filenames.
    See the corresponding argument to `Saver.save()`.

##### Returns:

  The full path to the latest checkpoint or `None` if no checkpoint was found.

