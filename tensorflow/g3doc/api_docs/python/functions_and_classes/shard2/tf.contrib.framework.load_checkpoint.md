### `tf.contrib.framework.load_checkpoint(filepattern)` {#load_checkpoint}

Returns CheckpointReader for latest checkpoint.

##### Args:


*  <b>`filepattern`</b>: Directory with checkpoints file or path to checkpoint.

##### Returns:

  `CheckpointReader` object.

##### Raises:


*  <b>`ValueError`</b>: if checkpoint_dir doesn't have 'checkpoint' file or checkpoints.

