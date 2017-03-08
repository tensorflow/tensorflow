#### `tf.contrib.learn.RunConfig.get_task_id()` {#RunConfig.get_task_id}

Returns task index from `TF_CONFIG` environmental variable.

If you have a ClusterConfig instance, you can just access its task_id
property instead of calling this function and re-parsing the environmental
variable.

##### Returns:

  `TF_CONFIG['task']['index']`. Defaults to 0.

