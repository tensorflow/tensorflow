This class specifies the specific configurations for the run.

Parameters:
  tf_master: TensorFlow master. Empty string is default for local.
  num_cores: Number of cores to be used. (default: 4)
  verbose: Controls the verbosity, possible values:
    0: the algorithm and debug information is muted.
    1: trainer prints the progress.
    2: log device placement is printed.
  gpu_memory_fraction: Fraction of GPU memory used by the process on
    each GPU uniformly on the same machine.
  tf_random_seed: Random seed for TensorFlow initializers.
    Setting this value, allows consistency between reruns.
  keep_checkpoint_max: The maximum number of recent checkpoint files to keep.
    As new files are created, older files are deleted.
    If None or 0, all checkpoint files are kept.
    Defaults to 5 (that is, the 5 most recent checkpoint files are kept.)
  keep_checkpoint_every_n_hours: Number of hours between each checkpoint
    to be saved. The default value of 10,000 hours effectively disables
    the feature.

Attributes:
  tf_master: Tensorflow master.
  tf_config: Tensorflow Session Config proto.
  tf_random_seed: Tensorflow random seed.
  keep_checkpoint_max: Maximum number of checkpoints to keep.
  keep_checkpoint_every_n_hours: Number of hours between each checkpoint.
- - -

#### `tf.contrib.learn.RunConfig.__init__(tf_master='', num_cores=4, verbose=1, gpu_memory_fraction=1, tf_random_seed=42, keep_checkpoint_max=5, keep_checkpoint_every_n_hours=10000)` {#RunConfig.__init__}




