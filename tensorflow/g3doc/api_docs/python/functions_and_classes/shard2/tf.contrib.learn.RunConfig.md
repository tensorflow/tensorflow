This class specifies the specific configurations for the run.

Parameters:
  execution_mode: Runners use this flag to execute different tasks, like
    training vs evaluation. 'all' (the default) executes both training and
    eval.
  master: TensorFlow master. Empty string (the default) for local.
  task: Task id of the replica running the training (default: 0).
  num_ps_replicas: Number of parameter server tasks to use (default: 0).
  training_worker_session_startup_stagger_secs: Seconds to sleep between the
    startup of each worker task session (default: 5).
  training_worker_max_startup_secs: Max seconds to wait before starting any
    worker (default: 60).
  eval_delay_secs: Number of seconds between the beginning of each eval run.
    If one run takes more than this amount of time, the next run will start
    immediately once that run completes (default 60).
  eval_steps: Number of steps to run in each eval (default: 100).
  num_cores: Number of cores to be used (default: 4).
  verbose: Controls the verbosity, possible values:
    0: the algorithm and debug information is muted.
    1: trainer prints the progress.
    2: log device placement is printed.
  gpu_memory_fraction: Fraction of GPU memory used by the process on
    each GPU uniformly on the same machine.
  tf_random_seed: Random seed for TensorFlow initializers.
    Setting this value allows consistency between reruns.
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

#### `tf.contrib.learn.RunConfig.__init__(execution_mode='all', master='', task=0, num_ps_replicas=0, training_worker_session_startup_stagger_secs=5, training_worker_max_startup_secs=60, eval_delay_secs=60, eval_steps=100, num_cores=4, verbose=1, gpu_memory_fraction=1, tf_random_seed=42, keep_checkpoint_max=5, keep_checkpoint_every_n_hours=10000)` {#RunConfig.__init__}




