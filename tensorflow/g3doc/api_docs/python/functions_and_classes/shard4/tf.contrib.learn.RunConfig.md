This class specifies the specific configurations for the run.

If you're a Google-internal user using command line flags with learn_runner.py
(for instance, to do distributed training or to use parameter servers), you
probably want to use learn_runner.EstimatorConfig instead.
- - -

#### `tf.contrib.learn.RunConfig.__init__(master=None, task=None, num_ps_replicas=None, num_cores=0, log_device_placement=False, gpu_memory_fraction=1, cluster_spec=None, tf_random_seed=None, save_summary_steps=100, save_checkpoints_secs=600, keep_checkpoint_max=5, keep_checkpoint_every_n_hours=10000, job_name=None, is_chief=None, evaluation_master='')` {#RunConfig.__init__}

Constructor.

If set to None, `master`, `task`, `num_ps_replicas`, `cluster_spec`,
`job_name`, and `is_chief` are set based on the TF_CONFIG environment
variable, if the pertinent information is present; otherwise, the defaults
listed in the Args section apply.

The TF_CONFIG environment variable is a JSON object with two relevant
attributes: `task` and `cluster_spec`. `cluster_spec` is a JSON serialized
version of the Python dict described in server_lib.py. `task` has two
attributes: `type` and `index`, where `type` can be any of the task types
in the cluster_spec. When TF_CONFIG contains said information, the
following properties are set on this class:

  * `job_name` is set to [`task`][`type`]
  * `task` is set to [`task`][`index`]
  * `cluster_spec` is parsed from [`cluster`]
  * 'master' is determined by looking up `job_name` and `task` in the
    cluster_spec.
  * `num_ps_replicas` is set by counting the number of nodes listed
    in the `ps` job of `cluster_spec`.
  * `is_chief`: true when `job_name` == "master" and `task` == 0.

Example:
```
  cluster = {'ps': ['host1:2222', 'host2:2222'],
             'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
  os.environ['TF_CONFIG'] = json.dumps({
      {'cluster': cluster,
       'task': {'type': 'worker', 'index': 1}}})
  config = RunConfig()
  assert config.master == 'host4:2222'
  assert config.task == 1
  assert config.num_ps_replicas == 2
  assert config.cluster_spec == server_lib.ClusterSpec(cluster)
  assert config.job_name == 'worker'
  assert not config.is_chief
```

##### Args:


*  <b>`master`</b>: TensorFlow master. Defaults to empty string for local.
*  <b>`task`</b>: Task id of the replica running the training (default: 0).
*  <b>`num_ps_replicas`</b>: Number of parameter server tasks to use (default: 0).
*  <b>`num_cores`</b>: Number of cores to be used. If 0, the system picks an
    appropriate number (default: 0).
*  <b>`log_device_placement`</b>: Log the op placement to devices (default: False).
*  <b>`gpu_memory_fraction`</b>: Fraction of GPU memory used by the process on
    each GPU uniformly on the same machine.
*  <b>`cluster_spec`</b>: a `tf.train.ClusterSpec` object that describes the cluster
    in the case of distributed computation. If missing, reasonable
    assumptions are made for the addresses of jobs.
*  <b>`tf_random_seed`</b>: Random seed for TensorFlow initializers.
    Setting this value allows consistency between reruns.
*  <b>`save_summary_steps`</b>: Save summaries every this many steps.
*  <b>`save_checkpoints_secs`</b>: Save checkpoints every this many seconds.
*  <b>`keep_checkpoint_max`</b>: The maximum number of recent checkpoint files to
    keep. As new files are created, older files are deleted. If None or 0,
    all checkpoint files are kept. Defaults to 5 (that is, the 5 most recent
    checkpoint files are kept.)
*  <b>`keep_checkpoint_every_n_hours`</b>: Number of hours between each checkpoint
    to be saved. The default value of 10,000 hours effectively disables
    the feature.
*  <b>`job_name`</b>: the type of task, e.g., 'ps', 'worker', etc. The `job_name`
    must exist in the `cluster_spec.jobs`.
*  <b>`is_chief`</b>: whether or not this task (as identified by the other parameters)
    should be the chief task.
*  <b>`evaluation_master`</b>: the master on which to perform evaluation.

##### Raises:


*  <b>`ValueError`</b>: if num_ps_replicas and cluster_spec are set (cluster_spec
    may fome from the TF_CONFIG environment variable).


- - -

#### `tf.contrib.learn.RunConfig.is_chief` {#RunConfig.is_chief}




- - -

#### `tf.contrib.learn.RunConfig.job_name` {#RunConfig.job_name}




