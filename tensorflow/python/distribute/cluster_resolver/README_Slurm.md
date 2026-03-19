# Slurm Cluster Resolver

The Slurm Cluster Resolver resolves cluster specification for distributing
TensorFlow work launched on HPC systems running on Slurm. This implementation is
able to handle homogeneous and heterogeneous tasks as long as the number of GPUs
per node and task are the same. This means on nodes with 4 GPUs each it will be
possible to allocate 4 processes on node A and only 2 on node B. The resolution
is done by determining job configuration through a number of Slurm variables and
can be overwritten by user input. By default everything is determined from the
Slurm environment, hence for most uses case no manual setting of parameters is
required.

## How it works

`SlurmClusterResolver` reads the environment variables that are set inside a job
step launched by Slurm. This means it will only work correctly for applications
launched via `srun`.

The process ID/rank is extracted from environment variable `SLURM_PROCID` and
the total number of tasks launched is extracted from `SLURM_STEP_NUM_TASKS`. The
hostnames are resolved by inspection `SLURM_STEP_NODELIST`. The number of tasks
per node is extracted from `SLURM_STEP_TASKS_PER_NODE`, unless a value is
specified by user. By using this variable heterogeneous task distributions are
possible. The user can set `tasks_per_node` to a single integer for homogeneous
tasks or a dictionary mapping node names to number of tasks for heterogeneous
distributions. However setting this is **NOT** recommended as there is a chance
it makes `SLURM_PROCID` be wrong.

A base port can be specified by user and in case there are more than one task
launched per node the port number will be incremented for each additional tasks
on that node. However a reasonable default is used.

The number of GPUs present on each node and number of GPUs for each tasks are
automatically detected. This is done by checking for `CUDA_VISIBLE_DEVICES`
first (which is set by Slurm to a list of GPUs for the current node) and has a
fallback to using `nvidia-smi`. If this doesn't work or non-NVIDIA GPUs are used
those 2 values have to be specified by the user. By default allocated GPUs will
be automatically exposed to processes according to specification by setting
`CUDA_VISIBLE_DEVICES`.

## Basic example

-   Slurm allocation in shell `salloc --nodes=2 -t 01:30:00 --ntasks-per-node=2
    --gres=gpu:k80:4 --exclusive`
-   Run the example `srun python tf_example.py`
-   Creating cluster in Python `import tensorflow as tf cluster_resolver =
    tf.distribute.cluster_resolver.SlurmClusterResolver() strategy =
    tf.distribute.experimental.MultiWorkerMirroredStrategy(cluster_resolver=cluster_resolver)
    with strategy.scope(): # Load and compile model and data`

The above example will allocate 4 jobs on 2 nodes with each node having 2 jobs
and 4 GPUs. `cluster_resolver.cluster_spec()` will return a cluster
specification object in protobuf format with the following value (host names may
vary): `job { name: "worker" tasks { key: 0 value: "t02n13:8888" } tasks { key:
1 value: "t02n13:8889" } tasks { key: 2 value: "t02n41:8888" } tasks { key: 3
value: "t02n41:8889" } }`

The `job_name` will be `worker` for all nodes and `task_index` will be `0` to
`3`. Also GPUs will be allocated automatically, so the first job on each node
will see GPU 0 and 1, and the second GPU 2 and 3.

## Advanced example

-   Assuming the same job parameters (`salloc` & `srun`) as above
-   Creating cluster in Python ``` cluster_resolver =
    tf.contrib.cluster_resolver.SlurmClusterResolver( {'ps': 1, 'worker': 3},
    port_base=1337, tasks_per_node=2, gpus_per_node=2, gpus_per_task=1,
    auto_set_gpu=False)

cluster = cluster_resolver.cluster_spec() job_name, task_index =
cluster_resolver.get_task_info() ```

In this case 1 parameter server job and 3 worker jobs are used. The resulting
protobuf specification will look similar to this: `job { name: "ps" tasks { key:
0 value: "t02n13:1337" } } job { name: "worker" tasks { key: 0 value:
"t02n13:1338" } tasks { key: 1 value: "t02n41:1337" } tasks { key: 2 value:
"t02n41:1338" } }`

The value of `job_name` will be `ps` for `t02n13:1337` and `worker` for all
others. There will be no GPU allocation done by the cluster resolver, so this
has to be done manually which is useful if e.g. GPUs 0 should go to the first
process and GPU 3 to the second process on each node. Also note that only 1 GPU
will be used per task.

## Extension points

The class `SlurmClusterResolver` provides some methods that are meant to be
overwritten by deriving classes:

-   `_resolve_own_rank`
-   `_resolve_num_tasks`
-   `_resolve_hostlist`
-   `_resolve_task_configuration`

    Those can be used to implement a cluster resolver that gets information from
    a different source, e.g. via MPI, a file or other environment variables. See
    the documentation of these methods on what to return.
