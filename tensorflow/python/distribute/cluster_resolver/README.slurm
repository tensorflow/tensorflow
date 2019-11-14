# Slurm Cluster Resolver

The Slurm Cluster Resolver resolves cluster specification for distribution TensorFlow work launched on HPC system running on Slurm. This implementation is able to handle homogeneous task allocation on computing nodes with default task distribution plane. The resolution is done by determining job configuration through a number of Slurm output variables and user input. The resolver requires the specification of total number of tasks launched, process ID/rank of the running process, number of tasks launched per node, number of GPUs present on each node and the number of GPUs to allocate for each task.

The process ID/rank is extracted from environment variable ```SLURM_PROCID``` and the total number of tasks launched is extract from ```SLURM_NTASKS```. The number of tasks per node is extracted from ```SLURM_NTASKS_PER_NODE```, unless a value is specified by user. The number of GPUs present on each node and number of GPUs for each task have to be specified by the user. A base port can be specified by user and in case there are more than one task launched per node the port number will be incremented for each additional tasks on that node. The hostnames are resolved by running command ```scontrol show hostname``` through a subprocess and a list of hostnames will be returned. The distribution of rank/process ID by default follows that order. By default allocated GPUs will be automatically exposed to processes according to specification by setting ```CUDA_VISIBLE_DEVICE```.

## Example
- Slurm allocation in shell  ```salloc --nodes=2 -t 01:30:00 -A <project ID> --ntasks-per-node=2 --gres=gpu:k80:2 --exclusive```
- Creating cluster in Python
```
cluster_resolver = tf.contrib.cluster_resolver.SlurmClusterResolver(
    {'ps': 1, 'worker': 3},
    port_base=8888,
    tasks_per_node=2,
    gpus_per_node=2,
    gpus_per_task=1,
    auto_set_gpu=True)

cluster = cluster_resolver.cluster_spec()
job_name, task_index = cluster_resolver.get_task_info()
```
The above example resolves a cluster specification for a Slurm job allocation with two computing nodes each having two GPUs and two tasks will be launched on each node. The jobs are specified in form of a dictionary where the key is a string representing job name and value is an integer that specifies the number of tasks in that job. ```cluster_resolver.cluster_spec()``` will return a cluster specificaiton object and the cluster specification will have the following specification as protobuf.

```
job {
  name: "ps"
  tasks {
    value: "t02n13:8888"
  }
}
job {
  name: "worker"
  tasks {
    value: "t02n13:8889"
  }
  tasks {
    key: 1
    value: "t02n41:8888"
  }
  tasks {
    key: 2
    value: "t02n41:8889"
  }
}
```

After calling ```cluster_resolver.cluster_spec()``` internal data structions of the resolver will be populated. By looking at the process ID/rank and comparing with cluster specification the task can 'realize' which task it belongs to. This can be retrieved by calling ```cluster_resolver.get_task_info()``` and a string specifying job name and an integer specifying the task index will be returned.

GPUs will be automatically allocated to the processes. For example in the above example ```
t02n41:8888``` will see GPU 0 and ```t02n41:8889``` will see GPU 1.
