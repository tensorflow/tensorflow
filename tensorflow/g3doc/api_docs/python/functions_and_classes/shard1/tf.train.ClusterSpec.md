Represents a cluster as a set of "tasks", organized into "jobs".

A `tf.train.ClusterSpec` represents the set of processes that
participate in a distributed TensorFlow computation. Every
[`tf.train.Server`](#Server) is constructed in a particular cluster.

To create a cluster with two jobs and five tasks, you specify the
mapping from job names to lists of network addresses (typically
hostname-port pairs).

```python
cluster = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                           "worker1.example.com:2222",
                                           "worker2.example.com:2222"],
                                "ps": ["ps0.example.com:2222",
                                       "ps1.example.com:2222"]})
```

Each job may also be specified as a sparse mapping from task indices
to network addresses. This enables a server to be configured without
needing to know the identity of (for example) all other worker
tasks:

```python
cluster = tf.train.ClusterSpec({"worker": {1: "worker1.example.com:2222"},
                                "ps": ["ps0.example.com:2222",
                                       "ps1.example.com:2222"]})
```

- - -

#### `tf.train.ClusterSpec.as_cluster_def()` {#ClusterSpec.as_cluster_def}

Returns a `tf.train.ClusterDef` protocol buffer based on this cluster.


- - -

#### `tf.train.ClusterSpec.as_dict()` {#ClusterSpec.as_dict}

Returns a dictionary from job names to their tasks.

For each job, if the task index space is dense, the corresponding
value will be a list of network addresses; otherwise it will be a
dictionary mapping (sparse) task indices to the corresponding
addresses.

##### Returns:

  A dictionary mapping job names to lists or dictionaries
  describing the tasks in those jobs.



#### Other Methods
- - -

#### `tf.train.ClusterSpec.__bool__()` {#ClusterSpec.__bool__}




- - -

#### `tf.train.ClusterSpec.__eq__(other)` {#ClusterSpec.__eq__}




- - -

#### `tf.train.ClusterSpec.__init__(cluster)` {#ClusterSpec.__init__}

Creates a `ClusterSpec`.

##### Args:


*  <b>`cluster`</b>: A dictionary mapping one or more job names to (i) a
    list of network addresses, or (ii) a dictionary mapping integer
    task indices to network addresses; or a `tf.train.ClusterDef`
    protocol buffer.

##### Raises:


*  <b>`TypeError`</b>: If `cluster` is not a dictionary mapping strings to lists
    of strings, and not a `tf.train.ClusterDef` protobuf.


- - -

#### `tf.train.ClusterSpec.__ne__(other)` {#ClusterSpec.__ne__}




- - -

#### `tf.train.ClusterSpec.__nonzero__()` {#ClusterSpec.__nonzero__}




- - -

#### `tf.train.ClusterSpec.job_tasks(job_name)` {#ClusterSpec.job_tasks}

Returns a mapping from task ID to address in the given job.

NOTE: For backwards compatibility, this method returns a list. If
the given job was defined with a sparse set of task indices, the
length of this list may not reflect the number of tasks defined in
this job. Use the [`num_tasks()`](#ClusterSpec.num_tasks) method
to find the number of tasks defined in a particular job.

##### Args:


*  <b>`job_name`</b>: The string name of a job in this cluster.

##### Returns:

  A list of task addresses, where the index in the list
  corresponds to the task index of each task. The list may contain
  `None` if the job was defined with a sparse set of task indices.

##### Raises:


*  <b>`ValueError`</b>: If `job_name` does not name a job in this cluster.


- - -

#### `tf.train.ClusterSpec.jobs` {#ClusterSpec.jobs}

Returns a list of job names in this cluster.

##### Returns:

  A list of strings, corresponding to the names of jobs in this cluster.


- - -

#### `tf.train.ClusterSpec.num_tasks(job_name)` {#ClusterSpec.num_tasks}

Returns the number of tasks defined in the given job.

##### Args:


*  <b>`job_name`</b>: The string name of a job in this cluster.

##### Returns:

  The number of tasks defined in the given job.

##### Raises:


*  <b>`ValueError`</b>: If `job_name` does not name a job in this cluster.


- - -

#### `tf.train.ClusterSpec.task_address(job_name, task_index)` {#ClusterSpec.task_address}

Returns the address of the given task in the given job.

##### Args:


*  <b>`job_name`</b>: The string name of a job in this cluster.
*  <b>`task_index`</b>: A non-negative integer.

##### Returns:

  The address of the given task in the given job.

##### Raises:


*  <b>`ValueError`</b>: If `job_name` does not name a job in this cluster,
  or no task with index `task_index` is defined in that job.


- - -

#### `tf.train.ClusterSpec.task_indices(job_name)` {#ClusterSpec.task_indices}

Returns a list of valid task indices in the given job.

##### Args:


*  <b>`job_name`</b>: The string name of a job in this cluster.

##### Returns:

  A list of valid task indices in the given job.

##### Raises:


*  <b>`ValueError`</b>: If `job_name` does not name a job in this cluster,
  or no task with index `task_index` is defined in that job.


