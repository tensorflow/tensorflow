Represents a cluster as a set of "tasks", organized into "jobs".

A `tf.train.ClusterSpec` represents the set of processes that
participate in a distributed TensorFlow computation. Every
[`tf.train.Server`](#Server) is constructed in a particular cluster.

To create a cluster with two jobs and five tasks, you specify the
mapping from job names to lists of network addresses (typically
hostname-port pairs).

```
cluster = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                           "worker1.example.com:2222",
                                           "worker2.example.com:2222"],
                                "ps": ["ps0.example.com:2222",
                                       "ps1.example.com:2222"]})
```

- - -

#### `tf.train.ClusterSpec.as_cluster_def()` {#ClusterSpec.as_cluster_def}

Returns a `tf.train.ClusterDef` protocol buffer based on this cluster.


- - -

#### `tf.train.ClusterSpec.as_dict()` {#ClusterSpec.as_dict}

Returns a dictionary from job names to lists of network addresses.



#### Other Methods
- - -

#### `tf.train.ClusterSpec.__init__(cluster)` {#ClusterSpec.__init__}

Creates a `ClusterSpec`.

##### Args:


*  <b>`cluster`</b>: A dictionary mapping one or more job names to lists of network
    addresses, or a `tf.train.ClusterDef` protocol buffer.

##### Raises:


*  <b>`TypeError`</b>: If `cluster` is not a dictionary mapping strings to lists
    of strings, and not a `tf.train.ClusterDef` protobuf.


- - -

#### `tf.train.ClusterSpec.job_tasks(job_name)` {#ClusterSpec.job_tasks}

Returns a list of tasks in the given job.

##### Args:


*  <b>`job_name`</b>: The string name of a job in this cluster.

##### Returns:

  A list of strings, corresponding to the network addresses of tasks in
  the given job, ordered by task index.

##### Raises:


*  <b>`ValueError`</b>: If `job_name` does not name a job in this cluster.


- - -

#### `tf.train.ClusterSpec.jobs` {#ClusterSpec.jobs}

Returns a list of job names in this cluster.

##### Returns:

  A list of strings, corresponding to the names of jobs in this cluster.


