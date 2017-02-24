# Distributed TensorFlow

This document shows how to create a cluster of TensorFlow servers, and how to
distribute a computation graph across that cluster. We assume that you are
familiar with the [basic concepts](../../get_started/basic_usage.md) of
writing TensorFlow programs.

## Hello distributed TensorFlow!

To see a simple TensorFlow cluster in action, execute the following:

```shell
# Start a TensorFlow server as a single-process "cluster".
$ python
>>> import tensorflow as tf
>>> c = tf.constant("Hello, distributed TensorFlow!")
>>> server = tf.train.Server.create_local_server()
>>> sess = tf.Session(server.target)  # Create a session on the server.
>>> sess.run(c)
'Hello, distributed TensorFlow!'
```

The
[`tf.train.Server.create_local_server()`](../../api_docs/python/train.md#Server.create_local_server)
method creates a single-process cluster, with an in-process server.

## Create a cluster

A TensorFlow "cluster" is a set of "tasks" that participate in the distributed
execution of a TensorFlow graph. Each task is associated with a TensorFlow
"server", which contains a "master" that can be used to create sessions, and a
"worker" that executes operations in the graph.  A cluster can also be divided
into one or more "jobs", where each job contains one or more tasks.

To create a cluster, you start one TensorFlow server per task in the cluster.
Each task typically runs on a different machine, but you can run multiple tasks
on the same machine (e.g. to control different GPU devices). In each task, do
the following:

1.  **Create a `tf.train.ClusterSpec`** that describes all of the tasks
    in the cluster. This should be the same for each task.

2.  **Create a `tf.train.Server`**, passing the `tf.train.ClusterSpec` to
    the constructor, and identifying the local task with a job name
    and task index.


### Create a `tf.train.ClusterSpec` to describe the cluster

The cluster specification dictionary maps job names to lists of network
adresses. Pass this dictionary to
the [`tf.train.ClusterSpec`](../../api_docs/python/train.md#ClusterSpec)
constructor.  For example:

<table>
  <tr><th><code>tf.train.ClusterSpec</code> construction</th><th>Available tasks</th>
  <tr>
    <td><pre>
tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
</pre></td>
<td><code>/job:local/task:0<br/>/job:local/task:1</code></td>
  </tr>
  <tr>
    <td><pre>
tf.train.ClusterSpec({
    "worker": [
        "worker0.example.com:2222",
        "worker1.example.com:2222",
        "worker2.example.com:2222"
    ],
    "ps": [
        "ps0.example.com:2222",
        "ps1.example.com:2222"
    ]})
</pre></td><td><code>/job:worker/task:0</code><br/><code>/job:worker/task:1</code><br/><code>/job:worker/task:2</code><br/><code>/job:ps/task:0</code><br/><code>/job:ps/task:1</code></td>
  </tr>
</table>

### Create a `tf.train.Server` instance in each task

A [`tf.train.Server`](../../api_docs/python/train.md#Server) object contains a
set of local devices, a set of connections to other tasks in its
`tf.train.ClusterSpec`, and a
["session target"](../../api_docs/python/client.md#Session) that can use these
to perform a distributed computation. Each server is a member of a specific
named job and has a task index within that job.  A server can communicate with
any other server in the cluster.

For example, to launch a cluster with two servers running on `localhost:2222`
and `localhost:2223`, run the following snippets in two different processes on
the local machine:

```python
# In task 0:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=0)
```
```python
# In task 1:
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=1)
```

**Note:** Manually specifying these cluster specifications can be tedious,
especially for large clusters. We are working on tools for launching tasks
programmatically, e.g. using a cluster manager like
[Kubernetes](http://kubernetes.io). If there are particular cluster managers for
which you'd like to see support, please raise a
[GitHub issue](https://github.com/tensorflow/tensorflow/issues).

## Specifying distributed devices in your model

To place operations on a particular process, you can use the same
[`tf.device()`](../../api_docs/python/framework.md#device)
function that is used to specify whether ops run on the CPU or GPU. For example:

```python
with tf.device("/job:ps/task:0"):
  weights_1 = tf.Variable(...)
  biases_1 = tf.Variable(...)

with tf.device("/job:ps/task:1"):
  weights_2 = tf.Variable(...)
  biases_2 = tf.Variable(...)

with tf.device("/job:worker/task:7"):
  input, labels = ...
  layer_1 = tf.nn.relu(tf.matmul(input, weights_1) + biases_1)
  logits = tf.nn.relu(tf.matmul(layer_1, weights_2) + biases_2)
  # ...
  train_op = ...

with tf.Session("grpc://worker7.example.com:2222") as sess:
  for _ in range(10000):
    sess.run(train_op)
```

In the above example, the variables are created on two tasks in the `ps` job,
and the compute-intensive part of the model is created in the `worker`
job. TensorFlow will insert the appropriate data transfers between the jobs
(from `ps` to `worker` for the forward pass, and from `worker` to `ps` for
applying gradients).

## Replicated training

A common training configuration, called "data parallelism," involves multiple
tasks in a `worker` job training the same model on different mini-batches of
data, updating shared parameters hosted in one or more tasks in a `ps`
job. All tasks typically run on different machines. There are many ways to
specify this structure in TensorFlow, and we are building libraries that will
simplify the work of specifying a replicated model. Possible approaches include:

* **In-graph replication.** In this approach, the client builds a single
  `tf.Graph` that contains one set of parameters (in `tf.Variable` nodes pinned
  to `/job:ps`); and multiple copies of the compute-intensive part of the model,
  each pinned to a different task in `/job:worker`.

* **Between-graph replication.** In this approach, there is a separate client
  for each `/job:worker` task, typically in the same process as the worker
  task. Each client builds a similar graph containing the parameters (pinned to
  `/job:ps` as before using
  [`tf.train.replica_device_setter()`](../../api_docs/python/train.md#replica_device_setter)
  to map them deterministically to the same tasks); and a single copy of the
  compute-intensive part of the model, pinned to the local task in
  `/job:worker`.

* **Asynchronous training.** In this approach, each replica of the graph has an
  independent training loop that executes without coordination. It is compatible
  with both forms of replication above.

* **Synchronous training.** In this approach, all of the replicas read the same
  values for the current parameters, compute gradients in parallel, and then
  apply them together. It is compatible with in-graph replication (e.g. using
  gradient averaging as in the
  [CIFAR-10 multi-GPU trainer](https://www.tensorflow.org/code/tensorflow_models/tutorials/image/cifar10/cifar10_multi_gpu_train.py)),
  and between-graph replication (e.g. using the
  [`tf.train.SyncReplicasOptimizer`](../../api_docs/python/train.md#SyncReplicasOptimizer)).

### Putting it all together: example trainer program

The following code shows the skeleton of a distributed trainer program,
implementing **between-graph replication** and **asynchronous training**. It
includes the code for the parameter server and worker tasks.

```python
import argparse
import sys

import tensorflow as tf

FLAGS = None


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):

      # Build model...
      loss = ...
      global_step = tf.contrib.framework.get_or_create_global_step()

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    hooks=[tf.train.StopAtStepHook(last_step=1000000)]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(FLAGS.task_index == 0),
                                           checkpoint_dir="/tmp/train_logs",
                                           hooks=hooks) as mon_sess:
      while not mon_sess.should_stop():
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # mon_sess.run handles AbortedError in case of preempted PS.
        mon_sess.run(train_op)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--worker_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
  )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
  )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
```

To start the trainer with two parameter servers and two workers, use the
following command line (assuming the script is called `trainer.py`):

```shell
# On ps0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=0
# On ps1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=ps --task_index=1
# On worker0.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=0
# On worker1.example.com:
$ python trainer.py \
     --ps_hosts=ps0.example.com:2222,ps1.example.com:2222 \
     --worker_hosts=worker0.example.com:2222,worker1.example.com:2222 \
     --job_name=worker --task_index=1
```

## Glossary

**Client**

A client is typically a program that builds a TensorFlow graph and constructs a
`tensorflow::Session` to interact with a cluster. Clients are typically written
in Python or C++. A single client process can directly interact with multiple
TensorFlow servers (see "Replicated training" above), and a single server can
serve multiple clients.

**Cluster**

A TensorFlow cluster comprises a one or more "jobs", each divided into lists of
one or more "tasks". A cluster is typically dedicated to a particular high-level
objective, such as training a neural network, using many machines in parallel. A
cluster is defined by
a [`tf.train.ClusterSpec`](../../api_docs/python/train.md#ClusterSpec) object.

**Job**

A job comprises a list of "tasks", which typically serve a common purpose.
For example, a job named `ps` (for "parameter server") typically hosts nodes
that store and update variables; while a job named `worker` typically hosts
stateless nodes that perform compute-intensive tasks. The tasks in a job
typically run on different machines. The set of job roles is flexible:
for example, a `worker` may maintain some state.

**Master service**

An RPC service that provides remote access to a set of distributed devices,
and acts as a session target. The master service implements the
`tensorflow::Session` interface, and is responsible for coordinating work across
one or more "worker services". All TensorFlow servers implement the master
service.

**Task**

A task corresponds to a specific TensorFlow server, and typically corresponds
to a single process. A task belongs to a particular "job" and is identified by
its index within that job's list of tasks.

**TensorFlow server** A process running
a [`tf.train.Server`](../../api_docs/python/train.md#Server) instance, which is
a member of a cluster, and exports a "master service" and "worker service".

**Worker service**

An RPC service that executes parts of a TensorFlow graph using its local devices.
A worker service implements [worker_service.proto](https://www.tensorflow.org/code/tensorflow/core/protobuf/worker_service.proto).
All TensorFlow servers implement the worker service.
