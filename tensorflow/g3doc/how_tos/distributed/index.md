# Distributed TensorFlow

This document shows how to create a cluster of TensorFlow servers, and how to
distribute a computation graph across that cluster. We assume that you are
familiar with the [basic concepts](../../get_started/basic_usage.md) of
writing TensorFlow programs.

## Hello distributed TensorFlow!

This tutorial assumes that you are using a TensorFlow nightly build. You
can test your installation by starting a local server as follows:

```shell
# Start a TensorFlow server as a single-process "cluster".
$ python
>>> import tensorflow as tf
>>> c = tf.constant("Hello, distributed TensorFlow!")
>>> server = tf.train.Server.create_local_server()
>>> sess = tf.Session(server.target)
>>> sess.run(c)
'Hello, distributed TensorFlow!'
```

The
[`tf.train.Server.create_local_server()`](../../api_docs/train.md#Server.create_local_server)
method creates a single-process cluster.

## Create a cluster

Most clusters have multiple tasks, divided into one or more jobs.  To create a
cluster with multiple processes or machines:

1.  **For each process or machine** in the cluster, run a TensorFlow program to
    do the following:

    1.  **Create a `tf.train.ClusterSpec`**, which describes all of the tasks
        in the cluster. This should be the same in each process.

    1.  **Create a `tf.train.Server`**, passing the `tf.train.ClusterSpec` to
        the constructor, and identifying the local process with a job name
        and task index.


### Create a `tf.train.ClusterSpec` to describe the cluster

The cluster specification dictionary maps job names to lists of network
adresses. Pass this dictionary to the `tf.train.ClusterSpec` constructor.  For
example:

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

### Create a `tf.train.Server` instance in each process

A [`tf.train.Server`](../../api_docs/python/train.md#Server) object contains a
set of local devices, and a
[`tf.Session`](../../api_docs/python/client.md#Session) target that can
participate in a distributed computation. Each server belongs to a particular
cluster (specified by a `tf.train.ClusterSpec`), and corresponds to a particular
task in a named job. The server can communicate with any other server in the
same cluster.

For example, to define and instantiate servers running on `localhost:2222` and
`localhost:2223`, run the following snippets in different processes:

```python
# In task 0:
cluster = tf.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.GrpcServer(cluster, job_name="local", task_index=0)
```
```python
# In task 1:
cluster = tf.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.GrpcServer(cluster, job_name="local", task_index=1)
```

**Note:** Manually specifying these cluster specifications can be tedious,
especially for large clusters. We are working on tools for launching tasks
programmatically, e.g. using a cluster manager like
[Kubernetes](http://kubernetes.io). If there are particular cluster managers for
which you'd like to see support, please raise a
[GitHub issue](https://github.com/tensorflow/tensorflow/issues).

## Specifying distributed devices in your model

To place operations on a particular process, you can use the same
[`tf.device()`](https://www.tensorflow.org/versions/master/api_docs/python/framework.html#device)
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

A common training configuration ("data parallel training") involves multiple
tasks in a `worker` job training the same model, using shared parameters hosted
in a one or more tasks in a `ps` job. Each task will typically run on a
different machine. There are many ways to specify this structure in TensorFlow,
and we are building libraries that will simplify the work of specifying a
replicated model. Possible approaches include:

* Building a single graph containing one set of parameters (in `tf.Variable`
  nodes pinned to `/job:ps`), and multiple copies of the "model" pinned to
  different tasks in `/job:worker`. Each copy of the model can have a different
  `train_op`, and one or more client threads can call `sess.run(train_ops[i])`
  for each worker `i`. This implements *asynchronous* training.

  This approach uses a single `tf.Session` whose target is one of the workers in
  the cluster.

* As above, but where the gradients from all workers are averaged. See the
  [CIFAR-10 multi-GPU trainer](https://www.tensorflow.org/code/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py)
  for an example of this form of replication. This implements *synchronous*
  training.

* The "distributed trainer" approach uses multiple graphs&mdash;one per
  worker&mdash;where each graph contains one set of parameters (pinned to
  `/job:ps`) and one copy of the model (pinned to a particular
  `/job:worker/task:i`). The "container" mechanism is used to share variables
  between different graphs: when each variable is constructed, the optional
  `container` argument is specified with the same value in each copy of the
  graph. For large models, this can be more efficient, because the overall graph
  is smaller.

  This approach uses multiple `tf.Session` objects: one per worker process,
  where the `target` of each is the address of a different worker. The
  `tf.Session` objects can all be created in a single Python client, or you can
  use multiple Python clients to better distribute the trainer load.

### Putting it all together: example trainer program

The following code shows the skeleton of a distributed trainer program. It
includes the code for the parameter server and worker processes.

```python
import tensorflow as tf

# Flags for defining the tf.train.ClusterSpec
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")

# Flags for defining the tf.train.Server
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

FLAGS = tf.app.flags.FLAGS


def main(_):
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts(",")

  cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
  server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=task_index)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":
    
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:worker/task:%d" % FLAGS.task_index,
        cluster=cluster)):
        
      # Build model...
      loss = ...
      global_step = tf.Variable(0)

      train_op = tf.train.AdagradOptimizer(0.01).minimize(
          loss, global_step=global_step)

      saver = tf.train.Saver()
      summary_op = tf.merge_all_summaries()
      init_op = tf.initialize_all_variables()

    # Create a "supervisor", which oversees the training process.
    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                             logdir="/tmp/train_logs",
                             init_op=init_op,
                             summary_op=summary_op,
                             saver=saver,
                             global_step=global_step,
                             save_model_secs=600)

    # The supervisor takes care of session initialization and restoring from
    # a checkpoint.
    sess = sv.prepare_or_wait_for_session(server.target)

    # Start queue runners for the input pipelines (if any).
    sv.start_queue_runners(sess)
    
    # Loop until the supervisor shuts down (or 1000000 steps have completed).
    step = 0
    while not sv.should_stop() and step < 1000000:
      # Run a training step asynchronously.
      # See `tf.train.SyncReplicasOptimizer` for additional details on how to
      # perform *synchronous* training.
      _, step = sess.run([train_op, global_step])


if __name__ == "__main__":
  tf.app.run()
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

<dl>
  <dt>Client</dt>
  <dd>
    A client is typically a program that builds a TensorFlow graph and
    constructs a `tensorflow::Session` to interact with a cluster. Clients are
    typically written in Python or C++. A single client process can directly
    interact with multiple TensorFlow servers (see "Replicated training" above),
    and a single server can serve multiple clients.
  </dd>
  <dt>Cluster</dt>
  <dd>
    A TensorFlow cluster comprises one or more TensorFlow servers, divided into
    a set of named jobs, which in turn comprise lists of tasks. A cluster is
    typically dedicated to a particular high-level objective, such as training a
    neural network, using many machines in parallel.
  </dd>
  <dt>Job</dt>
  <dd>
    A job comprises a list of "tasks", which typically serve a common
    purpose. For example, a job named `ps` (for "parameter server") typically
    hosts nodes that store and update variables; while a job named `worker`
    typically hosts stateless nodes that perform compute-intensive tasks.
    The tasks in a job typically run on different machines.
  </dd>
  <dt>Master service</dt>
  <dd>
    An RPC service that provides remote access to a set of distributed
    devices. The master service implements the <code>tensorflow::Session</code>
    interface, and is responsible for coordinating work across one or more
    "worker services".
  </dd>
  <dt>Task</dt>
  <dd>
    A task typically corresponds to a single TensorFlow server process,
    belonging to a particular "job" and with a particular index within that
    job's list of tasks.
  </dd>
  <dt>TensorFlow server</dt>
  <dd>
    A process running a <code>tf.train.Server</code> instance, which is a
    member of a cluster, and exports a "master service" and "worker service".
  </dd>
  <dt>Worker service</dt>
  <dd>
    An RPC service that executes parts of a TensorFlow graph using its local
    devices. A worker service implements <a href=
    "https://www.tensorflow.org/code/tensorflow/core/protobuf/worker_service.proto"
    ><code>worker_service.proto</code></a>.
  </dd>
</dl>
