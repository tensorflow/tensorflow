# TensorFlow Verbs Quick-Start

## Intro
This document provides a starting point for the ROCm support of the [community-contributed Verbs RDMA module](../tensorflow/contrib/verbs/README.md) for TensorFlow.

This document assumes you understand the steps of the [Basic Installation](tensorflow-install-basic.md#install-rocm) as well as how to [Build From Source](tensorflow-build-from-source.md).

When building TensorFlow from source, you enable the Verbs module by

- adding `--config=verbs` to your bazel build command, and
- you must have the OFED headers and libraries installed, e.g., verbs.h, libibverbs.so.

It is outside the scope of this document to explain the hardware and software setup required for InfiniBand clusters.  There are numerous resources available online.  Hereafter, we assume your systems are appropriately setup.  The remainder of this document details how to run a distributed TensorFlow application [TF CNN Benchmarks] while enabling high-speed Verbs communication.

## Distributed TensorFlow

Please read and understand the [official Distributed TensorFlow example](https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/distributed.md).

To enable Verbs at runtime, when you construct your `tf.train.Server` instances, you can specify the server protocol as a parameter to the constructor.  The default is "grpc", the google RPC service.  Change this to "grpc+verbs".  For example:

```python
cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name="local", task_index=0, protocol="grpc+verbs")
```

## Distributed TensorFlow CNN Benchmarks

The [TF CNN Benchmarks] have a good, accessible example of distributed training using verbs.  Please refer to their README for additional command-line parameters.  The following describes the command-line parameters specific for running distributed TensorFlow with verbs support.

The `ClusterSpec` you learned about from the [suggested documentation](https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/distributed.md#create-a-tftrainclusterspec-to-describe-the-cluster) above can be specified on the command-line as arguments to the benchmark script.  For example, the following cluster spec uses three hosts, placing one worker on each and co-locating one parameter server on the first two hosts:

```
tf.train.ClusterSpec({
    "worker": [
        "HostA:2222",
        "HostB:2222",
        "HostC:2222"
    ],
    "ps": [
        "HostA:2223",
        "HostB:2223"
    ]})
```

Becomes this:

```bash
python tf_cnn_benchmarks.py --ps_hosts=HostA:2223,HostB:2223 --worker_hosts=HostA:2222,HostB:2222,HostC:2222
```

You would run two instances of the benchmark on HostA (one for the `ps` process and one for the `worker`), two on HostB, and one on HostC.  You add additional command-line parameters to tell the benchmark instance whether it is a `ps` or a `worker` task, and which task ID to associate it with.  In the above case, you would use

```
--job_name=ps --task_index=0 # On HostA
--job_name=ps --task_index=1 # On HostB
--job_name=worker --task_index=0 # On HostA
--job_name=worker --task_index=1 # On HostB
--job_name=worker --task_index=2 # On HostC
```

The number of parameter servers to use will only impact performance; you must experiment until you find the ideal cluster spec for your particular model and input.  In general, efficient distributed training is an ongoing research problem.

### Server Protocol grpc+verbs
The above cluster spec setup via the command-line will run using the default "grpc" Google RPC protocol.  To instead enable verbs, specify the protocol using `--server_protocol=grpc+verbs`.  You must repeat all of these command-line settings, except for `job_name` and `task_index`, for each instance of the benchmark you run.

### Manually Killing Parameter Server Processes
When running either using the default grpc or verbs protocol, the parameter server processes will not terminate on their own.  This is a known issue with the upstream benchmarks.  You will need to manually kill those processes.  If you ran them in their own terminal sessions, you can press `Ctrl+Z` to pause the server and then within the same terminal run `kill %1` to kill the last running background process (the server you just paused).

[TF CNN Benchmarks]: https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks
