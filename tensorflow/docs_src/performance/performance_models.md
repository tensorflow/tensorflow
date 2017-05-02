# High-Performance Models

TensorFlow is a powerful and flexible machine learning platform.
It can be used to distribute model training and inference across a large number
of machines and computation devices.

Its software stack is made of a few layers:

* a fast and powerful C++ core
* low-level Python primitives that sit right above individual kernels
* a diverse range of high-level libraries that aim to make building real models
  easier

There are many existing examples and tutorials that explain useful features in
TensorFlow.  The goal of this set of scripts is to demonstrate that we can build
flexible and powerful high-performance models using the low-level APIs.
In the future, many of the high-performance primitives will be incorporated into
high-level APIs, and made available to more users transparently.
But meanwhile, we show that it is fairly easy for advanced users to build highly
scalable models targeting different system types, network topologies, etc.

We divide our effort to build high-performance models into three categories:

1. A fast input pipeline to read data from disk, preprocess it, and make it
   ready on the GPU.
2. A high-throughput model that trains on GPU very efficiently.
3. Fast variable and gradients distribution mechanisms that scale well across
   many machines and computation devices.

## Input Pipeline

The input pipeline is the part of a TensorFlow program that reads input data,
shuffles it, and preprocesses it.

Among the most important features to build a fast input pipeline:

* Avoid using feed-dictionary to feed a large amount of data for each step.
  * Instead, use reader ops to get data into TensorFlow directly.
* Parallelize data processing.
* Use software pipelining to feed data, so that data is available immediately
  when needed.

One way to implement software pipelining in TensorFlow is through
`tf.FifoQueue`, and it is possible to parallelize data processing through
`tf.train.queue_runner`, which uses Python threads as its underlying
implementation.
This lays the foundation for the current Inception input pipeline.
This design is well built for feeding older generation of GPUs,
but the overhead of Python threads is too large to feed newer GPUs that are four
to five times faster.

In this model, we explore an alternative design that uses the native
parallelism in TensorFlow.  In our example of an image model input pipeline,
there are a few important parts:

* Choose and read the image files from the disk.
* Decode the image data into images, transform and add distortion so they are
ready to be used.
* Organize the transformed images into a minibatch.
* Transfer the images from CPU to GPU, so they are ready for model training.

It is important to note that the dominant part of each stage can happen in
parallel with that of other stages:
the file IO uses DMA to transfer the data from hard disk to memory;
image decoding, transformation and distortion are CPU-heavy;
the data transfer from CPU to GPU uses the GPU's copy-engine unit;
and the GPU kernels use the main SMs of the GPU.
It is natural to cut our pipeline into those parts so they can run in parallel
with each other.

Also, as mentioned earlier, most of the current input pipeline heavily uses
Python threads.  However, the large overhead introduced by Python threads
severely limits its scalability when the newer GPUs are a lot faster; we can
alleviate this by making a single `session.run` call execute all parts of the
pipeline.

### Parallelize IO Reads

In this new model, we use the native parallelism in TensorFlow: TensorFlow
subscribes to an eager-execution model, which means that when nodes in the graph
became available, TensorFlow will try to execute as many of them as possible.

In order to parallelize reading from hard disk, we use `data_flow_ops.RecordInput`
in this model.
Given a list of input files of TFRecords, `RecordInput` continuously reads
records using background threads, placing the records into its own large,
internal pool of records.
When it is has loaded at least half of its capacity, it produces output tensors.

Since this op has its internal threads, and is dominated by IO time that doesn’t
consume much CPU time, it naturally runs in parallel with the rest of the model.

### Parallelize Image Processing

After reading from “RecordInput”, the tensors are passed to the input processing
pipeline.  For example, if we need to feed 8 GPUs, each with a batch-size of 32,
then for each step we do the following.

First, read 32x8=256 records, and process them individually, in
parallel. This starts with 256 independent RecordInput read ops in the graph.

Then, follow each read with identical set of ops for processing. Each set is
considered independent and will execute in parallel.  The operations include
image decoding, image distortion, and resizing.

Finally, once the images are ready, they will be concatenated together into 8
batch-size 32 tensors.
Note that we can use “tf.concat” for this purpose.
However, “tf.concat” is implemented as a single op, which waits for all
the inputs to be ready, and then concatenates them together. Since all
inputs are produced in parallel, there will be a long tail waiting for all
inputs to be available; and when concatenation happens, the op becomes memory
limited as all input tensors compete for memory bandwidth.
So for the final concatenation, we use `tf.parallel_stack` instead. This
allocates an uninitialized tensor as an output, and each input tensor is written
to its designated portion of the output tensor as soon as the input is
available.  When all the input tensors are finished, the output tensor is passed
along in the graph. This effectively hides all the memory latency with the long
tail of producing all the input tensors.

### Parallelize CPU-to-GPU Data Transfer

In our example, once all the input images are processed and concatenated
together by the CPU, we have 8 tensors, each of which has a batch-size of 32.
These tensors are then to be used by the GPU for the model training.

In TensorFlow, users can use tensors from one device on any other device
directly.  TensorFlow inserts implicit copies to make the tensors available on
any devices where they are used.  The runtime schedules the copy between devices
to run before the tensors are actually used.  However, if the copy cannot finish
in time, the computation that needs those tensors will stall.

For high-performance models, it is helpful to explicitly schedule the copy ahead
of the time in parallel, so when the computation starts on GPU, all the tensors
are already available on the right device.

### Software Pipelining

With all the stages capable of being driven by different processors, we insert
`data_flow_ops.StagingArea` in between them so they run in parallel.
`StagingArea` is a queue-like operator similar to `tf.FifoQueue`.
But it offers simpler functionalities and can be executed on both CPU and GPU.

Before the model starts running all the stages, we warm up the stages in order
so the staging buffers in between all have one set of data in them.
During each run step that follows, we will run all the stages.
They read one set of data from the staging buffers at the beginning of each
stage, and push one set at end end.

For example: if there are three stages: A, B and C.
There are two staging areas in between: S1 and S2.
During the warm up, we run:

```
Warm up:
Step 1: A0
Step 2: A1  B0

Actual execution:
Step 3: A2  B1  C0
Step 4: A3  B2  C1
Step 5: A4  B3  C2
```

After the warm up, S1 and S2 each have one set of data in them.
For each step of the actual execution, one set of data is consumed from each
staging area, and one set is added to each.

There are a few nice properties about the scheme:

* All the stages are non-blocking, since the staging areas always have one set
of data after the warm up.
* Each stage can run in parallel since they can all start immediately.
* The staging buffers have a fixed memory overhead. They will have at most one
  extra set of data.
* Only a single`session.run()` call is needed to run all stages of the step,
  which makes profiling and debugging much easier.

## Best Practices in Building High-Performance Models

The computation on GPU can happen immediately since the input data have already
been transferred onto GPU when the step starts.
But it is still important to build the model that runs as fast as possible.
Here are some tips for a high-performance convolutional neural network (CNN)
model:

### Build the model with both NHWC and NCHW

Most TensorFlow operations used by a CNN support both NHWC and NCHW data format.
On GPU, NCHW is faster.
But on CPU, NHWC is sometimes faster.

So it is a good idea to build the model that can work in both ways.
Our model shows a good way to do that effectively.
For GPU training, we should always use NCHW.
But if the model needs inference on CPU, we could use NHWC; weights obtained
from training with NCHW data format can be used for inference in NHWC data
format.

### Use Fused Batch-Normalization

The default batch-normalization in TensorFlow is implemented as composite
operations.
This is very general, but often leads to suboptimal performance.
An alternative is the fused batch-normalization, and the performance on GPU
is often much faster.

## Variable Distribution and Gradient Aggregation

During training, training variable values are updated using aggregated gradients
and deltas.  In this model, we demonstrate that with the flexible and
general-purpose TensorFlow primitives, it is fairly easy to build a diverse
range of high-performance distribution and aggregation schemes for different
types of systems.

For example:

* The standard parameter-server where each replica of the training model reads
  the variables directly, and updates the variable independently.  When each
  model needs the variables, they are copied over through the standard implicit
  copies added by the TensorFlow runtime. It is shown how to use this method
  in either local training, distributed synchronous training, and distributed
  asynchronous training.
* A replicated mode for local training where each GPU has an identical
  copy of the training parameters.  The forward and backward computation can
  start immediately as the variable data is immediately available.  Gradients
  are accumulated across all GPUs, and the aggregated total is applied to
  each GPU's copy of the variables so that they stay in sync.
* A distributed replicated mode of training where each GPU has an identical copy
  of the training parameters, and a master copy of the variables is stored
  on the parameter-servers.  The forward and backward computation can
  start immediately as the variable data is immediately available.  Gradients
  are accumulated across all GPUs on each server and then the per-server
  aggregated gradients are applied to the master copy. After all workers do
  this, each worker updates its copy of the variable from the master copy.

We show that most of the variable distribution and aggregation subsystem can
be implemented through TensorFlow low-level primitives with manageable
complexity at the model level. Here we discuss some more details.

### Parameter-server Variables

The most common way trainable variables are managed in TensorFlow models is the
parameter server mode.

In a distributed system, this means that each worker process runs the same
model, and parameter server processes own the master copies of the variables.
When a worker needs a variable from a parameter server, it refers to it
directly.  The TensorFlow runtime adds implicit copies to the graph to make the
variable value available on the computation device that needs it. When a
gradient is computed on a worker, it is sent to the parameter server that owns
the particular variable, and the corresponding optimizer is used to update the
variable.

There are some techniques to improve throughput:

* The variables are spread among parameter servers based on their size, for load
  balancing.
* When each worker has multiple GPUs, gradients are accumulated across the GPUs
  and a single aggregated gradient is sent to the parameter server. This reduces
  the network bandwidth and the amount of work done by the parameter servers.

For coordinating between workers, a very common mode is async updates, where
each worker updates the master copy of the variables without synchronizing with
other workers.  In our model, we demonstrate that it is fairly easy to introduce
synchronization across workers so updates for all workers are finished in one
step before the next step can start.

The parameter-server method can also be used for local training, In this case,
instead of spreading the master copies of variables across parameters servers,
they are either on the CPU or spread across the available GPUs.

Due to the simple nature of this setup, this architecture has gained a lot of
popularity within the community.

This is available in the benchmark scripts as the 'parameter_server'
variable_update mode.

![parameter_server mode in distributed
training](../images/perf_parameter_server_mode_doc.png){
width="900" style="max-width: inherit"}

### Replicated Variables

In this design, each GPU on the server has its own copy of each variable. The
values are kept in sync across GPUs by applying the fully aggregated gradient to
each GPU's copy of the variable.

The variables and data are available at the start of training, so the forward
pass of training can start immediately. Gradients are aggregated across the
devices and the fully aggregated gradient is then applied to each local copy.

Gradient aggregation across the server can be done in different ways:

* Using standard TensorFlow operations to accumulate the total on a single
  device (CPU or GPU) and then copy it back to all GPUs.
* Using NVIDIA NCCL, described below in the NCCL section.

This is available in the benchmark scripts for local execution only, as the
'replicated' variable_update mode.

### Replicated Variables in Distributed Training

The replicated method for variables can be extended to distributed training.
One way to do this like the replicated mode: aggregate the gradients fully
across the cluster and apply them to each local copy of the variable. This may
be shown in a future version of this scripts; the scripts do present a different
variation, described here.

In this mode, in addition to each GPU's copy of the variables, a master copy is
stored on the parameter servers. As with the replicated mode, training can start
immediately using the local copies of the variables.

As the gradients of the weights become available, they are sent back to the
parameter servers and all local copies are updated:

1. All the gradients from the GPU on the same worker are aggregated together.
2. Aggregated gradients from each worker are sent to the parameter server that
   owns the variable, where the specified optimizer is used to update the
   master copy of the variable.
3. Each worker updates its local copy of the variable from the master. In
   the example model, this is done with a cross-replica barrier that waits for
   all the workers to finish updating the variables, and fetches the new
   variable only after the barrier has been released by all replicas.  Once the
   copy finishes for all variables, this marks the end of a training step, and a
   new step can start.

Although this sounds similar to the standard use of parameter servers, the
performance is often better in many cases.  This is largely due to the fact the
computation can happen without any delay, and much of the copy latency of early
gradients can be hidden by later computation layers.

This is available in the benchmark scripts as the 'distributed_replicated'
variable_update mode.

![distributed_replicated mode](
../images/perf_distributed_replicated_mode_doc.png){
width="900" style="max-width: inherit"}

#### NCCL

In order to broadcast variables and aggregate gradients across different GPUs
within the same host machine, we can use the default TensorFlow implicit copy
mechanism.

However, we can instead use the optional NCCL support.  NCCL is an NVIDIA
library that can efficiently broadcast and aggregate data across different GPUs.
It schedules a cooperating kernel on each GPU that knows how to best utilize the
underlying hardware topology; this kernel uses a single SM of the GPU.

In our experiment, we demonstrate that although NCCL often leads to much faster
data aggregation by itself, it doesn't necessarily lead to faster training.  Our
hypothesis is that the implicit copies are essentially free since they go to the
copy engine on GPU, as long as its latency can be hidden by the main computation
itself.  Although NCCL can transfer data faster, it takes one SM away, and adds
more pressure to the underlying L2 cache.  Our results show that for 8-GPUs,
NCCL often leads to better performance.  However, for fewer GPUs, the implicit
copies often perform better.

#### Staged Variables

We further introduce a staged-variable mode where we use staging areas for both
the variable reads, and their updates.
Similar to software pipelining of the input pipeline, this can hide the data
copy latency.
If the computation time takes longer than the copy and aggregation, the copy
itself becomes essentially free.

The downside is that all the weights read are from the previous training step.
So it is a different algorithm from SGD.
But it is possible to improve its convergence by adjusting learning rate and
other hyperparameters.

## Conclusions

In this high-performance model, we present a number of options to build
high-performance models in TensorFlow.
Due to the flexible design in TensorFlow, advanced features like this often
requires no system-level changes, and can be largely achieved through
model-level changes.

We do not claim which combination works best for a particular model.
That should be left to the engineers who build the model and the training system.
Many of the ingredients of the high-performance model will find their ways
to high-level primitives that become transparent to users.
However, we have shown that advanced users can easily tune and modify the
underlying model behavior using low-level primitives.
This could be very useful when improving performance for particular system
setups and model configurations.
