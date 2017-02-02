# Using JIT Compilation

> Note: TensorFlow must be compiled from source to include XLA.

## Why use just-in-time (JIT) compilation?

The TensorFlow/XLA JIT compiler compiles and runs parts of TensorFlow graphs via
XLA. The benefit of this over the standard TensorFlow implementation is that XLA
can fuse multiple operators (kernel fusion) into a small number of compiled
kernels. Fusing operators can reduce memory bandwidth requirements and improve
performance compared to executing operators one-at-a-time, as the TensorFlow
executor does.

## Running TensorFlow graphs via XLA

There are two ways to run TensorFlow computations via XLA, either by
JIT-compiling operators placed on a CPU or GPU device, or by placing operators
on the `XLA_CPU` or `XLA_GPU` TensorFlow devices. Placing operators directly on
a TensorFlow XLA device forces the operator to run on that device and is mainly
used for testing.

> Note: The XLA CPU backend produces fast single-threaded code (in most cases),
> but does not yet parallelize as well as the TensorFlow CPU backend. The XLA
> GPU backend is competitive with the standard TensorFlow implementation,
> sometimes faster, sometimes slower.

### Turning on JIT compilation

JIT compilation can be turned on at the session level or manually for select
operations. Both of these approaches are zero-copy --- data does not need to be
copied when passing data between a compiled XLA kernel and a TensorFlow operator
placed on the same device.

#### Session

Turning on JIT compilation at the session level will result in all possible
operators being greedily compiled into XLA computations. Each XLA computation
will be compiled into one or more kernels for the underlying device. (This does
not mean everything will be fused into a single CUDA kernel, for example.)

Subject to a few constraints, if there are two adjacent operators in the graph
that both have XLA implementations, then they will be compiled into a single XLA
computation.

JIT compilation is turned on at the session level by setting the
`global_jit_level` config to `tf.OptimizerOptions.ON_1` and passing the config
during session initialization.

```python
# Config to turn on JIT compilation
config = tf.ConfigProto()
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

sess = tf.Session(config=config)
```

#### Manual

JIT compilation can also be turned on manually for one or more operators. This
is done by tagging the operators to compile with the attribute
`_XlaCompile=true`. The simplest way to do this is via the
`tf.contrib.compiler.jit.experimental_jit_scope()` scope defined in
[`tensorflow/contrib/compiler/jit.py`](https://www.tensorflow.org/code/tensorflow/contrib/compiler/jit.py).
Example usage:

```python
    jit_scope = tf.contrib.compiler.jit.experimental_jit_scope

    x = tf.placeholder(np.float32)
    with jit_scope():
      y = tf.add(x, x)  # The "add" will be compiled with XLA.
```

The `_XlaCompile` attribute is currently supported on a best-effort basis. If an
operator cannot be compiled, TensorFlow will silently fall back to the normal
implementation.

### Placing operators on XLA devices

Another way to run computations via XLA is to place an operator on a specific
XLA device. This method is normally only used for testing. Valid targets are
`XLA_CPU` or `XLA_GPU`.

```python
with tf.device("/job:localhost/replica:0/task:0/device:XLA_GPU:0"):
  output = tf.add(input1, input2)
```

Unlike JIT compilation on the standard CPU and GPU devices, these devices make a
copy of data when it is transferred on and off the device. The extra copy makes
it expensive to mix XLA and TensorFlow operators in the same graph.

## Tutorial

This tutorial covers training a simple version of MNIST softmax with JIT turned
on. While, the tutorial was created using CPU only, the steps are the same and
the artifacts are similar to running on GPU.

### Step #1: Prepare sample script

Download or move
[mnist_softmax_xla.py](https://www.tensorflow.org/code/tensorflow/examples/tutorials/mnist/mnist_softmax_xla.py)
into a folder outside of the TensorFlow source tree.

### Step #2: Run without XLA

Execute the python script to train the model without XLA.

```shell
python mnist_softmax_xla.py --xla=false
```

Using the Chrome Trace Event Profiler (browse to chrome://tracing),
open the timeline file created when the script finishes: `timeline.ctf.json`.
The rendered timeline should look similar to the picture below with multiple
green boxes labeled `MatMul`, possibly across multiple CPUs.
<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/jit_timeline_cpu.png">
</div>

### Step #3 Run with XLA

Execute the python script to train the model with XLA and turn on a debugging
feature of XLA via an environmental variable that outputs the XLA graph.

```shell
TF_XLA_FLAGS=--xla_generate_hlo_graph=.* python mnist_softmax_xla.py
```

Open the timeline file created (`timeline.ctf.json`).  The rendered timeline
should look similar to the picture below with one long bar labeled `_XlaLaunch`.
<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/jit_timeline_cpu_xla.png">
</div>

To understand what is happening in `_XlaLaunch`, look at the console output for
statements similar to the following:

```shell
computation cluster_0[_XlaCompiledKernel=true,_XlaNumConstantArgs=1].v82 [CPU:
pipeline start, before inline]: /tmp/hlo_graph_0.dot

```

The debug statements point to the location of `hlo_graph_xx.dot` files that
contain info about the graph created by XLA at runtime. To Render the .dot file
into a png, install [GraphViz](http://www.graphviz.org/Download..php) and run:

```shell
dot -Tpng hlo_graph_0.dot -o hlo_graph_0.png
```

The result will look like the following:
<div style="width:95%; margin:auto; margin-bottom:10px; margin-top:20px;">
  <img style="width:100%" src="../../images/jit_cpu_xla_graph.png">
</div>
