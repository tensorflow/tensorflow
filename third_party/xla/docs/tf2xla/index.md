# XLA: Optimizing Compiler for Machine Learning

[OpenXLA](https://openxla.org) is a domain-specific compiler for linear
algebra that can accelerate TensorFlow models with potentially no source code
changes.

## Introduction

When a TensorFlow program is run, all of the operations are executed
individually by the TensorFlow executor. Each TensorFlow operation has a
precompiled GPU kernel implementation that the executor dispatches to.

XLA provides an alternative mode of running models: it compiles the TensorFlow
graph into a sequence of computation kernels generated specifically for the
given model. Because these kernels are unique to the model, they can exploit
model-specific information for optimization. For example, let's look at an
optimization XLA does in the context of a simple TensorFlow computation:

```
def model_fn(x, y, z):
  return tf.reduce_sum(x + y * z)
```

Run without XLA, the graph launches three kernels: one for the multiplication,
one for the addition and one for the reduction. However, XLA can optimize the
graph so that it computes the result in a single kernel launch. It does this by
"fusing" the addition, multiplication and reduction into a single GPU kernel.
Moreover, this fused operation does not write out the intermediate values
produced by `y*z` and `x+y*z` to memory; instead it "streams" the results of
these intermediate computations directly to their users while keeping them
entirely in GPU registers. Fusion is XLA's single most important optimization.
Memory bandwidth is typically the scarcest resource on hardware accelerators, so
removing memory operations is one of the best ways to improve performance.

## Enable XLA for TensorFlow models

### Explicit compilation with `tf.function(jit_compile=True)`

Explicit compilation API offers a fine-grained control for choosing which
functions should be compiled. For example, the following TensorFlow function
which performs the MNIST training is compiled with XLA:

```
@tf.function(jit_compile=True)
def train_mnist(images, labels):
    images, labels = cast(images, labels)

    with tf.GradientTape() as tape:
      predicted_labels = layer(images)
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=predicted_labels, labels=labels
      ))
    layer_variables = layer.trainable_variables
    grads = tape.gradient(loss, layer_variables)
    optimizer.apply_gradients(zip(grads, layer_variables))
```

The `jit_compile` API has _must-compile_ semantics: either the entire
function is compiled with XLA, or an `errors.InvalidArgumentError` exception is
thrown. XLA can not currently compile functions where dimensions are not
_inferrable_: that is, if it's not possible to infer the dimensions of all
tensors without running the entire computation. For example, the following
function will not compile:

```
@tf.function
def not_compilable(x):
  return tf.unique(x)
```

Shapes can vary across the runs though:

```
@tf.function(jit_compile=True)
def recompiled_on_launch(a, b):
  return a + b

recompiled_on_launch(tf.ones([1, 10]), tf.ones([1, 10]))
recompiled_on_launch(tf.ones([1, 100]), tf.ones([1, 100]))
```

Note: Nesting behavior: the function will be compiled if at least one function
in its call stack has `jit_compile=True`.

See the [tutorial colab](./tutorials/jit_compile.ipynb) for a more detailed
usage example, and a
[tutorial video](https://www.youtube.com/watch?v=cPAD9vLKE0c) on
`jit_compile=True` usage.

### Usage with Keras

For Keras models, `jit_compile=True` can be set as an argument to
[`model.compile`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile):

```
model.compile(optimizer="adam", jit_compile=True)
```

### Usage with distributed strategy

XLA:GPU can be used with TF distributed strategy
([`MirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)
or
[`MultiWorkerMirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy))
by annotating step function with `jit_compile=True`:

```
@tf.function(jit_compile=True)
def step_fn():
  t = tf.ones(shape=[100], dtype=tf.float32)
  ctx = tf.distribute.get_replica_context()
  return ctx.all_reduce(tf.distribute.ReduceOp.SUM, t)

@tf.function
def run_fn():
  return strategy.run(step_fn)
```

### Auto-clustering

A simple way to start using XLA in TensorFlow models without any changes is to
enable _auto-clustering_, which automatically finds _clusters_ (connected
subgraphs) within the TensorFlow functions which can be compiled and executed
using XLA. Auto-clustering on GPU can be enabled by setting the `TF_XLA_FLAGS`
environment variable:

Note: In TF2, only the code inside `tf.function` will be clustered.

```
$ TF_XLA_FLAGS=--tf_xla_auto_jit=2 path/to/your/tf/program
```

Auto-clustering is currently optimized for GPU workloads, but it can also be
enabled on CPU by additionally using the flag `--tf_xla_cpu_global_jit`:

```
$ TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" path/to/your/program
```

Note: Auto-clustering support on CPU and on multi-GPU environments is
experimental.

For a detailed usage example see the
[auto-clustering tutorial colab](./tutorials/autoclustering_xla.ipynb).

## Inspect compiled programs

XLA provides introspection facilities which let you inspect the generated
programs. To dump the generated programs, use the environment variable
`XLA_FLAGS`:

```
$ XLA_FLAGS="--xla_dump_to=/tmp/generated" TF_XLA_FLAGS="--tf_xla_auto_jit=2" my/tensorflow/program
```

After the dumping is performed, you can find the following files in
`/tmp/generated`:

-   `module_XXXX.*_optimizations.txt` Generated
    [XLA programs](./operation_semantics.md), one per each compiled cluster.
    Attaching those when submitting XLA bug reports is extremely helpful!

-   `module_XXXX.ir-*.ll` Generated files in
    [LLVM](https://llvm.org/docs/LangRef.html) intermediate representation, with
    [NVPTX](https://llvm.org/docs/NVPTXUsage.html) intrinsics.

-   `module_XXXX.ptx` Generated
    [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
    files.

You can also dump the graph visualizing the embedding of XLA clusters inside of
the TensorFlow graph with:

```
$ TF_DUMP_GRAPH_PREFIX=/tmp/generated TF_XLA_FLAGS="--tf_xla_clustering_debug"
```

## Reproducible bug reports

A bug report is much easier to reproduce if it includes dumps for the generated
XLA programs and the used auto-clustering embedding.
To generate them for a TensorFlow program running with auto-clustering, launch:

```
$ TF_DUMP_GRAPH_PREFIX=/tmp/generated \
  TF_XLA_FLAGS="--tf_xla_clustering_debug --tf_xla_auto_jit=2" \
  XLA_FLAGS="--xla_dump_hlo_as_text --xla_dump_to=/tmp/generated" \
    my/tensorflow/program"
```

When filing bugs, attach the contents of the `/tmp/generated` directory
(referenced above).

If possible, try to isolate
a bug to a single XLA program by using the
[`run_hlo_module`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/tools/run_hlo_module_main.cc)
and iteratively running it on generated programs.

## Further reading

-   [OpenXLA Documentation](https://openxla.org) OpenXLA Documentation
-   [Known Issues](./known_issues.md) List of known issues with XLA+TF
-   [XLA - TensorFlow, Compiled](https://developers.googleblog.com/2017/03/xla-tensorflow-compiled.html):
    Read on Google Developers Blog
-   Check out the
    [XLA source](https://github.com/openxla/xla)
    on Github!

## XLA Frontends

Apart from TensorFlow, XLA programs can be generated by:

-   [JAX](https://github.com/google/jax): Composable transformations of
    Python+NumPy programs
-   [Julia](https://github.com/JuliaTPU/XLA.jl): The Julia language for
    scientific computing
-   [PyTorch](https://github.com/pytorch/xla): PyTorch framework
-   [Nx](https://github.com/elixir-nx/nx): Numerical computing library for the
    Elixir programming language

## Talks

### Using XLA from TF using `jit_compile=True`

<iframe frameborder="0" allow="accelerometer; autoplay;
encrypted-media; gyroscope; picture-in-picture; fullscreen" width="640" height="360"
src="https://www.youtube.com/embed/cPAD9vLKE0c?origin=https%3A%2F%2Fwww.tensorflow.org&amp;autohide=1&amp;showinfo=0&amp;video-id=kAOanJczHA0&amp;enablejsapi=1&amp;widgetid=1" id="widget2" data-title="YouTube video player"></iframe>

### XLA Overview

<iframe frameborder="0" allow="accelerometer; autoplay;
encrypted-media; gyroscope; picture-in-picture; fullscreen" width="640" height="360"
src="https://www.youtube.com/embed/kAOanJczHA0?origin=https%3A%2F%2Fwww.tensorflow.org&amp;autohide=1&amp;showinfo=0&amp;video-id=kAOanJczHA0&amp;enablejsapi=1&amp;widgetid=1"
id="widget2" data-title="YouTube video player"></iframe>
