# Performance Guide

This guide contains a collection of best practices for optimizing your
TensorFlow code. The best practices apply to both new and experienced
Tensorflow users.  As a complement to the best practices in this document, the
@{$performance_models$High-Performance Models} document links to example code
and details for creating models that scale on a variety of hardware.

## Best Practices
While optimizing implementations of different types of models can be different,
the topics below cover best practices to get the most performance from
TensorFlow. Although these suggestions focus on image-based models, we will
regularly add tips for all kinds of models. The following list highlights key
best practices:

*   Build and install from source
*   Utilize queues for reading data
*   Preprocessing on the CPU
*   Use `NCHW` image data format
*   Place shared parameters on the GPU
*   Use fused batch norm

The following sections detail the preceding suggestions.

### Build and install from source

To install the most optimized version of TensorFlow, build and install
TensorFlow from source by following [Installing TensorFlow from Source](../install/install_sources).
Building from source with compiler optimizations for the target hardware and
ensuring the latest CUDA platform and cuDNN libraries are installed results in
the highest performing installs.

For the most stable experience, build from the [latest release](https://github.com/tensorflow/tensorflow/releases)
branch. To get the latest performance changes and accept some stability risk,
build from [master](https://github.com/tensorflow/tensorflow).

If there is a need to build TensorFlow on a platform that has different hardware
than the target, then cross-compile with the highest optimizations for the target
platform.  The following command is an example of telling `bazel` to compile for
a specific platform:

```python
# This command optimizes for Intel’s Broadwell processor
bazel build -c opt --copt=-march="broadwell" --config=cuda //tensorflow/tools/pip_package:build_pip_package

```

#### Environment, build, and install tips

*   Compile with the highest level of compute the [GPU
    supports](http://developer.nvidia.com/cuda-gpus), e.g. P100: 6.0, Titan X
    (pascal): 6.2, Titan X (maxwell): 5.2, and K80: 3.7.
*   Install the latest CUDA platform and cuDNN libraries.
*   Make sure to use a version of gcc that supports all of the optimizations of
    the target CPU. The recommended minimum gcc version is 4.8.3.
*   TensorFlow checks on startup whether it has been compiled with the
    optimizations available on the CPU. If the optimizations are not included,
    TensorFlow will emit warnings, e.g. AVX, AVX2, and FMA instructions not
    included.

### Utilize queues for reading data

One common cause of poor performance is underutilizing GPUs, or essentially
"starving" them of data by not setting up an efficient pipeline. Make sure to
set up an input pipeline to utilize queues and stream data effectively. Review
the @{$reading_data#reading_from_files$Reading Data guide} for implementation
details. One way to identify a "starved" GPU is to generate and review
timelines. A detailed tutorial for timelines does not exist, but a quick example
of generating a timeline exists as part of the @{$jit$XLA JIT} tutorial. Another
simple way to check if a GPU is underutilized is to run `watch nvidia-smi`, and
if GPU utilization is not approaching 100% then the GPU is not getting data fast
enough.

Unless for a special circumstance or for example code, do not feed data
into the session from Python variables, e.g. `dictionary`.

```python
# Using feed_dict often results in suboptimal performance when using large inputs.
sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

### Preprocessing on the CPU

Placing preprocessing operations on the CPU can significantly improve
performance.  When preprocessing occurs on the GPU the flow of data is
CPU -> GPU (preprocessing) -> CPU -> GPU (training).  The data is bounced back
and forth between the CPU and GPU.  When preprocessing is placed on the CPU,
the data flow is CPU (preprocessing) -> GPU (training).  Another benefit is
preprocessing on the CPU frees GPU time to focus on training.

Placing preprocessing on the CPU can result in a 6X+ increase in samples/sec
processed, which could lead to training in 1/6th of the time.  To ensure
preprocessing is on the CPU, wrap the preprocessing operations as shown below:

```python
with tf.device('/cpu:0'):
  # function to get and process images or data.
  distorted_inputs = load_and_distort_images()
```

### Use large files

Under some circumstances, both the CPU and GPU can be starved for data by the
I/O system. If you are using many small files to form your input data set, you
may be limited by the speed of your filesystem. If your training loop runs
faster when using SSDs vs HDDs for storing your input data, you could could be
I/O bottlenecked.

If this is the case, you should pre-process your input data, creating a few
large TFRecord files.

### Use NCHW image data format

Image data format refers to the representation of batches of images. TensorFlow
supports `NHWC` (TensorFlow default) and `NCHW` (cuDNN default). N refers to the
number of images in a batch, H refers to the number of pixels in the vertical
dimension, W refers to the number of pixels in the horizontal dimension, and C
refers to the channels (e.g. 1 for black and white, 3 for RGB, etc.) Although
cuDNN can operate on both formats, it is faster to operate in its default
format.

The best practice is to build models that work with both `NCHW` and `NHWC` as it
is common to train using `NCHW` on GPU, and then do inference with NHWC on CPU.

The very brief history of these two formats is that TensorFlow started by using
`NHWC` because it was a little faster on CPUs. Then the TensorFlow team
discovered that `NCHW` performs better when using the NVIDIA cuDNN library.  The
current recommendation is that users support both formats in their models. In
the long term, we plan to rewrite graphs to make switching between the formats
transparent.

### Use fused batch norm

When using batch norm
@{tf.contrib.layers.batch_norm} set the attribute `fused=True`:

```python
bn = tf.contrib.layers.batch_norm(
          input_layer, fused=True, data_format='NCHW'
          scope=scope, **kwargs)
```

The non-fused batch norm does computations using several individual Ops. Fused
batch norm combines the individual operations into a single kernel, which runs
faster.

