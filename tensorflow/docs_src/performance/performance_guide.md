# Performance Guide

This guide contains a collection of best practices for optimizing TensorFlow
code. The guide is divided into a few sections:

*   [General best practices](#general_best_practices) covers topics that are
    common across a variety of model types and hardware.
*   [Optimizing for GPU](#optimizing_for_gpu) details tips specifically relevant
    to GPUs.
*   [Optimizing for CPU](#optimizing_for_cpu) details CPU specific information.

## General best practices

The sections below cover best practices that are relevant to a variety of
hardware and models. The best practices section is broken down into the
following sections:

*   [Input pipeline optimizations](#input-pipeline-optimization)
*   [Data formats](#data-formats)
*   [Common fused Ops](#common-fused-ops)
*   [RNN Performance](#rnn-performance)
*   [Building and installing from source](#building-and-installing-from-source)

### Input pipeline optimization

Typical models retrieve data from disk and preprocess it before sending the data
through the network. For example, models that process JPEG images will follow
this flow: load image from disk, decode JPEG into a tensor, crop and pad,
possibly flip and distort, and then batch. This flow is referred to as the input
pipeline. As GPUs and other hardware accelerators get faster, preprocessing of
data can be a bottleneck.

Determining if the input pipeline is the bottleneck can be complicated. One of
the most straightforward methods is to reduce the model to a single operation
(trivial model) after the input pipeline and measure the examples per second. If
the difference in examples per second for the full model and the trivial model
is minimal then the input pipeline is likely a bottleneck. Below are some other
approaches to identifying issues:

*   Check if a GPU is underutilized by running `nvidia-smi -l 2`. If GPU
    utilization is not approaching 80-100%, then the input pipeline may be the
    bottleneck.
*   Generate a timeline and look for large blocks of white space (waiting). An
    example of generating a timeline exists as part of the @{$jit$XLA JIT}
    tutorial.
*   Check CPU usage. It is possible to have an optimized input pipeline and lack
    the CPU cycles to process the pipeline.
*   Estimate the throughput needed and verify the disk used is capable of that
    level of throughput. Some cloud solutions have network attached disks that
    start as low as 50 MB/sec, which is slower than spinning disks (150 MB/sec),
    SATA SSDs (500 MB/sec), and PCIe SSDs (2,000+ MB/sec).

#### Preprocessing on the CPU

Placing input pipeline operations on the CPU can significantly improve
performance. Utilizing the CPU for the input pipeline frees the GPU to focus on
training. To ensure preprocessing is on the CPU, wrap the preprocessing
operations as shown below:

```python
with tf.device('/cpu:0'):
  # function to get and process images or data.
  distorted_inputs = load_and_distort_images()
```

If using `tf.estimator.Estimator` the input function is automatically placed on
the CPU.

#### Using the tf.data API

The @{$datasets$tf.data API} is replacing `queue_runner` as the recommended API
for building input pipelines. This
[ResNet example](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator/cifar10_main.py)
([arXiv:1512.03385](https://arxiv.org/abs/1512.03385))
training CIFAR-10 illustrates the use of the `tf.data` API along with
`tf.estimator.Estimator`.

The `tf.data` API utilizes C++ multi-threading and has a much lower overhead
than the Python-based `queue_runner` that is limited by Python's multi-threading
performance. A detailed performance guide for the `tf.data` API can be found
[here](#datasets_performance).

While feeding data using a `feed_dict` offers a high level of flexibility, in
general `feed_dict` does not provide a scalable solution. If only a single GPU
is used, the difference between the `tf.data` API and `feed_dict` performance
may be negligible. Our recommendation is to avoid using `feed_dict` for all but
trivial examples. In particular, avoid using `feed_dict` with large inputs:

```python
# feed_dict often results in suboptimal performance when using large inputs.
sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
```

#### Fused decode and crop

If inputs are JPEG images that also require cropping, use fused
@{tf.image.decode_and_crop_jpeg} to speed up preprocessing.
`tf.image.decode_and_crop_jpeg` only decodes the part of
the image within the crop window. This significantly speeds up the process if
the crop window is much smaller than the full image. For imagenet data, this
approach could speed up the input pipeline by up to 30%.

Example Usage:

```python
def _image_preprocess_fn(image_buffer):
    # image_buffer 1-D string Tensor representing the raw JPEG image buffer.

    # Extract image shape from raw JPEG image buffer.
    image_shape = tf.image.extract_jpeg_shape(image_buffer)

    # Get a crop window with distorted bounding box.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      image_shape, ...)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Decode and crop image.
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    cropped_image = tf.image.decode_and_crop_jpeg(image, crop_window)
```

`tf.image.decode_and_crop_jpeg` is available on all platforms. There is no speed
up on Windows due to the use of `libjpeg` vs. `libjpeg-turbo` on other
platforms.

#### Use large files

Reading large numbers of small files significantly impacts I/O performance.
One approach to get maximum I/O throughput is to preprocess input data into
larger (~100MB) `TFRecord` files. For smaller data sets (200MB-1GB), the best
approach is often to load the entire data set into memory. The document
[Downloading and converting to TFRecord format](https://github.com/tensorflow/models/tree/master/research/slim#downloading-and-converting-to-tfrecord-format)
includes information and scripts for creating `TFRecords` and this
[script](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.py)
converts the CIFAR-10 data set into `TFRecords`.

### Data formats

Data formats refers to the structure of the Tensor passed to a given Op. The
discussion below is specifically about 4D Tensors representing images. In
TensorFlow the parts of the 4D tensor are often referred to by the following
letters:

*   N refers to the number of images in a batch.
*   H refers to the number of pixels in the vertical (height) dimension.
*   W refers to the number of pixels in the horizontal (width) dimension.
*   C refers to the channels. For example, 1 for black and white or grayscale
    and 3 for RGB.

Within TensorFlow there are two naming conventions representing the two most
common data formats:

*   `NCHW` or `channels_first`
*   `NHWC` or `channels_last`

`NHWC` is the TensorFlow default and `NCHW` is the optimal format to use when
training on NVIDIA GPUs using [cuDNN](https://developer.nvidia.com/cudnn).

The best practice is to build models that work with both data formats. This
simplifies training on GPUs and then running inference on CPUs. If TensorFlow is
compiled with the [Intel MKL](#tensorflow_with_intel_mkl-dnn) optimizations,
many operations, especially those related to CNN based models, will be optimized
and support `NCHW`. If not using the MKL, some operations are not supported on
CPU when using `NCHW`.

The brief history of these two formats is that TensorFlow started by using
`NHWC` because it was a little faster on CPUs. In the long term, we are working
on tools to auto rewrite graphs to make switching between the formats
transparent and take advantages of micro optimizations where a GPU Op may be
faster using `NHWC` than the normally most efficient `NCHW`.

### Common fused Ops

Fused Ops combine multiple operations into a single kernel for improved
performance. There are many fused Ops within TensorFlow and @{$xla$XLA} will
create fused Ops when possible to automatically improve performance. Collected
below are select fused Ops that can greatly improve performance and may be
overlooked.

#### Fused batch norm

Fused batch norm combines the multiple operations needed to do batch
normalization into a single kernel. Batch norm is an expensive process that for
some models makes up a large percentage of the operation time. Using fused batch
norm can result in a 12%-30% speedup.

There are two commonly used batch norms and both support fusing. The core
@{tf.layers.batch_normalization} added fused starting in TensorFlow 1.3.

```python
bn = tf.layers.batch_normalization(
    input_layer, fused=True, data_format='NCHW')
```

The contrib @{tf.contrib.layers.batch_norm} method has had fused as an option
since before TensorFlow 1.0.

```python
bn = tf.contrib.layers.batch_norm(input_layer, fused=True, data_format='NCHW')
```

### RNN Performance

There are many ways to specify an RNN computation in TensorFlow and they have
trade-offs with respect to model flexibility and performance. The
@{tf.nn.rnn_cell.BasicLSTMCell} should be considered a reference implementation
and used only as a last resort when no other options will work.

When using one of the cells, rather than the fully fused RNN layers, you have a
choice of whether to use @{tf.nn.static_rnn} or @{tf.nn.dynamic_rnn}.  There
shouldn't generally be a performance difference at runtime, but large unroll
amounts can increase the graph size of the @{tf.nn.static_rnn} and cause long
compile times.  An additional advantage of @{tf.nn.dynamic_rnn} is that it can
optionally swap memory from the GPU to the CPU to enable training of very long
sequences.  Depending on the model and hardware configuration, this can come at
a performance cost.  It is also possible to run multiple iterations of
@{tf.nn.dynamic_rnn} and the underlying @{tf.while_loop} construct in parallel,
although this is rarely useful with RNN models as they are inherently
sequential.

On NVIDIA GPUs, the use of @{tf.contrib.cudnn_rnn} should always be preferred
unless you want layer normalization, which it doesn't support.  It is often at
least an order of magnitude faster than @{tf.contrib.rnn.BasicLSTMCell} and
@{tf.contrib.rnn.LSTMBlockCell} and uses 3-4x less memory than
@{tf.contrib.rnn.BasicLSTMCell}.

If you need to run one step of the RNN at a time, as might be the case in
reinforcement learning with a recurrent policy, then you should use the
@{tf.contrib.rnn.LSTMBlockCell} with your own environment interaction loop
inside a @{tf.while_loop} construct. Running one step of the RNN at a time and
returning to Python is possible, but it will be slower.

On CPUs, mobile devices, and if @{tf.contrib.cudnn_rnn} is not available on
your GPU, the fastest and most memory efficient option is
@{tf.contrib.rnn.LSTMBlockFusedCell}.

For all of the less common cell types like @{tf.contrib.rnn.NASCell},
@{tf.contrib.rnn.PhasedLSTMCell}, @{tf.contrib.rnn.UGRNNCell},
@{tf.contrib.rnn.GLSTMCell}, @{tf.contrib.rnn.Conv1DLSTMCell},
@{tf.contrib.rnn.Conv2DLSTMCell}, @{tf.contrib.rnn.LayerNormBasicLSTMCell},
etc., one should be aware that they are implemented in the graph like
@{tf.contrib.rnn.BasicLSTMCell} and as such will suffer from the same poor
performance and high memory usage.  One should consider whether or not those
trade-offs are worth it before using these cells. For example, while layer
normalization can speed up convergence, because cuDNN is 20x faster the fastest
wall clock time to convergence is usually obtained without it.


### Building and installing from source

The default TensorFlow binaries target the broadest range of hardware to make
TensorFlow accessible to everyone. If using CPUs for training or inference, it
is recommended to compile TensorFlow with all of the optimizations available for
the CPU in use. Speedups for training and inference on CPU are documented below
in [Comparing compiler optimizations](#comparing-compiler-optimizations).

To install the most optimized version of TensorFlow,
@{$install_sources$build and install} from source. If there is a need to build
TensorFlow on a platform that has different hardware than the target, then
cross-compile with the highest optimizations for the target platform. The
following command is an example of using `bazel` to compile for a specific
platform:

```python
# This command optimizes for Intel’s Broadwell processor
bazel build -c opt --copt=-march="broadwell" --config=cuda //tensorflow/tools/pip_package:build_pip_package

```

#### Environment, build, and install tips

*   `./configure` asks which compute capability to include in the build. This
    does not impact overall performance but does impact initial startup. After
    running TensorFlow once, the compiled kernels are cached by CUDA. If using
    a docker container, the data is not cached and the penalty is paid each time
    TensorFlow starts. The best practice is to include the
    [compute capabilities](http://developer.nvidia.com/cuda-gpus)
    of the GPUs that will be used, e.g. P100: 6.0, Titan X (Pascal): 6.1, Titan
    X (Maxwell): 5.2, and K80: 3.7.
*   Use a version of gcc that supports all of the optimizations of the target
    CPU. The recommended minimum gcc version is 4.8.3. On OS X, upgrade to the
    latest Xcode version and use the version of clang that comes with Xcode.
*   Install the latest stable CUDA platform and cuDNN libraries supported by
    TensorFlow.

## Optimizing for GPU

This section contains GPU-specific tips that are not covered in the
[General best practices](#general-best-practices). Obtaining optimal performance
on multi-GPUs is a challenge. A common approach is to use data parallelism.
Scaling through the use of data parallelism involves making multiple copies of
the model, which are referred to as "towers", and then placing one tower on each
of the GPUs. Each tower operates on a different mini-batch of data and then
updates variables, also known as parameters, that need to be shared between
each of the towers. How each tower gets the updated variables and how the
gradients are applied has an impact on the performance, scaling, and convergence
of the model.  The rest of this section provides an overview of variable
placement and the towering of a model on multiple GPUs.
@{$performance_models$High-Performance Models} gets into more details regarding
more complex methods that can be used to share and update variables between
towers.

The best approach to handling variable updates depends on the model, hardware,
and even how the hardware has been configured. An example of this, is that two
systems can be built with NVIDIA Tesla P100s but one may be using PCIe and the
other [NVLink](http://www.nvidia.com/object/nvlink.html). In that scenario, the
optimal solution for each system may be different. For real world examples, read
the @{$performance/benchmarks$benchmark} page which details the settings that
were optimal for a variety of platforms. Below is a summary of what was learned
from benchmarking various platforms and configurations:

*   **Tesla K80**: If the GPUs are on the same PCI Express root complex and are
    able to use [NVIDIA GPUDirect](https://developer.nvidia.com/gpudirect) Peer
    to Peer, then placing the variables equally across the GPUs used for
    training is the best approach. If the GPUs cannot use GPUDirect, then
    placing the variables on the CPU is the best option.

*   **Titan X (Maxwell and Pascal), M40, P100, and similar**: For models like
    ResNet and InceptionV3, placing variables on the CPU is the optimal setting,
    but for models with a lot of variables like AlexNet and VGG, using GPUs with
    `NCCL` is better.

A common approach to managing where variables are placed, is to create a method
to determine where each Op is to be placed and use that method in place of a
specific device name when calling `with tf.device():`. Consider a scenario where
a model is being trained on 2 GPUs and the variables are to be placed on the
CPU. There would be a loop for creating and placing the "towers" on each of the
2 GPUs. A custom device placement method would be created that watches for Ops
of type `Variable`, `VariableV2`, and `VarHandleOp` and indicates that they are
to be placed on the CPU. All other Ops would be placed on the target GPU.
The building of the graph would proceed as follows:

*   On the first loop a "tower" of the model would be created for `gpu:0`.
    During the placement of the Ops, the custom device placement method would
    indicate that variables are to be placed on `cpu:0` and all other Ops on
    `gpu:0`.

*   On the second loop, `reuse` is set to `True` to indicate that variables are
    to be reused and then the "tower" is created on `gpu:1`. During the
    placement of the Ops associated with the "tower", the variables that were
    placed on `cpu:0` are reused and all other Ops are created and placed on
    `gpu:1`.

The final result is all of the variables are placed on the CPU with each GPU
having a copy of all of the computational Ops associated with the model.

The code snippet below illustrates two different approaches for variable
placement: one is placing variables on the CPU; the other is placing variables
equally across the GPUs.

```python

class GpuParamServerDeviceSetter(object):
  """Used with tf.device() to place variables on the least loaded GPU.

    A common use for this class is to pass a list of GPU devices, e.g. ['gpu:0',
    'gpu:1','gpu:2'], as ps_devices.  When each variable is placed, it will be
    placed on the least loaded gpu. All other Ops, which will be the computation
    Ops, will be placed on the worker_device.
  """

  def __init__(self, worker_device, ps_devices):
    """Initializer for GpuParamServerDeviceSetter.
    Args:
      worker_device: the device to use for computation Ops.
      ps_devices: a list of devices to use for Variable Ops. Each variable is
      assigned to the least loaded device.
    """
    self.ps_devices = ps_devices
    self.worker_device = worker_device
    self.ps_sizes = [0] * len(self.ps_devices)

  def __call__(self, op):
    if op.device:
      return op.device
    if op.type not in ['Variable', 'VariableV2', 'VarHandleOp']:
      return self.worker_device

    # Gets the least loaded ps_device
    device_index, _ = min(enumerate(self.ps_sizes), key=operator.itemgetter(1))
    device_name = self.ps_devices[device_index]
    var_size = op.outputs[0].get_shape().num_elements()
    self.ps_sizes[device_index] += var_size

    return device_name

def _create_device_setter(is_cpu_ps, worker, num_gpus):
  """Create device setter object."""
  if is_cpu_ps:
    # tf.train.replica_device_setter supports placing variables on the CPU, all
    # on one GPU, or on ps_servers defined in a cluster_spec.
    return tf.train.replica_device_setter(
        worker_device=worker, ps_device='/cpu:0', ps_tasks=1)
  else:
    gpus = ['/gpu:%d' % i for i in range(num_gpus)]
    return ParamServerDeviceSetter(worker, gpus)

# The method below is a modified snippet from the full example.
def _resnet_model_fn():
    # When set to False, variables are placed on the least loaded GPU. If set
    # to True, the variables will be placed on the CPU.
    is_cpu_ps = False

    # Loops over the number of GPUs and creates a copy ("tower") of the model on
    # each GPU.
    for i in range(num_gpus):
      worker = '/gpu:%d' % i
      # Creates a device setter used to determine where Ops are to be placed.
      device_setter = _create_device_setter(is_cpu_ps, worker, FLAGS.num_gpus)
      # Creates variables on the first loop.  On subsequent loops reuse is set
      # to True, which results in the "towers" sharing variables.
      with tf.variable_scope('resnet', reuse=bool(i != 0)):
        with tf.name_scope('tower_%d' % i) as name_scope:
          # tf.device calls the device_setter for each Op that is created.
          # device_setter returns the device the Op is to be placed on.
          with tf.device(device_setter):
            # Creates the "tower".
            _tower_fn(is_training, weight_decay, tower_features[i],
                      tower_labels[i], tower_losses, tower_gradvars,
                      tower_preds, False)

```

In the near future the above code will be for illustration purposes only as
there will be easy to use high level methods to support a wide range of popular
approaches. This
[example](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator)
will continue to get updated as the API expands and evolves to address multi-GPU
scenarios.

## Optimizing for CPU

CPUs, which includes Intel® Xeon Phi™, achieve optimal performance when
TensorFlow is @{$install_sources$built from source} with all of the instructions
supported by the target CPU.

Beyond using the latest instruction sets, Intel® has added support for the
Intel® Math Kernel Library for Deep Neural Networks (Intel® MKL-DNN) to
TensorFlow. While the name is not completely accurate, these optimizations are
often simply referred to as 'MKL' or 'TensorFlow with MKL'. [TensorFlow
with Intel® MKL-DNN](#tensorflow_with_intel_mkl_dnn) contains details on the
MKL optimizations.

The two configurations listed below are used to optimize CPU performance by
adjusting the thread pools.

*   `intra_op_parallelism_threads`: Nodes that can use multiple threads to
    parallelize their execution will schedule the individual pieces into this
    pool.
*   `inter_op_parallelism_threads`: All ready nodes are scheduled in this pool.

These configurations are set via the `tf.ConfigProto` and passed to `tf.Session`
in the `config` attribute as shown in the snippet below.  For both configuration
options, if they are unset or set to 0, will default to the number of logical
CPU cores. Testing has shown that the default is effective for systems ranging
from one CPU with 4 cores to multiple CPUs with 70+ combined logical cores.
A common alternative optimization is to set the number of threads in both pools
equal to the number of physical cores rather than logical cores.

```python

  config = tf.ConfigProto()
  config.intra_op_parallelism_threads = 44
  config.inter_op_parallelism_threads = 44
  tf.session(config=config)

```

The [Comparing compiler optimizations](#comparing-compiler-optimizations)
section contains the results of tests that used different compiler
optimizations.

### TensorFlow with Intel® MKL DNN

Intel® has added optimizations to TensorFlow for Intel® Xeon® and Intel® Xeon
Phi™ though the use of Intel® Math Kernel Library for Deep Neural Networks
(Intel® MKL-DNN) optimized primitives. The optimizations also provide speedups
for the consumer line of processors, e.g. i5 and i7 Intel processors. The Intel
published paper
[TensorFlow* Optimizations on Modern Intel® Architecture](https://software.intel.com/en-us/articles/tensorflow-optimizations-on-modern-intel-architecture)
contains additional details on the implementation.

> Note: MKL was added as of TensorFlow 1.2 and currently only works on Linux. It
> also does not work when also using `--config=cuda`.

In addition to providing significant performance improvements for training CNN
based models, compiling with the MKL creates a binary that is optimized for AVX
and AVX2. The result is a single binary that is optimized and compatible with
most modern (post-2011) processors.

TensorFlow can be compiled with the MKL optimizations using the following
commands that depending on the version of the TensorFlow source used.

For TensorFlow source versions after 1.3.0:

```bash
./configure
# Pick the desired options
bazel build --config=mkl --config=opt //tensorflow/tools/pip_package:build_pip_package

```

For TensorFlow versions 1.2.0 through 1.3.0:

```bash
./configure
Do you wish to build TensorFlow with MKL support? [y/N] Y
Do you wish to download MKL LIB from the web? [Y/n] Y
# Select the defaults for the rest of the options.

bazel build --config=mkl --copt="-DEIGEN_USE_VML" -c opt //tensorflow/tools/pip_package:build_pip_package

```

#### Tuning MKL for the best performance

This section details the different configurations and environment variables that
can be used to tune the MKL to get optimal performance. Before tweaking various
environment variables make sure the model is using the `NCHW` (`channels_first`)
[data format](#data-formats). The MKL is optimized for `NCHW` and Intel is
working to get near performance parity when using `NHWC`.

MKL uses the following environment variables to tune performance:

*   KMP_BLOCKTIME - Sets the time, in milliseconds, that a thread should wait,
    after completing the execution of a parallel region, before sleeping.
*   KMP_AFFINITY - Enables the run-time library to bind threads to physical
    processing units.
*   KMP_SETTINGS - Enables (true) or disables (false) the printing of OpenMP*
    run-time library environment variables during program execution.
*   OMP_NUM_THREADS - Specifies the number of threads to use.

More details on the KMP variables are on
[Intel's](https://software.intel.com/en-us/node/522775) site and the OMP
variables on
[gnu.org](https://gcc.gnu.org/onlinedocs/libgomp/Environment-Variables.html)

While there can be substantial gains from adjusting the environment variables,
which is discussed below, the simplified advice is to set the
`inter_op_parallelism_threads` equal to the number of physical CPUs and to set
the following environment variables:

*   KMP_BLOCKTIME=0
*   KMP_AFFINITY=granularity=fine,verbose,compact,1,0

Example setting MKL variables with command-line arguments:

```bash
KMP_BLOCKTIME=0 KMP_AFFINITY=granularity=fine,verbose,compact,1,0 \
KMP_SETTINGS=1 python your_python_script.py
```

Example setting MKL variables with python `os.environ`:

```python
os.environ["KMP_BLOCKTIME"] = str(FLAGS.kmp_blocktime)
os.environ["KMP_SETTINGS"] = str(FLAGS.kmp_settings)
os.environ["KMP_AFFINITY"]= FLAGS.kmp_affinity
if FLAGS.num_intra_threads > 0:
  os.environ["OMP_NUM_THREADS"]= str(FLAGS.num_intra_threads)

```

There are models and hardware platforms that benefit from different settings.
Each variable that impacts performance is discussed below.

*   **KMP_BLOCKTIME**: The MKL default is 200ms, which was not optimal in our
    testing. 0 (0ms) was a good default for CNN based models that were tested.
    The best performance for AlexNex was achieved at 30ms and both GoogleNet and
    VGG11 performed best set at 1ms.

*   **KMP_AFFINITY**: The recommended setting is
    `granularity=fine,verbose,compact,1,0`.

*   **OMP_NUM_THREADS**: This defaults to the number of physical cores.
    Adjusting this parameter beyond matching the number of cores can have an
    impact when using Intel® Xeon Phi™ (Knights Landing) for some models. See
    [TensorFlow* Optimizations on Modern Intel® Architecture](https://software.intel.com/en-us/articles/tensorflow-optimizations-on-modern-intel-architecture)
    for optimal settings.

*   **intra_op_parallelism_threads**: Setting this equal to the number of
    physical cores is recommended. Setting the value to 0, which is the default
    and will result in the value being set to the number of logical cores, is an
    option to try for some architectures.  This value and `OMP_NUM_THREADS`
    should be equal.

*   **inter_op_parallelism_threads**: Setting this equal to the number of
    sockets is recommended. Setting the value to 0, which is the default,
    results in the value being set to the number of logical cores.

### Comparing compiler optimizations

Collected below are performance results running training and inference on
different types of CPUs on different platforms with various compiler
optimizations.  The models used were ResNet-50
([arXiv:1512.03385](https://arxiv.org/abs/1512.03385)) and
InceptionV3 ([arXiv:1512.00567](https://arxiv.org/abs/1512.00567)).

For each test, when the MKL optimization was used the environment variable
KMP_BLOCKTIME was set to 0 (0ms) and KMP_AFFINITY to
`granularity=fine,verbose,compact,1,0`.

#### Inference InceptionV3

**Environment**

*   Instance Type: AWS EC2 m4.xlarge
*   CPU: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz (Broadwell)
*   Dataset: ImageNet
*   TensorFlow Version: 1.2.0 RC2
*   Test Script: [tf_cnn_benchmarks.py](https://github.com/tensorflow/benchmarks/blob/mkl_experiment/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py)

**Batch Size: 1**

Command executed for the MKL test:

```bash
python tf_cnn_benchmarks.py --forward_only=True --device=cpu --mkl=True \
--kmp_blocktime=0 --nodistortions --model=inception3 --data_format=NCHW \
--batch_size=1 --num_inter_threads=1 --num_intra_threads=4 \
--data_dir=<path to ImageNet TFRecords>
```

| Optimization | Data Format | Images/Sec   | Intra threads | Inter Threads |
:              :             : (step time)  :               :               :
| ------------ | ----------- | ------------ | ------------- | ------------- |
| AVX2         | NHWC        | 7.0 (142ms)  | 4             | 0             |
| MKL          | NCHW        | 6.6 (152ms)  | 4             | 1             |
| AVX          | NHWC        | 5.0 (202ms)  | 4             | 0             |
| SSE3         | NHWC        | 2.8 (361ms)  | 4             | 0             |

**Batch Size: 32**

Command executed for the MKL test:

```bash
python tf_cnn_benchmarks.py --forward_only=True --device=cpu --mkl=True \
--kmp_blocktime=0 --nodistortions --model=inception3 --data_format=NCHW \
--batch_size=32 --num_inter_threads=1 --num_intra_threads=4 \
--data_dir=<path to ImageNet TFRecords>
```

| Optimization | Data Format | Images/Sec    | Intra threads | Inter Threads |
:              :             : (step time)   :               :               :
| ------------ | ----------- | ------------- | ------------- | ------------- |
| MKL          | NCHW        | 10.3          | 4             | 1             |
:              :             : (3,104ms)     :               :               :
| AVX2         | NHWC        | 7.5 (4,255ms) | 4             | 0             |
| AVX          | NHWC        | 5.1 (6,275ms) | 4             | 0             |
| SSE3         | NHWC        | 2.8 (11,428ms)| 4             | 0             |

#### Inference ResNet-50

**Environment**

*   Instance Type: AWS EC2 m4.xlarge
*   CPU: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz (Broadwell)
*   Dataset: ImageNet
*   TensorFlow Version: 1.2.0 RC2
*   Test Script: [tf_cnn_benchmarks.py](https://github.com/tensorflow/benchmarks/blob/mkl_experiment/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py)

**Batch Size: 1**

Command executed for the MKL test:

```bash
python tf_cnn_benchmarks.py --forward_only=True --device=cpu --mkl=True \
--kmp_blocktime=0 --nodistortions --model=resnet50 --data_format=NCHW \
--batch_size=1 --num_inter_threads=1 --num_intra_threads=4 \
--data_dir=<path to ImageNet TFRecords>
```

| Optimization | Data Format | Images/Sec   | Intra threads | Inter Threads |
:              :             : (step time)  :               :               :
| ------------ | ----------- | ------------ | ------------- | ------------- |
| AVX2         | NHWC        | 8.8 (113ms)  | 4             | 0             |
| MKL          | NCHW        | 8.5 (120ms)  | 4             | 1             |
| AVX          | NHWC        | 6.4 (157ms)  | 4             | 0             |
| SSE3         | NHWC        | 3.7 (270ms)  | 4             | 0             |

**Batch Size: 32**

Command executed for the MKL test:

```bash
python tf_cnn_benchmarks.py --forward_only=True --device=cpu --mkl=True \
--kmp_blocktime=0 --nodistortions --model=resnet50 --data_format=NCHW \
--batch_size=32 --num_inter_threads=1 --num_intra_threads=4 \
--data_dir=<path to ImageNet TFRecords>
```

| Optimization | Data Format | Images/Sec    | Intra threads | Inter Threads |
:              :             : (step time)   :               :               :
| ------------ | ----------- | ------------- | ------------- | ------------- |
| MKL          | NCHW        | 12.4          | 4             | 1             |
:              :             : (2,590ms)     :               :               :
| AVX2         | NHWC        | 10.4 (3,079ms)| 4             | 0             |
| AVX          | NHWC        | 7.3 (4,4416ms)| 4             | 0             |
| SSE3         | NHWC        | 4.0 (8,054ms) | 4             | 0             |

#### Training InceptionV3

**Environment**

*   Instance Type: Dedicated AWS EC2 r4.16xlarge (Broadwell)
*   CPU: Intel Xeon E5-2686 v4 (Broadwell) Processors
*   Dataset: ImageNet
*   TensorFlow Version: 1.2.0 RC2
*   Test Script: [tf_cnn_benchmarks.py](https://github.com/tensorflow/benchmarks/blob/mkl_experiment/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py)

Command executed for MKL test:

```bash
python tf_cnn_benchmarks.py --device=cpu --mkl=True --kmp_blocktime=0 \
--nodistortions --model=resnet50 --data_format=NCHW --batch_size=32 \
--num_inter_threads=2 --num_intra_threads=36 \
--data_dir=<path to ImageNet TFRecords>
```

Optimization | Data Format | Images/Sec | Intra threads | Inter Threads
------------ | ----------- | ---------- | ------------- | -------------
MKL          | NCHW        | 20.8       | 36            | 2
AVX2         | NHWC        | 6.2        | 36            | 0
AVX          | NHWC        | 5.7        | 36            | 0
SSE3         | NHWC        | 4.3        | 36            | 0

ResNet and [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
were also run on this configuration but in an ad hoc manner. There were not
enough runs executed to publish a coherent table of results. The incomplete
results strongly indicated the final result would be similar to the table above
with MKL providing significant 3x+ gains over AVX2.
