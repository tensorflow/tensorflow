# **Multiple Stream TensorFlow**

## **Introduction**
Multiple Stream TensorFlow is developed based on the official [TensorFlow 2.2](https://github.com/tensorflow/tensorflow/tree/r2.2). It leverages the features of modern GPUs to accelerate deep learning training and inference. This Multi-Stream implementation has successfully helped several customers migrate their TF models to the GPU and go online.

## **Key Features**

1. Multiple Streams

Now the user can specify how many stream groups to create for each GPU. each time the session.run() function is called, TF will schedule the most suitable stream group to execute. To reduce inter-stream dependencies at high concurrency, we also support different CUDA contexts for the stream groups.

2. Stream Merging

In the native TF, one computation stream and multiple copy streams are introduced to a stream group to achieve parallelism of computation and copy. However, synchronization between the streams can cause significant overhead, especially in scenarios where copying is frequent. Now, with the Multi-Stream implementation, the parallelism of copy and computation can be done between stream groups, so we allow users to use only one stream for computation and copy within a stream group to alleviate the stream synchronization overhead.

3. Resources management among streams

In the design, we make all stream groups reuse the same set of model parameters to avoid taking up too much GPU memory. For other resources, we separate them between stream groups as much as possible to reduce the dependencies and achieve better speedups, e.g.

* Per-stream GPU allocator
* Per-stream host allocator if `TF_PER_STREAM_HOST_ALLOCATOR=true`
* Per-stream StreamExecutor
* Per-stream threadpool if multiple threadpools are set in the ConfigProto
* Per-stream GPU threads if `TF_GPU_THREAD_MODE=gpu_private`

4. CUDA 11.0 and cuDNN 8 support

The native TF2.2 does not yet support CUDA 11.0 and cuDNN 8, and we add these supports for two reasons:

* To facilitate our testing on graphics cards such as the A100 model
* A fix in cuDNN 8.5 is crucial for the stable running of the multi-stream feature

## **Build Instruction**
1. Preparation
* Recommend building env and dependencies:
  - docker: ubuntu:18.04
  - CUDA: 11.0 (need manual installation)
  - cuDNN: 8.7 (need manual installation)
* Install packages:
```
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    openjdk-8-jdk\
    openjdk-8-jre-headless \
    libxml2 \
    nano \
    python3-dev \
    python3-pip \
    unzip
apt-get clean && \
    rm -rf /var/lib/apt/lists/*

BAZEL_VERSION=2.0.0
mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd -

ln -s /usr/bin/python3 /usr/bin/python
ln -s /usr/bin/pip3 /usr/bin/pip
pip --no-cache-dir install -U pip 'numpy<1.19.0' wheel packaging requests opt_einsum
pip --no-cache-dir install -U keras_preprocessing --no-deps
```

2. Configure TF

* Set environment variables
```
cuda_version="11.0"
cudnn_version="8"
compute_capabilities="8.0"
export TF_NEED_CUDA=1
export TF_NEED_TENSORRT=0
export TF_CUDA_VERSION=$cuda_version
export TF_CUDNN_VERSION=$cudnn_version
export TF_CUDA_COMPUTE_CAPABILITIES=$compute_capabilities
```
* Run ./configure
  - The generated .tf\_configure.bazelrc example:

```
build --action_env PYTHON_BIN_PATH="/usr/bin/python"
build --action_env PYTHON_LIB_PATH="/usr/local/lib/python3.6/dist-packages"
build --python_path="/usr/bin/python"
build --config=xla
build --action_env TF_CUDA_VERSION="11.0"
build --action_env TF_CUDNN_VERSION="8"
build --action_env CUDA_TOOLKIT_PATH="/usr/local/cuda-11.0"
build --action_env TF_CUDA_COMPUTE_CAPABILITIES="8.0"
build --action_env LD_LIBRARY_PATH="/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
build --action_env GCC_HOST_COMPILER_PATH="/usr/bin/x86_64-linux-gnu-gcc-7"
build --config=cuda
build:opt --copt=-march=native
build:opt --copt=-Wno-sign-compare
build:opt --host_copt=-march=native
build:opt --define with_default_optimizations=true
test --flaky_test_attempts=3
test --test_size_filters=small,medium
test --test_env=LD_LIBRARY_PATH
test:v1 --test_tag_filters=-benchmark-test,-no_oss,-no_gpu,-oss_serial
test:v1 --build_tag_filters=-benchmark-test,-no_oss,-no_gpu
test:v2 --test_tag_filters=-benchmark-test,-no_oss,-no_gpu,-oss_serial,-v1only
test:v2 --build_tag_filters=-benchmark-test,-no_oss,-no_gpu,-v1only
build --action_env TF_CONFIGURE_IOS="0"
```

3. Build the Multi-Stream demo

* Run the following command to build:

```
bazel build --config=opt --config=cuda --config=v2 \
    --verbose_failures \
    --output_filter=DONT_MATCH_ANYTHING \
    --noincompatible_do_not_split_linking_cmdline \
    //tensorflow/cc:tutorials_example_inference
```

## **Run Testing**

1. Basic settings.

```
export TF_GPU_STREAM_GROUP_COUNT=3
export TF_GPU_STREAM_MERGE=true

# command model_path number_of_input input_name number_of_output output_name batch_size iters number_of_threads
./bazel-bin/tensorflow/cc/tutorials_example_inference ./tensorflow/cc/tutorials/model_test.pbtxt 1 input0 1 output 16 1000 3
```

Set `TF_GPU_STREAM_GROUP_COUNT=N` to create N stream groups. `N=1` will fall back to the original TF performance.

Set `TF_GPU_STREAM_MERGE=true` to merge the compute stream, H2D copy stream, and D2H copy stream in one stream group into one stream.

2. Advanced settings.

```
export TF_GPU_STREAM_GROUP_COUNT=3
export TF_GPU_STREAM_MERGE=true
export TF_GPU_CONTEXT_COUNT=3
export TF_PER_STREAM_HOST_ALLOCATOR=true
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=1

# enable MPS
nvidia-cuda-mps-control -d

# command model_path number_of_input input_name number_of_output output_name batch_size iters number_of_threads use_xla number_of_threadpool number_of_threads_in_one_threadpool
./bazel-bin/tensorflow/cc/tutorials_example_inference ./tensorflow/cc/tutorials/model_test.pbtxt 1 input0 1 output 16 1000 3 1 3 1

# disable MPS
echo quit | nvidia-cuda-mps-control
```


Set `TF_GPU_CONTEXT_COUNT=N` to create the N stream groups in N CUDA contexts to reduce contention for the context lock. Enabling MPS by `nvidia-cuda-mps-control -d` is needed if multi-context is used.

Set `TF_PER_STREAM_HOST_ALLOCATOR=true` to create an exclusive GPU host allocator for every stream group. Caution: This may cause undefined behavior when multiple GPUs are used for one model.

If you set `use_xla` to 1, XLA JIT will be used for model compilation. This is useful for models with a lot of small kernels. (Doesn't work for our demo model because it's too small.)

If you set `number_of_threadpool` larger than 0, TF will create several thread pools in one session, rather than using a global thread pool. If you also use multi-stream, the `i`-th stream group will use the `i % number_of_threadpool`-th thread pool, so we recommend you set the number of the thread pool and the stream group the same. If multi-stream is not used, the user should specify which thread pool to use manually.

Furthermore, Set `number_of_threads_in_one_threadpool` to specify how many threads there are in each thread pool. Otherwise, TF will create as many threads as the number of CPU cores.

Set `TF_GPU_THREAD_MODE=gpu_private` and `TF_GPU_THREAD_COUNT=N` to use `N` additional threads for every stream group to launch kernels, instead of using the threads from the thread pool. Set `TF_GPU_THREAD_MODE=gpu_shared` to let all the stream groups share `N` additional threads.
