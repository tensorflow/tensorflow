# Build from source

This document describes how to build XLA components.

If you did not clone the XLA repository or install Bazel, check out the initial
sections of the [XLA Developer Guide](developer_guide.md).

## Linux

### Configure

XLA builds are configured by the `.bazelrc` file in the repository's root
directory. The `./configure.py` script can be used to adjust common settings.

If you need to change the configuration, run the `./configure.py` script from
the repository's root directory. This script has flags for the location of XLA
dependencies and additional build configuration options (compiler flags, for
example). Refer to the *Sample session* section for details.

### CPU support

We recommend using a suitable docker container to build/test XLA, such as
[TensorFlow's docker container](https://www.tensorflow.org/install/docker):

```
docker run --name xla -w /xla -it -d --rm -v $PWD:/xla tensorflow/build:latest-python3.9 bash
```

Using a docker container you can build XLA with CPU support using the following
commands:

```
docker exec xla ./configure.py --backend=CPU
docker exec xla bazel build //xla/...  --spawn_strategy=sandboxed --test_output=all
```

If you want to build XLA targets with CPU support without Docker you need to
install clang. XLA currently builds on CI with clang-17, but earlier versions
should also work:

```
apt install clang
```

Then configure and build targets using the following commands:

```sh
./configure.py --backend=CPU
bazel build --test_output=all --spawn_strategy=sandboxed //xla/...
```

### GPU support

We recommend using the same docker container as above to build XLA with GPU
support:

```
docker run --name xla_gpu -w /xla -it -d --rm -v $PWD:/xla tensorflow/build:latest-python3.9 bash
```

To build XLA with GPU support use the following command:

```
docker exec xla_gpu ./configure.py --backend=CUDA
docker exec xla_gpu bazel build --test_output=all --spawn_strategy=sandboxed //xla/...
```

For more details regarding
[TensorFlow's GPU docker images you can check out this document.](https://www.tensorflow.org/install/source#gpu_support_2)

You can build XLA targets with GPU support without Docker as well. Configure and
build targets using the following commands:

```
./configure.py --backend=CUDA

bazel build --test_output=all --spawn_strategy=sandboxed //xla/...
```

For more details regarding
[hermetic CUDA you can check out this document.](hermetic_cuda.md)
