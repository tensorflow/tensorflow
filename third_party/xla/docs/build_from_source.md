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

### Build XLA with CUDA/cuDNN Support Using the JAX CI/Release Container

XLA is a compiler used internally by JAX.
JAX is distributed via PyPI wheels.
The [JAX Continuous Integration documentation](https://github.com/jax-ml/jax/tree/main/ci#running-these-scripts-locally-on-your-machine)
explains how to build JAX wheels using
the [tensorflow/ml-build:latest](https://us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build) Docker container.

We can extend these instructions to build XLA targets within the JAX container
as well. This ensures that the XLA targets' build configuration is consistent
with the JAX/XLA build configuration, which can be useful if we want to
reproduce workload results using XLA tools that were originally created in JAX.

#### Build XLA Targets in the JAX CI Container

1. Clone the JAX repository and navigate to the 'jax' directory
```bash
git clone https://github.com/jax-ml/jax.git

cd jax
```

2. Start JAX CI/Release Docker container by running:
```bash
./ci/utilities/run_docker_container.sh
```
This will start a Docker container named 'jax'.

3. Build the jax-cuda-plugin target inside the container using:
```bash
docker exec jax ./ci/build_artifacts.sh jax-cuda-plugin
```
This will create the .jax_configure.bazelrc file with the required build
configuration, including CUDA/cuDNN support

4. Access an interactive shell inside the container:
```bash
docker exec -ti jax /bin/bash
```
You should now be in the `/jax` directory within the container

5. Build the XLA target with the following command, e.g.:
```bash
/usr/local/bin/bazel build \
  --config=cuda_libraries_from_stubs \
  --verbose_failures=true \
  @local_xla//xla/tools/multihost_hlo_runner:hlo_runner_main
```

Optionally, you can overwrite `HERMETIC` envs, e.g.:
```
--repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES="sm_90"
```

6. Copy the resulting artifacts to `/jax/dist` to access them from the host OS
if needed
```bash
cp bazel-bin/external/xla/xla/tools/multihost_hlo_runner/hlo_runner_main \
  ./dist/
```

7. Exit the interactive shell:
```bash
exit
```
