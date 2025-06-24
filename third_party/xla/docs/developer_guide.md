# XLA developer guide

This guide shows you how to get started developing the XLA project.

Before you begin, complete the following prerequisites:

1.  Go to [Contributing page](contributing.md) and review the contribution
    process.
2.  If you haven't already done so, sign the
    [Contributor License Agreement](https://cla.developers.google.com/about).
3.  Install or configure the following dependencies:
    -   A [GitHub](https://github.com/) account
    -   [Docker](https://www.docker.com/)

Then follow the steps below to get the source code, set up an environment, build
the repository, and create a pull request.

## Get the code

1.  Create a fork of the [XLA repository](https://github.com/openxla/xla).
2.  Clone your fork of the repo, replacing `{USER}` with your GitHub username:
    ```sh
    git clone https://github.com/{USER}/xla.git
    ```

3.  Change into the `xla` directory: `cd xla`

4.  Configure the remote upstream repo:
    ```sh
    git remote add upstream https://github.com/openxla/xla.git
    ```

## Set up an environment

1.  Install [Bazel](https://bazel.build/install).

    To build XLA, you must have Bazel installed. The recommended way to install
    Bazel is using [Bazelisk](https://github.com/bazelbuild/bazelisk#readme),
    which automatically downloads the correct Bazel version for XLA. If Bazelisk
    is unavailable, you can [install Bazel](https://bazel.build/install)
    manually.

2.  Create and run the
    [ml-build](https://us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build)
    Docker container.

    To set up a Docker container for building XLA with support for both CPU and
    GPU, run the following command:

    ```sh
    docker run -itd --rm \
      --name xla \
      -w /xla \
      -v $PWD:/xla \
      us-docker.pkg.dev/ml-oss-artifacts-published/ml-public-container/ml-build:latest \
      bash
    ```

    If building with GPU/CUDA support, add `--gpus all` to grant the container
    access to all available GPUs. This enables automatic detection of CUDA
    compute capabilities.

## Build

Configure for CPU:

```sh
docker exec xla ./configure.py --backend=CPU
```

Configure for GPU:

```sh
docker exec xla ./configure.py --backend=CUDA
```

CUDA compute capabilities will be detected automatically by running
`nvidia-smi`. If GPUs are not available during the build, you must specify
the compute capabilities manually. For example:

```sh
# Automatically detects compute capabilities (requires GPUs)
./configure.py --backend=CUDA

# Manually specify compute capabilities (for builds without GPUs)
./configure.py --backend=CUDA --cuda_compute_capabilities="9.0"
```

Build:

```sh
docker exec xla bazel build \
  --spawn_strategy=sandboxed \
  --test_output=all \
  //xla/...
```

**Note:** You can build XLA on a machine without GPUs. In that case:

- Do **not** use `--gpus all` flag when starting the Docker container.
- During `./configure.py`, manually specify the CUDA compute capabilities
using the `--cuda_compute_capabilities` flag.

**Note:** Thanks to hermetic CUDA rules, you don't need to build XLA inside a
Docker container. You can build XLA for GPU directly on your machine - even if
it doesn't have a GPU or the NVIDIA driver installed.

```sh
# Automatically detects compute capabilities (requires GPUs)
./configure.py --backend=CUDA

# Manually specify compute capabilities (for builds without GPUs)
./configure.py --backend=CUDA --cuda_compute_capabilities="9.0"

bazel build \
  --spawn_strategy=sandboxed \
  --test_output=all \
  //xla/...
```

Your first build will take quite a while because it has to build the entire
stack, including XLA, MLIR, and StableHLO.

To learn more about building XLA, see [Build from source](build_from_source.md).

## Create a pull request

When you're ready to send changes for review, create a
[pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

To learn about the XLA code review philosophy, see
[Review Process](contributing.md#review-process).
