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

2.  Create and run a
    [TensorFlow Docker container](https://www.tensorflow.org/install/docker).

    To get the TensorFlow Docker image for both CPU and GPU building, run the
    following command:

    ```sh
    docker run --name xla -w /xla -it -d --rm -v $PWD:/xla tensorflow/build:latest-python3.9 bash
    ```

## Build

Build for CPU:

```sh
docker exec xla ./configure.py --backend=CPU
docker exec xla bazel build --test_output=all --spawn_strategy=sandboxed //xla/...
```

Build for GPU:

```sh
docker exec xla ./configure.py --backend=CUDA
docker exec xla bazel build --test_output=all --spawn_strategy=sandboxed //xla/...
```

**NB:** please note that with hermetic CUDA rules, you don't have to build XLA
in Docker. You can build XLA for GPU on your machine without GPUs and without
NVIDIA driver installed:

```sh
./configure.py --backend=CUDA

bazel build --test_output=all --spawn_strategy=sandboxed //xla/...
```

Your first build will take quite a while because it has to build the entire
stack, including XLA, MLIR, and StableHLO.

To learn more about building XLA, see [Build from source](build_from_source.md).

## Create a pull request

When you're ready to send changes for review, create a
[pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

To learn about the XLA code review philosophy, see
[Review Process](contributing.md#review-process).
