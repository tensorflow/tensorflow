# Official CI Directory

> **Warning** This folder is still under construction. It is part of an ongoing
> effort to improve the structure of CI and build related files within the
> TensorFlow repo. This warning will be removed when the contents of this
> directory are stable and appropriate documentation around its usage is in
> place.

Maintainer: TensorFlow and TensorFlow DevInfra

Issue Reporting: File an issue against this repo and tag
[@devinfra](https://github.com/orgs/tensorflow/teams/devinfra)

********************************************************************************

This directory contains TensorFlow's official CI build scripts and tools. The
TensorFlow team uses these for:

1.  Publishing the `tf-nightly` Python packages and other automated TensorFlow
    release artifacts
2.  Building (but not publishing) the `tensorflow` Python packages and other
    non-automated release artifacts
3.  Performing presubmit, continuous, and scheduled tests to verify TensorFlow's
    correctness. These mostly consist of `bazel test` invocations.
4.  Performing maintenance on the TensorFlow build configuration, such as the
    tooling in [requirements_updater](requirements_updater)

This directory only contains build scripts, tools, and environment settings. It
does not include any orchestration. TensorFlow uses both Kokoro (a Google
internal system) and GitHub Actions for orchestration and scheduling, and those
are not configured in this directory.

TensorFlow's CI tests cover a number of different platforms and configurations,
such as:

-   Linux, MacOS, and Windows
-   CPU-only or with NVIDIA CUDA support (Linux only)
-   Support for different Python versions.

The scripts are configured with settings files (`env/`) to keep them tidy, and
each script reads its settings from the file denoted by the `TFCI` environment
variable. All settings are prepended by `TFCI_`. Executing a build script looks
like this:

```
cd <tensorflow-root-directory>
mkdir -p build_output
cp ci/official/envs/nightly_linux_x86_cuda_py311 build_output/env
vim build_output/env  # Fix TFCI_BAZEL_COMMON_ARGS
export TFCI=$(realpath build_output/env)
./ci/official/wheel.sh
ls build_output
```

The scripts are intended to be easy to use for both CI systems and for local
replication.

### The Important TensorFlow Test Scripts

Generally speaking, changes to TensorFlow are gated by these test scripts:

1.  `wheel.sh` builds the TensorFlow Pip package and verifies its contents.
2.  `pycpp.sh` runs an extensive `bazel test` suite whose targets vary depending
    on the platform (target selection is handled in TensorFlow's bazelrc)
3.  `libtensorflow.sh` builds
    [libtensorflow](https://www.tensorflow.org/install/lang_c).
4.  `code_check_full.sh` and `code_check_changed_files.sh` run some static code
    analysis checks.

Our CI runs these under a variety of environments that will receive additional
documentation in the future.

### Running Tests Yourself

To run tests yourself, you'll copy an `env` file, adjust it to match your
environment, `export TFCI=your-path`, and then simply run the script you want.
We generally use `<tensorflow-directory>/build_output` for all temporary storage
and build artifacts, and you'll find all output files there.

You can find out which env file a TensorFlow job used by looking at the `TFCI`
variable in the `BUILD_CONFIG` section of the Invocation Details for that job,
either in Sponge (internal) or ResultStore (external). The files in `envs/` are
configured to match TensorFlow's CI system and reference paths and settings.

**You MUST make one configuration adjustment to successfully run the tests
locally**: in your copy of the env file, look at `TFCI_BAZEL_COMMON_ARGS` and
change these flags if they are present (they may have spaces instead of equal
signs):

-   Replace `--config=sigbuild_remote_cache_push` with
    `--config=sigbuild_remote_cache`
-   Remove `--config=rbe`
-   Remove `--config=resultstore`

All of these settings rely on `--google_default_credentials`. Bazel will abort
if you are not logged in to a GCP account or lack significant permissions for
TensorFlow's Google Cloud Platform account. Google employees can re-enable RBE
and ResultStore and use those features when building locally. Read the sections
below for more details.

Here is a complete example of how to set up and run a script:

```
cd <tensorflow-root-directory>
mkdir -p build_output
cp ci/official/envs/nightly_linux_x86_cuda_py311 build_output/env
vim build_output/env  # Fix TFCI_BAZEL_COMMON_ARGS
export TFCI=$(realpath build_output/env)
./ci/official/wheel.sh
ls build_output
```

### `env` File Settings

Options in `env` files should mostly be self-explanatory. Search within this
directory for options and their usage. This section will explain usage that is
not as obvious.

Variables are not order-dependent and can reference any `TFCI` variable defined
in the same file. In the examples below, many settings modify arrays (`ARRAY=(
first second third )`). Arrays must be merged to combine behavior their
behavior. For example, you can't do this:

```
# I want to use a GPU
TFCI_DOCKER_ARGS=( --gpus all )
# I want to use a separate Bazel cache
TFCI_DOCKER_ARGS=( -v "$HOME/bazelcache:/root/bazelcache" )
```

Instead, those settings must be defined as:

```
TFCI_DOCKER_ARGS=( --gpus all -v "$HOME/bazelcache:/root/bazelcache" )
```

The scripts determine the root git directory automatically when invoked, and run
all subsequent commands from that directory. Variables in `env` files may either
use full paths (`$SOME_FULLPATH/foo`) or paths relative to the root directory
(`foo` would then point to `tensorflow/foo`).

#### Run Tests with Docker

See also: [utilities/docker.sh](utilities/docker.sh) and
[utilities/cleanup_docker.sh](utilities/cleanup_docker.sh)

TensorFlow uses
[TensorFlow Build Docker images](https://hub.docker.com/r/tensorflow/build) for
most of its testing, including Remote Build Execution (RBE). Running with Docker
is the best way to replicate errors in TF official CI. You can disable docker
and run locally by setting `TFCI_DOCKER_ENABLE=0`. If you leave it enabled, the
scripts will:

1.  Pull the `TFCI_DOCKER_IMAGE` if it is not present, as long as
    `TFCI_DOCKER_PULL_ENABLE=1`
2.  Create a container named `tf` only if it is not present, with the TensorFlow
    root git directory (`TFCI_GIT_DIR`) mounted as a volume inside the container
    with the same path as the real root directory.
3.  Execute any `tfrun` script commands inside of the container
4.  **Not** clean up the container after the job is complete. You need to
    `docker rm -f tf` if you change container settings or wish to clean up.

Docker does not handle `ctrl-c` correctly. If you interrupt a bazel command, you
will need to run `docker exec tf pkill bazel` to forcibly abort it.

As a Google-internal developer or someone else with access to Remote Build
Execution (RBE), ResultStore, or the Remote Build Cache, You may want to mount
your Google Cloud credential files so your builds use your own Google account:

```
# Make sure you've run "gcloud auth application-default login" first.
TFCI_DOCKER_ARGS=( -v "$HOME/.config/gcloud:/root/.config/gcloud" )
```

You can also enable GPU passthrough if you are using the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/overview.html):

```
TFCI_DOCKER_ARGS=( --gpus all )
```

You may want to mount a directory to use as a bazel cache:

```
TFCI_DOCKER_ARGS=( -v "$HOME/bazelcache:/root/bazelcache" )
TFCI_BAZEL_COMMON_ARGS=( --disk_cache=/root/bazelcache )
```

#### Speed up builds with a Build Cache

All bazel users can take advantage of Bazel's
[build cache](https://bazel.build/remote/caching) when building TensorFlow to
save a lot (sometimes hours) of build time. Some things to keep in mind:

-   The cache is not infallible and can sometimes be corrupted or give bad
    results. We're not entirely sure how often this happens, but the TF team has
    discovered that significant environmental changes (like new CUDA versions,
    changes to manylinux, compilers, etc.) can cause undefined behavior (build
    failures, tests pass when they shouldn't, etc.)
-   The cache is often invalidated by LLVM updates, which TensorFlow pulls in
    continuously. It can even be invalidated with no changes to the git
    repository, so keep that in mind if your build mysteriously takes a long
    time.

The simplest and most effective way to use the cache is to combine a local cache
with TensorFlow's remote cache. TensorFlow's official nightly builds push to a
publicly accessible cache, which you can combine with a local bazel cache:

```
TFCI_BAZEL_COMMON_ARGS=( --disk_cache=$TFCI_OUT_DIR/cache --config=sigbuild_remote_cache )
```

This will place a bazel cache in `$TFCI_OUT_DIR/cache`, which by default
resolves to `build_output/cache`. On Docker, since `build_output` is inside the
TensorFlow source code volume mount, the cache directory will not be deleted
when the container is removed.

If you're using Docker, you can also mount a specific directory to use as the
cache if you want to use a system-wide bazel cache:

```
TFCI_DOCKER_ARGS=( -v "$HOME/bazelcache:/root/bazelcache" )
TFCI_BAZEL_COMMON_ARGS=( --disk_cache=/root/bazelcache )
```

Keep these additional details in mind when using the cache:

-   The official nightly builds use `--config=sigbuild_remote_cache_push` to
    push the results of `wheel.sh` to a remote cache. Our CI must use Bazel's
    `--google_default_credentials` flag to pull upload credentials from the
    virtual machine, but the flag raises an error if no credentials are
    available.
-   It's safe to use `--config=sigbuild_remote_cache_push` (the default config)
    as a normal developer because you don't have upload permission, but you
    should switch it to `sigbuild_remote_cache` if you're a Google developer.
    There is no cache for `pycpp.sh`, because the official jobs use Remote Build
    Execution instead.

#### Speed up builds with Remote Build Execution

A limited set of authenticated users (mostly internal Google developers) can use
[Remote Build Execution](https://bazel.build/remote/rbe#:~:text=Remote%20execution%20of%20a%20Bazel,nodes%20available%20for%20parallel%20actions)
(RBE) on the same GCP project that TensorFlow itself uses. RBE is much faster
than a local cache and performs the bazel invocation on container clusters in
GCP. Make sure you have `gcloud` configured, and that you've run `gcloud auth
application-default login`.

```
TFCI_BAZEL_COMMON_ARGS=( --config=rbe )
```

If you're using Docker, you must mount your GCP credentials in the container:

```
TFCI_DOCKER_ARGS=( -v "$HOME/.config/gcloud:/root/.config/gcloud" )
```

#### Share build results by uploading to ResultStore

A limited set of authenticated users (mostly internal Google developers) can
upload Bazel results to GCP ResultStore, which is the same service that
TensorFlow uses to share its public build results. Make sure you have `gcloud`
configured, and that you've run `gcloud auth application-default login`.

```
TFCI_BAZEL_COMMON_ARGS=( --config=resultstore )
```

If you're using Docker, you must mount your GCP credentials in the container:

```
TFCI_DOCKER_ARGS=( -v "$HOME/.config/gcloud:/root/.config/gcloud" )
```

Running a build script will print a list of ResultStore URLs after the script
terminates.

#### Uploading Release Artifacts

Artifact uploads, part of the TF release process, are controlled by
`UPLOAD_ENABLE` variables. Normal users will not have the authentication
necessary to perform uploads, but it's possible a Google developer could. As
such, all `UPLOAD_ENABLE` variables are set to `${TFCI_GLOBAL_UPLOAD_ENABLE:-}`,
which quietly resolves to empty unless `TFCI_GLOBAL_UPLOAD_ENABLE=1`.
TensorFlow's CI system sets this variable explicitly. Developers have no reason
to set it.
