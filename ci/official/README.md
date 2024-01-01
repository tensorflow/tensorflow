# Official CI Directory

Maintainer: TensorFlow and TensorFlow DevInfra

Issue Reporting: File an issue against this repo and tag
[@devinfra](https://github.com/orgs/tensorflow/teams/devinfra)

********************************************************************************

## TensorFlow's Official CI and Build/Test Scripts

TensorFlow's official CI jobs run the scripts in this folder. Our internal CI
system, Kokoro, schedules our CI jobs by combining a build script with a file
from the `envs` directory that is filled with configuration options:

-   Nightly jobs (Run nightly on the `nightly` branch)
    -   Uses `wheel.sh`, `libtensorflow.sh`, `code_check_full.sh`
    -   Uses `envs/nightly_...`
-   Continuous jobs (Run on every GitHub commit)
    -   Uses `pycpp.sh`
    -   Uses `envs/continuous_...`
-   Presubmit jobs (Run on every GitHub PR)
    -   Uses `pycpp.sh`, `code_check_changed_files.sh`
    -   Also uses `envs/continuous_...`

These "env" files match up with an environment matrix that roughly covers:

-   Linux, MacOS, and Windows machines (these pool definitions are internal)
-   x86 and arm64
-   CPU-only, or with NVIDIA CUDA support (Linux only), or with TPUs
-   Different Python versions

## How to Test Your Changes to TensorFlow

You may check how your changes will affect TensorFlow by:

1. Creating a PR and observing the presubmit test results
2. Running the CI scripts locally, as explained below
3. **Google employees only**: Google employees can use an internal-only tool
called "MLCI" that makes testing more convenient: it can execute any full CI job
against a pending change. Search for "MLCI" internally to find it.

You may invoke a CI script of your choice by following these instructions:

```bash
# Required: create a file to hold your settings
cd tensorflow-git-dir
export TFCI=$(mktemp)

# Required: choose an environment (env) to copy.
#   If you've clicked on a test result from our CI (via a dashboard or GitHub link),
#   click to "Invocation Details" and find BUILD_CONFIG, which will contain a TFCI
#   value in the "env_vars" list that you can choose to copy that environment.
echo >>$TFCI source ci/official/envs/nightly_linux_x86_cpu_py311

# Required: Reset settings for local execution
echo >>$TFCI source ci/official/envs/local_default

# Recommended: use a local+remote cache.
#
#   Bazel will cache your builds in tensorflow/build_output/cache,
#   and will also try using public build cache results to speed up
#   your builds. This usually saves a lot of time, especially when
#   re-running tests. However, note that:
#
#    - New environments like new CUDA versions, changes to manylinux,
#      compilers, etc. can cause undefined behavior such as build failures
#      or tests passing incorrectly.
#    - Automatic LLVM updates are known to extend build time even with
#      the cache; this is unavoidable.
echo >>$TFCI source ci/official/envs/local_multicache

# Advanced: Use Remote Build Execution (RBE) (internal developers only)
#
#   RBE dramatically speeds up builds and testing. It also gives you a
#   public URL to share your build results with collaborators. However,
#   it is only available to a limited set of internal TensorFlow developers.
#
#   RBE is incompatible with local caching, so you must remove
#   ci/official/envs/local_multicache from your $TFCI file.
#
# To use RBE, you must first run `gcloud auth application-default login`, then:
# echo >>$TFCI source ci/official/envs/local_rbe

# Recommended: Configure Docker. (Linux only)
#
#   TF uses hub.docker.com/r/tensorflow/build containers for CI,
#   and scripts on Linux create a persistent container called "tf"
#   which mounts your TensorFlow directory into the container.
#
#   Important: because the container is persistent, you cannot change TFCI
#   variables in between script executions. To forcibly remove the
#   container and start fresh, run "docker rm -f tf". Removing the container
#   destroys some temporary bazel data and causes longer builds.
#
#   You will need the NVIDIA Container Toolkit for GPU testing:
#   https://github.com/NVIDIA/nvidia-container-toolkit
#
#   Note: if you interrupt a bazel command on docker (ctrl-c), you
#   will need to run `docker exec tf pkill bazel` to quit bazel.
#
#   Note: new files created from the container are owned by "root".
#   You can run e.g. `docker exec tf chown -R $(id -u):$(id -g) build_output`
#   to transfer ownership to your user.
#
# Docker is enabled by default on Linux. You may disable it if you prefer:
# echo >>$TFCI source ci/official/envs/local_nodocker

# Finally: Run your script of choice.
#   If you've clicked on a test result from our CI (via a dashboard or GitHub link),
#   click to "Invocation Details" and find BUILD_CONFIG, which will contain a
#   "build_file" item that indicates the script used.
ci/official/wheel.sh

# Advanced: Select specific build/test targets with "any.sh".
# TF_ANY_TARGETS=":your/target" TF_ANY_MODE="test" ci/official/any.sh

# Afterwards: Examine the results, which will include: The bazel cache,
# generated artifacts like .whl files, and "script.log", from the script.
# Note that files created under Docker will be owned by "root".
ls build_output
```

## Contribution & Maintenance

The TensorFlow team does not yet have guidelines in place for contributing to
this directory. We are working on it. Please join a TF SIG Build meeting (see:
bit.ly/tf-sig-build-notes) if you'd like to discuss the future of contributions.

### Brief System Overview

The top-level scripts and utility scripts should be fairly well-documented. Here
is a brief explanation of how they tie together:

1.  `envs/*` are lists of variables made with bash syntax. A user must set a
    `TFCI` env param pointing to one of the `env` files.
2.  `utilities/setup.sh`, initialized by all top-level scripts, reads and sets
    values from that `TFCI` path.
    -   `set -a` / `set -o allexport` exports the variables from `env` files so
        all scripts can use them.
    -   `utilities/setup_docker.sh` creates a container called `tf` with all
        `TFCI_` variables shared to it.
3.  Top-level scripts (`wheel.sh`, etc.) reference `env` variables and call
    `utilities/` scripts.
    -   The `tfrun` function makes a command run correctly in Docker if Docker
        is enabled.
