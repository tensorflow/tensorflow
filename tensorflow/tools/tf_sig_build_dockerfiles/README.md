# TF SIG Build Dockerfiles

Standard Dockerfiles for TensorFlow builds, used internally at Google.

Maintainer: @angerson (TensorFlow OSS DevInfra; SIG Build)

* * *

These docker containers are for building and testing TensorFlow in CI
environments (and for users replicating those CI builds). They are openly
developed in TF SIG Build, verified by Google developers, and published to
tensorflow/build on [Docker Hub](https://hub.docker.com/r/tensorflow/build/).
The TensorFlow OSS DevInfra team uses these containers for most of our
Linux-based CI, including `tf-nightly` tests and Pip packages and TF release
packages for TensorFlow 2.9 onwards.

## Tags

These Dockerfiles are built and deployed to [Docker
Hub](https://hub.docker.com/r/tensorflow/build/) via [Github
Actions](https://github.com/tensorflow/tensorflow/blob/master/.github/workflows/sigbuild-docker.yml).

The tags are defined as such:

- The `latest` tags are kept up-to-date to build TensorFlow's `master` branch.
- The `version number` tags target the corresponding TensorFlow version. We
  continuously build the `current-tensorflow-version + 1` tag, so when a new
  TensorFlow branch is cut, that Dockerfile is frozen to support that branch.
- We support the same Python versions that TensorFlow does.

## Updating the Containers

For simple changes, you can adjust the source files and then make a PR. Send it
to @angerson for review. We have presubmits that will make sure your change
still builds a container. After approval and submission, our GitHub Actions
workflow deploys the containers to Docker Hub.

- To update Python packages, look at `devel.requirements.txt`
- To update system packages, look at `devel.packages.txt`
- To update the way `bazel build` works, look at `devel.usertools/*.bazelrc`.

To rebuild the containers locally after making changes, use this command from
this directory:

For CUDA
```bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg PYTHON_VERSION=python3.9 --target=devel -t my-tf-devel .
```
For ROCM
```
DOCKER_BUILDKIT=1 docker build -f Dockerfile.rocm \
  --build-arg ROCM_VERSION=5.4.0 --build-arg PYTHON_VERSION=python3.9 -t my-tf-devel .
```

It will take a long time to build devtoolset and install CUDA packages. After
it's done, you can use the commands below to test your changes. Just replace
`tensorflow/build:latest-python3.9` with `my-tf-devel` to use your image
instead.

### Automatic GCR.io Builds for Presubmits

TensorFlow team members (i.e. Google employees) can apply a `Build and deploy
to gcr.io for staging` tag to their PRs to the Dockerfiles, as long as the PR
is being developed on a branch of this repository, not a fork. Unfortunately
this is not available for non-Googler contributors for security reasons.

## Run the TensorFlow Team's Nightly Test Suites with Docker

The TensorFlow DevInfra team runs a daily test suite that builds `tf-nightly`
and runs a `bazel test` suite on both the Pip package (the "pip" tests) and
on the source code itself (the "nonpip" tests). These test scripts are often
referred to as "The Nightly Tests" and can be a common reason for a TF PR to be
reverted. The build scripts aren't visible to external users, but they use
the configuration files which are included in these containers. Our test suites,
which include the build of `tf-nightly`, are easy to replicate with these
containers, and here is how you can do it.

Presubmits are not using these containers... yet.

Here are some important notes to keep in mind:

- The Ubuntu CI jobs that build the `tf-nightly` package build at the GitHub
  `nightly` tag. You can see the specific commit of a `tf-nightly` package on
  pypi.org in `tf.version.GIT_VERSION`, which will look something like
  `v1.12.1-67282-g251085598b7`. The final section, `g251085598b7`, is a short
  git hash.

- If you interrupt a `docker exec` command with `ctrl-c`, you will get your
  shell back but the command will continue to run. You cannot reattach to it,
  but you can kill it with `docker kill tf` (or `docker kill the-container-name`).
  This will destroy your container but will not harm your work since it's
  mounted.  If you have any suggestions for handling this better, let us know.

Now let's build `tf-nightly`.

1. Set up your directories:

    - A directory with the TensorFlow source code, e.g. `/tmp/tensorflow`
    - A directory for TensorFlow packages built in the container, e.g. `/tmp/packages`
    - A directory for your local bazel cache (can be empty), e.g. `/tmp/bazelcache`

2. Choose the Docker container to use from [Docker
   Hub](https://hub.docker.com/r/tensorflow/build/tags). The options for the
   `master` branch are:

For CUDA

    - `tensorflow/build:latest-python3.11`
    - `tensorflow/build:latest-python3.10`
    - `tensorflow/build:latest-python3.9`
    - `tensorflow/build:latest-python3.8`

For ROCM

    - `rocm/tensorflow-build:latest-python3.11`
    - `rocm/tensorflow-build:latest-python3.10`
    - `rocm/tensorflow-build:latest-python3.9`
    - `rocm/tensorflow-build:latest-python3.8`

    For this example we'll use `tensorflow/build:latest-python3.9`.

3. Pull the container you decided to use.

For CUDA

    ```bash
    docker pull tensorflow/build:latest-python3.9
    ```

For ROCM

    ```bash
    docker pull rocm/tensorflow-build:latest-python3.9
    ```

4. Start a backgrounded Docker container with the three folders mounted.

    - Mount the TensorFlow source code to `/tf/tensorflow`.
    - Mount the directory for built packages to `/tf/pkg`.
    - Mount the bazel cache to `/tf/cache`. You don't need `/tf/cache` if
      you're going to use the remote cache.

    Here are the arguments we're using:

    - `--name tf`: Names the container `tf` so we can refer to it later.
    - `-w /tf/tensorflow`: All commands run in the `/tf/tensorflow` directory,
      where the TF source code is.
    - `-it`: Makes the container interactive for running commands
    - `-d`: Makes the container start in the background, so we can send
      commands to it instead of running commands from inside.

    And `-v` is for mounting directories into the container.

    For CUDA
    ```bash
    docker run --name tf -w /tf/tensorflow -it -d \
      -v "/tmp/packages:/tf/pkg" \
      -v "/tmp/tensorflow:/tf/tensorflow" \
      -v "/tmp/bazelcache:/tf/cache" \
      tensorflow/build:latest-python3.9 \
      bash
    ```

    For ROCM
    ```
    docker run --name tf -w /tf/tensorflow -it -d --network=host \
    --device=/dev/kfd --device=/dev/dri \
    --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    -v "/tmp/packages:/tf/pkg" \
    -v "/tmp/tensorflow:/tf/tensorflow" \
    -v "/tmp/bazelcache:/tf/cache" \
    rocm/tensorflow-build:latest-python3.9 \
    bash
    ```

    Note: if you wish to use your own Google Cloud Platform credentials for
    e.g. RBE, you may also wish to set `-v
    $HOME/.config/gcloud:/root/.config/gcloud` to make your credentials
    available to bazel. You don't need to do this unless you know what you're
    doing.

Now you can continue on to any of:

- Build `tf-nightly` and then (optionally) run a test suite on the pip package
  (the "pip" suite)
- Run a test suite on the TF code directly (the "nonpip" suite)
- Build the libtensorflow packages (the "libtensorflow" suite)
- Run a code-correctness check (the "code_check" suite)

### Build `tf-nightly` and run Pip tests

1. Apply the `update_version.py` script that changes the TensorFlow version to
   `X.Y.Z.devYYYYMMDD`. This is used for `tf-nightly` on PyPI and is technically
   optional.

    ```bash
    docker exec tf python3 tensorflow/tools/ci_build/update_version.py --nightly
    ```

2. Build TensorFlow by following the instructions under one of the collapsed
   sections below. You can build both CPU and GPU packages without a GPU. TF
   DevInfra's remote cache is better for building TF only once, but if you
   build over and over, it will probably be better in the long run to use a
   local cache. We're not sure about which is best for most users, so let us
   know on [Gitter](https://gitter.im/tensorflow/sig-build).

   This step will take a long time, since you're building TensorFlow. GPU takes
   much longer to build. Choose one and click on the arrow to expand the
   commands:

    <details><summary>TF Nightly CPU - Remote Cache</summary>

    Build the sources with Bazel:

    ```
    docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
    build --config=sigbuild_remote_cache \
    tensorflow/tools/pip_package:build_pip_package
    ```

    And then construct the pip package:

    ```
    docker exec tf \
      ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
      /tf/pkg \
      --cpu \
      --nightly_flag
    ```

    </details>

    <details><summary>TF Nightly GPU - Remote Cache</summary>

    Build the sources with Bazel:

    ```
    docker exec tf bazel --bazelrc=/usertools/gpu.bazelrc \
    build --config=sigbuild_remote_cache --config=cuda \
    tensorflow/tools/pip_package:build_pip_package
    ```

    And then construct the pip package:

    ```
    docker exec tf \
      ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
      /tf/pkg \
      --nightly_flag
    ```

    </details>

    <details><summary>TF Nightly CPU - Local Cache</summary>

    Make sure you have a directory mounted to the container in `/tf/cache`!

    Build the sources with Bazel:

    ```
    docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
    build --config=sigbuild_local_cache \
    tensorflow/tools/pip_package:build_pip_package
    ```

    And then construct the pip package:

    ```
    docker exec tf \
      ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
      /tf/pkg \
      --cpu \
      --nightly_flag
    ```

    </details>

    <details><summary>TF Nightly GPU - Local Cache</summary>

    Make sure you have a directory mounted to the container in `/tf/cache`!

    For CUDA:
    Build the sources with Bazel and the cuda config:

    ```
    docker exec tf \
    bazel --bazelrc=/usertools/gpu.bazelrc \
    build --config=sigbuild_local_cache --config=cuda \
    tensorflow/tools/pip_package:build_pip_package
    ```

    And then construct the pip package:

    ```
    docker exec tf \
      ./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
      /tf/pkg \
      --nightly_flag
    ```

    For ROCM:

    Build the sources with Bazel and the rocm config:

    ```
    docker exec tf \
    bazel --bazelrc=/usertools/gpu.bazelrc \
    build --config=sigbuild_local_cache --config=rocm \
    tensorflow/tools/pip_package:build_pip_package --verbose_failures

    ```

    And then construct the nightly pip package:

    ```
    docker exec tf \
	./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
	/tf/pkg \
	--rocm \
	--nightly_flag
    ```

    Note: if you are creating a release (non-nightly) pip package:
    ```
    docker exec tf \
	./bazel-bin/tensorflow/tools/pip_package/build_pip_package \
	/tf/pkg \
	--rocm \
	--project_name tensorflow_rocm
    ```

    </details>

3. Run the helper script that checks for manylinux compliance, renames the
   wheels, and then checks the size of the packages.

    For CUDA

    ```
    docker exec tf /usertools/rename_and_verify_wheels.sh
    ```

    For ROCM

    ```
    docker exec tf /usertools/rename_and_verify_ROCM_wheels.sh
    ```

4. Take a look at the new wheel packages you built! They may be owned by `root`
   because of how Docker volume permissions work.

    ```
    ls -al /tmp/packages
    ```

5. To continue on to running the Pip tests, create a venv and install the
   testing packages:

    ```
    docker exec tf /usertools/setup_venv_test.sh bazel_pip "/tf/pkg/tf_nightly*.whl"
    ```

6. And now run the tests depending on your target platform: `--config=pip`
   includes the same test suite that is run by the DevInfra team every night.
   If you want to run a specific test instead of the whole suite, pass
   `--config=pip_venv` instead, and then set the target on the command like
   normal.

    <details><summary>TF Nightly CPU - Remote Cache</summary>

    Build the sources with Bazel:

    ```
    docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
    test --config=sigbuild_remote_cache \
    --config=pip
    ```

    </details>

    <details><summary>TF Nightly GPU - Remote Cache</summary>

    Build the sources with Bazel:

    ```
    docker exec tf bazel --bazelrc=/usertools/gpu.bazelrc \
    test --config=sigbuild_remote_cache \
    --config=pip
    ```

    </details>

    <details><summary>TF Nightly CPU - Local Cache</summary>

    Make sure you have a directory mounted to the container in `/tf/cache`!

    Build the sources with Bazel:

    ```
    docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
    test --config=sigbuild_local_cache \
    --config=pip
    ```

    </details>

    <details><summary>TF Nightly GPU - Local Cache</summary>

    Make sure you have a directory mounted to the container in `/tf/cache`!

    Build the sources with Bazel:

    ```
    docker exec tf \
    bazel --bazelrc=/usertools/gpu.bazelrc \
    test --config=sigbuild_local_cache \
    --config=pip
    ```

    </details>

### Run Nonpip Tests

1. Run the tests depending on your target platform. `--config=nonpip` includes
   the same test suite that is run by the DevInfra team every night. If you
   want to run a specific test instead of the whole suite, you do not need
   `--config=nonpip` at all; just set the target on the command line like usual.

    <details><summary>TF Nightly CPU - Remote Cache</summary>

    Build the sources with Bazel:

    ```
    docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
    test --config=sigbuild_remote_cache \
    --config=nonpip
    ```

    </details>

    <details><summary>TF Nightly GPU - Remote Cache</summary>

    Build the sources with Bazel:

    ```
    docker exec tf bazel --bazelrc=/usertools/gpu.bazelrc \
    test --config=sigbuild_remote_cache \
    --config=nonpip
    ```

    </details>

    <details><summary>TF Nightly CPU - Local Cache</summary>

    Make sure you have a directory mounted to the container in `/tf/cache`!

    Build the sources with Bazel:

    ```
    docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
    test --config=sigbuild_local_cache \
    --config=nonpip
    ```

    </details>

    <details><summary>TF Nightly GPU - Local Cache</summary>

    Make sure you have a directory mounted to the container in `/tf/cache`!

    Build the sources with Bazel:

    ```
    docker exec tf \
    bazel --bazelrc=/usertools/gpu.bazelrc \
    test --config=sigbuild_local_cache \
    --config=nonpip
    ```

    </details>

### Test, build and package libtensorflow

1. Run the tests depending on your target platform.
   `--config=libtensorflow_test` includes the same test suite that is run by
   the DevInfra team every night. If you want to run a specific test instead of
   the whole suite, just set the target on the command line like usual.

    <details><summary>TF Nightly CPU - Remote Cache</summary>

    ```
    docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
    test --config=sigbuild_remote_cache \
    --config=libtensorflow_test
    ```

    </details>

    <details><summary>TF Nightly GPU - Remote Cache</summary>

    ```
    docker exec tf bazel --bazelrc=/usertools/gpu.bazelrc \
    test --config=sigbuild_remote_cache \
    --config=libtensorflow_test
    ```

    </details>

    <details><summary>TF Nightly CPU - Local Cache</summary>

    Make sure you have a directory mounted to the container in `/tf/cache`!

    ```
    docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
    test --config=sigbuild_local_cache \
    --config=libtensorflow_test
    ```

    </details>

    <details><summary>TF Nightly GPU - Local Cache</summary>

    Make sure you have a directory mounted to the container in `/tf/cache`!

    ```
    docker exec tf \
    bazel --bazelrc=/usertools/gpu.bazelrc \
    test --config=sigbuild_local_cache \
    --config=libtensorflow_test
    ```

    </details>

1. Build the libtensorflow packages.

    <details><summary>TF Nightly CPU - Remote Cache</summary>

    ```
    docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
    build --config=sigbuild_remote_cache \
    --config=libtensorflow_build
    ```

    </details>

    <details><summary>TF Nightly GPU - Remote Cache</summary>

    ```
    docker exec tf bazel --bazelrc=/usertools/gpu.bazelrc \
    build --config=sigbuild_remote_cache \
    --config=libtensorflow_build
    ```

    </details>

    <details><summary>TF Nightly CPU - Local Cache</summary>

    Make sure you have a directory mounted to the container in `/tf/cache`!

    ```
    docker exec tf bazel --bazelrc=/usertools/cpu.bazelrc \
    build --config=sigbuild_local_cache \
    --config=libtensorflow_build
    ```

    </details>

    <details><summary>TF Nightly GPU - Local Cache</summary>

    Make sure you have a directory mounted to the container in `/tf/cache`!

    ```
    docker exec tf \
    bazel --bazelrc=/usertools/gpu.bazelrc \
    build --config=sigbuild_local_cache \
    --config=libtensorflow_build
    ```

    </details>

1. Run the `repack_libtensorflow.sh` utility to repack and rename the archives.

    <details><summary>CPU</summary>

    ```
    docker exec tf /usertools/repack_libtensorflow.sh /tf/pkg "-cpu-linux-x86_64"
    ```

    </details>

    <details><summary>GPU</summary>

    ```
    docker exec tf /usertools/repack_libtensorflow.sh /tf/pkg "-gpu-linux-x86_64"
    ```

    </details>

### Run a code check

1. Every night the TensorFlow team runs `code_check_full`, which contains a
   suite of checks that were gradually introduced over TensorFlow's lifetime
   to prevent certain unsable code states. This check has supplanted the old
   "sanity" or "ci_sanity" checks.

    ```
    docker exec tf bats /usertools/code_check_full.bats --timing --formatter junit
    ```

### Clean Up

1. Shut down and remove the container when you are finished.

    ```
    docker stop tf
    docker rm tf
    ```
