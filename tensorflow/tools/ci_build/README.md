# TensorFlow Builds

This directory contains all the files and setup instructions to run all
the important builds and tests. **You can trivially run it yourself!** It also
run continuous integration [ci.tensorflow.org](https://ci.tensorflow.org).



## Run It Yourself

1. Install [Docker](http://www.docker.com/). Follow instructions
   [on the Docker site](https://docs.docker.com/installation/).

   You can run all the jobs **without docker** if you are on mac or on linux
   and you just don't want docker. Just install all the dependencies from
   [Installing TensorFlow](https://www.tensorflow.org/install/).
   Then run any of the one liners below without the
   `tensorflow/tools/ci_build/ci_build.sh` in them.

2. Clone tensorflow repository.

   ```bash
   git clone https://github.com/tensorflow/tensorflow.git
   ```

3. Go to tensorflow directory

   ```bash
   cd tensorflow
   ```

4. Build what you want, for example

   ```bash
   tensorflow/tools/ci_build/ci_build.sh CPU bazel test //tensorflow/...
   ```



## Jobs

The jobs run by [ci.tensorflow.org](https://ci.tensorflow.org) include following:

```bash
# Note: You can run the following one-liners yourself if you have Docker. Run
# without `tensorflow/tools/ci_build/ci_build.sh` on mac or linux without Docker.

# build and run cpu tests
tensorflow/tools/ci_build/ci_build.sh CPU bazel test //tensorflow/...

# build and run gpu tests (note if you get unstable results you may be running
# out of gpu memory - if so add "--jobs=1" argument)
tensorflow/tools/ci_build/ci_build.sh GPU bazel test -c opt --config=cuda //tensorflow/...

# build pip with gpu support
tensorflow/tools/ci_build/ci_build.sh GPU tensorflow/tools/ci_build/builds/pip.sh GPU -c opt --config=cuda

# build and run gpu tests using python 3
CI_DOCKER_EXTRA_PARAMS="-e CI_BUILD_PYTHON=python3" tensorflow/tools/ci_build/ci_build.sh GPU tensorflow/tools/ci_build/builds/pip.sh GPU -c opt --config=cuda

# build android example app
tensorflow/tools/ci_build/ci_build.sh ANDROID tensorflow/tools/ci_build/builds/android.sh

# cmake cpu build and test
tensorflow/tools/ci_build/ci_build.sh CPU tensorflow/tools/ci_build/builds/cmake.sh

# run bash inside the container
CI_DOCKER_EXTRA_PARAMS='-it --rm' tensorflow/tools/ci_build/ci_build.sh CPU /bin/bash
```

**Note**: The set of jobs and how they are triggered is still evolving.
There are builds for master branch on cpu, gpu and android. There is a build
for incoming gerrit changes. Gpu tests and benchmark are coming soon. Check
[ci.tensorflow.org](https://ci.tensorflow.org) for current jobs.



## How Does TensorFlow Continuous Integration Work

We use [jenkins](https://jenkins-ci.org/) as our continuous integration.
It is running at [ci.tensorflow.org](https://ci.tensorflow.org).
All the jobs are run within [docker](http://www.docker.com/) containers.

Builds can be triggered by push to master, push a change set or manually.
The build started in jenkins will first pull the git tree. Then jenkins builds
a docker container (using one of those Dockerfile.* files in this directory).
The build itself is run within the container itself.

Source tree lives in jenkins job workspace. Docker container for jenkins
are transient - deleted after the build. Containers build very fast thanks
to docker caching. Individual builds are fast thanks to bazel caching.



## Implementation Details

* The ci_build.sh script create and run docker container with all dependencies.
  The builds/with_the_same_user together with ci_build.sh creates an environment
  which is the same inside the container as it is outside. The same user, group,
  path, so that docker symlinks work inside and outside the container. You can
  use it for your development. Edit files in your git clone directory. If you
  run the ci_build.sh it gets this directory mapped inside the container and
  build your tree.

* The unusual `bazel-ci_build-cache` directory is mapped to docker container
  performing the build using docker's --volume parameter. This way we cache
  bazel output between builds.

* The `builds` directory within this folder contains shell scripts to run within
  the container. They essentially contains workarounds for current limitations
  of bazel.
