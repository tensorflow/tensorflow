# TensorFlow.org Continuous Integration

This directory contains all the files and setup instructions to run
continuous integration [ci.tensorflow.org](http://ci.tensorflow.org).



## How it works

We use [jenkins](https://jenkins-ci.org/) as our continuous integration.
It is running at [ci.tensorflow.org](http://ci.tensorflow.org).
All the jobs are run within [docker](http://www.docker.com/) containers.

Builds can be triggered by push to master, push a change set or manually.
The build started in jenkins will first pull the git tree. Then jenkins builds
a docker container (using one of those Dockerfile.* files in this directory).
The build itself is run within the container itself.

Source tree lives in jenkins job workspace. Docker container for jenkins
are transient - deleted after the build. Containers build very fast thanks
to docker caching. Individual builds are fast thanks to bazel caching.



## Implementation Details

* The unusual `bazel-ci_build-cache` directory is mapped to docker
  container performing the build using docker's --volume parameter.
  This way we cache bazel output between builds.

* The `builds` directory within this folder contains shell scripts to run within
  the container. They essentially contains workarounds for current limitations
  of bazel.



## Run It Yourself

1. Install [Docker](http://www.docker.com/). Follow instructions
   [on the Docker site](https://docs.docker.com/installation/).

2. Clone tensorflow repository.

   ```bash
git clone --recurse-submodules https://github.com/tensorflow/tensorflow.git
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

The jobs run by [ci.tensorflow.org](http://ci.tensorflow.org) include following:

```bash
# Note: You can run the following one-liners yourself if you have Docker.

# build and run cpu tests
tensorflow/tools/ci_build/ci_build.sh CPU bazel test //tensorflow/...

# build gpu
tensorflow/tools/ci_build/ci_build.sh GPU bazel build -c opt --config=cuda //tensorflow/...

# build pip with gpu support
tensorflow/tools/ci_build/ci_build.sh GPU tensorflow/tools/ci_build/builds/gpu_pip.sh

# build android example app
tensorflow/tools/ci_build/ci_build.sh ANDROID tensorflow/tools/ci_build/builds/android.sh

# run bash inside the container
CI_DOCKER_EXTRA_PARAMS='-it --rm' CI_COMMAND_PREFIX='' tensorflow/tools/ci_build/ci_build.sh CPU /bin/bash
```

**Note**: The set of jobs and how they are triggered is still evolving.
There are builds for master branch on cpu, gpu and android. There is a build
for incoming gerrit changes. Gpu tests and benchmark are coming soon. Check
[ci.tensorflow.org](http://ci.tensorflow.org) for current jobs.
