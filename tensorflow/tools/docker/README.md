# Using TensorFlow via Docker

This directory contains `Dockerfile`s to make it easy to get up and running with
TensorFlow via [Docker](http://www.docker.com/).

## Installing Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/installation/), but we give some
quick links here:

* [OSX](https://www.docker.com/products/docker#/mac)
* [Ubuntu](https://docs.docker.com/engine/installation/linux/ubuntulinux/)

## Which containers exist?

We currently maintain two Docker container images:

* `gcr.io/tensorflow/tensorflow` - TensorFlow with all dependencies - CPU only!

* `gcr.io/tensorflow/tensorflow:latest-gpu` - TensorFlow with all dependencies
  and support for NVidia CUDA

Note: We also publish the same containers into
[Docker Hub](https://hub.docker.com/r/tensorflow/tensorflow/tags/).


## Running the container

Run non-GPU container using

    $ docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow

For GPU support install NVidia drivers (ideally latest) and
[nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Run using

    $ nvidia-docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow:latest-gpu


Note: If you would have a problem running nvidia-docker you may try the old method
we have used. But it is not recommended. If you find a bug in nvidia-docker, please report
it there and try using nvidia-docker as described above.

    $ export CUDA_SO=$(\ls /usr/lib/x86_64-linux-gnu/libcuda.* | xargs -I{} echo '-v {}:{}')
    $ export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    $ docker run -it -p 8888:8888 $CUDA_SO $DEVICES gcr.io/tensorflow/tensorflow:latest-gpu


## More containers

See all available [tags](https://hub.docker.com/r/tensorflow/tensorflow/tags/)
for additional containers, such as release candidates or nightly builds.


## Rebuilding the containers

Building TensorFlow Docker containers should be done through the
[parameterized_docker_build.sh](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/parameterized_docker_build.sh)
script. The raw Dockerfiles should not be used directly as they contain strings
to be replaced by the script during the build.

To use the script, specify the container type (`CPU` vs. `GPU`), the desired
Python version (`PYTHON2` vs. `PYTHON3`) and whether the developer Docker image
is to be built (`NO` vs. `YES`). In addition, you need to specify the central
location from where the pip package of TensorFlow will be downloaded.

For example, to build a CPU-only non-developer Docker image for Python 2, using
TensorFlow's nightly pip package:

``` bash
export TF_DOCKER_BUILD_IS_DEVEL=NO
export TF_DOCKER_BUILD_TYPE=CPU
export TF_DOCKER_BUILD_PYTHON_VERSION=PYTHON2

export NIGHTLY_VERSION="1.head"
export TF_DOCKER_BUILD_CENTRAL_PIP=$(echo ${TF_DOCKER_BUILD_PYTHON_VERSION} | sed s^PYTHON2^http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=${TF_DOCKER_BUILD_PYTHON_VERSION},label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-${NIGHTLY_VERSION}-cp27-cp27mu-manylinux1_x86_64.whl^ | sed s^PYTHON3^http://ci.tensorflow.org/view/Nightly/job/nightly-python35-linux-cpu/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-${NIGHTLY_VERSION}-cp35-cp35m-manylinux1_x86_64.whl^)

tensorflow/tools/docker/parameterized_docker_build.sh
```

If successful, the image will be tagged as `${USER}/tensorflow:latest` by default.

Rebuilding GPU images requires [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
