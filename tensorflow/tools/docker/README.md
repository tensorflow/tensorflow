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

Just pick the dockerfile corresponding to the container you want to build, and run

    $ docker build --pull -t $USER/tensorflow-suffix -f Dockerfile.suffix .
