# Using TensorFlow via Docker

This directory contains `Dockerfile`s to make it easy to get up and running with
TensorFlow via [Docker](http://www.docker.com/).

## Installing Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/installation/), but we give some
quick links here:

* [OSX](https://docs.docker.com/installation/mac/): [docker toolbox](https://www.docker.com/toolbox)
* [ubuntu](https://docs.docker.com/installation/ubuntulinux/)

## Which containers exist?

We currently maintain three Docker container images:

* `gcr.io/tensorflow/tensorflow`, which is a minimal VM with TensorFlow and
  all dependencies.

* `gcr.io/tensorflow/tensorflow-full`, which contains a full source
  distribution and all required libraries to build and run TensorFlow from
  source.

* `gcr.io/tensorflow/tensorflow-full-gpu`, which is the same as the previous
  container, but built with GPU support.

## Running the container

Each of the containers is published to a Docker registry; for the non-GPU
containers, running is as simple as

    $ docker run -it -p 8888:8888 gcr.io/tensorflow/tensorflow

For the container with GPU support, we require the user to make the appropriate
NVidia libraries available on their system, as well as providing mappings so
that the container can see the host's GPU. For most purposes, this can be
accomplished via

    $ export CUDA_SO=$(\ls /usr/lib/x86_64-linux-gnu/libcuda.* | xargs -I{} echo '-v {}:{}')
    $ export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    $ docker run -it -p 8888:8888 $CUDA_SO $DEVICES gcr.io/tensorflow/tensorflow-devel-gpu

Alternately, you can use the `docker_run_gpu.sh` script in this directory.

## Rebuilding the containers

Just pick the dockerfile corresponding to the container you want to build, and run;

    $ docker build --pull -t $USER/tensorflow-suffix -f Dockerfile.suffix .
