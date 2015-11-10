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

* `b.gcr.io/tensorflow/tensorflow`, which is a minimal VM with TensorFlow and
  all dependencies.

* `b.gcr.io/tensorflow/tensorflow-full`, which contains a full source
  distribution and all required libraries to build and run TensorFlow from
  source.

* `b.gcr.io/tensorflow/tensorflow-full-gpu`, which is the same as the previous
  container, but built with GPU support.

## Running the container

Each of the containers is published to a Docker registry; for the non-GPU
containers, running is as simple as

    $ docker run -it b.gcr.io/tensorflow/tensorflow

For the container with GPU support, we require the user to make the appropriate
NVidia libraries available on their system, as well as providing mappings so
that the container can see the host's GPU. For most purposes, this can be
accomplished via

    $ export CUDA_SO=$(\ls /usr/lib/x86_64-linux-gnu/libcuda* | xargs -I{} echo '-v {}:{}')
    $ export DEVICES=$(\ls /dev/nvidia* | xargs -I{} echo '--device {}:{}')
    $ export CUDA_SRCS="-v /usr/local/cuda:/usr/local/cuda -v /usr/share/nvidia:/usr/share/nvidia"
    $ docker run -it $CUDA_SO $CUDA_SRCS $DEVICES b.gcr.io/tensorflow/tensorflow-full-gpu

Alternately, you can use the `docker_run_gpu.sh` script in this directory.

## Rebuilding the containers

### tensorflow/tensorflow

This one requires no extra setup -- just

    $ docker build -t $USER/tensorflow -f Dockerfile.lite .

### tensorflow/tensorflow-full

This one requires a copy of the tensorflow source tree at `./tensorflow` (since
we don't keep the `Dockerfile`s at the top of the tree). With that in place,
just run

    $ git clone https://github.com/tensorflow/tensorflow
    $ docker build -t $USER/tensorflow-full -f Dockerfile.cpu .

### tensorflow/tensorflow-gpu

This one requires a few steps, since we need the NVidia headers to be available
*during* the build step, but we don't want them included in the final container
image. We need to start by installing the NVidia libraries as described in the
[CUDA setup instructions](/get_started/os_setup.md#install_cuda). With that
complete, we can build via

    $ cp -a /usr/local/cuda .
    $ docker build -t $USER/tensorflow-gpu-base -f Dockerfile.gpu_base .
    # Flatten the image
    $ export TC=$(docker create $USER/tensorflow-gpu-base)
    $ docker export $TC | docker import - $USER/tensorflow-gpu-flat
    $ docker rm $TC
    $ export TC=$(docker create $USER/tensorflow-gpu-flat /bin/bash)
    $ docker commit --change='CMD ["/bin/bash"]'  --change='ENV CUDA_PATH /usr/local/cuda' --change='ENV LD_LIBRARY_PATH /usr/local/cuda/lib64' --change='WORKDIR /root' $TC $USER/tensorflow-full-gpu
    $ docker rm $TC

This final image is a full TensorFlow image with GPU support.
