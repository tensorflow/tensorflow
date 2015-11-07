# Using TensorFlow via Docker

This directory contains `Dockerfile`s to make it easy to get up and running with
TensorFlow via [Docker](http://www.docker.com/).

## Installing Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/installation/), but we give some
quick links here:

* [OSX](https://docs.docker.com/installation/mac/): [docker toolbox](https://www.docker.com/toolbox)
* [ubuntu](https://docs.docker.com/installation/ubuntulinux/)

## Running the container

Before you build your container, you can add notebooks you need
to a subdirectory of your working directory `notebooks/` and any python
libraries you need for them to `notebooks/requirements.txt` to have them
installed with `pip`.

To build a container image from this `Dockerfile`, just run

    $ docker build -t $USER/tensorflow_docker .

This will create a new container from the description, and print out an
identifying hash. You can then run this container locally:

    $ docker run -p 8888:8888 -it $USER/tensorflow_docker

This will start the container (inside a VM locally), and expose the running
IPython endpoint locally on port 8888. (The `-it` flags keep stdin connected to
a tty in the container, which is helpful when you want to stop the server;
`docker help run` explains all the possibilities.)

**NOTE**: If you want to be able to add data to your IPython Notebook while it's
running you can do this in a subdirectory of the /notebook volume as follows:

    $ docker run -p 8888:8888 -it -v ./notebook/data:/notebook/data \
        $USER/tensorflow_docker

**Caveat**: Note that `docker build` uses the first positional argument as the
*context* for the build; in particular, it starts by collecting all files in
that directory and shipping them to the docker daemon to build the image itself.
This means you shouldn't use the `-f` flag to use this Dockerfile from a
different directory, or you'll end up copying around more files than you'd like.
So:

    # ok
    $ docker build .                     # inside tools/docker
    $ docker build path/to/tools/docker  # further up the tree
    # bad
    $ docker build -f tools/docker/Dockerfile . # will pick up all files in .

## Experimenting in the container:

When the container starts up, it launches an IPython notebook server, populated
with several "Getting Started with TensorFlow" notebooks.

# TODO

* Decide how much of this is handled by the native
  [docker support in bazel](http://bazel.io/blog/2015/07/28/docker_build.html).
