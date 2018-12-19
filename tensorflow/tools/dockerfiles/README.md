# TensorFlow Dockerfiles

This directory houses TensorFlow's Dockerfiles and the infrastructure used to
create and deploy them to
[Docker Hub](https://hub.docker.com/r/tensorflow/tensorflow).

**DO NOT EDIT THE DOCKERFILES/ DIRECTORY MANUALLY!** The files within are
maintained by `assembler.py`, which builds Dockerfiles from the files in
`partials/` and the rules in `spec.yml`. See
[the Contributing section](#contributing) for more information.

These Dockerfiles are planned to replace the Dockerfiles used to generate
[TensorFlow's official Docker images](https://hub.docker.com/r/tensorflow/tensorflow).

## Building

The Dockerfiles in the `dockerfiles` directory must have their build context set
to **the directory with this README.md** to copy in helper files. For example:

```bash
$ docker build -f ./dockerfiles/cpu.Dockerfile -t tf .
```

Each Dockerfile has its own set of available `--build-arg`s which are documented
in the Dockerfile itself.

## Running Locally Built Images

After building the image with the tag `tf` (for example), use `docker run` to
run the images.

Note for new Docker users: the `-v` and `-u` flags share directories and
permissions between the Docker container and your machine. Without `-v`, your
work will be wiped once the container quits, and without `-u`, files created by
the container will have the wrong file permissions on your host machine.
Check out the [Docker run
documentation](https://docs.docker.com/engine/reference/run/) for more info.

```bash
# Volume mount (-v) is optional but highly recommended, especially for Jupyter.
# User permissions (-u) are required if you use (-v).

# CPU-based images
$ docker run -u $(id -u):$(id -g) -v $(pwd):/my-devel -it tf

# GPU-based images (set up nvidia-docker2 first)
$ docker run --runtime=nvidia -u $(id -u):$(id -g) -v $(pwd):/my-devel -it tf

# Images with Jupyter run on port 8888 and need a volume for your notebooks
# You can change $(PWD) to the full path to a directory if your notebooks
# live outside the current directory.
$ docker run --user $(id -u):$(id -g) -p 8888:8888 -v $(PWD):/tf/notebooks -it tf
```

These images do not come with the TensorFlow source code -- but the development
images have git included, so you can `git clone` it yourself.

## Contributing

To make changes to TensorFlow's Dockerfiles, you'll update `spec.yml` and the
`*.partial.Dockerfile` files in the `partials` directory, then run
`assembler.py` to re-generate the full Dockerfiles before creating a pull
request.

You can use the `Dockerfile` in this directory to build an editing environment
that has all of the Python dependencies you'll need:

```bash
# Build the tools-helper image so you can run the assembler
$ docker build -t tf-tools -f tools.Dockerfile .

# Set --user to set correct permissions on generated files
$ docker run --user $(id -u):$(id -g) -it -v $(pwd):/tf tf-tools bash

# Next you can make a handy alias depending on what you're doing. When building
# Docker images, you need to run as root with docker.sock mounted so that the
# container can run Docker commands. When assembling Dockerfiles, though, you'll
# want to run as your user so that new files have the right permissions.

# If you're BUILDING OR DEPLOYING DOCKER IMAGES, run as root with docker.sock:
$ alias asm_images="docker run --rm -v $(pwd):/tf -v /var/run/docker.sock:/var/run/docker.sock tf-tools python3 assembler.py "

# If you're REBUILDING OR ADDING DOCKERFILES, remove docker.sock and add -u:
$ alias asm_dockerfiles="docker run --rm -u $(id -u):$(id -g) -v $(pwd):/tf tf-tools python3 assembler.py "

# Check assembler flags
$ asm_dockerfiles --help

# Assemble all of the Dockerfiles
$ asm_dockerfiles --release dockerfiles --construct_dockerfiles

# Build all of the "nightly" images on your local machine:
$ asm_images --release nightly --build_images

# Save the list of built images to a file:
$ asm_images --release nightly --build_images > tf-built.txt

# Build version release for version 99.0, except "gpu" tags:
$ asm_images --release versioned --arg _TAG_PREFIX=99.0 --build_images --exclude_tags_matching '.*gpu.*'

# Test your changes to the devel images:
$ asm_images --release nightly --build_images --run_tests_path=$(realpath tests) --only_tags_matching="^devel-gpu-py3$"
```
