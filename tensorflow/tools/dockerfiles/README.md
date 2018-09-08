# TensorFlow Dockerfiles

This directory houses TensorFlow's Dockerfiles. **DO NOT EDIT THE DOCKERFILES
MANUALLY!** They are maintained by `assembler.py`, which builds Dockerfiles from
the files in `partials/` and the rules in `spec.yml`. See [the Contributing
section](#contributing) for more information.

## Building

The Dockerfiles in the `dockerfiles` directory must have their build context set
to **the directory with this README.md** to copy in helper files. For example:

```bash
$ docker build -f ./dockerfiles/cpu.Dockerfile -t tf .
```

Each Dockerfile has its own set of available `--build-arg`s which are documented
in the Dockerfile itself.

## Running

After building the image with the tag `tf` (for example), use `docker run` to
run the images. Examples are below.

Note for new Docker users: the `-v` and `-u` flags share directories between
the Docker container and your machine, and very important. Without
`-v`, your work will be wiped once the container quits, and without `-u`, files
created by the container will have the wrong file permissions on your host
machine. If you are confused, check out the [Docker run
documentation](https://docs.docker.com/engine/reference/run/).

```bash
# Volume mount (-v) is optional but highly recommended, especially for Jupyter.
# User permissions (-u) are required if you use (-v).

# CPU-based images
$ docker run -u $(id -u):$(id -g) -v $(PWD):/my-devel -it tf

# GPU-based images (set up nvidia-docker2 first)
$ docker run --runtime=nvidia -u $(id -u):$(id -g) -v $(PWD):/my-devel -it tf

# Images with Jupyter run on port 8888, and needs a volume for notebooks
$ docker run --user $(id -u):$(id -g) -p 8888:8888 -v $(PWD):/notebooks -it tf
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
$ docker build -t tf-assembler -f assembler.Dockerfile .

# Set --user to set correct permissions on generated files
$ docker run --user $(id -u):$(id -g) -it -v $(pwd):/tf tf-assembler bash 

# In the container...
/tf $ python3 ./assembler.py -o dockerfiles -s spec.yml
```
