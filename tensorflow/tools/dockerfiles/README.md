# TensorFlow Dockerfiles

This directory houses TensorFlow's Dockerfiles. **DO NOT EDIT THE DOCKERFILES
MANUALLY!** They are maintained by `assembler.py`, which builds Dockerfiles from
the files in `partials/` and the rules in `spec.yml`. See [the Maintaining
section](#maintaining) for more information.

## Building

The Dockerfiles in the `dockerfiles` directory must have their build context set
to **the directory with this README.md** to copy in helper files. For example:

```bash
$ docker build -f ./dockerfiles/cpu.Dockerfile -t tf-cpu .
```

Each Dockerfile has its own set of available `--build-arg`s which are documented
in the Dockerfile itself.

## Maintaining

To make changes to TensorFlow's Dockerfiles, you'll update `spec.yml` and the
`*.partial.Dockerfile` files in the `partials` directory, then run
`assembler.py` to re-generate the full Dockerfiles before creating a pull
request.

You can use the `Dockerfile` in this directory to build an editing environment
that has all of the Python dependencies you'll need:

```bash
$ docker build -t tf-assembler .

# Set --user to set correct permissions on generated files
$ docker run --user $(id -u):$(id -g) -it -v $(pwd):/tf tf-assembler bash 

# In the container...
/tf $ python3 ./assembler.py -o dockerfiles -s spec.yml --validate
```
