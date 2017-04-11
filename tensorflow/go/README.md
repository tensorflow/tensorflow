# TensorFlow in Go

Construct and execute TensorFlow graphs in Go.

[![GoDoc](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go?status.svg)](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go)

> *WARNING*: The API defined in this package is not stable and can change
> without notice. The same goes for the awkward package path
> (`github.com/tensorflow/tensorflow/tensorflow/go`).

## Quickstart

1.  Download and extract the TensorFlow C library, preferably into `/usr/local`.
    GPU-enabled versions require CUDA 8.0 and cuDNN 5.1. For other versions, the
    TensorFlow C library will have to be built from source (see below).

    -   Linux:
        [CPU-only](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.0.0.tar.gz),
        [GPU-enabled](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.0.0.tar.gz)
    -   OS X
        [CPU-only](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-darwin-x86_64-1.0.0.tar.gz),
        [GPU-enabled](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-darwin-x86_64-1.0.0.tar.gz)

    The following shell snippet downloads and extracts into `/usr/local`:

    ```sh
    TF_TYPE="cpu" # Set to "gpu" for GPU support
    curl -L \
      "https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-${TF_TYPE}-$(go env GOOS)-x86_64-1.0.0.tar.gz" |
    sudo tar -C /usr/local -xz
    ```

2.  `go get` this package (and run tests):

    ```sh
    go get github.com/tensorflow/tensorflow/tensorflow/go
    go test github.com/tensorflow/tensorflow/tensorflow/go
    ```

3.  Done!

### Installing into locations other than `/usr/local`

The TensorFlow C library (`libtensorflow.so`) needs to be available at build
time (e.g., `go build`) and run time (`go test` or executing binaries). If the
library has not been extracted into `/usr/local`, then it needs to be made
available through the `LIBRARY_PATH` environment variable at build time and the
`LD_LIBRARY_PATH` environment variable (`DYLD_LIBRARY_PATH` on OS X) at run
time.

For example, if the TensorFlow C library was extracted into `/dir`, then:

```sh
export LIBRARY_PATH=/dir/lib
export LD_LIBRARY_PATH=/dir/lib   # For Linux
export DYLD_LIBRARY_PATH=/dir/lib # For OS X
```

## Building the TensorFlow C library from source

If the "Quickstart" instructions above do not work (perhaps the release archives
are not available for your operating system or architecture, or you're using a
different version of CUDA/cuDNN), then the TensorFlow C library must be built
from source.

### Prerequisites

-   [bazel](https://www.bazel.build/versions/master/docs/install.html)
-   Environment to build TensorFlow from source code
    ([Linux](https://www.tensorflow.org/versions/master/get_started/os_setup.html#prepare-environment-for-linux)
    or [OS
    X](https://www.tensorflow.org/versions/master/get_started/os_setup.html#prepare-environment-for-mac-os-x)).
    If you don't need GPU support, then try the following: `sh # Linux sudo
    apt-get install python swig python-numpy # OS X with homebrew brew install
    swig`

### Build

1.  Download the source code

    ```sh
    go get -d github.com/tensorflow/tensorflow/tensorflow/go
    ```

2.  Build the TensorFlow C library:

    ```sh
    cd ${GOPATH}/src/github.com/tensorflow/tensorflow
    ./configure
    bazel build --config opt //tensorflow:libtensorflow.so
    ```

    This can take a while (tens of minutes, more if also building for GPU).

3.  Make `libtensorflow.so` available to the linker. This can be done by either:

    a. Copying it to a system location, e.g.,

    ```sh
    sudo cp ${GOPATH}/src/github.com/tensorflow/tensorflow/bazel-bin/tensorflow/libtensorflow.so /usr/local/lib
    ```

    OR

    b. Setting environment variables:

    ```sh
    export LIBRARY_PATH=${GOPATH}/src/github.com/tensorflow/tensorflow/bazel-bin/tensorflow
    # Linux
    export LD_LIBRARY_PATH=${GOPATH}/src/github.com/tensorflow/tensorflow/bazel-bin/tensorflow
    # OS X
    export DYLD_LIBRARY_PATH=${GOPATH}/src/github.com/tensorflow/tensorflow/bazel-bin/tensorflow
    ```

4.  Build and test:

    ```sh
    go test github.com/tensorflow/tensorflow/tensorflow/go
    ```

### Generate wrapper functions for ops

Go functions corresponding to TensorFlow operations are generated in `op/wrappers.go`. To regenerate them:

Prerequisites:
- [Protocol buffer compiler (protoc) 3.x](https://github.com/google/protobuf/releases/)
- The TensorFlow repository under GOPATH

```sh
go generate github.com/tensorflow/tensorflow/tensorflow/go/op
```

## Support

Use [stackoverflow](http://stackoverflow.com/questions/tagged/tensorflow) and/or
[Github issues](https://github.com/tensorflow/tensorflow/issues).

## Contributions

Contributions are welcome. If making any signification changes, probably best to
discuss on a [Github issue](https://github.com/tensorflow/tensorflow/issues)
before investing too much time. Github pull requests are used for contributions.
