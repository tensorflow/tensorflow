# TensorFlow Go API

Construct and execute TensorFlow graphs in Go.

[![GoDoc](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go?status.svg)](https://godoc.org/github.com/tensorflow/tensorflow/tensorflow/go)

> *WARNING*: The API defined in this package is not stable and can change
> without notice. The same goes for the awkward package path
> (`github.com/tensorflow/tensorflow/tensorflow/go`).

## Requirements

-   Go version 1.7+
-   [bazel](https://www.bazel.build/versions/master/docs/install.html)
-   Environment to build TensorFlow from source code
    ([Linux](https://www.tensorflow.org/versions/master/get_started/os_setup.html#prepare-environment-for-linux)
    or [Mac OS
    X](https://www.tensorflow.org/versions/master/get_started/os_setup.html#prepare-environment-for-mac-os-x)).
    If you'd like to skip reading those details and do not care about GPU
    support, try the following:

    ```sh
    # On Linux
    sudo apt-get install python swig python-numpy

    # On Mac OS X with homebrew
    brew install swig
    ```

## Installation

1.  Download the TensorFlow source code:

    ```sh
    go get -d github.com/tensorflow/tensorflow/tensorflow/go
    ```

2.  Build the TensorFlow library (`libtensorflow.so`):

    ```sh
    cd ${GOPATH}/src/github.com/tensorflow/tensorflow
    ./configure
    bazel build -c opt //tensorflow:libtensorflow.so
    ```

    This can take a while (tens of minutes, more if also building for GPU).

3.  Make `libtensorflow.so` available to the linker. This can be done by either:

    a. Copying it to a system location, e.g.,

    ```sh
    cp ${GOPATH}/src/github.com/tensorflow/tensorflow/bazel-bin/tensorflow/libtensorflow.so /usr/local/lib
    ```

    OR

    b. Setting the
    `LD_LIBRARY_PATH=${GOPATH}/src/github.com/tensorflow/tensorflow/bazel-bin/tensorflow`
    environment variable (`DYLD_LIBRARY_PATH` on Mac OS X).

4.  Generate wrapper functions for TensorFlow ops:

    ```sh
    go generate github.com/tensorflow/tensorflow/tensorflow/go/op
    ```

After this, the `go` tool should be usable as normal. For example:

```sh
go test -v github.com/tensorflow/tensorflow/tensorflow/go
```

## Contributions

This API has been built on top of the [C
API](https://www.tensorflow.org/code/tensorflow/c/c_api.h),
which is intended for building language bindings for TensorFlow functionality.
However, this is far from complete. Contributions are welcome.
