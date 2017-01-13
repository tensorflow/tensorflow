Bazel rules to package the TensorFlow C-library and [header
files](https://www.tensorflow.org/code/tensorflow/c/c_api.h)
into an archive.

## TensorFlow C library

The TensorFlow [C
API](https://www.tensorflow.org/code/tensorflow/c/c_api.h)
is typically a requirement of TensorFlow APIs in other languages such as
[Go](https://www.tensorflow.org/code/tensorflow/go)
and [Rust](https://github.com/tensorflow/rust).

The command:

```sh
bazel build -c opt //tensorflow/tools/lib_package:libtensorflow
```

produces `bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz`, which
can be distributed and installed using something like:

```sh
tar -C /usr/local -xzf libtensorflow.tar.gz
```

## Release

Scripts to generate archives using these rules for release are in
[tensorflow/tools/ci_build/linux](https://www.tensorflow.org/code/tensorflow/tools/ci_build/linux)
and
[tensorflow/tools/ci_build/osx](https://www.tensorflow.org/code/tensorflow/tools/ci_build/osx)
