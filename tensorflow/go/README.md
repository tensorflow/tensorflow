# tensorflow for Go

This package provides a SWIG-based Go package to run Tensorflow graphs in Go programs.

It is not an immediate goal to support Graph generation via this package.

A higher level API is presented in the 'github.com/tensorflow/tensorflow' package.

## Troubleshooting

```ld: library not found for -ltensorflow```

This package expects the linker to find the 'libtensorflow' shared library. 

To generate this file run:

```sh
$ go generate github.com/tensorflow/tensorflow
```

`libtensorflow.so` will end up at `${GOPATH}/src/github.com/tensorflow/bazel-bin/tensorflow/libtensorflow.so`.

