Bazel rules to package the TensorFlow APIs in languages other than Python into
archives.

## C library

The TensorFlow [C
API](https://www.tensorflow.org/code/tensorflow/c/c_api.h)
is typically a requirement of TensorFlow APIs in other languages such as
[Go](https://www.tensorflow.org/code/tensorflow/go)
and [Rust](https://github.com/tensorflow/rust).

The following commands:

```sh
bazel test --config opt //tensorflow/tools/lib_package:libtensorflow_test
bazel build --config opt //tensorflow/tools/lib_package:libtensorflow
```

test and produce the archive at
`bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz`, which can be
distributed and installed using something like:

```sh
tar -C /usr/local -xzf libtensorflow.tar.gz
```

### Examples
A simple C API example can be found [here](https://github.com/tensorflow/tensorflow/pull/24963#issuecomment-454800809).

## C++ library
The TensorFlow [C++
API](https://www.tensorflow.org/guide/extend/cc)
provides mechanisms for constructing and executing a data flow graph. This is a higher level API based on the C API that allows user to train and inference existing models with C++.

The following command:

```sh
bazel build --config opt //tensorflow/tools/lib_package:libtensorflow_cc
```
produces the archive at
`bazel-bin/tensorflow/tools/lib_package/libtensorflow_cc.tar.gz`, which can be
distributed and installed using something like:

```sh
tar -C /usr/local -xzf libtensorflow_cc.tar.gz
```

### Examples
Training example can be found [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/tutorials/example_trainer.cc).

### Library Linking Issue
TensorFlow uses several libraries that may also be used by applications linking against the C and C++ APIs (such as libjpeg).  When we create
the shared library, only export the minimal core TF API functions to avoid causing library conflicts (e.g., those reported in [github issue]( https://github.com/tensorflow/tensorflow/issues/1924)).

### Headers
The current C++ API includes following headers:

```
$ tree
.
└── include
    └── tensorflow
        ├── c
        │   └── eager
 		├── cc
 		│	├── client
 		│	├── framework
 		│	├── gradient
 		│	├── ops
 		│	├── profiler
 		│	├── saved_model
 		│	├── tools
 		│	└── training
 		└── core
 			├── framework
 			├── graph
 			├── lib
 			│	├── core
 			│	└── strings
 			├── platform
 			└── public
```

## Java library

The TensorFlow [Java
API](https://www.tensorflow.org/code/tensorflow/java/README.md)
consists of a native library (`libtensorflow_jni.so`) and a Java archive (JAR).
The following commands:

```sh
bazel test --config opt //tensorflow/tools/lib_package:libtensorflow_test
bazel build --config opt \
  //tensorflow/tools/lib_package:libtensorflow_jni.tar.gz \
  //tensorflow/java:libtensorflow.jar \
  //tensorflow/java:libtensorflow-src.jar
```

test and produce the following:

-   The native library (`libtensorflow_jni.so`) packaged in an archive at:
    `bazel-bin/tensorflow/tools/lib_package/libtensorflow_jni.tar.gz`
-   The Java archive at:
    `bazel-bin/tensorflow/java/libtensorflow.jar`
-   The Java archive for Java sources at:
    `bazel-bin/tensorflow/java/libtensorflow-src.jar`

## Release

Scripts to build these archives for TensorFlow releases are in
[tensorflow/tools/ci_build/linux](https://www.tensorflow.org/code/tensorflow/tools/ci_build/linux)
and
[tensorflow/tools/ci_build/osx](https://www.tensorflow.org/code/tensorflow/tools/ci_build/osx)