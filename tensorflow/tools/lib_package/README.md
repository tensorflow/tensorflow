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

## Java library

The TensorFlow [Java
API](https://www.tensorflow.org/code/tensorflow/java/README.md)
consists of a native library (`libtensorflow_jni.so`) and a Java archive (JAR).
The following commands:

```sh
bazel test --config opt //tensorflow/tools/lib_package:libtensorflow_test
bazel build --config opt \
  //tensorflow/tools/lib_package:libtensorflow_jni.tar.gz \
  //tensorflow/tools/lib_package:libtensorflow.jar \
  //tensorflow/tools/lib_package:libtensorflow-src.jar
```

test and produce the following:

-   The native library (`libtensorflow_jni.so`) packaged in an archive at:
    `bazel-bin/tensorflow/tools/lib_package/libtensorflow_jni.tar.gz`
-   The Java archive at:
    `bazel-bin/tensorflow/tools/lib_package/libtensorflow.jar`
-   The Java archive for Java sources at:
    `bazel-bin/tensorflow/tools/lib_package/libtensorflow-src.jar`

## Release

Scripts to build these archives for TensorFlow releases are in
[tensorflow/tools/ci_build/linux](https://www.tensorflow.org/code/tensorflow/tools/ci_build/linux)
and
[tensorflow/tools/ci_build/osx](https://www.tensorflow.org/code/tensorflow/tools/ci_build/osx)
