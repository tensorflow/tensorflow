# TensorFlow Lite C API

This directory contains C APIs for TensorFlow Lite. This includes C APIs
for common types, like kernels and delegates, as well as an explicit C API
for inference.

## Header summary

Each public C header contains types and methods for specific uses:

*   `common.h` - Contains common C enums, types and methods used throughout
    TensorFlow Lite. This includes everything from error codes, to the kernel
    and delegate APIs.
*    `builtin_op_data.h` - Contains op-specific data that is used for builtin
     kernels. This should only be used when (re)implementing a builtin operator.
*   `c_api.h` - Contains the TensorFlow Lite C API for inference. The
     functionality here is largely equivalent (though a strict subset of) the
     functionality provided by the C++ `Interpreter` API.
*   `c_api_experimental.h` - Contains experimental C API methods for inference.
     These methods are useful and usable, but aren't yet part of the stable API.

## Using the C API

See the [`c_api.h`](c_api.h) header for API usage details.

## Building the C API

A native shared library target that contains the C API for inference has been
provided. Assuming a working [bazel](https://bazel.build/versions/master/docs/install.html)
configuration, this can be built as follows:

```sh
bazel build -c opt //tensorflow/lite/c:tensorflowlite_c
```

and for Android (replace `android_arm` with `android_arm64` for 64-bit),
assuming you've [configured your project for Android builds](../g3doc/guide/android.md):

```sh
bazel build -c opt --cxxopt=--std=c++11 --config=android_arm \
  //tensorflow/lite/c:tensorflowlite_c
```

The generated shared library will be available in your
`bazel-bin/tensorflow/lite/c` directory. A target which packages the shared
library together with the necessary headers (`c_api.h`, `c_api_experimental.h`
and `common.h`) will be available soon, and will also be released as a prebuilt
archive (together with existing prebuilt packages for Android/iOS).
