# TF Lite Experimental Unity Plugin

This directory contains an experimental sample Unity (2017) Plugin, based on
the experimental TF Lite C API. The sample demonstrates running inference within
Unity by way of a C# `Interpreter` wrapper.

Note that the native TF Lite plugin(s) *must* be built before using the Unity
Plugin, and placed in Assets/TensorFlowLite/SDK/Plugins/. For the editor (note
that this has only been tested on Linux; the syntax may differ on Mac/Windows):

```sh
bazel build -c opt --cxxopt=--std=c++11 \
  //tensorflow/lite/experimental/c:libtensorflowlite_c.so
```

and for Android:

```sh
bazel build -c opt --cxxopt=--std=c++11 \
  --crosstool_top=//external:android/crosstool \
  --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
  --cpu=armeabi-v7a \
  //tensorflow/lite/experimental/c:libtensorflowlite_c.so
```

If you encounter issues with native plugin discovery on Mac ("Darwin")
platforms, try renaming `libtensorflowlite_c.so` to `tensorflowlite_c.bundle`.
Similarly, on Windows you'll likely need to rename `libtensorflowlite_c.so` to
`tensorflowlite_c.dll`.
