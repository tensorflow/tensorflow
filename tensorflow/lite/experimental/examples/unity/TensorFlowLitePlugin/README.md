# TF Lite Experimental Unity Plugin

This directory contains an experimental sample Unity (2017) Plugin, based on
the experimental TF Lite C API. The sample demonstrates running inference within
Unity by way of a C# `Interpreter` wrapper.

Note that the native TF Lite plugin(s) *must* be built before using the Unity
Plugin, and placed in Assets/TensorFlowLite/SDK/Plugins/. For the editor (note
that the generated shared library name and suffix are platform-dependent):

```sh
bazel build -c opt --cxxopt=--std=c++11 //tensorflow/lite/c:tensorflowlite_c
```

and for Android (replace `android_arm` with `android_arm64` for 64-bit):

```sh
bazel build -c opt --cxxopt=--std=c++11 --config=android_arm \
  //tensorflow/lite/c:tensorflowlite_c
```

If you encounter issues with native plugin discovery on Mac ("Darwin")
platforms, try renaming `libtensorflowlite_c.dylib` to `tensorflowlite_c.bundle`.
