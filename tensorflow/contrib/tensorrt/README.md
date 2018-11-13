# Using TensorRT in TensorFlow

This module provides necessary bindings and introduces TRT_engine_op operator
that wraps a subgraph in TensorRT. This is still a work in progress but should
be useable with most common graphs.

## Compilation

In order to compile the module, you need to have a local TensorRT installation
(libnvinfer.so and respective include files). During the configuration step,
TensorRT should be enabled and installation path should be set. If installed
through package managers (deb,rpm), configure script should find the necessary
components from the system automatically. If installed from tar packages, user
has to set path to location where the library is installed during configuration.

```shell
bazel build --config=cuda --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/
```

After the installation of tensorflow package, TensorRT transformation will be
available. An example use can be found in test/test_tftrt.py script

## Installing TensorRT 3.0.4

In order to make use of TensorRT integration, you will need a local installation
of TensorRT 3.0.4 from the [NVIDIA Developer website](https://developer.nvidia.com/tensorrt).
Installation instructions for compatibility with TensorFlow are provided on the
[TensorFlow GPU support](https://www.tensorflow.org/install/gpu) guide.
