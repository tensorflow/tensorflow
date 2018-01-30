Using TensorRT in TensorFlow
============================

This module provides necessary bindings and introduces TRT_engine_op
operator that wraps a subgraph in TensorRT.

Compilation
-----------

In order to compile the module, you need to have a local TensorRT
installation (libnvinfer.so and respective include files). During the
configuration step, TensorRT should be enabled and installation path
should be set. If installed through package managers (deb,rpm),
configure script should find the necessary components from the system
automatically. If installed from tar packages, user has to set path to
location where the library is installed during configuration.


```
bazel build --config=cuda --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/
```

After the installation of tensorflow package, TensorRT transformation
will be available. An example use is shown below.

```python
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
#... create and train or load model
gdef = sess.graph.as_graph_def()
trt_gdef = trt.CreateInferenceGraph(
    gdef, #original graph_def
    ["output"], #name of output node(s)
    max_batch_size, #maximum batch size to run the inference
    max_workspace_size) # max memory for TensorRT to use
tf.reset_default_graph()
tf.import_graph_def(graph_def=trt_gdef)
#...... run inference
```
