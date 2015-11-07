# TensorFlow C++ Session API reference documentation

TensorFlow's public C++ API includes only the API for executing graphs, as of
version 0.5. To control the execution of a graph from C++:

1. Build the computation graph using the [Python API](../python/).
1. Use [tf.train.write_graph()](../python/train.md?cl=head#write_graph) to
write the graph to a file.
1. Load the graph using the C++ Session API. For example:

  ```c++
  // Reads a model graph definition from disk, and creates a session object you
  // can use to run it.
  Status LoadGraph(string graph_file_name, Session** session) {
    GraphDef graph_def;
    TF_RETURN_IF_ERROR(
        ReadBinaryProto(Env::Default(), graph_file_name, &graph_def));
    TF_RETURN_IF_ERROR(NewSession(SessionOptions(), session));
    TF_RETURN_IF_ERROR((*session)->Create(graph_def));
    return Status::OK();
  }
```

1. Run the graph with a call to `session->Run()`


##Classes

* [tensorflow::Env](ClassEnv.md)
* [tensorflow::EnvWrapper](ClassEnvWrapper.md)
* [tensorflow::RandomAccessFile](ClassRandomAccessFile.md)
* [tensorflow::Session](ClassSession.md)
* [tensorflow::Status](ClassStatus.md)
* [tensorflow::Tensor](ClassTensor.md)
* [tensorflow::TensorBuffer](ClassTensorBuffer.md)
* [tensorflow::TensorShape](ClassTensorShape.md)
* [tensorflow::TensorShapeIter](ClassTensorShapeIter.md)
* [tensorflow::TensorShapeUtils](ClassTensorShapeUtils.md)
* [tensorflow::Thread](ClassThread.md)
* [tensorflow::WritableFile](ClassWritableFile.md)

##Structs

* [tensorflow::SessionOptions](StructSessionOptions.md)
* [tensorflow::Status::State](StructState.md)
* [tensorflow::TensorShapeDim](StructTensorShapeDim.md)
* [tensorflow::ThreadOptions](StructThreadOptions.md)


<div class='sections-order' style="display: none;">
<!--
<!-- ClassEnv.md -->
<!-- ClassEnvWrapper.md -->
<!-- ClassRandomAccessFile.md -->
<!-- ClassSession.md -->
<!-- ClassStatus.md -->
<!-- ClassTensor.md -->
<!-- ClassTensorBuffer.md -->
<!-- ClassTensorShape.md -->
<!-- ClassTensorShapeIter.md -->
<!-- ClassTensorShapeUtils.md -->
<!-- ClassThread.md -->
<!-- ClassWritableFile.md -->
<!-- StructSessionOptions.md -->
<!-- StructState.md -->
<!-- StructTensorShapeDim.md -->
<!-- StructThreadOptions.md -->
-->
</div>






