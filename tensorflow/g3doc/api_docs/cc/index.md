# TensorFlow C++ Session API reference documentation

TensorFlow's public C++ API includes only the API for executing graphs, as of
version 0.5. To control the execution of a graph from C++:

1. Build the computation graph using the [Python API](../python/).
1. Use [`tf.train.write_graph()`](../python/train.md#write_graph) to
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

## Env

* [tensorflow::Env](ClassEnv.md)
* [tensorflow::RandomAccessFile](ClassRandomAccessFile.md)
* [tensorflow::WritableFile](ClassWritableFile.md)
* [tensorflow::EnvWrapper](ClassEnvWrapper.md)

## Session

* [tensorflow::Session](ClassSession.md)
* [tensorflow::SessionOptions](StructSessionOptions.md)

## Status

* [tensorflow::Status](ClassStatus.md)
* [tensorflow::Status::State](StructState.md)

## Tensor

* [tensorflow::Tensor](ClassTensor.md)
* [tensorflow::TensorShape](ClassTensorShape.md)
* [tensorflow::TensorShapeDim](StructTensorShapeDim.md)
* [tensorflow::TensorShapeUtils](ClassTensorShapeUtils.md)
* [tensorflow::PartialTensorShape](ClassPartialTensorShape.md)
* [tensorflow::PartialTensorShapeUtils](ClassPartialTensorShapeUtils.md)
* [TF_Buffer](StructTF_Buffer.md)

## Thread

* [tensorflow::Thread](ClassThread.md)
* [tensorflow::ThreadOptions](StructThreadOptions.md)

