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

* [tensorflow::Env](classEnv.md)
* [tensorflow::RandomAccessFile](classRandomAccessFile.md)
* [tensorflow::WritableFile](classWritableFile.md)
* [tensorflow::EnvWrapper](classEnvWrapper.md)

## Session

* [tensorflow::Session](classSession.md)
* [tensorflow::SessionOptions](structSessionOptions.md)

## Status

* [tensorflow::Status](classStatus.md)
* [tensorflow::Status::State](structState.md)

## Tensor

* [tensorflow::Tensor](classTensor.md)
* [tensorflow::TensorShape](classTensorShape.md)
* [tensorflow::TensorShapeDim](structTensorShapeDim.md)
* [tensorflow::TensorShapeUtils](classTensorShapeUtils.md)
* [tensorflow::PartialTensorShape](classPartialTensorShape.md)
* [tensorflow::PartialTensorShapeUtils](classPartialTensorShapeUtils.md)
* [TF_Buffer](structTF_Buffer.md)

## Thread

* [tensorflow::Thread](classThread.md)
* [tensorflow::ThreadOptions](structThreadOptions.md)

