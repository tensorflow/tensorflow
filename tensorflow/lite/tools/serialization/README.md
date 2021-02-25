# TFLite Serialization Tool

**NOTE:** This tool is intended for advanced users only, and should be used with
care.

The (C++) serialization library generates and writes a TFLite flatbuffer given
an `Interpreter` or `Subgraph`. Example use-cases include authoring models with
the `Interpreter` API, or updating models on-device (by modifying `tensor.data`
for relevant tensors).

## Serialization

### Writing flatbuffer to file

To write a TFLite model from an `Interpreter` (see `lite/interpreter.h`):
`std::unique_ptr<tflite::Interpreter> interpreter; // ...build/modify
interpreter... tflite::ModelWriter writer(interpreter.get()); std::string
filename = "/tmp/model.tflite"; writer.Write(filename);`

Note that the above API does not support custom I/O tensors or custom ops yet.
However, it does support model with Control Flow.

To generate/write a flatbuffer for a particular `Subgraph` (see
`lite/core/subgraph.h`) you can use `SubgraphWriter`.

```
std::unique_ptr<tflite::Interpreter> interpreter;
// ...build/modify interpreter...
// The number of subgraphs can be obtained by:
// const int num_subgraphs = interpreter_->subgraphs_size();
// Note that 0 <= subgraph_index < num_subgraphs
tflite::SubgraphWriter writer(&interpreter->subgraph(subgraph_index));
std::string filename = "/tmp/model.tflite";
writer.Write(filename);
```

`SubgraphWriter` supports custom ops and/or custom I/O tensors.

### Generating flatbuffer in-memory

Both `ModelWriter` and `SubgraphWriter` support a `GetBuffer` method to return
the generated flatbuffer in-memory:

```
std::unique_ptr<uint8_t[]> output_buffer;
size_t output_buffer_size;
tflite::ModelWriter writer(interpreter.get());
writer.GetBuffer(&output_buffer, &output_buffer_size);
```

## De-serialization

The flatbuffers written as above can be de-serialized just like any other TFLite
model, for eg:

```
std::unique_ptr<FlatBufferModel> model =
    FlatBufferModel::BuildFromFile(filename);
tflite::ops::builtin::BuiltinOpResolver resolver;
InterpreterBuilder builder(*model, resolver);
std::unique_ptr<Interpreter> new_interpreter;
builder(&new_interpreter);
```
