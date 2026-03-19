# Proto Splitter

Utilities for splitting large protos.

For a more detailed overview of the library, see our [in-depth guide](g3doc/in-depth-guide.md).

## The Python `Splitter` class

Users can apply the Splitter implementations by calling:

```python
splitter = MySplitterClass(proto)

# Export the chunks to a file.
splitter.write(file_prefix)

# Access the chunks created in the splitter.
chunks, chunked_message = splitter.split()
```

### Composable Riegeli splitter

The `split.py` class provides a `ComposableSplitter` class that is implemented
to write to the Riegeli format, and allows combinable implementations of
different message splitters.

Recommended steps to subclass `ComposableSplitter`:

1.  (required) Override `build_chunks()`. This method sets the values of
    `self._chunks` and `self._chunked_message` based on the user-passed proto.
2.  Update `version_def`. This is important to ensure that users are able to
    apply the Merger to the chunked proto, or get understandable version errors.
3.  If `__init__` is overridden: call `super().__init__(proto, **kwargs)`. This
    is optional but highly recommended since it sets up basic attributes that
    may be needed by other splitters.

#### Example

Consider the `SavedModel` protobuf, which is composed of many different messages
(shown below). It contains two message types that can be at risk of running `>
2GB`: `SavedObjectGraph` and `GraphDef`.

We can write a `SavedModelSplitter` that contains logic for chunking both types,
or re-use splitter that specifically work on each. Considering that `GraphDef`
is used widely outside of `SavedModel`, the latter option is preferable.

```proto
message SavedModel {
  ...
  repeated MetaGraphDef meta_graphs = 2;
}
message MetaGraphDef {
  ...
  GraphDef graph_def = 2;
  SavedObjectGraph object_graph_def = 7;
}
message GraphDef {
  repeated NodeDef node = 1;
  FunctionDefLibrary library = 2;
  ...
}
```

The SavedModel splitter implementation would look like:

```python
class SavedModelSplitter(ComposableSplitter):
  def build_chunks(self):
    ObjectGraphSplitter(
      saved_model.meta_graphs[0].object_graph_def,
      parent_splitter=self,
      fields_in_parent=["meta_graphs", 0, "object_graph_def"]
    ).build_chunks()

    GraphDefSplitter(
      saved_model.meta_graphs[0].graph_def,
      parent_splitter=self,
      fields_in_parent=["meta_graphs", 0, "graph_def"],
    ).build_chunks()

# See the results:
A.split()  # [...chunks from B, ...chunks from C]
```

When B.split() and C.split() are called, chunks are added to A's chunk list, and
A's ChunkedMessage proto is updated directly.

## The C++ `Merger` class

Once the proto has been split and written to disk using the aforementioned
`Splitter` class, it can be merged back into its original form using these
methods:

```c++
absl::Status Merger::Merge(
  const std::vector<std::unique_ptr<tsl::protobuf::Message>>& chunks,
  const ::proto_splitter::ChunkedMessage& chunked_message,
  tsl::protobuf::Message* merged);

absl::Status Merger::Read(std::string prefix, tsl::protobuf::Message* merged);
```

`Merger::Merge` requires the user to already have a collection of chunks in
memory, while `Merger::Read` merges a chunked proto directly from disk. The
methods can be used like so:

```c++
// Merge
std::vector<std::unique_ptr<tsl::protobuf::Message>> my_chunks = GetMyChunks();
::proto_splitter::ChunkedMessage chunked_message = GetMyChunkedMessage();
my_project::MyProto my_proto;
Merger::Merge(my_chunks, chunked_message, &my_proto);

// Read
my_project::MyOtherProto my_other_proto;
Merger::Read("path/to/saved_model", &my_other_proto);
```

##### In-Depth Guide

Looking for a more detailed overview of the library? See our [in-depth guide](g3doc/in-depth-guide.md).
