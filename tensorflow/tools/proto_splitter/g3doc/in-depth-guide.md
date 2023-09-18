# Proto Splitter / Merger Library

This doc lists implementation details about the [Proto Splitter/Merger library](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/proto_splitter). New Splitters should take these details into consideration to generate valid chunks and metadata that are compatible with the Merger. If you'd just like to use the new feature when exporting a SavedModel, simply add the following flag to `tf.saved_model.SaveOptions`:

```python
tf.saved_model.save(
  ...,
  options=tf.saved_model.SaveOptions(experimental_image_format=True)
)
```

The Merger has been integrated with `tf.saved_model.load`, so no change needs to be made to SavedModel loading code.

## Chunking Schema

A proto larger than 2GB cannot be serialized. This is a limit of the protobuf implementation that we must work around, which is why we created a proto Splitter/Merger solution. The Splitter takes a proto as input and produces **chunks** and **metadata**. Chunks are parts of a proto that have been split into units of binary data, and can be merged together to form the original proto. Metadata refers to the auxiliary information about where these chunks are extracted from the original proto. This structural information of the proto is contained in the tree-like `ChunkedMessage`. When writing to disk, the metadata takes the form of `ChunkMetadata`, which contains the `ChunkedMessage` as well as information about the chunks' location within the file. When simply splitting the message in memory, only the `ChunkedMessage` is needed. On the Merger side of things, the metadata is used to build the proto back from its disjointed chunks.

`ChunkedMessage` contains an optional `chunk_index`, which references a `chunk` that contains the corresponding message. This message may be further chunked and have one or more of its fields with their own chunks. Therefore, `ChunkedMessage` also contains a list of `ChunkedField`s.

A `ChunkedField` represents a field within a message that has been delegated to its own `chunk`. It contains `field_tag`s that specify where it is located relative to the message `ChunkedField` belongs to. It also contains a `ChunkedMessage`, which allows for a structure that resembles a tree, which is a natural fit for proto metadata.

As an example, consider the following message `A` and its corresponding `ChunkedMessage`:

```proto
message A {
  int num = 1;
  string str = 2;
  B b = 3;
}

message B {
  ...
}
```

#### Metadata:
```proto
ChunkedMessage {
  chunk_index: 0
  chunked_fields: [
    ChunkedField {
      field_tag: [b]
      message: ChunkedMessage {
        chunk_index: 1
      }
    }
  ]
}
```

#### View of memory (deserialized):
```proto
chunks [
  0: A {
    num: ...
    str: ...
  }
  1: B {
    ...
  }
]
```

Here, `A`'s `ChunkedMessage` has the optional `chunk_index`, so we see in memory that `chunks[0]` does indeed contain the message `A`. Note that the `A` in `chunks[0]` lacks the `b` field, which has been chunked out. We see this reflected in `A`'s `ChunkedMessage`, whose `chunked_field`s contains the `ChunkedField` that corresponds to this `b` field. The `field_tag`s contain the (very short) path to the `b` field, and the `ChunkedMessage` within the `ChunkedField` references the location of the `chunk` in memory. Indeed, we see the `B` message in memory at `chunks[1]`.

## Field Tag Serialization

A `chunked_field`'s location within the proto is specified by its `field_tag`s.

```proto
message ChunkedField {
  repeated FieldIndex field_tag = 1;
}

message FieldIndex {
  message MapKey {
    oneof type {
      string s = 1;
      bool boolean = 2;
      uint32 ui32 = 3;
      uint64 ui64 = 4;
      int32 i32 = 5;
      int64 i64 = 6;
    }
  }
  oneof kind {
    uint32 field = 1;
    MapKey map_key = 2;
    uint64 index = 3;
  }
}
```

Consider the following messages `A`, `B`, and `C`:

```proto
message A {
  map<string,B> b = 1;
}

message B {
  repeated C c = 1;
}

message C {
  BigMessage big = 1;
}

message BigMessage {
  ...
}
```

Say we were given an `A` proto and wanted to chunk out `big`, since it is quite large. To reference `big`, we use the following path: `A.b["example_string"].c[3].big`. In this case, our list of `field_tag`s would look something like: `[ b, "example_string", c, 3, big ]`. The `field_tag`s for a `chunked_field` (`big`) specify its location relative to the given proto.

These tags represent either a `field`, `map_key`, or `index`, depending on what exactly is being referenced. For example, this allows us to differentiate between `G1 = GraphDef.node.1.attr.value.tensor` and `G2 = GraphDef.node[1].attr["value"].tensor`, even though their lists of `field_tag`s appear to be very similar. `G1`'s `node` field is simply a message containing a field `1`, while `G2`'s `node` field is a repeated message, who's `1`st element is being referenced. Similarly, `G1`'s `attr` field is a message containing a field called `attr`, while `G2`'s `attr` is a map, with the `value` key being referenced. Technically, we could use the proto reflection API to tell whether these ambiguous fields are repeated/map fields or not. However, it is better to be explicit, since it avoids bugs and the extra information makes for a better debugging experience.

## Chunk Extraction and Storage

Proto fields relevant to splitting/merging are classified using their type and occurrence:

 - Field type: **Scalar** or **Message**
 - Field occurrence: **Singular**, **Repeated**, or **Map**

Other proto field qualifiers like `oneof`, `required`, `optional`, and `packed` do not affect splitting and merging, so they are not taken into account in the implementation.

### Singular Fields

Scalar fields are simply serialized as bytes. Numerical types, such as ints, are serialized in numpy-readable binary. Message fields are also serialized as bytes, once they have been chunked down to <2GB.

### Repeated Fields

When repeated fields are split, they are stored in a chunk that has the same type as the parent of that repeated field. The order of the `chunked_field` for repeated fields is the same order in which the chunks should be merged.

For example, consider the message `A` which contains a repeated field `i`:

```proto
message A {
  repeated int i = 1;
}

A(i=[1, 2, 3, 4, 5])
```

#### Metadata
```proto
ChunkedMessage {
  chunked_fields: [
    ChunkedField {
      field_tag = [],
      chunk = 0
    },
    ChunkedField {
      field_tag = [],
      chunk = 1
    },
  ]
}
```

#### View of memory (deserialized)
```proto
chunks [
  0: A {
    i=[1, 2]
  }
  1: A {
    i=[3, 4, 5]
  }
]
```

`A`'s `ChunkedMessage` contains two `ChunkedField`s, one for the indices `[1, 2]` and another for the indices `[3, 4, 5]`. The `field_tag`s for both are empty, because the chunks are also of type `A`, and not a field within `A`. During merging, `chunks[0]` must be merged into the in-memory message `A` before `chunks[1]` so that the ordering of the repeated field elements is correct.

### Map Fields

Protobuf maps, like repeated fields, are not a distinct structure within the proto specification. Instead, maps are actually represented by repeated messages with `key` and `value` fields. (This means proto maps aren't really associative containers, but that isn't important here.) Here's an example of a map:

```proto
message A {
  map<string, int> my_map = 1;
}
A(my_map={"abc": 123, "def": 456})
```

#### Underlying proto structure:
```proto
A: {
  my_map: {
    key: "abc"
    value: 123
  }
  my_map: {
    key: "def"
    value: 456
  }
}
```

Since maps are really just repeated fields under the hood, we can chunk them the same way we chunk repeated fields:

```proto
message A {
  map<int, int> m = 1;
}

A(i={1:2, 3:4, 5:6})
```

#### Metadata
```proto
ChunkedMessage {
  chunked_fields: [
    ChunkedField {
      field_tag = [],
      chunk = 0
    },
    ChunkedField {
      field_tag = [],
      chunk = 1
    },
  ]
}
```

#### View of memory (deserialized)
```proto
chunks [
  0: A {
    i={3: 4}
  }
  1: A {
    i={1: 2, 5: 6}
  }
]
```

However, we can also chunk out the values in the map entry directly if we'd like:

```proto
message A {
  map<int, B> m = 1;
}

message B {
  int i = 1;
}

A(i={1:B(i=3), 2:B(i=4)})
```

#### Metadata
```proto
ChunkedMessage {
  chunked_fields: [
    ChunkedField {
      field_tag = [m, 3],
      chunk = 0
    },
    ChunkedField {
      field_tag = [m, 2],
      chunk = 1
    },
  ]
}
```

#### View of memory (deserialized)
```proto
chunks [
  0: B {
    i=3
  }
  1: B {
    i=4
  }
]
```

### Blank Message Compression

In general, we assume the first chunk to be the base message from which all the chunks are extracted (during the split), or the chunk that exists. **However, it's important to note that this isn't required.** If all data is extracted from the user-provided proto into chunks, there is no need for the initial chunk to be the base message. Here's an example with message `A`:

```proto
message A {
  B b = 1;
  C c = 2;
}

a = A(b=B(...), c=C(...))
```

Message `a` can be split into chunks `[b, c]` in two ways:

*First chunk is the same as the parent type*

```proto
chunked_message {
  chunk_index: 0  // Chunk index is set as the parent message type
  chunked_fields {  // First field is chunked
    field_tag { field: 1 }
    message { chunk_index: 1 }
  }
  chunked_fields {  // Second field stored in a separate chunk
    field_tag { field: 2 }
    message { chunk_index: 2 }
  }
}
```

#### View of memory (deserialized)
```proto
chunks [
  0: A {...}
  1: B {...}
  2: C {...}
]
```

*First chunk is not the parent type*

```proto
chunked_message {
  // Chunk index is not set in the parent message type
  chunked_fields {  // First field is chunked
    field_tag { field: 1 }
    message { chunk_index: 0 }
  }
  chunked_fields {  // Second field stored in a separate chunk
    field_tag { field: 2 }
    message { chunk_index: 1 }
  }
}
```

#### View of memory (deserialized)
```proto
chunks [
  0: B {...}
  1: C {...}
]
```

This second method is viable since Message `A` only contains data from fields `b` and `c`. Once `b` and `c` are chunked, there's no other data from `A` to include, so we don't bother creating a chunk for `A`. The merging implementation should not make an assumption on the type of the first chunk, and in this case must create a new (blank) `A` message to merge the `b` and `c` chunks into.

**tldr: A chunked_message may not have a parent chunk to merge its chunked_fields into**

## Creating a Splitter

Now that we've covered the format used by the Splitters/Merger, we can work on implementing our own Splitter. By now you can understand why each proto requires its own bespoke Splitter, since automatic splitting wouldn't take advantage of the knowledge we have as proto designers of bottlenecks and opportunities for optimization. So, let's walk through the process of creating a Splitter for our message `ModelConfig`:

```proto
enum ActivationFunction {
  RELU = 0;
  SIGMOID = 1;
  TANH = 2;
}

message Layer {
  string name = 1;
  int32 num_units = 2;
  ActivationFunction activation_function = 3;
}

message ModelConfig {
  string model_name = 1;
  int32 input_shape = 2;
  repeated Layer hidden_layers = 3;
  int32 output_units = 4;
  ActivationFunction output_activation = 5;
  map<string, float> hyperparameters = 6;
}
```

To create a `ModelConfig` Splitter, we have to decide what exactly is being split. As the designers of `ModelConfig`, we know that the `hidden_layers` tend to be quite large, so that makes the `Layer`s messages good candidates to split out into their own chunks. For the sake of example, we're also going to split out the `hyperparameters` field.

To create a Splitter, we must subclass the `ComposableSplitter` class and override its `build_chunks` method. If we wanted to store state in a Splitter, we could also override the `__init__` method, but it isn't required. In our example this would be enough to split and chunk out the fields we settled on (`hidden_layers` and `hyperparameters`), but we'll also create a Splitter for the `Layer` message to showcase Splitter composition.

```python
class ModelConfigSplitter(ComposableSplitter):
  def build_chunks(self):
    for k, v in self._proto.hyperparameters:
      self.add_chunk(bytes(str(v), "utf-8"), ["hyperparameters", k])

    for i, layer in enumerate(self._proto.hidden_layers):
      LayerSplitter(
        layer,
        parent_splitter=self,
        fields_in_parent=["hidden_layers", i]
      ).build_chunks()

class LayerSplitter(ComposableSplitter):
  def build_chunks(self):
    self.add_chunk(self._proto, [])

ModelConfigSplitter(
  proto=ModelConfig(...)
)
```

`build_chunks` generates chunks from `self._proto`, then for each chunk, calls `add_chunk` to add it to `self._chunks` and update `self._chunked_message`. `ModelConfigSplitter` does this once for `hyperparameters`, by simply converting the float value to a string and then to bytes. The Splitter does it again for `hidden_layers`, which get chunked by a dedicated `LayerSplitter` class. `LayerSplitter` doesn't actually do any chunking, but is here to showcase the ability to have a hierarchy of Splitters.

## Merging

There are two ways of merging a chunked proto using the provided Merger:

 - `Merger::Read()`, merges directly into a user-provided merged_message from a .cpb file on disk
 - `Merger::Merge()`, requires that the chunks and chunked metadata be stored in memory

`Merge()` should be called at runtime with the C++ Splitter, and allows one to skip any unnecessary disk reads/writes. `Read()` is therefore more holistic, handling both file IO and merging, so we'll consider its implementation below. The provided Merger is independent of any Splitter or protobuf, so developers will not have to write their own in the vast majority of cases.

### Riegeli

Since chunked protos use the riegeli file format, we use the riegeli api for file IO. The `riegeli::RecordReader` makes it easy to `Seek()` to a position in the file and `ReadRecord()` at that location.

### Reflection

We also make use of the protobuf reflection api to add and modify fields in `merged_message` using `FieldDescriptor`s.

### ChunkedMetadata

But to understand what should be read and where to read it from, we need the `ChunkedMetadata`. The metadata is always stored in the last chunk of the chunked proto, so we simply read that record to begin the merging process. Within the `ChunkedMetadata`, the sequence of `ChunkInfo` tells us where in the chunked proto to find the chunk we're looking for. And the `ChunkedMessage` contains a tree of metadata that we can use to reconstruct the desired proto.

### Field Processing

Starting at the root `ChunkedMessage`, we first check to see if it references a chunk by specifying a `chunk_index`. If so, we need to merge that chunk into the target proto (let's call it `A`) before processing each of its `chunked_field`s. If there is no `chunk_index`, then `A` only contains fields that have been chunked out. Before merging in the `chunked_field`s, they must be sorted by depth and index. For example, we need to merge in `GraphDef.library` before `GraphDef.library.function[0]`, which needs to be merged in before `GraphDef.library.function[1]`. We must merge in the `library` field first so that the `library.function`s have some place to be merged into, and the `0`th `function` must be merged before the `1`st `function` to maintain the proper ordering. Now we're ready to merge in the `chunked_field`s.

For each `ChunkedField` in a `ChunkedMessage`:

1. Read in the `chunk` specified by the `chunks_info[chunked_field.message.chunk_index]`
2. If the `chunked_field` has no `field_tag`s, then it does not reference a field within the parent message, but rather part of the parent message itself. For example, consider the following message and its corresponding `chunked_message`:
   ```proto
   message A {
     ...
   }
   
   chunked_message = {
     chunked_fields {  // empty field_tag, belongs to the parent chunked_message
       field_tag { }
       message { chunk_index: 0}
     }
     chunked_fields {  // also belongs to the parent
    chunk
       field_tag { }
       message { chunk_index: 1}
     }
   }
   ```
   In this case, a message `A` has been split into multiple chunks (here `A1` and `A2`, but hypothetically up to `An`), rather than splitting its fields into their own chunks. Splitting a message into chunks directly or splitting a message's fields into chunks are simply two different approaches that we offer in our api. So, the `chunk` should be merged directly into the parent message (`A`), and we skip the remaining steps to move on to the next `chunked_field`.
3. Navigate the `merged_message` using the `field_tag`s, until reaching the target field. Fields may need to be constructed along the way if they were not kept during the splitting process (see [Blank Message Compression above](#blank_message_compression)).
4. If the field is not a message, it is a primitive data type like bool or int, so we simply convert the `chunk` string to the appropriate type and set the field using reflection. If it is a message, then we recursively process it using its corresponding `ChunkedMessage`.

When the recursive process is complete, the `chunk`s have been successfully merged into the `merged_message`, so it's ready to be used in your program.

## Putting It All Together

Now that we've covered the entire splitting and merging process, let's go over an end-to-end example. We'll use the `ModelConfigSplitter` class we created in the [Creating a Splitter](#creating_a_splitter) section above. To write our proto to disk, we simply call `Splitter.write()`:

```python
my_proto = ModelConfig(...)
export_dir = "..."
my_splitter = ModelConfigSplitter(my_proto)
my_splitter.write(export_dir)
```

And in C++, we can use the Merger to read in our chunked proto:

```c++
ModelConfig my_proto;
string export_dir = "...";
Merger::Read(export_dir, &my_proto);
```

If we'd instead like to split and merge our proto directly in memory, we'd need `ModelConfigSplitter` to be a C++ class, but the process is very similar:

```c++
class ModelConfigSplitter : public ComposableSplitter {
  ...
};

ModelConfig my_proto{...};
string export_dir = "...";
ModelConfigSplitter my_splitter(my_proto);

// std::pair<std::vector<MessageBytes>*, ::proto_splitter::ChunkedMessage*>
auto[chunks, chunked_message] = my_splitter.Split();

// chunks, chunked_message are processed

ModelConfig my_new_proto;
Merger::Merge(chunks, chunked_message, &my_new_proto);
```
