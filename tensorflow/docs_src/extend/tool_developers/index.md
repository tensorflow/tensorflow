# A Tool Developer's Guide to TensorFlow Model Files

Most users shouldn't need to care about the internal details of how TensorFlow
stores data on disk, but you might if you're a tool developer. For example, you
may want to analyze models, or convert back and forth between TensorFlow and
other formats. This guide tries to explain some of the details of how you can
work with the main files that hold model data, to make it easier to develop
those kind of tools.

[TOC]

## Protocol Buffers

All of TensorFlow's file formats are based on
[Protocol Buffers](https://developers.google.com/protocol-buffers/?hl=en), so to
start it's worth getting familiar with how they work. The summary is that you
define data structures in text files, and the protobuf tools generate classes in
C, Python, and other languages that can load, save, and access the data in a
friendly way. We often refer to Protocol Buffers as protobufs, and I'll use
that convention in this guide.

## GraphDef

The foundation of computation in TensorFlow is the `Graph` object. This holds a
network of nodes, each representing one operation, connected to each other as
inputs and outputs. After you've created a `Graph` object, you can save it out
by calling `as_graph_def()`, which returns a `GraphDef` object.

The GraphDef class is an object created by the ProtoBuf library from the
definition in
[tensorflow/core/framework/graph.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto). The protobuf tools parse
this text file, and generate the code to load, store, and manipulate graph
definitions. If you see a standalone TensorFlow file representing a model, it's
likely to contain a serialized version of one of these `GraphDef` objects
saved out by the protobuf code.

This generated code is used to save and load the GraphDef files from disk. The code that actually loads the model looks like this:

```python
graph_def = graph_pb2.GraphDef()
```

This line creates an empty `GraphDef` object, the class that's been created
from the textual definition in graph.proto. This is the object we're going to
populate with the data from our file.

```python
with open(FLAGS.graph, "rb") as f:
```

Here we get a file handle for the path we've passed in to the script

```python
  if FLAGS.input_binary:
    graph_def.ParseFromString(f.read())
  else:
    text_format.Merge(f.read(), graph_def)
```

## Text or Binary?

There are actually two different formats that a ProtoBuf can be saved in.
TextFormat is a human-readable form, which makes it nice for debugging and
editing, but can get large when there's numerical data like weights stored in
it. You can see a small example of that in
[graph_run_run2.pbtxt](https://github.com/tensorflow/tensorboard/blob/master/tensorboard/demo/data/graph_run_run2.pbtxt).

Binary format files are a lot smaller than their text equivalents, even though
they're not as readable for us. In this script, we ask the user to supply a
flag indicating whether the input file is binary or text, so we know the right
function to call. You can find an example of a large binary file inside the
[inception_v3 archive](https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz),
as `inception_v3_2016_08_28_frozen.pb`.

The API itself can be a bit confusing - the binary call is actually
`ParseFromString()`, whereas you use a utility function from the `text_format`
module to load textual files.

## Nodes

Once you've loaded a file into the `graph_def` variable, you can now access the
data inside it. For most practical purposes, the important section is the list
of nodes stored in the node member. Here's the code that loops through those:

```python
for node in graph_def.node
```

Each node is a `NodeDef` object, defined in
[tensorflow/core/framework/node_def.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/node_def.proto). These
are the fundamental building blocks of TensorFlow graphs, with each one defining
a single operation along with its input connections. Here are the members of a
`NodeDef`, and what they mean.

### `name`

Every node should have a unique identifier that's not used by any other nodes
in the graph. If you don't specify one as you're building a graph using the
Python API, one reflecting the name of operation, such as "MatMul",
concatenated with a monotonically increasing number, such as "5", will be
picked for you. The name is used when defining the connections between nodes,
and when setting inputs and outputs for the whole graph when it's run.

### `op`

This defines what operation to run, for example `"Add"`, `"MatMul"`, or
`"Conv2D"`. When a graph is run, this op name is looked up in a registry to
find an implementation. The registry is populated by calls to the
`REGISTER_OP()` macro, like those in
[tensorflow/core/ops/nn_ops.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/ops/nn_ops.cc).

### `input`

A list of strings, each one of which is the name of another node, optionally
followed by a colon and an output port number. For example, a node with two
inputs might have a list like `["some_node_name", "another_node_name"]`, which
is equivalent to `["some_node_name:0", "another_node_name:0"]`, and defines the
node's first input as the first output from the node with the name
`"some_node_name"`, and a second input from the first output of
`"another_node_name"`

### `device`

In most cases you can ignore this, since it defines where to run a node in a
distributed environment, or when you want to force the operation onto CPU or
GPU.

### `attr`

This is a key/value store holding all the attributes of a node. These are the
permanent properties of nodes, things that don't change at runtime such as the
size of filters for convolutions, or the values of constant ops. Because there
can be so many different types of attribute values, from strings, to ints, to
arrays of tensor values, there's a separate protobuf file defining the data
structure that holds them, in
[tensorflow/core/framework/attr_value.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto).

Each attribute has a unique name string, and the expected attributes are listed
when the operation is defined. If an attribute isn't present in a node, but it
has a default listed in the operation definition, that default is used when the
graph is created.

You can access all of these members by calling `node.name`, `node.op`, etc. in
Python. The list of nodes stored in the `GraphDef` is a full definition of the
model architecture.

## Freezing

One confusing part about this is that the weights usually aren't stored inside
the file format during training. Instead, they're held in separate checkpoint
files, and there are `Variable` ops in the graph that load the latest values
when they're initialized. It's often not very convenient to have separate files
when you're deploying to production, so there's the
[freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) script that takes a graph definition and a set
of checkpoints and freezes them together into a single file.

What this does is load the `GraphDef`, pull in the values for all the variables
from the latest checkpoint file, and then replace each `Variable` op with a
`Const` that has the numerical data for the weights stored in its attributes
It then strips away all the extraneous nodes that aren't used for forward
inference, and saves out the resulting `GraphDef` into an output file.

## Weight Formats

If you're dealing with TensorFlow models that represent neural networks, one of
the most common problems is extracting and interpreting the weight values. A
common way to store them, for example in graphs created by the freeze_graph
script, is as `Const` ops containing the weights as `Tensors`. These are
defined in
[tensorflow/core/framework/tensor.proto](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto), and contain information
about the size and type of the data, as well as the values themselves. In
Python, you get a `TensorProto` object from a `NodeDef` representing a `Const`
op by calling something like `some_node_def.attr['value'].tensor`.

This will give you an object representing the weights data. The data itself
will be stored in one of the lists with the suffix _val as indicated by the
type of the object, for example `float_val` for 32-bit float data types.

The ordering of convolution weight values is often tricky to deal with when
converting between different frameworks. In TensorFlow, the filter weights for
the `Conv2D` operation are stored on the second input, and are expected to be
in the order `[filter_height, filter_width, input_depth, output_depth]`, where
filter_count increasing by one means moving to an adjacent value in memory.

Hopefully this rundown gives you a better idea of what's going on inside
TensorFlow model files, and will help you if you ever need to manipulate them.
