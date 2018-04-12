# TensorFlow in other languages

## Background

This document is intended as a guide for those interested in the creation or
development of TensorFlow functionality in other programming languages. It
describes the features of TensorFlow and recommended steps for making the same
available in other programming languages.

Python was the first client language supported by TensorFlow and currently
supports the most features. More and more of that functionality is being moved
into the core of TensorFlow (implemented in C++) and exposed via a [C API].
Client languages should use the language's [foreign function interface
(FFI)](https://en.wikipedia.org/wiki/Foreign_function_interface) to call into
this [C API] to provide TensorFlow functionality.

## Overview

Providing TensorFlow functionality in a programming language can be broken down
into broad categories:

-   *Run a predefined graph*: Given a `GraphDef` (or
    `MetaGraphDef`) protocol message, be able to create a session, run queries,
    and get tensor results. This is sufficient for a mobile app or server that
    wants to run inference on a pre-trained model.
-   *Graph construction*: At least one function per defined
    TensorFlow op that adds an operation to the graph. Ideally these functions
    would be automatically generated so they stay in sync as the op definitions
    are modified.
-   *Gradients (AKA automatic differentiation)*: Given a graph and a list of
    input and output operations, add operations to the graph that compute the
    partial derivatives (gradients) of the inputs with respect to the outputs.
    Allows for customization of the gradient function for a particular operation
    in the graph.
-   *Functions*: Define a subgraph that may be called in multiple places in the
    main `GraphDef`. Defines a `FunctionDef` in the `FunctionDefLibrary`
    included in a `GraphDef`.
-   *Control Flow*: Construct "If" and "While" with user-specified subgraphs.
    Ideally these work with gradients (see above).
-   *Neural Network library*: A number of components that together support the
    creation of neural network models and training them (possibly in a
    distributed setting). While it would be convenient to have this available in
    other languages, there are currently no plans to support this in languages
    other than Python. These libraries are typically wrappers over the features
    described above.

At a minimum, a language binding should support running a predefined graph, but
most should also support graph construction. The TensorFlow Python API provides
all these features.

## Current Status

New language support should be built on top of the [C API]. However, as you can
see in the table below, not all functionality is available in C yet. Providing
more functionality in the [C API] is an ongoing project.

Feature                                        | Python                                                      | C
:--------------------------------------------- | :---------------------------------------------------------- | :--
Run a predefined Graph                         | `tf.import_graph_def`, `tf.Session`                         | `TF_GraphImportGraphDef`, `TF_NewSession`
Graph construction with generated op functions | Yes                                                         | Yes (The C API supports client languages that do this)
Gradients                                      | `tf.gradients`                                              |
Functions                                      | `tf.python.framework.function.Defun`                        |
Control Flow                                   | `tf.cond`, `tf.while_loop`                                  |
Neural Network library                         | `tf.train`, `tf.nn`, `tf.contrib.layers`, `tf.contrib.slim` |

## Recommended Approach

### Run a predefined graph

A language binding is expected to define the following classes:

-   `Graph`: A graph representing a TensorFlow computation. Consists of
    operations (represented in the client language by `Operation`s) and
    corresponds to a `TF_Graph` in the C API. Mainly used as an argument when
    creating new `Operation` objects and when starting a `Session`. Also
    supports iterating through the operations in the graph
    (`TF_GraphNextOperation`), looking up operations by name
    (`TF_GraphOperationByName`), and converting to and from a `GraphDef`
    protocol message (`TF_GraphToGraphDef` and `TF_GraphImportGraphDef` in the C
    API).
-   `Operation`: Represents a computation node in the graph. Corresponds to a
    `TF_Operation` in the C API.
-   `Output`: Represents one of the outputs of an operation in the graph. Has a
    `DataType` (and eventually a shape). May be passed as an input argument to a
    function for adding operations to a graph, or to a `Session`'s `Run()`
    method to fetch that output as a tensor. Corresponds to a `TF_Output` in the
    C API.
-   `Session`: Represents a client to a particular instance of the TensorFlow
    runtime. Its main job is to be constructed with a `Graph` and some options
    and then field calls to `Run()` the graph. Corresponds to a `TF_Session` in
    the C API.
-   `Tensor`: Represents an N-dimensional (rectangular) array with elements all
    the same `DataType`. Gets data into and out of a `Session`'s `Run()` call.
    Corresponds to a `TF_Tensor` in the C API.
-   `DataType`: An enumerant with all the possible tensor types supported by
    TensorFlow. Corresponds to `TF_DataType` in the C API and often referred to
    as `dtype` in the Python API.

### Graph construction

TensorFlow has many ops, and the list is not static, so we recommend generating
the functions for adding ops to a graph instead of writing them by individually
by hand (though writing a few by hand is a good way to figure out what the
generator should generate). The information needed to generate a function is
contained in an `OpDef` protocol message.

There are a few ways to get a list of the `OpDef`s for the registered ops:

-   `TF_GetAllOpList` in the C API retrieves all registered `OpDef` protocol
    messages. This can be used to write the generator in the client language.
    This requires that the client language have protocol buffer support in order
    to interpret the `OpDef` messages.
-   The C++ function `OpRegistry::Global()->GetRegisteredOps()` returns the same
    list of all registered `OpDef`s (defined in
    [`tensorflow/core/framework/op.h`](https://www.tensorflow.org/code/tensorflow/core/framework/op.h)). This can be used to write the generator
    in C++ (particularly useful for languages that do not have protocol buffer
    support).
-   The ASCII-serialized version of that list is periodically checked in to
    [`tensorflow/core/ops/ops.pbtxt`](https://www.tensorflow.org/code/tensorflow/core/ops/ops.pbtxt) by an automated process.

The `OpDef` specifies the following:

-   Name of the op in CamelCase. For generated functions follow the conventions
    of the language. For example, if the language uses snake_case, use that
    instead of CamelCase for the op's function name.
-   A list of inputs and outputs. The types for these may be polymorphic by
    referencing attributes, as described in the inputs and outputs section of
    @{$adding_an_op$Adding an     op}.
-   A list of attributes, along with their default values (if any). Note that
    some of these will be inferred (if they are determined by an input), some
    will be optional (if they have a default), and some will be required (no
    default).
-   Documentation for the op in general and the inputs, outputs, and
    non-inferred attributes.
-   Some other fields that are used by the runtime and can be ignored by the
    code generators.

An `OpDef` can be converted into the text of a function that adds that op to the
graph using the `TF_OperationDescription` C API (wrapped in the language's FFI):

-   Start with `TF_NewOperation()` to create the `TF_OperationDescription*`.
-   Call `TF_AddInput()` or `TF_AddInputList()` once per input (depending on
    whether the input has a list type).
-   Call `TF_SetAttr*()` functions to set non-inferred attributes. May skip
    attributes with defaults if you don't want to override the default value.
-   Set optional fields if necessary:
    -   `TF_SetDevice()`: force the operation onto a specific device.
    -   `TF_AddControlInput()`: add requirements that another operation finish
        before this operation starts running
    -   `TF_SetAttrString("_kernel")` to set the kernel label (rarely used)
    -   `TF_ColocateWith()` to colocate one op with another
-   Call `TF_FinishOperation()` when done. This adds the operation to the graph,
    after which it can't be modified.

The existing examples run the code generator as part of the build process (using
a Bazel genrule). Alternatively, the code generator can be run by an automated
cron process, possibly checking in the result. This creates a risk of divergence
between the generated code and the `OpDef`s checked into the repository, but is
useful for languages where code is expected to be generated ahead of time like
`go get` for Go and `cargo ops` for Rust. At the other end of the spectrum, for
some languages the code could be generated dynamically from
[`tensorflow/core/ops/ops.pbtxt`](https://www.tensorflow.org/code/tensorflow/core/ops/ops.pbtxt).

#### Handling Constants

Calling code will be much more concise if users can provide constants to input
arguments. The generated code should convert those constants to operations that
are added to the graph and used as input to the op being instantiated.

#### Optional parameters

If the language allows for optional parameters to a function (like keyword
arguments with defaults in Python), use them for optional attributes, operation
names, devices, control inputs etc. In some languages, these optional parameters
can be set using dynamic scopes (like "with" blocks in Python). Without these
features, the library may resort to the "builder pattern", as is done in the C++
version of the TensorFlow API.

#### Name scopes

It is a good idea to have support for naming graph operations using some sort of
scoping hierarchy, especially considering the fact that TensorBoard relies on it
to display large graphs in a reasonable way. The existing Python and C++ APIs
take different approaches: In Python, the "directory" part of the name
(everything up to the last "/") comes from `with` blocks. In effect, there is a
thread-local stack with the scopes defining the name hierarchy. The last
component of the name is either supplied explicitly by the user (using the
optional `name` keyword argument) or defaults to the name of the type of the op
being added. In C++ the "directory" part of the name is stored in an explicit
`Scope` object. The `NewSubScope()` method appends to that part of the name and
returns a new `Scope`. The last component of the name is set using the
`WithOpName()` method, and like Python defaults to the name of the type of op
being added. `Scope` objects are explicitly passed around to specify the name of
the context.

#### Wrappers

It may make sense to keep the generated functions private for some ops so that
wrapper functions that do a little bit of additional work can be used instead.
This also gives an escape hatch for supporting features outside the scope of
generated code.

One use of a wrapper is for supporting `SparseTensor` input and output. A
`SparseTensor` is a tuple of 3 dense tensors: indices, values, and shape. values
is a vector size [n], shape is a vector size [rank], and indices is a matrix
size [n, rank]. There are some sparse ops that use this triple to represent a
single sparse tensor.

Another reason to use wrappers is for ops that hold state. There are a few such
ops (e.g. a variable) that have several companion ops for operating on that
state. The Python API has classes for these ops where the constructor creates
the op, and methods on that class add operations to the graph that operate on
the state.

#### Other Considerations

-   It is good to have a list of keywords used to rename op functions and
    arguments that collide with language keywords (or other symbols that will
    cause trouble, like the names of library functions or variables referenced
    in the generated code).
-   The function for adding a `Const` operation to a graph typically is a
    wrapper since the generated function will typically have redundant
    `DataType` inputs.

### Gradients, functions and control flow

At this time, support for gradients, functions and control flow operations ("if"
and "while") is not available in languages other than Python. This will be
updated when the [C API] provides necessary support.

[C API]: https://www.tensorflow.org/code/tensorflow/c/c_api.h
