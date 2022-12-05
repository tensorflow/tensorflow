#  TensorFlow Graph IR

This directory contains the definition of the Intermediate Representation (IR)
for TensorFlow graphs using MLIR.

## Introduction

This directory defined an MLIR dialect, the “TensorFlow Graph dialect”, that
represents accurately TensorFlow graphs. Contrary to the previous TensorFlow
dialect which made some opinionated choices that diverged from GraphDef and
TensorFlow Graph semantics, this dialect embraces TensorFlow Graph as it is. In
particular the concepts of control dependencies, requested device, assigned
device, and node name are all first-class attributes on the MLIR operations in
this dialect.

The main principle that drove the development of this dialect has been to ensure
perfect round-trip and general compatibility with existing TensorFlow semantics,
so that this solution can be deployed by default in any situation where "Graph
Optimization" and Grappler transformations are involved today, regardless of
TensorFlow V1 or V2. This new approach is also made possible by evolutions in
MLIR that allow representing graphs in a way that wasn’t possible before (more
in the [Graph operation design](#graph_operation_design) section below).

## History of Dialects for TensorFlow

MLIR started with a basic structure reflecting LLVM in that it defined a
`Module` containing a list of `Functions`. Each of these was defining a body
constrained to be a Control-Flow Graph (CFG): a list of `Blocks`, each of them
containing a list of `Operations`. A fundamental aspect of the CFG
representation is the notion of “control”: the abstract semantic model considers
that a single `Operation` executes at a given time, and the next `Operation` to
execute is necessarily the one listed immediately after[^1]. The last
`Operation` in a `Block` is a `Terminator`: it decides what is the next `Block`
where the control will be transferred (think of a branch).

When MLIR started, a first dialect -- that we were referring to as “TF control
dialect” -- was developed to model TensorFlow graphs. This dialect supported
control dependencies, but didn’t allow cycles in the graph, which forced some
tricks to model TensorFlow V1 loops and in particular the `NextIteration`
operation. While this dialect enabled some experimentation, it wasn’t seen as
really practical and another dialect was co-existing: the “tf” dialect that
we’re using currently. This dialect was designed before TF2.0
[was released](https://blog.tensorflow.org/2019/09/tensorflow-20-is-now-available.html),
and made strong assumptions about TensorFlow evolving towards a world where
eager execution and function execution become unified and V1 specific constructs
would be deprecated and disappear. As such control dependencies are not
supported and are instead implicit, control-flow V1 ops (such as Switch & Merge)
and deadness aren’t supported[^2], new device placement modelling solutions were
considered. These choices in the model enabled us to write graph transformations
as stateless DAG-to-DAG patterns that can be applied to a subgraph, without
considering the entire graph.

## Motivation

The combination of the TensorFlow and executor dialects allows for importing
most TensorFlow graphs and the TensorFlow dialect has proven enough to implement
the TF/XLA bridge, TFLite converter, and TFRT . However, the intent was for
TensorFlow 2.0 to trace TensorFlow functions directly in the TensorFlow dialect,
leaving the executor dialect only as a way to provide limited support for
TensorFlow V1 graphs.

However, the implementation of TensorFlow 2.0 didn't break away from TensorFlow
V1 entirely, instead TensorFlow functions are wrapped above TensorFlow V1 and
expose a leaky abstraction over the classical graph. As a result, the TensorFlow
dialect never got in a position to be enabled by default in TensorFlow. In
particular there are many subtle way in which TensorFlow functions diverges from
the sequential eager interpretation. For example the following pattern has been
recommended to users who intended to call a function `bar` knowing that the
first argument wasn’t necessary if they only used the first result.

```
  @tf.function
  def foo(z):
    x = tf.Placeholder(tf.int32)
    y, _ = bar(x, z)
    return y
```

The use of a placeholder would throw an exception in eager mode, but “works” in
graph mode as long as inlining and pruning ensure the placeholder is removed
before execution.

Other cases involve the need for control dependencies beyond what the
auto-control-dependency tracking offers. For example the
[tf.recompute\_grad](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/custom_gradient.py#L497)
creates control-dependencies on non-side-effecting ops to have a finer grain
control of memory usage.

Finally, the error modelling in TensorFlow can also be surprising. While in
eager op-by-op mode the execution is interrupted as soon as an error occurs,
`tf.function` tracing does not consider error handling as side-effecting
(otherwise it would have to add a control dependency between every node!) and as
such a program like:

```
@tf.function
def foo(x, y, variable):
   b = tf.matmul(x, y)
   variable.assign(1.0)
   return b
```

Does not guarantee that the assignment to the variable won’t occur if an error
occurs while processing the matmul, so calling:

```
foo(1., 2., variable)
```

Throws an exception because `tf.matmul` expects rank-2 tensors, but the variable
may or may not have been assigned. As such a user may want to opt in a safer
behavior for their function:

```
@tf.function
def foo(x, y, variable):
   b = tf.matmul(x, y)
   with tf.control_dependencies([b]):
     variable.assign(1.0)
   return b
```

However, this control dependency cannot be modelled in the TensorFlow dialect:
it will be just dropped! There is no solution today to prevent the variable
assignment to be executed ahead of the `matmul` in the TensorFlow Dialect.

While many of these cases could be modeled with different constructs at the
source level, this would be a major overhaul of TensorFlow itself, and more
importantly its ecosystem. Instead, we recognize that the TensorFlow dialect as
it exists today cannot support all of these use-cases, and it prevented MLIR
from providing a general graph transformation solution for TensorFlow,
contributing to more fragmentation instead of reducing it as promised.

The rest of this document describe how this new dialect follows a more pragmatic
approach to enable MLIR deployment in TensorFlow.

## Design

This new dialect intends to allow us to replace Grappler and existing graph
transformations, for TensorFlow V1 and V2 without constraints. As such the main
principle is to support perfect roundtrip between TensorFlow Graph/GraphDef and
MLIR.

### General Operations

An individual TensorFlow `NodeDef` is translated into an individual MLIR
operation using the following form:

```
%AddV2, %ctl = tfg.AddV2(%placeholder, %placeholder_1) [%ctl_1, %ctl_2]
                     device("GPU") assigned_device("TPU") name("add")
                     {some_attribute = "some attr!"}
                     : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
```

*   Each operation returns an optional variadic number of tensors as well as a
    control token to express potential control dependencies.
*   The node type is carried in the operation mnemonic.
*   The list of regular inputs is in-between parentheses.
*   Optional control dependencies are exposed after the regular inputs and
    printed between square brackets.
*   The pre-placement “requested device” as well as the post-placement “assigned
    device” information are preserved.
*   The node name is carried as a first-class attribute.
*   Optional “op specific” attributes can be listed between curly brackets.
*   Finally, the type signature follows, omitting the control dependencies.

This structure allows for a perfect round-trip to NodeDef, while still being
ergonomic when manipulating it in MLIR (compared to the `tf\_executor` dialect
for example). The tradeoff we are making here is that we preserve all
attributes, including the “derived” ones[^3], which creates some amount of
redundancy with the signature. We may consider pruning these redundant
attributes in the future in the same way as we do in the TensorFlow dialect.

### Graph Operation

A structural operation is introduced as a container: `tfg.graph` acts as a bag
of unordered TensorFlow operations, and carries a “version” attribute that
corresponds to the
[VersionDef](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/versions.proto)
present in GraphDef:

```
tfg.graph #tfg.version<producer = 42, min_consumer = 33> {
  %arg0, %ctl_0 = tfg.placeholder() : () -> (tensor<*xi32>)
  %add, %ctl_1 = tfg.AddV2(%arg0, %arg1)
                    : (tensor<*xi32>, tensor<*xi32>) -> (tensor<*xi32>)
  %arg1, %ctl_2 = tfg.placeholder() : () -> (tensor<*xi32>)
}
```

Note that the `AddV2` operation is using the result of a `placeholder` operation
that is defined later in the list. This wasn’t possible in MLIR 2 years ago when
the TensorFlow dialect was designed. It was actually
[attempted to allow such unordered semantics](https://groups.google.com/a/tensorflow.org/g/mlir/c/gPQFIy9XpVw/m/hfxmBGF8AQAJ)
and break away from the CFG-centric representation, but we couldn’t reach a
consensus, and some key members of the team believed that a departure from
CFG/SSA would limit the reusability of many algorithms. On the other hand, this
choice prevented us to design a graph dialect that can just replace TensorFlow
Graph structure as-is. Since then MLIR evolved to become more general and this
feature is now available (it was motivated by the
[support for HW synthesis tools](https://llvm.discourse.group/t/rfc-allowing-dialects-to-relax-the-ssa-dominance-condition/833)).
Another recent development that made it also more friendly is the
[removal of the requirement for terminators](https://llvm.discourse.group/t/rfc-making-terminator-optional-for-single-block-graph-regions/2997):
the `tfg.graph` operation above contains a single block listing operations, and
a terminator does not have any role to play. Finally, a Dialect can now
[act as fallback for OpInterfaces](https://llvm.discourse.group/t/rfc-dialect-fallback-for-opinterface/3074),
which allows us to reuse more of the TensorFlow registry to provide information
to MLIR passes about TensorFlow operation without having to register them with
MLIR in the first place.

The `tfg.graph` operation round-trips almost perfectly to
[Graph](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/graph/graph.h#L504),
except for the `Function Library`, which I address below.

### Function Library

Functions in TensorFlow are stored as
[FunctionDef](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/function.proto),
which has a signature, holds attributes, identifies argument and returned
values, and finally contains a list of nodes for its body. While on the surface
this `repeated NodeDef node_def` field looks identical to the body of
[GraphDef](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/graph.proto#L17),
there are fundamental differences in the representation, and in particular the
format the edges are represented is different.

To understand these differences, it is important to realize that a key aspect of
`FunctionsDef` is that they are stored uninstantiated, and can be considered in
a similar way to a C++ template function. The signature is actually an `OpDef`,
and just like any regular TensorFlow operation the types of the arguments and
the results are encoded and constrained with attributes. These attributes are
only provided or inferred based on the function’s use: the call-site is
responsible for instantiating a function before it’s body can be represented as
a Graph. Because of this, the body of an uninstantiated function is modeled
differently than Graph body:

```
  tfg.func generic @foo(%arg0 : !tfg.tensor {tfg.name = "input"},
                        %arg1 : !tfg.tensor {tfg.name = "another_input"})
      -> (!tfg.tensor {tfg.name = "result1"},
          !tfg.tensor {tfg.name = "result2"})
      attributes {description = "function foo"} {
    %Greater, %ctl_0 = tfg.Greater(%arg0, %arg1) name("Greater")
    %G_z = tfg.get_result(%Greater) "z" : 0
    %Switch, %ctl_1 = tfg.Switch(%G_z, %G_z) name("cond/Switch")
    %s_true = tfg.get_result %Switch "output_true" : 0
    %s_false = tfg.get_result %Switch "output_false" : 0
    tfg.return(%s_true, %s_false) [%ctl_0]
  }
```

Note how the tensor types `!tfg.tensor` are opaque, and every operation returns
a single tensor output and a control token. The tensor output is then unpacked
by looking up individual results by name. This is particularly visible with the
`Switch` operation where the two results are accessed using `tfg.get_result`
looking them up by name `output_true:0` and `output_false:0`. This is required
because the OpDef can define the number of output based on the attribute present
on the NodeDef, and these attributes can in turn be dependent on the attributes
added on the function during instantiation (you can read more about it in the
[description of the placeholder attribute value](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/attr_value.proto#L48-L55)).

Post-instantiation, a function body is similar to the one of a graph:

```
  tfg.func @foo(%arg0 : tensor<*xf32> {tfg.name = "input"},
                %arg1 : tensor<*xf32> {tfg.name = "another_input"})
      -> (tensor<*xi1> {tfg.name = "result1"},
          tensor<*xi1> {tfg.name = "result2"})
      attributes {description = "function foo"} {
    %Greater, %ctl_0 = tfg.Greater(%arg0, %arg1) [%arg1.ctl] name("Greater")
                          : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xi1>
    %Switch:2, %ctl_1 = tfg.Switch(%Greater, %Greater) name("cond/Switch")
                          : (tensor<*xi1>, tensor<*xi1>) -> tensor<*xi1>
   tfg.return(%Switch#0, %Switch#1) [%ctl_0]
  }
```

The operations aren’t ordered, except for the `tfg.return` which is a terminator
and must be the last operation. The only remaining difference with a graph is in
the handling of the function signature (arguments and returned values), and
attributes.

There is one aspect of the modelling worth mentioning from the MLIR point of
view: FunctionDef allows for nodes in a graph to express input control
dependencies from function arguments. However, in MLIR you need an actual
[SSA](https://en.wikipedia.org/wiki/Static_single_assignment_form) value to add
an edge between two operations. These values are typed and this is why
operations define a control token (like `%ctl_0`). We apply the same recipe for
arguments and for each of them we define a control token. We omit these “shadow
arguments” from the textual form, but in-memory the MLIR function has really 4
arguments:

```
 tfg.func @foo(%arg0 : tensor<*xf32> {tfg.name = "input"}, %arg0.ctl : !tfg.control
      %arg1 : tensor<*xf32> {tfg.name = "another_input"}, %arg1.ctl : !tfg.control)
      -> (tensor<*xi1> {tfg.name = "result1"},
          tensor<*xi1> {tfg.name = "result2"})
      attributes {description = "function foo"} {
   ...
```

The convention is that callers are only exposed to the non-control input
(`%arg0` and `%arg1`) while the control tokens are only intended to be visible
and used in the body. This makes it very aligned with how TensorFlow works.
Inside the body, values for the control dependencies on the arguments are
available with a `.ctl` suffix (i.e. `%arg0.ctl` and `%arg1.ctl`).

### Saved Model

The basic blocks above are enough to model `GraphDef`, but not the entirety of
SavedModel. However, most of the use cases that we’re targeting right now are in
the scope of the existing GraphOptimization and Grappler APIs, which aren’t
really coupled to SavedModel. The user can load a SavedModel independently of
MLIR and invoke MLIR transformations on a Function or Graph from there. There is
also already a dialect to model the specific aspects of SavedModel, it is
currently wrapping around the TensorFlow executor dialect and the TensorFlow
dialect, and we may look into integrating it with the `tfg` dialect in the
future. For these reasons, we mostly leave out modeling the Saved Model for
future work right now.

### Future Enhancements

Functional control-flow is modeled with nodes in the graph invoking functions in
the library. MLIR supports `region`s, which is a concept that allows attaching
subgraphs directly inside a graph, making it more friendly to optimizations. For
example a conditional operation can represent the two branches subgraph in the
TensorFlow dialect directly as follows:

```
  %0, %1, %2 = "tf.IfRegion"(%arg0) ({
     %t0 = "tf.Abs"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     %t1 = "tf.Acos"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     %t2 = "tf.Acosh"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
    "tf.Yield"(%t0, %t1, %t2) : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> ()
  }, {
     %e0 = "tf.Neg"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     %e1 = "tf.Relu"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     %e2 = "tf.Sin"(%arg1) : (tensor<2xf32>) -> tensor<2xf32>
     "tf.Yield"(%e0, %e1, %e2) : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
  }): (tensor<i1>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
  %3 = "tf.Add"(%0, %1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  %4 = "tf.Add"(%2, %3) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
```

## Integration

MLIR transformations in this dialect will operate on a module that will contain
at most one `graph` operation as well as a list of functions. This interface
will make such transformations suitable for fit within Grappler or as
GraphOptimization interchangeably.

Instead of a flat graph, an entry function will be provided when feeds/fetches
are available for the main graph (PRE_PLACEMENT graph optimizations execute in
Session before feeds/fetches are provided).

## FAQ

### Why not just use the TensorFlow Executor Dialect?

The executor dialect wasn’t designed to write transformation: it is designed as
a wrapper around the TensorFlow dialect: the intent was for it to be a stepping
stone to integrate MLIR and TensorFlow, and then disappear when TensorFlow V1
graphs would be deprecated. This new dialect embraces TensorFlow as it is
instead of as I wish it would be.

In particular the executor dialect represents each TensorFlow node as an
isolated “subgraph” nested under an “island” operation. This requires 3
operations and an extra region for each TensorFlow node, which is quite
inefficient in memory as well as requiring extra indirection when pattern
matching or updating nodes in the graph.

### What happens to the existing TensorFlow Dialects?

The existing TensorFlow dialect is suitable for representing a large subset of
TensorFlow programs (like models that intend to convert to TFLite, or XLA), and
for such cases we will continue to use it.

### What happens to the existing TensorFlow Executor Dialect?

This new TensorFlow Graph Dialect could be used to replace the Executor Dialect
as the standalone staging importing format. Importing from GraphDef/Graph would
always go through the TensorFlow Graph Dialect before using some clustering or
promotion algorithms to raise some subgraphs to the TensorFlow Dialect, just
like we do now to cluster islands operations in TensorFlow Executor Dialect.
The details of such mechanisms are left for future work.

<!-- Footnotes -->

[^1]: While the semantic model is sequential, this does not prevent an
    implementation to execute operation in parallel when proven safe. This is
    similar to how a superscalar CPU involves implicit parallelism. For
    example when mapping the TensorFlow dialect to TFRT, only side-effecting
    operations (Variable accesses for example) are sequenced.
[^2]: One of the first tools built with this was the TF->TFlite converter
    (replacing TOCO). Since V1 control-flow isn’t supported on TFLite this
    wasn’t a limitation.
[^3]: Derived attributes is a concept used in the TensorFlow dialect: since MLIR
    models type and shape information on each individual result produced by an
    operation, some attributes that are inserted for the sole purpose of
    typing are redundant and eliminated in MLIR.
