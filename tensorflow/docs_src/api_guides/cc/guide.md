# C++ API

Note: By default [tensorflow.org](http://tensorflow.org) shows docs for the
most recent stable version. The instructions in this doc require building from
source. You will probably want to build from the `master` version of tensorflow.
You should, as a result, be sure you are following the
[`master` version of this doc](https://www.tensorflow.org/versions/master/api_guides/cc/guide),
in case there have been any changes.

[TOC]

TensorFlow's C++ API provides mechanisms for constructing and executing a data
flow graph. The API is designed to be simple and concise: graph operations are
clearly expressed using a "functional" construction style, including easy
specification of names, device placement, etc., and the resulting graph can be
efficiently run and the desired outputs fetched in a few lines of code. This
guide explains the basic concepts and data structures needed to get started with
TensorFlow graph construction and execution in C++.

## The Basics

Let's start with a simple example that illustrates graph construction and
execution using the C++ API.

```c++
// tensorflow/cc/example/example.cc

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

int main() {
  using namespace tensorflow;
  using namespace tensorflow::ops;
  Scope root = Scope::NewRootScope();
  // Matrix A = [3 2; -1 0]
  auto A = Const(root, { {3.f, 2.f}, {-1.f, 0.f} });
  // Vector b = [3 5]
  auto b = Const(root, { {3.f, 5.f} });
  // v = Ab^T
  auto v = MatMul(root.WithOpName("v"), A, b, MatMul::TransposeB(true));
  std::vector<Tensor> outputs;
  ClientSession session(root);
  // Run and fetch v
  TF_CHECK_OK(session.Run({v}, &outputs));
  // Expect outputs[0] == [19; -3]
  LOG(INFO) << outputs[0].matrix<float>();
  return 0;
}
```

Place this example code in the file `tensorflow/cc/example/example.cc` inside a
clone of the
TensorFlow
[github repository](http://www.github.com/tensorflow/tensorflow). Also place a
`BUILD` file in the same directory with the following contents:

```python
load("//tensorflow:tensorflow.bzl", "tf_cc_binary")

tf_cc_binary(
    name = "example",
    srcs = ["example.cc"],
    deps = [
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/core:tensorflow",
    ],
)
```

Use `tf_cc_binary` rather than Bazel's native `cc_binary` to link in necessary
symbols from `libtensorflow_framework.so`. You should be able to build and run
the example using the following command (be sure to run `./configure` in your
build sandbox first):

```shell
bazel run -c opt //tensorflow/cc/example:example
```

This example shows some of the important features of the C++ API such as the
following:

* Constructing tensor constants from C++ nested initializer lists
* Constructing and naming of TensorFlow operations
* Specifying optional attributes to operation constructors
* Executing and fetching the tensor values from the TensorFlow session.

We will delve into the details of each below.

## Graph Construction

### Scope

@{tensorflow::Scope} is the main data structure that holds the current state
of graph construction. A `Scope` acts as a handle to the graph being
constructed, as well as storing TensorFlow operation properties. The `Scope`
object is the first argument to operation constructors, and operations that use
a given `Scope` as their first argument inherit that `Scope`'s properties, such
as a common name prefix. Multiple `Scope`s can refer to the same graph, as
explained further below.

Create a new `Scope` object by calling `Scope::NewRootScope`. This creates
some resources such as a graph to which operations are added. It also creates a
@{tensorflow::Status} object which will be used to indicate errors encountered
when constructing operations. The `Scope` class has value semantics, thus, a
`Scope` object can be freely copied and passed around.

The `Scope` object returned by `Scope::NewRootScope` is referred
to as the root scope. "Child" scopes can be constructed from the root scope by
calling various member functions of the `Scope` class, thus forming a hierarchy
of scopes. A child scope inherits all of the properties of the parent scope and
typically has one property added or changed. For instance, `NewSubScope(name)`
appends `name` to the prefix of names for operations created using the returned
`Scope` object.

Here are some of the properties controlled by a `Scope` object:

* Operation names
* Set of control dependencies for an operation
* Device placement for an operation
* Kernel attribute for an operation

Please refer to @{tensorflow::Scope} for the complete list of member functions
that let you create child scopes with new properties.

### Operation Constructors

You can create graph operations with operation constructors, one C++ class per
TensorFlow operation. Unlike the Python API which uses snake-case to name the
operation constructors, the C++ API uses camel-case to conform to C++ coding
style. For instance, the `MatMul` operation has a C++ class with the same name.

Using this class-per-operation method, it is possible, though not recommended,
to construct an operation as follows:

```c++
// Not recommended
MatMul m(scope, a, b);
```

Instead, we recommend the following "functional" style for constructing
operations:

```c++
// Recommended
auto m = MatMul(scope, a, b);
```

The first parameter for all operation constructors is always a `Scope` object.
Tensor inputs and mandatory attributes form the rest of the arguments.

For optional arguments, constructors have an optional parameter that allows
optional attributes.  For operations with optional arguments, the constructor's
last optional parameter is a `struct` type called `[operation]:Attrs` that
contains data members for each optional attribute. You can construct such
`Attrs` in multiple ways:

* You can specify a single optional attribute by constructing an `Attrs` object
using the `static` functions provided in the C++ class for the operation. For
example:

```c++
auto m = MatMul(scope, a, b, MatMul::TransposeA(true));
```

* You can specify multiple optional attributes by chaining together functions
  available in the `Attrs` struct. For example:

```c++
auto m = MatMul(scope, a, b, MatMul::TransposeA(true).TransposeB(true));

// Or, alternatively
auto m = MatMul(scope, a, b, MatMul::Attrs().TransposeA(true).TransposeB(true));
```

The arguments and return values of operations are handled in different ways
depending on their type:

* For operations that return single tensors, the object returned by
  the operation object can be passed directly to other operation
  constructors. For example:

```c++
auto m = MatMul(scope, x, W);
auto sum = Add(scope, m, bias);
```

* For operations producing multiple outputs, the object returned by the
  operation constructor has a member for each of the outputs. The names of those
  members are identical to the names present in the `OpDef` for the
  operation. For example:

```c++
auto u = Unique(scope, a);
// u.y has the unique values and u.idx has the unique indices
auto m = Add(scope, u.y, b);
```

* Operations producing a list-typed output return an object that can
  be indexed using the `[]` operator. That object can also be directly passed to
  other constructors that expect list-typed inputs. For example:

```c++
auto s = Split(scope, 0, a, 2);
// Access elements of the returned list.
auto b = Add(scope, s[0], s[1]);
// Pass the list as a whole to other constructors.
auto c = Concat(scope, s, 0);
```

### Constants

You may pass many different types of C++ values directly to tensor
constants. You may explicitly create a tensor constant by calling the
@{tensorflow::ops::Const} function from various kinds of C++ values. For
example:

* Scalars

```c++
auto f = Const(scope, 42.0f);
auto s = Const(scope, "hello world!");
```

* Nested initializer lists

```c++
// 2x2 matrix
auto c1 = Const(scope, { {1, 2}, {2, 4} });
// 1x3x1 tensor
auto c2 = Const(scope, { { {1}, {2}, {3} } });
// 1x2x0 tensor
auto c3 = ops::Const(scope, { { {}, {} } });
```

* Shapes explicitly specified

```c++
// 2x2 matrix with all elements = 10
auto c1 = Const(scope, 10, /* shape */ {2, 2});
// 1x3x2x1 tensor
auto c2 = Const(scope, {1, 2, 3, 4, 5, 6}, /* shape */ {1, 3, 2, 1});
```

You may directly pass constants to other operation constructors, either by
explicitly constructing one using the `Const` function, or implicitly as any of
the above types of C++ values. For example:

```c++
// [1 1] * [41; 1]
auto x = MatMul(scope, { {1, 1} }, { {41}, {1} });
// [1 2 3 4] + 10
auto y = Add(scope, {1, 2, 3, 4}, 10);
```

## Graph Execution

When executing a graph, you will need a session. The C++ API provides a
@{tensorflow::ClientSession} class that will execute ops created by the
operation constructors. TensorFlow will automatically determine which parts of
the graph need to be executed, and what values need feeding. For example:

```c++
Scope root = Scope::NewRootScope();
auto c = Const(root, { {1, 1} });
auto m = MatMul(root, c, { {42}, {1} });

ClientSession session(root);
std::vector<Tensor> outputs;
session.Run({m}, &outputs);
// outputs[0] == {42}
```

Similarly, the object returned by the operation constructor can be used as the
argument to specify a value being fed when executing the graph. Furthermore, the
value to feed can be specified with the different kinds of C++ values used to
specify tensor constants. For example:

```c++
Scope root = Scope::NewRootScope();
auto a = Placeholder(root, DT_INT32);
// [3 3; 3 3]
auto b = Const(root, 3, {2, 2});
auto c = Add(root, a, b);
ClientSession session(root);
std::vector<Tensor> outputs;

// Feed a <- [1 2; 3 4]
session.Run({ {a, { {1, 2}, {3, 4} } } }, {c}, &outputs);
// outputs[0] == [4 5; 6 7]
```

Please see the @{tensorflow::Tensor} documentation for more information on how
to use the execution output.
