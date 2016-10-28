# XLA: The TensorFlow compiler framework

This document describes a compiler framework for linear algebra called *XLA*
that will be released as part of TensorFlow.  Most users of TensorFlow will not
invoke XLA directly, but will benefit from it through improvements in speed,
memory usage, and portability.

We are providing this preview for parties who are interested in details of
TensorFlow compilation and may want to provide feedback.  We will provide more
documentation with the code release.

[TOC]

## Compiling TensorFlow

The XLA compilation framework is invoked on subgraphs of TensorFlow
computations. The framework requires all tensor shapes to be fixed, so compiled
code is specialized to concrete shapes. This means, for example, that the
compiler may be invoked multiple times for the same subgraph if it is executed
on batches of different sizes. We had several goals in mind when designing the
TensorFlow compilation strategy:

*   *Improved execution speed.* Compiling subgraphs reduces the execution time
    of short-lived Ops by eliminating overhead from the TensorFlow runtime. The
    framework also fuses pipelined operations, reducing memory
    overheads. Specializing to known tensor shapes improves performance by
    allowing more aggressive constant propagation.
*   *Improved tensor buffer memory usage.* The compiler framework has an
    opportunity to analyze and schedule memory usage, in principle eliminating
    many intermediate storage buffers.
*   *Reduce reliance on custom Ops.* Many TensorFlow custom Ops are equivalent
    to subgraphs of existing lower-level Ops. By focusing on the preceding two
    goals we aim as far as possible to make the performance of low-level Ops be
    the same as that of hand-written fused implementations, removing the need
    for many custom Ops.
*   *Much improved mobile footprint.* When the compiled subgraph is an entire
    TensorFlow computation, it is possible to eliminate the TensorFlow runtime
    altogether and simply emit an object/header file pair that can be linked
    directly into another application. This is particularly useful for mobile
    inference, and can reduce the footprint of a TensorFlow computation by
    several orders of magnitude.
*   *Improved portability.* The compiler-based framework is designed to target
    different back-end hardware, including a variety of CPUs, GPUs, and custom
    accelerator hardware such as TPUs. The CPU and GPU back-ends currently use
    LLVM, while the internal Google TPU back-end (which will not be open-sourced
    at this time) uses custom code generation. The goal for this and other
    accelerators is that it should be relatively easy to write a new back-end
    for novel hardware, at which point a large fraction of TensorFlow programs
    will run unmodified on that hardware. This is in contrast with the approach
    of specializing individual monolithic Ops for new hardware, which requires
    TensorFlow programs to be rewritten to make use of those Ops.

## XLA: Accelerated Linear Algebra

XLA is a domain-specific compiler for linear algebra. The semantics of
operations are high level, e.g., arbitrary sized vector and matrix
operations. This makes the compiler easy to target from TensorFlow, and
preserves enough information to allow sophisticated scheduling and optimization.
The following tutorial provides introductory information about XLA. More details
follow in the [Operation Semantics](#operation_semantics) section.

It is important to note that the XLA framework is not set in stone. In
particular, while it is unlikely that the semantics of existing operations will
be changed, it is expected that more operations will be added as necessary to
cover important use cases, and we welcome feedback from the community about
missing functionality.

### Getting started - basic example

The following code sample shows how to use XLA to compute a simple vector
expression: $$\alpha x+y$$ ("axpy").

This sample presents a self-contained function - `ComputeAxpyParameters`, that
takes data as input, uses XLA to build a graph to compute the expression and
returns the resulting data.

This is done in several steps:

1.  Construct an XLA graph that encodes the expression we want to compute.
    The graph's nodes are XLA operations (sometimes called "ops" or HLOs for
    "high-level operations"), and its edges represent the data flow between
    operations.
2.  Ask XLA to create a "computation" based on this graph. This process
    JIT-compiles the graph into optimized native code for the chosen platform
    and returns a handle.
3.  Use the computation handle and the input data to calculate the result.

The XLA graph we construct for axpy is:
<div style="text-align:center">
![axpy params graph](../images/xla-axpy-tutorial-params.svg)
</div>

Note that all operations have predefined [shapes](#shapes_and_layout). A shape
describes the rank of the array, the size of each dimension and the primitive
element type. For example, `f32[10]` is a rank-1 array of single-precision
floats. `f32[]` is a single-precision float scalar.

In XLA, shapes are statically determined, including the size of each
dimension in an array. This permits the XLA compiler to produce very
efficient code for all backends. When constructing the graph, only the shapes of
input nodes (parameters or constants) have to be provided explicitly - the rest
is automatically inferred by XLA; therefore, the burden on the developer is
minimal.

Here is the part of the axpy sample code that constructs the graph (step 1):

```c++
std::unique_ptr<xla::Literal> ComputeAxpyParameters(
    const xla::Literal& alpha, const xla::Literal& x,
    const xla::Literal& y) {
  // Get the singleton handle for an XLA client library and create a new
  // computation builder.
  xla::Client* client(xla::ClientLibrary::ClientLibraryOrDie());
  xla::ComputationBuilder builder(client, "axpy");

  // Build the actual XLA computation graph. It's a function taking
  // three parameters and computing a single output.
  auto param_alpha = builder.Parameter(0, alpha.shape(), "alpha");
  auto param_x = builder.Parameter(1, x.shape(), "x");
  auto param_y = builder.Parameter(2, y.shape(), "y");
  auto axpy = builder.Add(builder.Mul(param_alpha, param_x), param_y);
```

XLA features a client-server design. `xla::ClientLibrary` provides a
simple way to instantiate an XLA server in the backend and connect to it with
an `xla::Client` object.

The `ComputationBuilder` class provides a convenient programming interface to
construct XLA computations. The semantics of XLA operations with links
to `ComputationBuilder` methods are documented in
[Operation Semantics](#operation_semantics).

Here is the part that JIT-compiles the graph (step 2):

```c++
  // We're done building the graph. Create a computation on the server.
  util::StatusOr<std::unique_ptr<xla::Computation>> computation_status =
      builder.Build();
  std::unique_ptr<xla::Computation> computation =
      computation_status.ConsumeValueOrDie();
```

Here is the part that runs the compiled code on the input (step 3):

```c++
  // Transfer the parameters to the server and get data handles that refer to
  // them.
  std::unique_ptr<xla::GlobalData> alpha_data =
      client->TransferToServer(alpha).ConsumeValueOrDie();
  std::unique_ptr<xla::GlobalData> x_data =
      client->TransferToServer(x).ConsumeValueOrDie();
  std::unique_ptr<xla::GlobalData> y_data =
      client->TransferToServer(y).ConsumeValueOrDie();

  // Now we have all we need to execute the computation on the device. We get
  // the result back in the form of a Literal.
  util::StatusOr<std::unique_ptr<xla::Literal>> result_status =
      client->ExecuteAndTransfer(
          *computation, {alpha_data.get(), x_data.get(), y_data.get()});
  return result_status.ConsumeValueOrDie();
}
```

There is one thing noticeably absent from the above code: no specification of
the device to use. The choice of device is orthogonal to the computation
specified and can be selected by choosing the appropriate service plugin.

### Moving data into and out of XLA

The main way to move data into and out of XLA is by populating
`xla::Literal` objects. This enables maximal generality for the XLA
client-server model of computation. When the service is running in the same
process as the client, the `xla::Client::TransferInProcess` method may be
used to transfer arrays to and from the service more efficiently.

### Constants vs. parameters

For the simple axpy computation we've seen earlier, we can construct an
alternative XLA graph:
<div style="text-align:center">
![axpy constants graph](../images/xla-axpy-tutorial-constants.svg)
</div>

The code to construct and run this computation is:

```c++
std::unique_ptr<xla::Literal> ComputeAxpyConstants(
    float alpha, gtl::ArraySlice<float> x,
    gtl::ArraySlice<float> y) {
  // Get the singleton handle for an XLA client library and create a new
  // computation builder.
  xla::Client* client(xla::ClientLibrary::ClientLibraryOrDie());
  xla::ComputationBuilder builder(client, "axpy");

  auto constant_alpha = builder.ConstantR0<float>(alpha);
  auto constant_x = builder.ConstantR1<float>(x);
  auto constant_y = builder.ConstantR1<float>(y);
  auto axpy = builder.Add(builder.Mul(constant_alpha, constant_x), constant_y);

  // We're done building the graph. Tell the server to create a Computation from
  // it, and then execute this computation on the device, transferring the
  // result back as a literal.
  util::StatusOr<std::unique_ptr<xla::Computation>> computation_status =
      builder.Build();
  std::unique_ptr<xla::Computation> computation =
      computation_status.ConsumeValueOrDie();
  // No need to pass arguments into the computation since it accepts no
  // parameters.
  util::StatusOr<std::unique_ptr<xla::Literal>> result_status =
      client->ExecuteAndTransfer(*computation, {});
  return result_status.ConsumeValueOrDie();
}
```

This computation has no user-provided inputs - the inputs are constants that are
embedded into the graph itself. It highlights an important design tradeoff that
should be considered when using XLA.

XLA is a JIT compiler. An XLA graph is created during the runtime of the host
program, and JIT-compiled to native code for the desired backend(s). This
compilation may take a non-trivial amount of time, which presents a tradeoff.

Many uses will want to compile a single graph and then run it repeatedly with
different inputs. This is what `parameter` ops are most suitable for. Re-running
the computation with different data doesn't require recompiling the graph.
Sometimes, however, some of the inputs may be constant (or at least constant
throughout some subset of the host program's runtime). In those cases, it makes
sense to create an XLA graph where these inputs are `constant` ops instead of
parameters. This will permit the XLA compiler to perform constant folding
and other advanced optimizations that may result in significantly more efficient
code. On the other hand, this means a computation needs to be recompiled every
time the "constant" value actually needs to change.

## Shapes and Layout

The XLA `Shape` proto describes the rank, size, and data type of an
N-dimensional array (*array* in short).

### Terminology, Notation, and Conventions

*   The rank of an array is equal to the number of dimensions. The *true rank*
    of an array is the number of dimensions which have a size greater than 1.

*   Dimensions are numbered from `0` up to `N-1` for an `N` dimensional array.
    The dimensions numbers are simply convenient labels. The order of these
    dimension numbers does not imply a particular minor/major ordering in the
    layout of the shape. The layout is determined by the `Layout` proto.

*   By convention, dimensions are listed in increasing order of dimension
    number. For example, for a 3-dimensional array of size `[A x B x C]`,
    dimension 0 has size `A`, dimension 1 has size `B` and dimension 2 has size
    `C`.

*   Two, three, and four dimensional arrays often have specific letters
    associated with dimensions. For example, for a 2D array:

    *   dimension 0: `y`
    *   dimension 1: `x`

    For a 3D array:

    *   dimension 0: `z`
    *   dimension 1: `y`
    *   dimension 2: `x`

    For a 4D array:

    *   dimension 0: `p`
    *   dimension 1: `z`
    *   dimension 2: `y`
    *   dimension 3: `x`

*   Functions in the XLA API which take dimensions do so in increasing order
    of dimension number. This matches the ordering used when passing dimensions
    as an `initializer_list`; e.g.

    `ShapeUtil::MakeShape(F32, {A, B, C, D})`

    Will create a shape whose dimension array consists of the sequence `[A, B,
    C, D]`.

### Layout

The `Layout` proto describes how an array is represented in memory. The `Layout`
proto includes the following fields:

```
message Layout {
  repeated int64 minor_to_major = 1;
  repeated int64 padded_dimensions = 2;
  optional PaddingValue padding_value = 3;
}
```

#### Minor-to-major dimension ordering

The only required field is `minor_to_major`. This field describes the
minor-to-major ordering of the dimensions within a shape. Values in
`minor_to_major` are an ordering of the dimensions of the array (`0` to `N-1`
for an `N` dimensional array) with the first value being the most-minor
dimension up to the last value which is the most-major dimension. The most-minor
dimension is the dimension which changes most rapidly when stepping through the
elements of the array laid out in linear memory.

For example, consider the following 2D array of size `[2 x 3]`:

```
a b c
d e f
```

Here dimension `0` is size 2, and dimension `1` is size 3. If the
`minor_to_major` field in the layout is `[0, 1]` then dimension `0` is the
most-minor dimension and dimension `1` is the most-major dimension. This
corresponds to the following layout in linear memory:

```
a d b e c f
```

This minor-to-major dimension order of `0` up to `N-1` is akin to *column-major*
(at rank 2). Assuming a monotonic ordering of dimensions, another name we may
use to refer to this layout in the code is simply "dim 0 is minor".

On the other hand, if the `minor_to_major` field in the layout is `[1, 0]` then
the layout in linear memory is:

```
a b c d e f
```

A minor-to-major dimension order of `N-1` down to `0` for an `N` dimensional
array is akin to *row-major* (at rank 2). Assuming a monotonic ordering of
dimensions, another name we may use to refer to this layout in the code is
simply "dim 0 is major".

#### Padding

Padding is defined in the optional `padded_dimensions` and `padding_value`
fields. The field `padded_dimensions` describes the sizes (widths) to which each
dimension is padded. If present, the number of elements in `padded_dimensions`
must equal the rank of the shape.

For example, given the `[2 x 3]` array defined above, if `padded_dimension` is
`[3, 5]` then dimension 0 is padded to a width of 3 and dimension 1 is padded to
a width of 5. The layout in linear memory (assuming a padding value of 0 and
column-major layout) is:

```
a d 0 b e 0 c f 0 0 0 0 0 0 0
```

This is equivalent to the layout of the following array with the same
minor-to-major dimension order:

```
a b c 0 0
d e f 0 0
0 0 0 0 0
```

## Operation Semantics

The following describes the semantics of operations defined in the
`ComputationBuilder` interface.

A note on nomenclature: the generalized data type XLA deals with is an
N-dimensional array holding elements of some uniform type (such as 32-bit
float). Throughout the documentation, we use *array* to denote an
arbitrary-dimensional array. For convenience, special cases have more specific
and familiar names; for example a *vector* is a 1-dimensional array and a
*matrix* is a 2-dimensional array.

### Broadcast

Adds dimensions to an array by duplicating the data in the array.

<b> `Broadcast(operand, broadcast_sizes)` </b>

Arguments         | Type                    | Semantics
----------------- | ----------------------- | -------------------------------
`operand`         | `ComputationDataHandle` | The array to duplicate
`broadcast_sizes` | `ArraySlice<int64>`     | The sizes of the new dimensions

The new dimensions are inserted on the left, i.e. if `broadcast_sizes` has
values `{a0, ..., aN}` and the operand shape has dimensions `{b0, ..., bM}` then
the shape of the output has dimensions `{a0, ..., aN, b0, ..., bM}`.

The new dimensions index into copies of the operand, i.e.

```
output[i0, ..., iN, j0, ..., jM] = operand[j0, ..., jM]
```

For example, if `operand` is a scalar `f32` with value `2.0f`, and
`broadcast_sizes` is `{2, 3}`, then the result will be an array with shape
`f32[2, 3]` and all the values in the result will be `2.0f`.

### Collapse

See also `ComputationBuilder::Collapse` and the [`Reshape`](#reshape) operation.

Collapses dimensions of an array into one dimension.

<b> `Collapse(operand, dimensions)` </b>

| Arguments    | Type                    | Semantics                           |
| ------------ | ----------------------- | ----------------------------------- |
| `operand`    | `ComputationDataHandle` | array of type T                     |
| `dimensions` | `int64` vector          | in-order, consecutive subset of T's |
:              :                         : dimensions.                         :

Collapse replaces the given subset of the operand's dimensions by a single
dimension. The input arguments are an arbitrary array of type T and a
compile-time-constant vector of dimension indices. The dimension indices must be
an in-order (low to high dimension numbers), consecutive subset of T's
dimensions. Thus, {0, 1, 2}, {0, 1}, or {1, 2} are all valid dimension sets, but
{1, 0} or {0, 2} are not. They are replaced by a single new dimension, in the
same position in the dimension sequence as those they replace, with the new
dimension size equal to the product of original dimension sizes. The lowest
dimension number in `dimensions` is the slowest varying dimension (most major)
in the loop nest which collapses these dimension, and the highest dimension
number is fastest varying (most minor). See the [`Reshape`](#reshape) operator
if more general collapse ordering is needed.

For example, let v be an array of 24 elements:

```
let v = f32[4x2x3] {{{10, 11, 12},  {15, 16, 17}},
                    {{20, 21, 22},  {25, 26, 27}},
                    {{30, 31, 32},  {35, 36, 37}},
                    {{40, 41, 42},  {45, 46, 47}}};

// Collapse to a single dimension, leaving one dimension.
let v012 = Collapse(v, {0,1,2});
then v012 == f32[24] {10, 11, 12, 15, 16, 17,
                      20, 21, 22, 25, 26, 27,
                      30, 31, 32, 35, 36, 37,
                      40, 41, 42, 45, 46, 47};

// Collapse the two lower dimensions, leaving two dimensions.
let v01 = Collapse(v, {0,1});
then v01 == f32[4x6] {{10, 11, 12, 15, 16, 17},
                      {20, 21, 22, 25, 26, 27},
                      {30, 31, 32, 35, 36, 37},
                      {40, 41, 42, 45, 46, 47}};

// Collapse the two higher dimensions, leaving two dimensions.
let v12 = Collapse(v, {1,2});
then v12 == f32[8x3] {{10, 11, 12},
                      {15, 16, 17},
                      {20, 21, 22},
                      {25, 26, 27},
                      {30, 31, 32},
                      {35, 36, 37},
                      {40, 41, 42},
                      {45, 46, 47}};

```

### Concatenate

See also `ComputationBuilder::ConcatInDim`

Concatenate composes an array from multiple array operands. The array is of the
same rank as each of the input array operands (which must be of the same rank as
each other) and contains the arguments in the order that they were specified.

<b> `Concatenate(operands..., dimension)` </b>

| Arguments   | Type                    | Semantics                            |
| ----------- | ----------------------- | ------------------------------------ |
| `operands`  | sequence of N           | N arrays of type T with dimensions   |
:             : `ComputationDataHandle` : [L0, L1, ...]                        :
| `dimension` | `int64`                 | A value in the interval `[0, N)`     |
:             :                         : that names the dimension to be       :
:             :                         : concatenated between the `operands`. :

With the exception of `dimension` all dimensions must be the same. This is
because XLA does not support "ragged" arrays -- the dimension which is being
concatenated must be the only one that differs between the operands. Also note
that rank-0 values cannot be concatenated (as it's impossible to name the
dimension along which the concatenation occurs).

1-dimensional example:

```
Concat({{2, 3}, {4, 5}, {6, 7}}, 0)
>>> {2, 3, 4, 5, 6, 7}
```

2-dimensional example:

```
let a = {
  {1, 2},
  {3, 4},
  {5, 6},
};
let b = {
  {7, 8},
};
Concat({a, b}, 0)
>>> {
  {1, 2},
  {3, 4},
  {5, 6},
  {7, 8},
}
```

Diagram:

<center><iframe src="../images/xla-concatenate.svg" width="920" height="400" title="Concatenate Diagram" frameborder="0"></iframe></center>

### ConvertElementType

See `ComputationBuilder::ConvertElementType`

Similar to an element-wise `static_cast` in C++, performs an element-wise
conversion operation from a data shape to a target shape. The dimensions must
match, and the conversion is an element-wise one; e.g. `s32` elements become
`f32` elements via an `s32`-to-`f32` conversion routine.

<b> `ConvertElementType(operand, new_element_type)` </b>

Arguments          | Type                    | Semantics
------------------ | ----------------------- | ---------------------------
`operand`          | `ComputationDataHandle` | array of type T with dims D
`new_element_type` | `PrimitiveType`         | type U

If the dimensions of the operand and the target shape do not match, or an
invalid conversion is requested (e.g. to/from a tuple) an error will be
produced.

A conversion such as `T=s32` to `U=f32` will perform a normalizing int-to-float
conversion routine such as round-to-nearest-even.

> Note: The precise float-to-int and visa-versa conversions are currently
> unspecified, but may become additional arguments to the convert operation in
> the future.

```
let a: s32[3] = {0, 1, 2};
let b: f32[3] = convert(a, f32);
then b == f32[3]{0.0, 1.0, 2.0}
```

### Conv (convolution)

See `ComputationBuilder::Conv`

As ConvWithGeneralPadding, but the padding is specified in a short-hand way as
either SAME or VALID. SAME padding pads the input (`lhs`) with zeroes so that
the output has the same shape as the input when not taking striding into
account. VALID padding simply means no padding.

### ConvWithGeneralPadding (convolution)

See `ComputationBuilder::ConvWithGeneralPadding`

Computes a convolution of the kind used in neural networks. Here, a convolution
can be thought of as a 2d window moving across a 2d base area and a computation
is performed for each possible position of the window.

| Arguments        | Type                    | Semantics                       |
| ---------------- | ----------------------- | ------------------------------- |
| `lhs`            | `ComputationDataHandle` | rank-4 array of inputs          |
| `rhs`            | `ComputationDataHandle` | rank-4 array of kernel weights  |
| `window_strides` | `ArraySlice<int64>`     | 2d array of kernel strides      |
| `padding`        | `ArraySlice<pair<int64, | 2d array of (low, high) padding |
:                  : int64>>`                :                                 :

The `lhs` argument is a rank 4 array describing the base area. We will call this
the input, even though of course the rhs is also an input. In a neural network,
these are the input activations. The 4 dimensions are, in this order:

*   `batch`: Each coordinate in this dimension represents an independent input
    for which convolution is carried out.
*   `z/depth/features`: Each (y,x) position in the base area has a vector
    associated to it, which goes into this dimension.
*   `y` and `x`: Describes the two spatial dimensions that define the 2d base
    area that the window moves across.

The `rhs` argument is a rank 4 array describing the convolutional
filter/kernel/window. The dimensions are, in this order:

*   `output-z`: The `z` dimension of the output.
*   `input-z`: The size of this dimension should equal the size of the `z`
    dimension in lhs.
*   `y` and `x`: Describes the two spatial dimensions that define the 2d window
    that moves across the base area.

The `window_strides` argument specifies the stride of the convolutional window
in the `y` and `x` dimensions. For example, if the stride in dimension `y` is 3,
then the window can only be placed at coordinates where the `y` index is
divisible by 3.

The `padding` argument specifies the amount of zero padding to be applied to the
base area. `padding[0]` specifies the padding for dimension `y` and `padding[1]`
specifies the padding for dimension `x`. Each pair has the low padding as the
first element and the high padding as the second element. The low padding is
applied in the direction of lower indices while the high padding is applied in
the direction of higher indices. For example, if `padding[1]` is `(2,3)` then
there will be a padding by 2 zeroes on the left and by 3 zeroes on the right in
the `x` dimension. Using padding is equivalent to inserting those same zero
values into the input (`lhs`) before doing the convolution.

The output shape has these dimensions, in this order:

*   `batch`: Same size as `batch` on the input (`lhs`).
*   `z`: Same size as `output-z` on the kernel (`rhs`).
*   `y` and `x`: One value for each valid placement of the convolutional window.

The valid placements of the convolutional window are determined by the strides
and the size of the base area after padding.

To describe what a convolution does, pick some fixed `batch`, `z`, `y`, `x`
coordinates in the output. Then `(y,x)` is a position of a corner of the window
within the base area (e.g. the upper left corner, depending on how you interpret
the spatial dimensions). We now have a 2d window, taken from the base area,
where each 2d point is associated to a 1d vector, so we get a 3d box. From the
convolutional kernel, since we fixed the output coordinate `z`, we also have a
3d box. The two boxes have the same dimensions, so we can take the sum of the
element-wise products between the two boxes (similar to a dot product). That is
the output value.

Note that if `output-z` is e.g. 5, then each position of the window produces 5
values in the output into the `z` dimension of the output. These values differ
in what part of the convolutional kernel is used - there is a separate 3d box of
values used for each `output-z` coordinate. So you could think of it as 5
separate convolutions with a different filter for each of them.

Here is pseudo-code for a convolution with padding and striding:

```
for (b, oz, oy, ox) {  // output coordinates
  value = 0;
  for (iz, ky, kx) {  // kernel coordinates and input z
    iy = oy*stride_y + ky - pad_low_y;
    ix = ox*stride_x + kx - pad_low_x;
    if ((iy, ix) inside the base area considered without padding) {
      value += input(b, iz, iy, ix) * kernel(oz, iz, ky, kx);
    }
  }
  output(b, oz, oy, ox) = value;
}
```

### Dot

See also `ComputationBuilder::Dot`

<b> `Dot(lhs, rhs)` </b>

Arguments | Type                    | Semantics
--------- | ----------------------- | ---------------
`lhs`     | `ComputationDataHandle` | array of type T
`rhs`     | `ComputationDataHandle` | array of type T

The exact semantics of this operation depend on the ranks of the operands:

| Input                   | Output                | Semantics               |
| ----------------------- | --------------------- | ----------------------- |
| scalar `dot` scalar     | scalar                | scalar multiplication   |
| vector [n] `dot` vector | scalar                | vector dot product      |
: [n]                     :                       :                         :
| matrix [m x k] `dot`    | vector [m]            | matrix-vector           |
: vector [k]              :                       : multiplication          :
| matrix [m x k] `dot`    | matrix [m x n]        | matrix-matrix           |
: matrix [k x n]          :                       : multiplication          :
| array [p x q x r] `dot` | array [p x q x s x t] | array dot product (read |
: array [s x r x t]       :                       : below)                  :

The operation performs sum of products over dimension 0 of `lhs` and dimensions
1 of `rhs`. These are the "contracted" dimensions. If the dimension to contract
exceeds the rank of the operand, the last dimension is contracted. This happens
when the `lhs` operand is a scalar or the `rhs` operand is a scalar or a vector.
The contracted dimensions of `lhs` and `rhs` must be of the same size.

The rank of the result array is `max(rank(lhs) - 1, 0) + max(rank(rhs) - 1, 0)`.
The result dimensions are ordered in the original order within each operand,
with the `rhs` dimensions followed by the `lhs` dimensions except the contracted
dimensions. For example, a dot product of two arrays `[p x q x r]` and `[s x r x
t]` produces a 4 dimensional array of `[p x q x s x t]` by contracting the
dimension of size `r`.

Notes:

1.  This follows the typical definition of a dot operator, as in other numeric
    libraries such as [numpy](http://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html).
2.  There is currently no support for the general tensor dot operator
    [numpy.tensordot](http://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html#numpy.tensordot).

### Element-wise binary arithmetic operations

See also `ComputationBuilder::Add`

A set of element-wise binary arithmetic operations is supported.

<b> `Op(lhs, rhs)` </b>

Where `Op` is one of `Add` (addition), `Sub` (subtraction), `Mul`
(multiplication), `Div` (division), `Rem` (remainder), `Max` (maximum), `Min`
(minimum).

Arguments | Type                    | Semantics
--------- | ----------------------- | ----------------------------------------
`lhs`     | `ComputationDataHandle` | left-hand-side operand: array of type T
`rhs`     | `ComputationDataHandle` | right-hand-side operand: array of type T

The arguments' shapes have to be either similar or compatible. See the
[broadcasting](#broadcasting_semantics) documentation about what it means for
shapes to be compatible. The result of an operation has a shape which is the
result of broadcasting the two input arrays. In this variant, operations between
arrays of different ranks are *not* supported, unless one of the operands is a
scalar.

When `Op` is `Rem`, the sign of the result is taken from the dividend.

An alternative variant with different-rank broadcasting support exists for these
operations:

<b> `Op(lhs, rhs, broadcast_dimensions)` </b>

Where `Op` is the same as above. This variant of the operation should be used
for arithmetic operations between arrays of different ranks (such as adding a
matrix to a vector).

The additional `broadcast_dimensions` operand is a slice of integers used to
expand the rank of the lower-rank operand up to the rank of the higher-rank
operand. `broadcast_dimensions` maps the dimensions of the lower-rank shape to
the dimensions of the higher-rank shape. The unmapped dimensions of the expanded
shape are filled with dimensions of size one. Degenerate-dimension broadcasting
then broadcasts the shapes along these degenerate dimension to equalize the
shapes of both operands. The semantics are described in detail in the
[broadcasting](#broadcasting_semantics) documentation.

### Element-wise comparison operations

See also `ComputationBuilder::Eq`

A set of standard element-wise binary comparison operations is supported. Note
that standard IEEE 754 floating-point comparison semantics apply when comparing
floating-point types.

<b> `Op(lhs, rhs)` </b>

Where `Op` is one of `Eq` (equal-to), `Ne` (not equal-to), `Ge`
(greater-or-equal-than), `Gt` (greater-than), `Le` (less-or-equal-than), `Le`
(less-than).

Arguments | Type                    | Semantics
--------- | ----------------------- | ----------------------------------------
`lhs`     | `ComputationDataHandle` | left-hand-side operand: array of type T
`rhs`     | `ComputationDataHandle` | right-hand-side operand: array of type T

The arguments' shapes have to be either similar or compatible. See the
[broadcasting](#broadcasting_semantics) documentation about what it means for
shapes to be compatible. The result of an operation has a shape which is the
result of broadcasting the two input arrays with the element type `PRED`. In
this variant, operations between arrays of different ranks are *not* supported,
unless one of the operands is a scalar.

An alternative variant with different-rank broadcasting support exists for these
operations:

<b> `Op(lhs, rhs, broadcast_dimensions)` </b>

Where `Op` is the same as above. This variant of the operation should be used
for comparison operations between arrays of different ranks (such as adding a
matrix to a vector).

The additional `broadcast_dimensions` operand is a slice of integers specifying
the dimensions to use for broadcasting the operands. The semantics are described
in detail in the [broadcasting](#broadcasting_semantics) documentation.

### Element-wise unary functions

ComputationBuilder supports these element-wise unary functions:

<b>`Exp(operand)`</b> Element-wise natural exponential `x -> e^x`.

<b>`Log(operand)`</b> Element-wise natural logarithm `x -> ln(x)`.

<b>`Neg(operand)`</b> Element-wise negation `x -> -x`.

<b>`Floor(operand)`</b> Element-wise floor `x -> ⌊x⌋`.

<b>`Ceil(operand)`</b> Element-wise ceil `x -> ⌈x⌉`.

<b>`Tanh(operand)`</b> Element-wise hyperbolic tangent `x -> tanh(x)`.

Arguments | Type                    | Semantics
--------- | ----------------------- | ---------------------------
`operand` | `ComputationDataHandle` | The operand to the function

The function is applied to each element in the `operand` array, resulting in an
array with the same shape. It is allowed for `operand` to be a scalar (rank 0).

### GetTupleElement

See also `ComputationBuilder::GetTupleElement`

Indexes into a tuple with a compile-time-constant value.

The value must be a compile-time-constant so that shape inference can determine
the type of the resulting value.

This is analogous to `std::get<int N>(t)` in C++. Conceptually:

```
let v: f32[10] = f32[10]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
let s: s32 = 5;
let t: (f32[10], s32) = tuple(v, s);
let element_1: s32 = gettupleelement(t, 1);  // Inferred shape matches s32.
```

See also [`Tuple`](#tuple).

### Infeed

See also `ComputationBuilder::Infeed`

<b> `Infeed(shape)` </b>

| Argument | Type    | Semantics                                             |
| -------- | ------- | ----------------------------------------------------- |
| `shape`  | `Shape` | Shape of the data read from the Infeed interface. The |
:          :         : layout field of the shape must be set to match the    :
:          :         : layout of the data sent to the device; otherwise its  :
:          :         : behavior is undefined.                                :

Devices have an abstraction for feeding data to long-running computations, e.g.,
feeding inputs to be consumed within the body of a [`While`](#while) loop.
`Infeed` reads a single data item from the implicit Infeed streaming interface
of the device, interpreting the data as the given shape and its layout, and
returns a `ComputationDataHandle` of the data. Multiple Infeed operations are
allowed in a computation, but there must be a total order among the Infeed
operations. For example, two Infeeds in the code below have a total order since
there is a dependency between the while loops. The compiler issues an error if
there isn't a total order.

```
result1 = while (condition, init = init_value) {
  Infeed(shape)
}

result2 = while (condition, init = result1) {
  Infeed(shape)
}
```

Nested tuple shapes are not supported. For an empty tuple shape, the Infeed
operation is effectively a nop and proceeds without reading any data from the
Infeed of the device.

> Note: We plan to allow multiple Infeed operations without a total order, in
> which case the compiler will provide information about how the Infeed
> operations are serialized in the compiled program.

### Map

See also `ComputationBuilder::Map`

<b> `Map(operands..., computation)` </b>

| Arguments         | Type                     | Semantics                     |
| ----------------- | ------------------------ | ----------------------------- |
| `operands`        | sequence of N            | N arrays of type T            |
:                   : `ComputationDataHandle`s :                               :
| `computation`     | `Computation`            | computation of type `T_0,     |
:                   :                          : T_1, ..., T_{N + M -1}` -> S` :
:                   :                          : with N parameters of type T   :
:                   :                          : and M of arbitrary type       :
| `static_operands` | sequence of M            | M arrays of arbitrary type    |
:                   : `ComputationDataHandle`s :                               :

Applies a scalar function over the given `operands` arrays, producing an array
of the same dimensions where each element is the result of the mapped function
applied to the corresponding elements in the input arrays with `static_operands`
given as additional input to `computation`.

The mapped function is an arbitrary computation with the restriction that it has
N inputs of scalar type `T` and a single output with type `S`. The output has
the same dimensions as the operands except that the element type T is replaced
with S.

For example: `Map(op1, op2, op3, computation, par1)` maps `elem_out <-
computation(elem1, elem2, elem3, par1)` at each (multi-dimensional) index in the
input arrays to produce the output array.

### Pad

See also `ComputationBuilder::Pad`

<b> `Pad(operand, padding_value, padding_config)` </b>

| Arguments        | Type                    | Semantics                     |
| ---------------- | ----------------------- | ----------------------------- |
| `operand`        | `ComputationDataHandle` | array of type `T`             |
| `padding_value`  | `ComputationDataHandle` | scalar of type `T` to fill in |
:                  :                         : the added padding             :
| `padding_config` | `PaddingConfig`         | padding amount on both edges  |
:                  :                         : (low, high) and between the   :
:                  :                         : elements of each dimension    :

Expands the given `operand` array by padding around the array as well as between
the elements of the array with the given `padding_value`. `padding_config`
specifies the amount of edge padding and the interior padding for each
dimension.

`PaddingConfig` is a repeated field of `PaddingConfigDimension`, which contains
three fields for each dimension: `edge_padding_low`, `edge_padding_high`, and
`interior_padding`. `edge_padding_low` and `edge_padding_high` specifies the
amount of padding added at the low-end (next to index 0) and the high-end (next
to the highest index) of each dimension respectively. `interior_padding`
specifies the amount of padding added between any two elements in each
dimension. This operation is a no-op if the edge padding pairs are all (0, 0)
and the interior padding values are all 0. Figure below shows examples of
different `edge_padding` and `interior_padding` values for a two dimensional
array.

<center><iframe src="../images/xla-pad.svg" width="920" height="600" title="Pad Diagram" frameborder="0"></iframe></center>

### Reduce

See also `ComputationBuilder::Reduce`

Applies a reduction function to an array.

<b> `Reduce(operand, init_value, computation, dimensions)` </b>

| Arguments     | Type                    | Semantics                        |
| ------------- | ----------------------- | -------------------------------- |
| `operand`     | `ComputationDataHandle` | array of type `T`                |
| `init_value`  | `ComputationDataHandle` | scalar of type `T`               |
| `computation` | `Computation`           | computation of type `T, T -> T`  |
| `dimensions`  | `int64` array           | unordered array of dimensions to |
:               :                         : reduce                           :

Conceptually, this operation reduces one or more dimensions in the input array
into scalars. The rank of the result array is `rank(operand) - len(dimensions)`.
`init_value` is the initial value used for every reduction and may also be
inserted anywhere during computation if the back-end chooses to do so. So in
most cases `init_value` should be an identity of the reduction function (for
example, 0 for addition).

The evaluation order of the reduction function across the reduction dimensions
is arbitrary and may be non-deterministic. Therefore, the reduction function
should not be overly sensitive to reassociation[^1].

As an example, when reducing across the one dimension in a 1D array with values
[10, 11, 12, 13], with reduction function `f` (this is `computation`) then that
could be computed as

`f(10, f(11, f(12, f(init_value, 13)))`

but there are also many other possibilities, e.g.

`f(init_value, f(f(10, f(init_value, 11)), f(f(init_value, 12), f(13,
init_value))))`

The following is a rough pseudo-code example of how reduction could be
implemented, using summation as the reduction computation with an initial value
of 0.

```python
result_shape <- remove all dims in dimensions from operand_shape

# Iterate over all elements in result_shape. The number of r's here is equal
# to the rank of the result
for r0 in range(result_shape[0]), r1 in range(result_shape[1]), ...:
  # Initialize this result element
  result[r0, r1...] <- 0

  # Iterate over all the reduction dimensions
  for d0 in range(dimensions[0]), d1 in range(dimensions[1]), ...:
    # Increment the result element with the value of the operand's element.
    # The index of the operand's element is constructed from all ri's and di's
    # in the right order (by construction ri's and di's together index over the
    # whole operand shape).
    result[r0, r1...] += operand[ri... di]
```

Here's an example of reducing a 2D array (matrix). The shape has rank 2,
dimension 0 of size 2 and dimension 1 of size 3:

<div/>
<div style="text-align:center">
![2d array](../images/xla-2d-matrix.svg)
</div>

Results of reducing dimensions 0 or 1 with an "add" function:

<div/>
<div style="text-align:center">
![reducing 2d array](../images/xla-reduce-from-2d.svg)
</div>

Note that both reduction results are 1D arrays. The diagram shows one as column
and another as row just for visual convenience.

For a more complex example, here is a 3D array. Its rank is 3, dimension 0 of
size 4, dimension 1 of size 2 and dimension 2 of size 3. For simplicity, the
values 1 to 6 are replicated across dimension 0.

<div/>
<div style="text-align:center">
![3d array](../images/xla-reduce-from-3d.svg)
</div>

Similarly to the 2D example, we can reduce just one dimension. If we reduce
dimension 0, for example, we get a rank-2 array where all values across
dimension 0 were folded into a scalar:

```text
| 4  8  12 |
| 4  8  12 |
```

If we reduce dimension 2, we also get a rank-2 array where all values across
dimension 2 were folded into a scalar:

```text
| 6  15 |
| 6  15 |
| 6  15 |
| 6  15 |
```

Note that the relative order between the remaining dimensions in the input is
preserved in the output, but some dimensions may get assigned new numbers (since
the rank changes).

We can also reduce multiple dimensions. Add-reducing dimensions 0 and 1 produces
the 1D array `| 20 28 36 |`.

Reducing the 3D array over all its dimensions produces the scalar `84`.

### ReduceWindow

See also `ComputationBuilder::ReduceWindow`

Applies a reduction function to all elements in each window of the input
multi-dimensional array, producing an output multi-dimensional array with the
same number of elements as the number of valid positions of the window. A
pooling layer can be expressed as a `ReduceWindow`.

<b> `ReduceWindow(operand, computation, window, init_value)` </b>

| Arguments           | Type                    | Semantics                    |
| ------------------- | ----------------------- | ---------------------------- |
| `operand`           | `ComputationDataHandle` | N dimensional array          |
:                     :                         : containing elements of type  :
:                     :                         : T. This is the base area on  :
:                     :                         : which the window is placed.  :
| `init_value`        | `ComputationDataHandle` | Starting value for the       |
:                     :                         : reduction. See [Reduce]      :
:                     :                         : (#reduce) for details.       :
| `computation`       | `Computation`           | Reduction function of type   |
:                     :                         : `T, T -> T`, to apply to all :
:                     :                         : elements in each window      :
| `window_dimensions` | `ArraySlice<int64>`     | array of integers for window |
:                     :                         : dimension values             :
| `window_strides`    | `ArraySlice<int64>`     | array of integers for window |
:                     :                         : stride values                :
| `padding`           | `Padding`               | padding type for window      |
:                     :                         : (Padding\:\:kSame or         :
:                     :                         : Padding\:\:kValid)           :

Below code and figure shows an example of using `ReduceWindow`. Input is a
matrix of size [4x6] and both window_dimensions and window_stride_dimensions are
[2x3].

```
// Create a computation for the reduction (maximum).
std::unique_ptr<Computation> max;
{
  ComputationBuilder builder(client_, "max");
  auto y = builder.Parameter(0, ShapeUtil::MakeShape(F32, {}), "y");
  auto x = builder.Parameter(1, ShapeUtil::MakeShape(F32, {}), "x");
  builder.Max(y, x);
  max = builder.Build().ConsumeValueOrDie();
}

// Create a ReduceWindow computation with the max reduction computation.
ComputationBuilder builder(client_, "reduce_window_2x3");
auto shape = ShapeUtil::MakeShape(F32, {4, 6});
auto input = builder.Parameter(0, shape, "input");
builder.ReduceWindow(
    input, *max,
    /*init_val=*/builder_.ConstantR0<float>(std::numeric_limits<float>::min()),
    /*window_dimensions=*/{2, 3},
    /*window_stride_dimensions=*/{2, 3},
    Padding::kValid);
```

<center><iframe src="../images/xla-reduce-window.svg" width="920" height="300" title="ReduceWindow Diagram" frameborder="0"></iframe></center>

Stride of 1 in a dimension specifies that the position of a window in the
dimension is 1 element away from its adjacent window. In order to specify that
no windows overlap with each other, window_stride_dimensions should be equal to
window_dimensions. The figure below illustrates the use of two different stride
values. Padding is applied to each dimension of the input and the calculations
are the same as though the input came in with the dimensions it has after
padding.

<center><iframe src="../images/xla-reduce-window-stride.svg" width="920" height="300" title="ReduceWindow Stride Diagram" frameborder="0"></iframe></center>

### Reshape

See also `ComputationBuilder::Reshape` and the [`Collapse`](#collapse)
operation.

Reshapes the dimensions of an array into a new configuration.

<b> `Reshape(operand, dimensions, new_sizes)` </b>

Arguments    | Type                    | Semantics
------------ | ----------------------- | ---------------------------------------
`operand`    | `ComputationDataHandle` | array of type T
`dimensions` | `int64` vector          | order in which dimensions are collapsed
`new_sizes`  | `int64` vector          | vector of sizes of new dimensions

Conceptually, reshape first flattens an array into a one-dimensional vector of
data values, and then refines this vector into a new shape. The input arguments
are an arbitrary array of type T, a compile-time-constant vector of dimension
indices, and a compile-time-constant vector of dimension sizes for the result.
The values in the `dimension` vector must be a permutation of all of T's
dimensions. The order of the dimensions in `dimensions` is from slowest-varying
dimension (most major) to fastest-varying dimension (most minor) in the loop
nest which collapses the input array into a single dimension. The `new_sizes`
vector determines the size of the output array. The value at index 0 in
`new_sizes` is the size of dimension 0, the value at index 1 is the size of
dimension 1, and so on. The product of the `new_size` dimensions must equal the
product of the operand's dimension sizes. When refining the collapsed array into
the multidimensional array defined by `new_sizes`, the dimensions in `new_sizes`
are ordered from slowest varying (most major) and to fastest varying (most
minor).

For example, let v be an array of 24 elements:

```
let v = f32[4x2x3] {{{10, 11, 12}, {15, 16, 17}},
                    {{20, 21, 22}, {25, 26, 27}},
                    {{30, 31, 32}, {35, 36, 37}},
                    {{40, 41, 42}, {45, 46, 47}}};

In-order collapse:
let v012_24 = Reshape(v, {0,1,2}, {24});
then v012_24 == f32[24] {10, 11, 12, 15, 16, 17,
                         20, 21, 22, 25, 26, 27,
                         30, 31, 32, 35, 36, 37,
                         40, 41, 42, 45, 46, 47};

let v012_83 = Reshape(v, {0,1,2}, {8,3});
then v012_83 == f32[8x3] {{10, 11, 12}, {15, 16, 17},
                          {20, 21, 22}, {25, 26, 27},
                          {30, 31, 32}, {35, 36, 37},
                          {40, 41, 42}, {45, 46, 47}};

Out-of-order collapse:
let v021_24 = Reshape(v, {1,2,0}, {24});
then v012_24 == f32[24] {10, 11, 12, 20, 21, 22,
                         30, 31, 32, 40, 41, 42,
                         15, 16, 17, 25, 26, 27,
                         35, 36, 37, 45, 46, 47};

let v021_83 = Reshape(v, {1,2,0}, {8,3});
then v021_83 == f32[8x3] {{10, 11, 12}, {20, 21, 22},
                          {30, 31, 32}, {40, 41, 42},
                          {15, 16, 17}, {25, 26, 27},
                          {35, 36, 37}, {45, 46, 47}};

let v021_262 = Reshape(v, {1,2,0}, {2,6,2});
then v021_262 == f32[2x6x2] {{{10, 11}, {12, 20}, {21, 22},
                              {30, 31}, {32, 40}, {41, 42}},
                             {{15, 16}, {17, 25}, {26, 27},
                              {35, 36}, {37, 45}, {46, 47}}};

```

As a special case, reshape can transform a single-element array to a scalar and
vice versa. For example, `Reshape(f32[1x1] {{5}}, {0,1}, {}) == 5; Reshape(5,
{}, {1,1}) == f32[1x1] {{5}};`

### Rev (reverse)

See `ComputationBuilder::Rev`

<b>`Rev(operand, dimensions)`</b>

Arguments    | Type                    | Semantics
------------ | ----------------------- | ---------------------
`operand`    | `ComputationDataHandle` | array of type T
`dimensions` | `ArraySlice<int64>`     | dimensions to reverse

Reverses the order of elements in the `operand` array along the specified
`dimensions`, generating an output array of the same shape. Each element of the
operand array at a multidimensional index is stored into the output array at a
transformed index. The multidimensional index is transformed by reversing the
index in each dimension to be reversed (i.e., if a dimension of size N is one of
the reversing dimensions, its index i is transformed into N - 1 - i).

One use for the `Rev` operation is to reverse the convolution weight array along
the two window dimensions during the gradient computation in neural networks.

### RngBernoulli

See also `ComputationBuilder::RngBernoulli`

Constructs an output of a given shape with random numbers generated following
the Bernoulli distribution. The parameter needs to be a scalar valued F32
operand while the output shape needs to have elemental type U32.

<b>`RngBernoulli(mean, shape)`</b>

| Arguments | Type                    | Semantics                             |
| --------- | ----------------------- | ------------------------------------- |
| `mean`    | `ComputationDataHandle` | Scalar of type F32 specifying mean of |
:           :                         : generated numbers                     :
| `shape`   | `Shape`                 | Output shape of type U32              |

### RngNormal

See also `ComputationBuilder::RngNormal`

Constructs an output of a given shape with random numbers generated following
the $$N(\mu, \sigma)$$ normal distribution. The parameters `mu` and `sigma`, and
output shape have to have elemental type F32. The parameters furthermore have to
be scalar valued.

<b>`RngNormal(mean, sigma, shape)`</b>

| Arguments | Type                    | Semantics                              |
| --------- | ----------------------- | -------------------------------------- |
| `mu`      | `ComputationDataHandle` | Scalar of type F32 specifying mean of  |
:           :                         : generated numbers                      :
| `sigma`   | `ComputationDataHandle` | Scalar of type F32 specifying standard |
:           :                         : deviation of generated numbers         :
| `shape`   | `Shape`                 | Output shape of type F32               |

### RngUniform

See also `ComputationBuilder::RngUniform`

Constructs an output of a given shape with random numbers generated following
the uniform distribution over the interval $$[a,b]$$. The parameters and output
shape may be either F32, S32 or U32, but the types have to be consistent.
Furthermore, the parameters need to be scalar valued.

<b>`RngUniform(a, b, shape)`</b>

| Arguments | Type                    | Semantics                         |
| --------- | ----------------------- | --------------------------------- |
| `a`       | `ComputationDataHandle` | Scalar of type T specifying lower |
:           :                         : limit of interval                 :
| `b`       | `ComputationDataHandle` | Scalar of type T specifying upper |
:           :                         : limit of interval                 :
| `shape`   | `Shape`                 | Output shape of type T            |

### SelectAndScatter

See also `ComputationBuilder::SelectAndScatter`

This operation can be considered as a composite operation that first computes
`ReduceWindow` on the `operand` array to select an element from each window, and
then scatters the `source` array to the indices of the selected elements to
construct an output array with the same shape as the operand array. The binary
`select` function is used to select an element from each window by applying it
across each window, and it is called with the property that the first
parameter's index vector is lexicographically less than the second parameter's
index vector. The `select` function returns `true` if the first parameter is
selected and returns `false` if the second parameter is selected, and the
function must hold transitivity (i.e., if `select(a, b)` and `select(b, c)` are
`true`, then `select(a, c)` is also `true`) so that the selected element does
not depend on the order of the elements traversed for a given window.

The function `scatter` is applied at each selected index in the output array. It
takes two scalar parameters:

1.  Current value at the selected index in the output array
2.  The scatter value from `source` that applies to the selected index

It combines the two parameters and returns a scalar value that's used to update
the value at the selected index in the output array. Initially, all indices of
the output array are set to `init_value`.

The output array has the same shape as the `operand` array and the `source`
array must have the same shape as the result of applying a `ReduceWindow`
operation on the `operand` array. `SelectAndScatter` can be used to
backpropagate the gradient values for a pooling layer in a neural network.

<b>`SelectAndScatter(operand, select, window_dimensions, window_strides,
padding, source, init_value, scatter)`</b>

| Arguments           | Type                    | Semantics                    |
| ------------------- | ----------------------- | ---------------------------- |
| `operand`           | `ComputationDataHandle` | array of type T over which   |
:                     :                         : the windows slide            :
| `select`            | `Computation`           | binary computation of type   |
:                     :                         : `T, T -> PRED`, to apply to  :
:                     :                         : all elements in each window; :
:                     :                         : returns `true` if the first  :
:                     :                         : parameter is selected and    :
:                     :                         : returns `false` if the       :
:                     :                         : second parameter is selected :
| `window_dimensions` | `ArraySlice<int64>`     | array of integers for window |
:                     :                         : dimension values             :
| `window_strides`    | `ArraySlice<int64>`     | array of integers for window |
:                     :                         : stride values                :
| `padding`           | `Padding`               | padding type for window      |
:                     :                         : (Padding\:\:kSame or         :
:                     :                         : Padding\:\:kValid)           :
| `source`            | `ComputationDataHandle` | array of type T with the     |
:                     :                         : values to scatter            :
| `init_value`        | `ComputationDataHandle` | scalar value of type T for   |
:                     :                         : the inital value of the      :
:                     :                         : output array                 :
| `scatter`           | `Computation`           | binary computation of type   |
:                     :                         : `T, T -> T`, to apply each   :
:                     :                         : scatter source element with  :
:                     :                         : its destination element      :

The figure below shows examples of using `SelectAndScatter`, with the `select`
function computing the maximal value among its parameters. Note that when the
windows overlap, as in the figure (2) below, an index of the `operand` array may
be selected multiple times by different windows. In the figure, the element of
value 9 is selected by both of the top windows (blue and red) and the binary
addition `scatter` function produces the output element of value 8 (2 + 6).

<center><iframe src="../images/xla-scatter-to-selected-window-element.svg" width="1000" height="700" title="SelectAndScatter Diagram" frameborder="0"></iframe></center>

### Select

See also `ComputationBuilder::Select`

Constructs an output array from elements of two input arrays, based on the
values of a predicate array.

<b> `Select(pred, on_true, on_false)` </b>

Arguments  | Type                    | Semantics
---------- | ----------------------- | ------------------
`pred`     | `ComputationDataHandle` | array of type PRED
`on_true`  | `ComputationDataHandle` | array of type T
`on_false` | `ComputationDataHandle` | array of type T

The arrays `on_true` and `on_false` must have the same shape. This is also the
shape of the output array. The array `pred` must have the same dimensionality as
`on_true` and `on_false`, with the `PRED` element type.

For each element `P` of `pred`, the corresponding element of the output array is
taken from `on_true` if the value of `P` is `true`, and from `on_false` if the
value of `P` is `false`. As a restricted form of
[broadcasting](#broadcasting_semantics), `pred` can be a scalar of type `PRED`.
In this case, the output array is taken wholly from `on_true` if `pred` is
`true`, and from `on_false` if `pred` is `false`.

Example with non-scalar `pred`:

```
let pred: PRED[4] = {true, false, false, true};
let v1: s32[4] = {1, 2, 3, 4};
let v2: s32[4] = {100, 200, 300, 400};
==>
Select(pred, v1, v2) = s32[4]{1, 200, 300, 4};
```

Example with scalar `pred`:

```
let pred: PRED = true;
let v1: s32[4] = {1, 2, 3, 4};
let v2: s32[4] = {100, 200, 300, 400};
==>
Select(pred, v1, v2) = s32[4]{1, 2, 3, 4};
```

Selections between tuples are supported. Tuples are considered to be scalar
types for this purpose. If `on_true` and `on_false` are tuples (which must have
the same shape!) then `pred` has to be a scalar of type `PRED`.

### Slice

See also `ComputationBuilder::Slice`

Slicing extracts a sub-array from the input array. The sub-array is of the same
rank as the input and contains the values inside a bounding box within the input
array where the dimensions and indices of the bounding box are given as
arguments to the slice operation.

<b> `Slice(operand, start_indices, limit_indices)` </b>

| Arguments       | Type                    | Semantics                        |
| --------------- | ----------------------- | -------------------------------- |
| `operand`       | `ComputationDataHandle` | N dimensional array of type T    |
| `start_indices` | `ArraySlice<int64>`     | List of N integers containing    |
:                 :                         : the starting indices of the      :
:                 :                         : slice for each dimension. Values :
:                 :                         : must be greater than or equal to :
:                 :                         : zero.                            :
| `limit_indices` | `ArraySlice<int64>`     | List of N integers containing    |
:                 :                         : the ending indices (exclusive)   :
:                 :                         : for the slice for each           :
:                 :                         : dimension. Each value must be    :
:                 :                         : strictly greater than the        :
:                 :                         : respective `start_indices` value :
:                 :                         : for the dimension and less than  :
:                 :                         : or equal to the size of the      :
:                 :                         : dimension.                       :

1-dimensional example:

```
let a = {0.0, 1.0, 2.0, 3.0, 4.0}
Slice(a, {2}, {4}) produces:
  {2.0, 3.0}
```

2-dimensional example:

```
let b =
 { {0.0,  1.0,  2.0},
   {3.0,  4.0,  5.0},
   {6.0,  7.0,  8.0},
   {9.0, 10.0, 11.0} }

Slice(b, {2, 1}, {4, 3}) produces:
  { { 7.0,  8.0},
    {10.0, 11.0} }
```

### Sort

See `ComputationBuilder::Sort`

Sorts the elements in the operand.

<b>`Sort(operand)`</b>

Arguments | Type                    | Semantics
--------- | ----------------------- | -------------------
`operand` | `ComputationDataHandle` | The operand to sort

### Trans (transpose)

See also the [`Reshape`](#reshape) operation.

<b>`Trans(operand)`</b>

Arguments | Type                    | Semantics
--------- | ----------------------- | -------------------------
`operand` | `ComputationDataHandle` | The operand to transpose.

Returns the transpose of `operand`. `operand` must have rank 2.

This is the same as Reshape(operand, {1, 0}, {operand.shape.dimensions[1],
operand.shape,dimensions[0]}).

### Tuple

See also `ComputationBuilder::Tuple`

A tuple containing a variable number of data handles, each of which has its own
shape.

This is analogous to `std::tuple` in C++. Conceptually:

```
let v: f32[10] = f32[10]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
let s: s32 = 5;
let t: (f32[10], s32) = tuple(v, s);
```

Tuples can be deconstructed (accessed) via the
[`GetTupleElement`](#gettupleelement) operation.

### While

See also `ComputationBuilder::While`

<b> `While(condition, body, init)` </b>

| Arguments   | Type          | Semantics                                      |
| ----------- | ------------- | ---------------------------------------------- |
| `condition` | `Computation` | Computation of type `T -> PRED` which defines  |
:             :               : the termination condition of the loop.         :
| `body`      | `Computation` | Computation of type `T -> T` which defines the |
:             :               : body of the loop.                              :
| `init`      | `T`           | Initial value for the parameter of `condition` |
:             :               : and `body`.                                    :

Sequentially executes the `body` until the `condition` fails. This is similar to
a typical while loop in many other languages except for the differences and
restrictions listed below.

*   A `While` node returns a value of type `T`, which is the result from the
    last execution of the `body`.
*   The shape of the type `T` is statically determined and must be the same
    across all iterations.
*   `While` nodes are not allowed to be nested. (This restriction may be lifted
    in the future on some targets.)

The T parameters of the computations are initialized with the `init` value in
the first iteration and are automatically updated to the new result from `body`
in each subsequent iteration.

One main use case of the `While` node is to implement the repeated execution of
training in neural networks. Simplified pseudocode is shown below with an graph
that represents the computation. The type `T` in this example is a `Tuple`
consisting of an `int32` for the iteration count and a `vector[10]` for the
accumulator. For 1000 iterations, the loop keeps adding a constant vector to the
accumulator.

```
// Pseudocode for the computation.
init = {0, zero_vector[10]} // Tuple of int32 and float[10].
result = init;
while (result(0) < 1000) {
  iteration = result(0) + 1;
  new_vector = result(1) + constant_vector[10];
  result = {iteration, new_vector};
}
```

## Broadcasting semantics

This section describes how the broadcasting semantics in XLA work.

### What is broadcasting

Broadcasting may be required for operations between multi-dimensional arrays of
different ranks, or between multi-dimensional arrays with different but
compatible shapes. Consider the addition `X+v` where `X` is a matrix (an array
of rank 2) and `v` is a vector (an array of rank 1). To perform element-wise
addition, XLA needs to "broadcast" the vector `v` to the same rank as the
matrix `X`, by replicating `v` a certain number of times. The vector's length
has to match at least one of the dimensions of the matrix.

For example:

    |1 2 3| + |7 8 9|
    |4 5 6|

The matrix's dimensions are (2,3), the vector's are (3). We broadcast the vector
by replicating it over rows to get:

    |1 2 3| + |7 8 9| = |8  10 12|
    |4 5 6|   |7 8 9|   |11 13 15|

In Numpy, this is called
[broadcasting](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

### Principles

We see XLA as a low-level infrastructure. Therefore, we want to make the
XLA language as strict and explicit as possible, avoiding implicit and
"magical" features that may make some computations slightly easier to define, at
the cost of more assumptions baked into user code that will be difficult to
change in the long term. If necessary, implicit and magical features can be
added in client-level wrappers.

Specifically w.r.t. broadcasting, we will require explicit broadcasting
specifications on operations between arrays of different ranks, instead of
inferring a possible broadcasting like Numpy does.

### Broadcasting a lower-rank array onto a higher-rank array

*Scalars* can always be broadcast over arrays without an explicit specification
of broadcasting dimensions. An element-wise binary operation between a scalar
and an array means applying the operation with the scalar for each element in
the array. For example, adding a scalar to a matrix means producing a matrix
each element of which is a sum of the scalar with the corresponding input
matrix's element.

    |1 2 3| + 7 = |8  9  10|
    |4 5 6|       |11 12 13|

Most broadcasting needs can be captured by using a tuple of dimensions on a
binary operation. When the inputs to the operation have different ranks, this
broadcasting tuple specifies which dimension(s) in the **higher-rank** array to
match with the **lower-rank** array.

Consider the previous example of adding a matrix with dimensions (2,3) to a
vector with dimension (3). *Without specifying broadcasting, this operation is
invalid.* Based on XLA convention, the left-most dimension is 0, and the
number grows as we walk the dimensions right-wards. For a (2,3) matrix we'd
index into it with `matrix[i,j]` with `i` running to 2 and `j` running to 3. `i`
indexes over dimension 0 and `j` indexes over dimension 1.

To correctly request our matrix-vector addition the user will specify the
broadcasting dimension to be (1), meaning that the vector's dimension is matched
to dimension 1 of the matrix. In 2D, if we consider dimension 0 as rows and
dimension 1 as columns, this means that each element of the vector becomes a
column of a size matching the number of rows in the matrix:

    |7 8 9| ==> |7 8 9|
                |7 8 9|

As a more complex example, consider adding a 3-element vector (dimension (3)) to
a 3x3 matrix (dimensions (3,3)). There are two ways broadcasting can happen
here:

Broadcasting dimension is 1, as before. Each vector element becomes a column -
the vector is duplicated for each row in the matrix.

    |7 8 9| ==> |7 8 9|
                |7 8 9|
                |7 8 9|

Broadcasting dimension is 0. Each vector element becomes a row - the vector is
duplicated for each column in the matrix.

     |7| ==> |7 7 7|
     |8|     |8 8 8|
     |9|     |9 9 9|

Note: when adding a 2x3 matrix to a 3-element vector, a broadcasting dimension
of 0 is invalid.

The broadcasting dimensions can be a tuple that describes how a smaller rank
shape is broadcast into a larger rank shape. For example, given a 2x3x4 cuboid
and a 3x4 matrix, a broadcasting tuple (1,2) means matching the matrix to
dimensions 1 and 2 of the cuboid.

This type of broadcast is used in the binary ops in `ComputationBuilder`, if the
`broadcast_dimensions` argument is given. In the XLA source code, this type
of broadcasting is sometimes called "InDim" broadcasting.

#### Formal definition

The broadcasting attribute allows matching a lower-rank array to a higher-rank
array, by specifying which dimensions of the higher-rank array to match. For
example, for an array with dimensions MxNxPxQ, we can match a vector with
dimension T as follows:

              MxNxPxQ

    dim 3:          T
    dim 2:        T
    dim 1:      T
    dim 0:    T

In each case, T has to be equal to the matching dimension of the higher-rank
array. The vector's values are then broadcast from the matched dimension to all
the other dimensions.

If we want to match a TxV matrix onto the MxNxPxQ array, we have to use a pair
of broadcasting dimensions:

              MxNxPxQ
    dim 2,3:      T V
    dim 1,2:    T V
    dim 0,3:  T     V
    etc...

The order of dimensions in the broadcasting tuple has to be the order in which
the lower-rank array's dimensions are expected to match the higher-rank array's
dimensions. The first element in the tuple says which dimension in the
higher-rank array has to match dimension 0 in the lower-rank array. The second
element for dimension 1, and so on. The order of broadcast dimensions has to be
strictly increasing. E.g. in the previous example, it's illegal to match V to N
and T to P; also, it's illegal to match V to both P and N.

### Broadcasting similar-rank arrays with degenerate dimensions

A related broadcasting problem is broadcasting two arrays that have the same
rank but different dimension sizes. Similarly to Numpy's rules, this is only
possible when the arrays are *compatible*. Two arrays are compatible when all
their dimensions are compatible. Two dimensions are compatible if:

*   They are equal, or
*   One of them is 1 (a "degenerate" dimension)

When we encounter two compatible arrays, the result shape has the maximum among
the two inputs at every dimension index.

Examples:

1.  (2,1) and (2,3) broadcast to (2,3).
2.  (1,2,5) and (7,2,5) broadcast to (7,2,5)
3.  (7,2,5) and (7,1,5) broadcast to (7,2,5)
4.  (7,2,5) and (7,2,6) are incompatible and cannot be broadcast.

A special case arises, and is also supported, where each of the input arrays has
a degenerate dimension at a different index. In this case, we get an "outer
operation": (2,1) and (1,3) broadcast to (2,3). For more examples, consult the
[Numpy documentation on broadcasting](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

### Broadcast composition

Broadcasting of a lower-rank array to a higher-rank array **and** broadcasting
using degenerate dimensions can both be performed in the same binary operation.
For example, a vector of size 4 and an matrix of size 1x2 can be added together
using broadcast dimensions value of (0):

    |1 2 3 4| + [5 6]    // [5 6] is a 1x2 matrix, not a vector.

First the vector is broadcast up to rank 2 (matrix) using the broadcast
dimensions. The single value (0) in the broadcast dimensions indicates that
dimension zero of the vector matches to dimension zero of the matrix. This
produces an matrix of size 4xM where the value M is chosen to match the
corresponding dimension size in the 1x2 array. Therefore, a 4x2 matrix is
produced:

    |1 1| + [5 6]
    |2 2|
    |3 3|
    |4 4|

Then "degenerate dimension broadcasting" broadcasts dimension zero of the 1x2
matrix to match the corresponding dimension size of the right hand side:

    |1 1| + |5 6|     |6  7|
    |2 2| + |5 6|  =  |7  8|
    |3 3| + |5 6|     |8  9|
    |4 4| + |5 6|     |9 10|

A more complicated example is a matrix of size 1x2 added to an array of size
4x3x1 using broadcast dimensions of (1, 2). First the 1x2 matrix is broadcast up
to rank 3 using the broadcast dimensions to produces an intermediate Mx1x2 array
where the dimension size M is determined by the size of the larger operand (the
4x3x1 array) producing a 4x1x2 intermediate array. The M is at dimension 0
(left-most dimension) because the dimensions 1 and 2 are mapped to the
dimensions of the original 1x2 matrix as the broadcast dimension are (1, 2).
This intermediate array can be added to the 4x3x1 matrix using broadcasting of
degenerate dimensions to produce a 4x3x2 array result.

[^1]: Some obvious reductions like "add reduction" are not strictly associative
    for floats. However, if the range of the data is limited, floating-point
    addition is close enough to being associative for most practical uses. It
    is possible to conceive some complete un-associative reductions, however,
    and these will produce wrong results in XLA reductions.

## C++ interface

The following is a fragment of the class definition for the client
`ComputationBuilder` interface, for reference:

```c++
class ComputationBuilder {
 public:
  // client: client in which to build the computation.
  // computation_name: name to use for the built computation.
  ComputationBuilder(Client* client, const string& computation_name);

  ~ComputationBuilder();

  // Returns the client the builder was initialized with.
  Client* client() { return client_; }

  // Returns the computation name.
  const string& name() { return name_; }

  // Sets the builder to a mode where it will die immediately when an error is
  // encountered, rather than producing it in a deferred fashion when Build() is
  // called (which is the default).
  void set_die_immediately_on_error(bool enabled) {
    die_immediately_on_error_ = enabled;
  }

  // Enqueues a "retrieve parameter value" instruction for a parameter that was
  // passed to the computation.
  ComputationDataHandle Parameter(int64 parameter_number, const Shape& shape,
                                  const string& name);

  // Retrieves the (inferred) shape of the operand in the computation.
  util::StatusOr<std::unique_ptr<Shape>> GetShape(
      const ComputationDataHandle& operand);

  // Checks that the operand has the given expected shape. Returns the operand
  // if yes, fails with a CHECK error if no.
  ComputationDataHandle CheckShape(const ComputationDataHandle& operand,
                                   const Shape& expected_shape);

  // Checks that the lhs and rhs results have the same shape.
  void CheckSameShape(const ComputationDataHandle& lhs,
                      const ComputationDataHandle& rhs);

  // Enqueues a constant with the value of the given literal onto the
  // computation.
  ComputationDataHandle ConstantLiteral(const Literal& literal);

  // Enqueues a constant onto the computation. Methods are templated on the
  // native host type (NativeT) which corresponds to a specific XLA
  // PrimitiveType as given in the following table:
  //
  //  Native Type   PrimitiveType
  // -----------------------------
  //   bool           PRED
  //   int32          S32
  //   int64          S64
  //   uint32         U32
  //   uint64         U64
  //   float          F32
  //   double         F64
  //
  // Note: not all primitive types defined in xla.proto have a corresponding
  // native type yet.
  template <typename NativeT>
  ComputationDataHandle ConstantR0(NativeT value);
  template <typename NativeT>
  ComputationDataHandle ConstantR1(gtl::ArraySlice<NativeT> values);
  template <typename NativeT>
  ComputationDataHandle ConstantR2(
      std::initializer_list<std::initializer_list<NativeT>> values);
  template <typename NativeT>
  ComputationDataHandle ConstantR2FromArray2D(const Array2D<NativeT>& values);
  template <typename NativeT>
  ComputationDataHandle ConstantR3FromArray3D(const Array3D<NativeT>& values);
  template <typename NativeT>
  ComputationDataHandle ConstantR4FromArray4D(const Array4D<NativeT>& values);

  // Enqueues a rank one constant (vector) onto the computation. The
  // vector has size 'length' and every element has the value 'value'.
  template <typename NativeT>
  ComputationDataHandle ConstantR1(int64 length, NativeT value);

  // Adds dimensions to an array by duplicating the data in the array.
  //
  // The new dimensions are inserted on the left, i.e. if
  // broadcast_sizes has values {a0, ..., aN} and the operand shape
  // has dimensions {b0, ..., bM} then the shape of the output has
  // dimensions {a0, ..., aN, b0, ..., bM}.
  //
  // The new dimensions index into copies of the operand, i.e.
  //
  //   output[i0, ..., iN, j0, ..., jM] = operand[j0, ..., jM]
  ComputationDataHandle Broadcast(const ComputationDataHandle& operand,
                                  gtl::ArraySlice<int64> broadcast_sizes);

  // Enqueues a pad operation onto the computation that pads the given value on
  // the edges as well as between the elements of the input. padding_config
  // specifies the padding amount for each dimension.
  ComputationDataHandle Pad(const ComputationDataHandle& operand,
                            const ComputationDataHandle& padding_value,
                            const PaddingConfig& padding_config);

  // Enqueues an operation onto the computation that flattens the operand based
  // on the dimension order (major/slowest-varying to minor/fastest-varying)
  // given, followed by reshaping it into the shape with the given dimension
  // sizes (also major to minor). Conceptually, this is a limited form of
  // "shape casting".
  ComputationDataHandle Reshape(const ComputationDataHandle& operand,
                                gtl::ArraySlice<int64> dimensions,
                                gtl::ArraySlice<int64> new_sizes);

  // Wrapper for Reshape.
  // Enqueues an operation to collapse the provided dimensions; e.g. an
  // operand with dimensions {x=256, y=2, z=2, p=32} can be collapsed to
  // {x=1024, y=32} by collapsing dims {0, 1, 2}. Collapsing dimensions must
  // be a consecutive, in-order subsequence of the operand dimensions.
  //
  // This could potentially cause data to be moved -- it provides a more
  // structured form of reshaping than an arbitrary Reshape operation.
  ComputationDataHandle Collapse(const ComputationDataHandle& operand,
                                 gtl::ArraySlice<int64> dimensions);

  // Enqueues a slice operation onto the computation that slices the operand
  // from the start indices to the limit indices; e.g.
  //
  //        x
  //   [ 0 1 2 3 ]
  // y [ 4 5 6 7 ] => slice(start={1, 1}, limit={2, 3}) => [ 5 6 ]
  //   [ 8 9 a b ]
  //
  // Note that "limit" means up-to-but-not-including; i.e. [start, limit) in 1D
  // range notation.
  ComputationDataHandle Slice(const ComputationDataHandle& operand,
                              gtl::ArraySlice<int64> start_indices,
                              gtl::ArraySlice<int64> limit_indices);

  // Enqueues a concatenate instruction onto the computation.
  ComputationDataHandle ConcatInDim(
      gtl::ArraySlice<ComputationDataHandle> operands, int64 dimension);

  // Enqueue a tracing operation onto the computation; the computation will emit
  // a logging message with the operand.
  void Trace(const string& tag, const ComputationDataHandle& operand);

  // Enqueues a conditional-move-like select operation onto the computation;
  // predicated on pred, selects between on_true and on_false.
  ComputationDataHandle Select(const ComputationDataHandle& pred,
                               const ComputationDataHandle& on_true,
                               const ComputationDataHandle& on_false);

  // Enqueues a tuple-creation instruction onto the computation.
  ComputationDataHandle Tuple(gtl::ArraySlice<ComputationDataHandle> elements);

  // Enqueues a tuple-element-get instruction onto the computation.
  ComputationDataHandle GetTupleElement(const ComputationDataHandle& tuple_data,
                                        int64 index);

  // Enqueues an equal-to comparison instruction onto the computation.
  ComputationDataHandle Eq(const ComputationDataHandle& lhs,
                           const ComputationDataHandle& rhs,
                           gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a not-equal comparison instruction onto the computation.
  ComputationDataHandle Ne(const ComputationDataHandle& lhs,
                           const ComputationDataHandle& rhs,
                           gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a greater-or-equal comparison instruction onto the computation.
  ComputationDataHandle Ge(const ComputationDataHandle& lhs,
                           const ComputationDataHandle& rhs,
                           gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a greater-than comparison instruction onto the computation.
  ComputationDataHandle Gt(const ComputationDataHandle& lhs,
                           const ComputationDataHandle& rhs,
                           gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a less-than comparison instruction onto the computation.
  ComputationDataHandle Lt(const ComputationDataHandle& lhs,
                           const ComputationDataHandle& rhs,
                           gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a less-or-equal comparison instruction onto the computation.
  ComputationDataHandle Le(const ComputationDataHandle& lhs,
                           const ComputationDataHandle& rhs,
                           gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a dot instruction onto the computation.
  ComputationDataHandle Dot(const ComputationDataHandle& lhs,
                            const ComputationDataHandle& rhs);

  // Default dimension numbers used for a convolution.
  static constexpr int64 kConvBatchDimension = 0;
  static constexpr int64 kConvFeatureDimension = 1;
  static constexpr int64 kConvFirstSpatialDimension = 2;
  static constexpr int64 kConvSecondSpatialDimension = 3;
  static constexpr int64 kConvKernelOutputDimension = 0;
  static constexpr int64 kConvKernelInputDimension = 1;
  static constexpr int64 kConvKernelFirstSpatialDimension = 2;
  static constexpr int64 kConvKernelSecondSpatialDimension = 3;

  // Creates a default ConvolutionDimensionNumbers. For the input operand
  // {batch, feature, height, width} = {0, 1, 2, 3} and for the weight operand
  // {kernel_output_feature, kernel_input_feature, height, width = {0, 1, 2, 3}.
  static ConvolutionDimensionNumbers CreateDefaultConvDimensionNumbers();

  // Creates a ConvolutionDimensionNumbers with the given arguments. Returns an
  // error if either the input or the weight dimension numbers have conflicts.
  static util::StatusOr<ConvolutionDimensionNumbers> CreateConvDimensionNumbers(
      int64 batch, int64 feature, int64 first_spatial, int64 second_spatial,
      int64 kernel_output_feature, int64 kernel_input_feature,
      int64 kernel_first_spatial, int64 kernel_second_spatial);

  // Enqueues a convolution instruction onto the computation, which uses the
  // default convolution dimension numbers.
  ComputationDataHandle Conv(const ComputationDataHandle& lhs,
                             const ComputationDataHandle& rhs,
                             gtl::ArraySlice<int64> window_strides,
                             Padding padding);

  // Enqueues a convolution instruction onto the computation, with the caller
  // provided padding configuration in the format returned by MakePadding().
  ComputationDataHandle ConvWithGeneralPadding(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      gtl::ArraySlice<int64> window_strides,
      gtl::ArraySlice<std::pair<int64, int64>> padding);

  // Enqueues a convolution instruction onto the computation, with the caller
  // provided dimension numbers configuration.
  ComputationDataHandle ConvWithGeneralDimensions(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      gtl::ArraySlice<int64> window_strides, Padding padding,
      const ConvolutionDimensionNumbers& dimension_numbers);

  // Enqueues a convolution instruction onto the computation, with the caller
  // provided padding configuration as well as the dimension numbers.
  ComputationDataHandle ConvGeneral(
      const ComputationDataHandle& lhs, const ComputationDataHandle& rhs,
      gtl::ArraySlice<int64> window_strides,
      gtl::ArraySlice<std::pair<int64, int64>> padding,
      const ConvolutionDimensionNumbers& dimension_numbers);

  // Enqueues an infeed instruction onto the computation, which reads data of
  // the given shape from the infeed buffer of the device.
  ComputationDataHandle Infeed(const Shape& shape);

  // Enqueues a custom call instruction onto the computation.
  // During code generation, a call instruction is emitted which targets a
  // symbol with the name |call_target_name|.  The |operands| are passed to the
  // call instruction.  |shape| is the resultant shape.
  ComputationDataHandle CustomCallOp(
      tensorflow::StringPiece call_target_name,
      gtl::ArraySlice<ComputationDataHandle> operands, const Shape& shape);

  // The following methods enqueue element-wise binary arithmetic operations
  // onto the computation. The shapes of the operands have to match unless one
  // of the operands is a scalar, or an explicit broadcast dimension is given
  // (see g3doc for more details).

  // Enqueues an add instruction onto the computation.
  ComputationDataHandle Add(const ComputationDataHandle& lhs,
                            const ComputationDataHandle& rhs,
                            gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a subtract instruction onto the computation.
  ComputationDataHandle Sub(const ComputationDataHandle& lhs,
                            const ComputationDataHandle& rhs,
                            gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a multiply instruction onto the computation.
  ComputationDataHandle Mul(const ComputationDataHandle& lhs,
                            const ComputationDataHandle& rhs,
                            gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a divide instruction onto the computation.
  ComputationDataHandle Div(const ComputationDataHandle& lhs,
                            const ComputationDataHandle& rhs,
                            gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a remainder instruction onto the computation.
  ComputationDataHandle Rem(const ComputationDataHandle& lhs,
                            const ComputationDataHandle& rhs,
                            gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a max instruction onto the computation.
  ComputationDataHandle Max(const ComputationDataHandle& lhs,
                            const ComputationDataHandle& rhs,
                            gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Enqueues a min instruction onto the computation.
  ComputationDataHandle Min(const ComputationDataHandle& lhs,
                            const ComputationDataHandle& rhs,
                            gtl::ArraySlice<int64> broadcast_dimensions = {});

  // Reduces an array among the provided dimensions, given "computation" as a
  // reduction operator.
  ComputationDataHandle Reduce(const ComputationDataHandle& operand,
                               const ComputationDataHandle& init_value,
                               const Computation& computation,
                               gtl::ArraySlice<int64> dimensions_to_reduce);

  // Enqueues a windowed reduce instruction onto the computation.
  ComputationDataHandle ReduceWindow(const ComputationDataHandle& operand,
                                     const ComputationDataHandle& init_value,
                                     const Computation& computation,
                                     gtl::ArraySlice<int64> window_dimensions,
                                     gtl::ArraySlice<int64> window_strides,
                                     Padding padding);

  // As ReduceWindow(), but the padding is given in the format
  // returned by MakePadding().
  ComputationDataHandle ReduceWindowWithGeneralPadding(
      const ComputationDataHandle& operand,
      const ComputationDataHandle& init_value, const Computation& computation,
      gtl::ArraySlice<int64> window_dimensions,
      gtl::ArraySlice<int64> window_strides,
      gtl::ArraySlice<std::pair<int64, int64>> padding);

  // Enqueues an operation that scatters the `source` array to the selected
  // indices of each window.
  ComputationDataHandle SelectAndScatter(
      const ComputationDataHandle& operand, const Computation& select,
      gtl::ArraySlice<int64> window_dimensions,
      gtl::ArraySlice<int64> window_strides, Padding padding,
      const ComputationDataHandle& source,
      const ComputationDataHandle& init_value, const Computation& scatter);

  // As SelectAndScatter(), but the padding is given in the format
  // returned by MakePadding().
  ComputationDataHandle SelectAndScatterWithGeneralPadding(
      const ComputationDataHandle& operand, const Computation& select,
      gtl::ArraySlice<int64> window_dimensions,
      gtl::ArraySlice<int64> window_strides,
      gtl::ArraySlice<std::pair<int64, int64>> padding,
      const ComputationDataHandle& source,
      const ComputationDataHandle& init_value, const Computation& scatter);

  // Enqueues an exp instruction onto the computation.
  ComputationDataHandle Exp(const ComputationDataHandle& operand);

  // Enqueues a floor instruction onto the computation.
  ComputationDataHandle Floor(const ComputationDataHandle& operand);

  // Enqueues a ceil instruction onto the computation.
  ComputationDataHandle Ceil(const ComputationDataHandle& operand);

  // Enqueues an log instruction (natural logarithm) onto the computation.
  ComputationDataHandle Log(const ComputationDataHandle& operand);

  // Enqueues a tanh instruction onto the computation.
  ComputationDataHandle Tanh(const ComputationDataHandle& operand);

  // Enqueues a float32 sqrt instruction onto the computation.
  // (float32 is specified as there is an implicit float32 0.5f constant
  // exponent).
  ComputationDataHandle SqrtF32(const ComputationDataHandle& operand);

  // Enqueues a float32 square instruction onto the computation.
  // (float32 is specified as there is an implicit float32 2.0f constant
  // exponent).
  ComputationDataHandle SquareF32(const ComputationDataHandle& operand);

  // Enqueues a lhs^rhs computation onto the computation.
  ComputationDataHandle Pow(const ComputationDataHandle& lhs,
                            const ComputationDataHandle& rhs);

  // Enqueues a convert instruction onto the computation that changes the
  // element type of the operand array to primitive_type.
  ComputationDataHandle ConvertElementType(const ComputationDataHandle& operand,
                                           PrimitiveType new_element_type);

  // Enqueues a float32 reciprocal instruction onto the computation.
  // (float32 is specified as there is an implicit float32 -1.0f constant
  // exponent).
  //
  // TODO(leary) axe F32 suffix, can be determined by reflecting on the shape of
  // the operand.
  ComputationDataHandle ReciprocalF32(const ComputationDataHandle& operand);

  // Enqueues a negate instruction onto the computation.
  ComputationDataHandle Neg(const ComputationDataHandle& operand);

  // Enqueues a transpose instruction onto the computation.
  ComputationDataHandle Trans(const ComputationDataHandle& operand);

  // Enqueues a reverse instruction onto the computation. The order of the
  // elements in the given dimensions is reversed (i.e., the element at index i
  // is moved to index dimension_size - 1 - i).
  ComputationDataHandle Rev(const ComputationDataHandle& operand,
                            gtl::ArraySlice<int64> dimensions);

  // Enqueues a sort (as increasing order) instruction onto the computation.
  ComputationDataHandle Sort(const ComputationDataHandle& operand);

  // Enqueues a clamp instruction onto the computation.
  ComputationDataHandle Clamp(const ComputationDataHandle& min,
                              const ComputationDataHandle& operand,
                              const ComputationDataHandle& max);

  // Enqueues a map instruction onto the computation.
  ComputationDataHandle Map(
      gtl::ArraySlice<ComputationDataHandle> operands,
      const Computation& computation,
      gtl::ArraySlice<ComputationDataHandle> static_operands = {});

  // Enqueues a N(mu, sigma) random number generation instruction onto the
  // computation.
  ComputationDataHandle RngNormal(const ComputationDataHandle& mu,
                                  const ComputationDataHandle& sigma,
                                  const Shape& shape);

  // Enqueues a U(a, b) random number generation instruction onto the
  // computation.
  ComputationDataHandle RngUniform(const ComputationDataHandle& a,
                                   const ComputationDataHandle& b,
                                   const Shape& shape);

  // Enqueues a B(1, p) random number generation instruction onto the
  // computation.
  ComputationDataHandle RngBernoulli(const ComputationDataHandle& mean,
                                     const Shape& shape);

  // Enqueues a while node onto the computation.
  ComputationDataHandle While(const Computation& condition,
                              const Computation& body,
                              const ComputationDataHandle& init);

  // Computes the value of a constant indicated by a
  // ComputationDataHandle.
  //
  // The handle must be from the computation currently being built -
  // i.e. returned from this builder with no intervening call to
  // Build(). This happens to currently work regardless of that, but
  // that may stop working at any time.
  //
  // The handle must represent a constant value, which in this case
  // means that it must not statically depend on a parameter to the
  // computation that is being built. Note this allows the output of
  // an Rng() node to count as constant - in that case you may receive
  // different values if you call this method several times. Let us
  // know if you have a use-case where that is a problem.
  //
  // This functionality can be useful when translating a computation
  // into XLA where something that looked dynamic is required by XLA
  // to be specified as a constant. E.g. the source computation
  // (outside of XLA) may include a dynamic computation of the shape
  // of something and ComputeConstant lets you determine what the
  // value of that computation is in the case where the value can be
  // determined at compile time.
  //
  // If output_layout is non-null, then the output of the computation
  // will be stored using that layout.
  util::StatusOr<std::unique_ptr<GlobalData>> ComputeConstant(
      const ComputationDataHandle& handle,
      const Layout* output_layout = nullptr);

  // Returns a new ComputationBuilder whose resultant Computation is used only
  // by this ComputationBuilder. The sub-ComputationBuilder has the same
  // die_immediately_on_error behavior as the parent.
  std::unique_ptr<ComputationBuilder> CreateSubBuilder(
      const string& computation_name);

  // Modifies the computation being built so that executions of it
  // will return the value associated with operand, rather than the
  // last expression enqueued on the ComputationBuilder. Any subsequent
  // operations added to the ComputationBuilder will not have any effect unless
  // SetReturnValue is called again.
  util::Status SetReturnValue(const ComputationDataHandle& operand);

  // Builds the computation with the requested operations, or returns a non-ok
  // status.
  util::StatusOr<std::unique_ptr<Computation>> Build();

  // Builds the computation with the requested operations, or notes an error in
  // the parent ComputationBuilder and returns an empty computation if building
  // failed. This function is intended to be used where the returned
  // Computation is only used by the parent ComputationBuilder and hence further
  // operation on the returned Computation will simply be error'ed out if an
  // error occurred while building this computation. If the built computation is
  // to be used by a ComputationBuilder other than the parent ComputationBuilder
  // then Build() should be used instead.
  std::unique_ptr<Computation> BuildAndNoteError();
};
```
