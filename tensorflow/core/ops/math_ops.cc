#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("AddN")
    .Input("inputs: N * T")
    .Output("sum: T")
    .Attr("N: int >= 1")
    .Attr("T: numbertype")
    .SetIsCommutative()
    .SetIsAggregate()
    .Doc(R"doc(
Add all input tensors element wise.

inputs: Must all be the same size and shape.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("BatchMatMul")
    .Input("x: T")
    .Input("y: T")
    .Output("out: T")
    .Attr("T: {float, double, int32, complex64}")
    .Attr("adj_x: bool = false")
    .Attr("adj_y: bool = false")
    .Doc(R"doc(
Multiplies slices of two tensors in batches.

Multiplies all slices of `Tensor` `x` and `y` (each slice can be
viewed as an element of a batch), and arranges the individual results
in a single output tensor of the same batch size. Each of the
individual slices can optionally be adjointed (to adjoint a matrix
means to transpose and conjugate it) before multiplication by setting
the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

The input tensors `x` and `y` are 3-D or higher with shape `[..., r_x, c_x]`
and `[..., r_y, c_y]`.

The output tensor is 3-D or higher with shape `[..., r_o, c_o]`, where:

    r_o = c_x if adj_x else r_x
    c_o = r_y if adj_y else c_y

It is computed as:

    out[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

x: 3-D or higher with shape `[..., r_x, c_x]`.
y: 3-D or higher with shape `[..., r_y, c_y]`.
out: 3-D or higher with shape `[..., r_o, c_o]`
adj_x: If `True`, adjoint the slices of `x`. Defaults to `False`.
adj_y: If `True`, adjoint the slices of `y`. Defaults to `False`.
)doc");

// --------------------------------------------------------------------------
// Casting Ops
//
// NOTE: Only a smaller number of types are supported by
// Cast. The exact casting rule is TBD. The current
// implementation uses C++ static cast rules for numeric
// types, which may be changed in the future.
REGISTER_OP("Cast")
    .Input("x: SrcT")
    .Output("y: DstT")
    .Attr("SrcT: type")
    .Attr("DstT: type")
    .Doc(R"doc(
Cast x of type SrcT to y of DstT.
)doc");

REGISTER_OP("_HostCast")
    .Input("x: SrcT")
    .Output("y: DstT")
    .Attr("SrcT: type")
    .Attr("DstT: type")
    .Doc(R"doc(
Cast x of type SrcT to y of DstT.

_HostCast requires its input and produces its output in host memory.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("Abs")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float, double, int32, int64}")
    .Doc(R"doc(
Computes the absolute value of a tensor.

Given a tensor `x`, this operation returns a tensor containing the absolute
value of each element in `x`. For example, if x is an input element and y is
an output element, this operation computes \\(y = |x|\\).
)doc");

REGISTER_OP("ComplexAbs")
    .Input("x: complex64")
    .Output("y: float")
    .Doc(R"doc(
Computes the complex absolute value of a tensor.

Given a tensor `x` of complex numbers, this operation returns a tensor of type
`float` that is the absolute value of each element in `x`. All elements in `x`
must be complex numbers of the form \\(a + bj\\). The absolute value is
computed as \\( \sqrt{a^2 + b^2}\\).

For example:

```
# tensor 'x' is [[-2.25 + 4.75j], [-3.25 + 5.75j]]
tf.complex_abs(x) ==> [5.25594902, 6.60492229]
```
)doc");

// Declares cwise unary operations signature: 't -> 't
#define UNARY()                      \
  Input("x: T").Output("y: T").Attr( \
      "T: {float, double, int32, complex64, int64}")

REGISTER_OP("Neg")
    .UNARY()
    .Doc(R"doc(
Computes numerical negative value element-wise.
I.e., \\(y = -x\\).
)doc");

REGISTER_OP("Inv")
    .UNARY()
    .Doc(R"doc(
Computes the reciprocal of x element-wise.
I.e., \\(y = 1 / x\\).
)doc");

REGISTER_OP("Square")
    .UNARY()
    .Doc(R"doc(
Computes square of x element-wise.
I.e., \\(y = x * x = x^2\\).
)doc");

REGISTER_OP("Sqrt")
    .UNARY()
    .Doc(R"doc(
Computes square root of x element-wise.
I.e., \\(y = \sqrt{x} = x^{1/2}\\).
)doc");

REGISTER_OP("Rsqrt")
    .UNARY()
    .Doc(R"doc(
Computes reciprocal of square root of x element-wise.
I.e., \\(y = 1 / \sqrt{x}\\).
)doc");

REGISTER_OP("Exp")
    .UNARY()
    .Doc(R"doc(
Computes exponential of x element-wise.  \\(y = e^x\\).
)doc");

REGISTER_OP("Log")
    .UNARY()
    .Doc(R"doc(
Computes natural logrithm of x element-wise.
I.e., \\(y = \log_e x\\).
)doc");

REGISTER_OP("Tanh")
    .UNARY()
    .Doc(R"doc(
Computes hyperbolic tangent of `x` element-wise.
)doc");

REGISTER_OP("Sigmoid")
    .UNARY()
    .Doc(R"doc(
Computes sigmoid of `x` element-wise.

Specifically, `y = 1 / (1 + exp(-x))`.
)doc");

REGISTER_OP("Sin")
    .UNARY()
    .Doc(R"doc(
Computes sin of x element-wise.
)doc");

REGISTER_OP("Cos")
    .UNARY()
    .Doc(R"doc(
Computes cos of x element-wise.
)doc");

#undef UNARY

REGISTER_OP("IsNan")
    .Input("x: T")
    .Output("y: bool")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Returns which elements of x are NaN.
)doc");

REGISTER_OP("IsInf")
    .Input("x: T")
    .Output("y: bool")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Returns which elements of x are Inf.
)doc");

REGISTER_OP("IsFinite")
    .Input("x: T")
    .Output("y: bool")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Returns which elements of x are finite.
)doc");

REGISTER_OP("Sign")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float, double, int32, int64}")
    .Doc(R"doc(
Returns an element-wise indication of the sign of a number.

y = sign(x) = -1 if x < 0; 0 if x == 0; 1 if x > 0.
)doc");

REGISTER_OP("Floor")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Returns element-wise largest integer not greater than x.
)doc");

REGISTER_OP("Ceil")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Returns element-wise smallest integer in not less than x.
)doc");

// Declares cwise binary operations signature: 't, 't -> 't.

#define BINARY_MORE()                              \
  Input("x: T").Input("y: T").Output("z: T").Attr( \
      "T: {float, double, int8, int16, int32, complex64, int64}")

#define BINARY_FEWER()                             \
  Input("x: T").Input("y: T").Output("z: T").Attr( \
      "T: {float, double, int32, complex64, int64}")

REGISTER_OP("Add")
    .BINARY_MORE()
    .SetIsCommutative()
    .Doc(R"doc(
Returns x + y element-wise.

*NOTE*: Add supports broadcasting. AddN does not.
)doc");

REGISTER_OP("Sub")
    .BINARY_FEWER()
    .Doc(R"doc(
Returns x - y element-wise.
)doc");

REGISTER_OP("Mul")
    .BINARY_MORE()
    .SetIsCommutative()
    .Doc(R"doc(
Returns x * y element-wise.
)doc");

REGISTER_OP("Div")
    .BINARY_FEWER()
    .Doc(R"doc(
Returns x / y element-wise.
)doc");

#undef BINARY_FEWER
#undef BINARY_MORE

REGISTER_OP("Maximum")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {float, double, int32, int64}")
    .SetIsCommutative()
    .Doc(R"doc(
Returns the max of x and y (i.e. x > y ? x : y) element-wise, broadcasts.
)doc");

REGISTER_OP("Minimum")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {float, double, int32, int64}")
    .SetIsCommutative()
    .Doc(R"doc(
Returns the min of x and y (i.e. x < y ? x : y) element-wise, broadcasts.
)doc");

REGISTER_OP("Mod")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {int32, int64, float, double}")
    .Doc(R"doc(
Returns element-wise remainder of division.
)doc");

REGISTER_OP("Pow")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {float, double, int32, complex64, int64}")
    .Doc(R"doc(
Computes the power of one value to another.

Given a tensor `x` and a tensor `y`, this operation computes \\(x^y\\) for
corresponding elements in `x` and `y`. For example:

```
# tensor 'x' is [[2, 2]], [3, 3]]
# tensor 'y' is [[8, 16], [2, 3]]
tf.pow(x, y) ==> [[256, 65536], [9, 27]]
```
)doc");

// --------------------------------------------------------------------------

// Declares cwise binary comparison operations signature: 't, 't -> bool,
// where 't has a natural total order.
#define COMPARISON()                                  \
  Input("x: T").Input("y: T").Output("z: bool").Attr( \
      "T: {float, double, int32, int64}")

REGISTER_OP("Less")
    .COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x < y) element-wise.
)doc");

REGISTER_OP("LessEqual")
    .COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x <= y) element-wise.
)doc");

REGISTER_OP("Greater")
    .COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x > y) element-wise.
)doc");

REGISTER_OP("GreaterEqual")
    .COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x >= y) element-wise.
)doc");

#undef COMPARISON

// --------------------------------------------------------------------------

#define COMPARISON()                                                     \
  Input("x: T").Input("y: T").Output("z: bool").SetIsCommutative().Attr( \
      "T: {float, double, int32, int64, complex64, quint8, qint8, qint32}")

REGISTER_OP("Equal")
    .COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x == y) element-wise.
)doc");

REGISTER_OP("NotEqual")
    .COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x != y) element-wise.
)doc");

#undef COMPARISON

// --------------------------------------------------------------------------

REGISTER_OP("LogicalNot")
    .Input("x: bool")
    .Output("y: bool")
    .Doc(R"doc(
Returns the truth value of NOT x element-wise.
)doc");

#define BINARY_LOGICAL() \
  Input("x: bool").Input("y: bool").Output("z: bool").SetIsCommutative()

REGISTER_OP("LogicalAnd")
    .BINARY_LOGICAL()
    .Doc(R"doc(
Returns the truth value of x AND y element-wise.
)doc");

REGISTER_OP("LogicalOr")
    .BINARY_LOGICAL()
    .Doc(R"doc(
Returns the truth value of x OR y element-wise.
)doc");

#undef BINARY_LOGICAL

// --------------------------------------------------------------------------

REGISTER_OP("Select")
    .Input("condition: bool")
    .Input("t: T")
    .Input("e: T")
    .Output("out: T")
    .Attr("T: type")
    .Doc(R"doc(
Selects elements from `t` or `e`, depending on `condition`.

The `condition`, `t`, and `e` tensors must all have the same shape,
and the output will also have that shape. The `condition` tensor acts
as an element-wise mask that chooses, based on the value at each
element, whether the corresponding element in the output should be
taken from `t` (if true) or `e` (if false). For example:

For example:

```prettyprint
# 'condition' tensor is [[True, False]
#                        [True, False]]
# 't' is [[1, 1],
#         [1, 1]]
# 'e' is [[2, 2],
#         [2, 2]]
select(condition, t, e) ==> [[1, 2],
                             [1, 2]]
```

t:= A `Tensor` with the same shape as `condition`.
e:= A `Tensor` with the same type and shape as `t`.
out:= A `Tensor` with the same type and shape as `t` and `e`.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("MatMul")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {float, double, int32, complex64}")
    .Doc(R"doc(
Multiply the matrix "a" by the matrix "b".

The inputs must be two-dimensional matrices and the inner dimension of
"a" (after being transposed if transpose_a is true) must match the
outer dimension of "b" (after being transposed if transposed_b is
true).

*Note*: The default kernel implementation for MatMul on GPUs uses
cublas.

transpose_a: If true, "a" is transposed before multiplication.
transpose_b: If true, "b" is transposed before multiplication.
)doc");

REGISTER_OP("SparseMatMul")
    .Input("a: float")
    .Input("b: float")
    .Output("product: float")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("a_is_sparse: bool = false")
    .Attr("b_is_sparse: bool = false")
    .Doc(R"doc(
Multiply matrix "a" by matrix "b".

The inputs must be two-dimensional matrices and the inner dimension of "a" must
match the outer dimension of "b". This op is optimized for the case where at
least one of "a" or "b" is sparse. The breakeven for using this versus a dense
matrix multiply on one platform was 30% zero values in the sparse matrix.
)doc");

// --------------------------------------------------------------------------

// For operations where the output is a reduction function along some
// dimensions of the input.
REGISTER_OP("Sum")
    .Input("input: T")
    .Input("reduction_indices: int32")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Doc(R"doc(
Computes the sum of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

input: The tensor to reduce.
reduction_indices: The dimensions to reduce.
keep_dims: If true, retain reduced dimensions with length 1.
output: The reduced tensor.
)doc");

REGISTER_OP("Mean")
    .Input("input: T")
    .Input("reduction_indices: int32")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Doc(R"doc(
Computes the mean of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

input: The tensor to reduce.
reduction_indices: The dimensions to reduce.
keep_dims: If true, retain reduced dimensions with length 1.
output: The reduced tensor.
)doc");

REGISTER_OP("Prod")
    .Input("input: T")
    .Input("reduction_indices: int32")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Doc(R"doc(
Computes the product of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

input: The tensor to reduce.
reduction_indices: The dimensions to reduce.
keep_dims: If true, retain reduced dimensions with length 1.
output: The reduced tensor.
)doc");

REGISTER_OP("Min")
    .Input("input: T")
    .Input("reduction_indices: int32")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Doc(R"doc(
Computes the minimum of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

input: The tensor to reduce.
reduction_indices: The dimensions to reduce.
keep_dims: If true, retain reduced dimensions with length 1.
output: The reduced tensor.
)doc");

REGISTER_OP("Max")
    .Input("input: T")
    .Input("reduction_indices: int32")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Doc(R"doc(
Computes the maximum of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

input: The tensor to reduce.
reduction_indices: The dimensions to reduce.
keep_dims: If true, retain reduced dimensions with length 1.
output: The reduced tensor.
)doc");

REGISTER_OP("ArgMax")
    .Input("input: T")
    .Input("dimension: int32")
    .Output("output: int64")
    .Attr("T: numbertype")
    .Doc(R"doc(
Returns the index with the largest value across dimensions of a tensor.

dimension: int32, 0 <= dimension < rank(input).  Describes which dimension
  of the input Tensor to reduce across. For vectors, use dimension = 0.
)doc");

REGISTER_OP("ArgMin")
    .Input("input: T")
    .Input("dimension: int32")
    .Output("output: int64")
    .Attr("T: numbertype")
    .Doc(R"doc(
Returns the index with the smallest value across dimensions of a tensor.

dimension: int32, 0 <= dimension < rank(input).  Describes which dimension
  of the input Tensor to reduce across. For vectors, use dimension = 0.
)doc");

REGISTER_OP("SegmentSum")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .Doc(R"doc(
Computes the sum along segments of a tensor.

Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \sum_j data_j\\) where sum is over `j` such
that `segment_ids[j] == i`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/SegmentSum.png" alt>
</div>

segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
first dimension.  Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension_0 which
has size `k`, the number of segments.
)doc");

REGISTER_OP("SegmentMean")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .Doc(R"doc(
Computes the mean along segments of a tensor.

Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Computes a tensor such that
\\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
over `j` such that `segment_ids[j] == i` and `N` is the total number of
values summed.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/SegmentMean.png" alt>
</div>

segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
first dimension.  Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension_0 which
has size `k`, the number of segments.
)doc");

REGISTER_OP("SegmentProd")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .Doc(R"doc(
Computes the product along segments of a tensor.

Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Computes a tensor such that
\\(output_i = \prod_j data_j\\) where the product is over `j` such
that `segment_ids[j] == i`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/SegmentProd.png" alt>
</div>

segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
first dimension.  Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension_0 which
has size `k`, the number of segments.
)doc");

REGISTER_OP("SegmentMin")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .Doc(R"doc(
Computes the minimum along segments of a tensor.

Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Computes a tensor such that
\\(output_i = \min_j(data_j)\\) where `min` is over `j` such
that `segment_ids[j] == i`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/SegmentMin.png" alt>
</div>

segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
first dimension.  Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension_0 which
has size `k`, the number of segments.
)doc");

REGISTER_OP("SegmentMax")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .Doc(R"doc(
Computes the maximum along segments of a tensor.

Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \max_j(data_j)\\) where `max` is over `j` such
that `segment_ids[j] == i`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/SegmentMax.png" alt>
</div>

segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
first dimension.  Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension_0 which
has size `k`, the number of segments.
)doc");

REGISTER_OP("UnsortedSegmentSum")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Input("num_segments: int32")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .Doc(R"doc(
Computes the sum along segments of a tensor.

Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Computes a tensor such that
\\(output_i = \sum_j data_j\\) where sum is over `j` such
that `segment_ids[j] == i`. Unlike `SegmentSum`, `segment_ids`
need not be sorted and need not cover all values in the full
  range of valid values.

If the sum is empty for a given segment ID `i`, `output[i] = 0`.

`num_segments` should equal the number of distinct segment IDs.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../images/UnsortedSegmentSum.png" alt>
</div>

segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
first dimension.

output: Has same shape as data, except for dimension_0 which
has size `num_segments`.

)doc");

REGISTER_OP("SparseSegmentSum")
    .Input("data: T")
    .Input("indices: int32")
    .Input("segment_ids: int32")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Computes the sum along sparse segments of a tensor.

Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
dimension, selecting a subset of dimension_0, specified by `indices`.

For example:

```prettyprint
c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

# Select two rows, one segment.
tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
  ==> [[0 0 0 0]]

# Select two rows, two segment.
tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
  ==> [[ 1  2  3  4]
       [-1 -2 -3 -4]]

# Select all rows, two segments.
tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
  ==> [[0 0 0 0]
       [5 6 7 8]]

# Which is equivalent to:
tf.segment_sum(c, tf.constant([0, 0, 1]))
```

indices: A 1-D tensor. Has same rank as `segment_ids`.

segment_ids: A 1-D tensor. Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension_0 which
has size `k`, the number of segments.
)doc");

REGISTER_OP("SparseSegmentMean")
    .Input("data: T")
    .Input("indices: int32")
    .Input("segment_ids: int32")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Computes the mean along sparse segments of a tensor.

Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
dimension, selecting a subset of dimension_0, specified by `indices`.

indices: A 1-D tensor. Has same rank as `segment_ids`.

segment_ids: A 1-D tensor. Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension_0 which
has size `k`, the number of segments.

)doc");

REGISTER_OP("SparseSegmentMeanGrad")
    .Input("grad: T")
    .Input("indices: int32")
    .Input("segment_ids: int32")
    .Input("output_dim0: int32")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Computes gradients for SparseSegmentMean.

Returns tensor "output" with same shape as grad, except for dimension_0 whose
value is output_dim0.

grad: gradient propagated to the SparseSegmentMean op.
indices: indices passed to the corresponding SparseSegmentMean op.
segment_ids: segment_ids passed to the corresponding SparseSegmentMean op.
output_dim0: dimension_0 of "data" passed to SparseSegmentMean op.
)doc");

REGISTER_OP("All")
    .Input("input: bool")
    .Input("reduction_indices: int32")
    .Output("output: bool")
    .Attr("keep_dims: bool = false")
    .Doc(R"doc(
Computes the "logical and" of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

input: The tensor to reduce.
reduction_indices: The dimensions to reduce.
keep_dims: If true, retain reduced dimensions with length 1.
output: The reduced tensor.
)doc");

REGISTER_OP("Any")
    .Input("input: bool")
    .Input("reduction_indices: int32")
    .Attr("keep_dims: bool = false")
    .Output("output: bool")
    .Doc(R"doc(
Computes the "logical or" of elements across dimensions of a tensor.

Reduces `input` along the dimensions given in `reduction_indices`. Unless
`keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
`reduction_indices`. If `keep_dims` is true, the reduced dimensions are
retained with length 1.

input: The tensor to reduce.
reduction_indices: The dimensions to reduce.
keep_dims: If true, retain reduced dimensions with length 1.
output: The reduced tensor.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("Range")
    .Input("start: int32")
    .Input("limit: int32")
    .Input("delta: int32")
    .Output("output: int32")
    .Doc(R"doc(
Creates a sequence of integers.

This operation creates a sequence of integers that begins at `start` and
extends by increments of `delta` up to but not including `limit`.

For example:

```
# 'start' is 3
# 'limit' is 18
# 'delta' is 3
tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]
```

start: 0-D (scalar). First entry in the sequence.
limit: 0-D (scalar). Upper limit of sequence, exclusive.
delta: 0-D (scalar). Optional. Default is 1. Number that increments `start`.
output: 1-D.
)doc");

REGISTER_OP("LinSpace")
    .Input("start: T")
    .Input("stop: T")
    .Input("num: int32")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Generates values in an interval.

A sequence of `num` evenly-spaced values are generated beginning at `start`.
If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
so that the last one is exactly `stop`.

For example:

```
tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
```

start: First entry in the range.
stop: Last entry in the range.
num: Number of values to generate.
output: 1-D. The generated values.
)doc");

REGISTER_OP("Complex")
    .Input("real: float")
    .Input("imag: float")
    .Output("out: complex64")
    .Doc(R"doc(
Converts two real numbers to a complex number.

Given a tensor `real` representing the real part of a complex number, and a
tensor `imag` representing the imaginary part of a complex number, this
operation returns complex numbers elementwise of the form \\(a + bj\\), where
*a* represents the `real` part and *b* represents the `imag` part.

The input tensors `real` and `imag` must have the same shape.

For example:

```
# tensor 'real' is [2.25, 3.25]
# tensor `imag` is [4.75, 5.75]
tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
```
)doc");

REGISTER_OP("Real")
    .Input("in: complex64")
    .Output("out: float")
    .Doc(R"doc(
Returns the real part of a complex number.

Given a tensor `in` of complex numbers, this operation returns a tensor of type
`float` that is the real part of each element in `in`. All elements in `in`
must be complex numbers of the form \\(a + bj\\), where *a* is the real part
returned by this operation and *b* is the imaginary part.

For example:

```
# tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.real(in) ==> [-2.25, 3.25]
```
)doc");

REGISTER_OP("Imag")
    .Input("in: complex64")
    .Output("out: float")
    .Doc(R"doc(
Returns the imaginary part of a complex number.

Given a tensor `in` of complex numbers, this operation returns a tensor of type
`float` that is the imaginary part of each element in `in`. All elements in `in`
must be complex numbers of the form \\(a + bj\\), where *a* is the real part
and *b* is the imaginary part returned by this operation.

For example:

```
# tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.imag(in) ==> [4.75, 5.75]
```
)doc");

REGISTER_OP("Conj")
    .Input("in: complex64")
    .Output("out: complex64")
    .Doc(R"doc(
Returns the complex conjugate of a complex number.

Given a tensor `in` of complex numbers, this operation returns a tensor of
complex numbers that are the complex conjugate of each element in `in`. The
complex numbers in `in` must be of the form \\(a + bj\\), where *a* is the real
part and *b* is the imaginary part.

The complex conjugate returned by this operation is of the form \\(a - bj\\).

For example:

```
# tensor 'in' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.conj(in) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
```
)doc");

}  // namespace tensorflow
