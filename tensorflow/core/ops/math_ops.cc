/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::Dimension;
using shape_inference::InferenceContext;
using shape_inference::Shape;

REGISTER_OP("AddN")
    .Input("inputs: N * T")
    .Output("sum: T")
    .Attr("N: int >= 1")
    .Attr("T: numbertype")
    .SetIsCommutative()
    .SetIsAggregate()
    .SetShapeFn([](InferenceContext* c) {
      const Shape* cur = c->input(c->num_inputs() - 1);
      for (int i = c->num_inputs() - 2; i >= 0; --i) {
        TF_RETURN_WITH_CONTEXT_IF_ERROR(c->Merge(c->input(i), cur, &cur),
                                        "From merging shape ", i,
                                        " with other shapes.");
      }
      c->set_output(0, cur);
      return Status::OK();
    })
    .Doc(R"doc(
Add all input tensors element wise.

inputs: Must all be the same size and shape.
)doc");

namespace {

// Shape inference function for binary operators that broadcast their inputs.
Status BroadcastBinaryOpShapeFn(InferenceContext* c) {
  const Shape* shape_x = c->input(0);
  const Shape* shape_y = c->input(1);
  if (!c->RankKnown(shape_x) || !c->RankKnown(shape_y)) {
    c->set_output(0, c->UnknownShape());
    return Status::OK();
  }
  const int32 rank_x = c->Rank(shape_x);
  const int32 rank_y = c->Rank(shape_y);
  const int32 rank_out = std::max(rank_x, rank_y);

  // To compute the broadcast dimensions, we zip together shape_x and shape_y
  // and
  // pad with 1 to make them the same length.
  std::vector<const Dimension*> dims;
  const Dimension* dim_one = rank_x == rank_y ? nullptr : c->MakeDim(1);
  for (int i = 0; i < rank_out; ++i) {
    const auto* dim_x = i < (rank_out - rank_x)
                            ? dim_one
                            : c->Dim(shape_x, i - (rank_out - rank_x));
    const auto* dim_y = i < (rank_out - rank_y)
                            ? dim_one
                            : c->Dim(shape_y, i - (rank_out - rank_y));
    if (!c->ValueKnown(dim_x) || !c->ValueKnown(dim_y)) {
      // One or both dimensions is unknown.
      //
      // - If either dimension is greater than 1, we assume that the program is
      // correct, and the other dimension will be broadcast to match it.
      // TODO(cwhipkey): For shape inference, if we eliminate the shape checks
      // in C++ op code, we must still assert that the unknown dim is either 1
      // or the same as the known dim.
      // - If either dimension is 1, the other dimension is the output.
      if (c->Value(dim_x) > 1) {
        dims.push_back(dim_x);
      } else if (c->Value(dim_y) > 1) {
        dims.push_back(dim_y);
      } else if (c->Value(dim_x) == 1) {
        dims.push_back(dim_y);
      } else if (c->Value(dim_y) == 1) {
        dims.push_back(dim_x);
      } else {
        dims.push_back(c->UnknownDim());
      }
    } else if (c->Value(dim_x) == 1 || c->Value(dim_y) == 1) {
      if (c->Value(dim_x) == 1 && dim_y != dim_one) {
        // We will broadcast dim_x to dim_y.
        dims.push_back(dim_y);
      } else {
        DCHECK_EQ(c->Value(dim_y), 1);
        // We will broadcast dim_y to dim_x.
        dims.push_back(dim_x);
      }
    } else {
      const Dimension* dim;
      TF_RETURN_IF_ERROR(c->Merge(dim_x, dim_y, &dim));
      dims.push_back(dim);
    }
  }

  c->set_output(0, c->MakeShape(dims));
  return Status::OK();
}

}  // namespace

// --------------------------------------------------------------------------

REGISTER_OP("BatchMatMul")
    .Input("x: T")
    .Input("y: T")
    .Output("output: T")
    .Attr("T: {half, float, double, int32, complex64, complex128}")
    .Attr("adj_x: bool = false")
    .Attr("adj_y: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* a_shape;
      const Shape* b_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 3, &a_shape));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 3, &b_shape));

      // Determine output rows and cols.
      bool adj_x;
      bool adj_y;
      TF_RETURN_IF_ERROR(c->GetAttr("adj_x", &adj_x));
      TF_RETURN_IF_ERROR(c->GetAttr("adj_y", &adj_y));
      const Dimension* output_rows = c->Dim(a_shape, adj_x ? -1 : -2);
      const Dimension* output_cols = c->Dim(b_shape, adj_y ? -2 : -1);

      // Batch dims match between inputs.
      const Shape* a_batch_dims;
      const Shape* b_batch_dims;
      const Shape* batch_dims;
      TF_RETURN_IF_ERROR(c->Subshape(a_shape, 0, -2, &a_batch_dims));
      TF_RETURN_IF_ERROR(c->Subshape(b_shape, 0, -2, &b_batch_dims));
      TF_RETURN_IF_ERROR(c->Merge(a_batch_dims, b_batch_dims, &batch_dims));

      // Assert inner dims match.
      const Dimension* unused;
      TF_RETURN_IF_ERROR(c->Merge(c->Dim(a_shape, adj_x ? -2 : -1),
                                  c->Dim(b_shape, adj_y ? -1 : -2), &unused));

      const Shape* out;
      TF_RETURN_IF_ERROR(c->Concatenate(
          batch_dims, c->Matrix(output_rows, output_cols), &out));
      c->set_output(0, out);
      return Status::OK();
    })
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

    output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

x: 3-D or higher with shape `[..., r_x, c_x]`.
y: 3-D or higher with shape `[..., r_y, c_y]`.
output: 3-D or higher with shape `[..., r_o, c_o]`
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
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Cast x of type SrcT to y of DstT.
)doc");

REGISTER_OP("_HostCast")
    .Input("x: SrcT")
    .Output("y: DstT")
    .Attr("SrcT: type")
    .Attr("DstT: type")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Cast x of type SrcT to y of DstT.

_HostCast requires its input and produces its output in host memory.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("Abs")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {half, float, double, int32, int64}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Computes the absolute value of a tensor.

Given a tensor `x`, this operation returns a tensor containing the absolute
value of each element in `x`. For example, if x is an input element and y is
an output element, this operation computes \\(y = |x|\\).
)doc");

REGISTER_OP("ComplexAbs")
    .Input("x: T")
    .Output("y: Tout")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .Attr("Tout: {float, double} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Computes the complex absolute value of a tensor.

Given a tensor `x` of complex numbers, this operation returns a tensor of type
`float` or `double` that is the absolute value of each element in `x`. All
elements in `x` must be complex numbers of the form \\(a + bj\\). The absolute
value is computed as \\( \sqrt{a^2 + b^2}\\).

For example:

```
# tensor 'x' is [[-2.25 + 4.75j], [-3.25 + 5.75j]]
tf.complex_abs(x) ==> [5.25594902, 6.60492229]
```
)doc");

// Declares cwise unary operations signature: 't -> 't
#define UNARY()                                                              \
  Input("x: T")                                                              \
      .Output("y: T")                                                        \
      .Attr("T: {half, float, double, int32, int64, complex64, complex128}") \
      .SetShapeFn(shape_inference::UnchangedShape)

#define UNARY_REAL()                    \
  Input("x: T")                         \
      .Output("y: T")                   \
      .Attr("T: {half, float, double}") \
      .SetShapeFn(shape_inference::UnchangedShape)

#define UNARY_COMPLEX()                                        \
  Input("x: T")                                                \
      .Output("y: T")                                          \
      .Attr("T: {half, float, double, complex64, complex128}") \
      .SetShapeFn(shape_inference::UnchangedShape)

#define UNARY_GRADIENT_COMPLEX()                               \
  Input("x: T")                                                \
      .Input("y: T")                                           \
      .Output("z: T")                                          \
      .Attr("T: {half, float, double, complex64, complex128}") \
      .SetShapeFn(shape_inference::UnchangedShape)

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
    .UNARY_COMPLEX()
    .Doc(R"doc(
Computes square root of x element-wise.
I.e., \\(y = \sqrt{x} = x^{1/2}\\).
)doc");

REGISTER_OP("Rsqrt")
    .UNARY_COMPLEX()
    .Doc(R"doc(
Computes reciprocal of square root of x element-wise.
I.e., \\(y = 1 / \sqrt{x}\\).
)doc");

REGISTER_OP("Exp")
    .UNARY_COMPLEX()
    .Doc(R"doc(
Computes exponential of x element-wise.  \\(y = e^x\\).
)doc");

REGISTER_OP("Log")
    .UNARY_COMPLEX()
    .Doc(R"doc(
Computes natural logarithm of x element-wise.
I.e., \\(y = \log_e x\\).
)doc");

REGISTER_OP("Tanh")
    .UNARY_COMPLEX()
    .Doc(R"doc(
Computes hyperbolic tangent of `x` element-wise.
)doc");

REGISTER_OP("TanhGrad").UNARY_GRADIENT_COMPLEX().Doc(R"doc(
Computes the gradient for the tanh of `x` wrt its input.

Specifically, `grad = dy * (1 - y*y)`, where `y = tanh(x)`, and `dy`
is the corresponding input gradient.
)doc");

REGISTER_OP("Lgamma")
    .UNARY_REAL()
    .Doc(R"doc(
Computes the log of the absolute value of `Gamma(x)` element-wise.
)doc");

REGISTER_OP("Digamma")
    .UNARY_REAL()
    .Doc(R"doc(
Computes Psi, the derivative of Lgamma (the log of the absolute value of
`Gamma(x)`), element-wise.
)doc");

REGISTER_OP("Erf")
    .UNARY_REAL()
    .Doc(R"doc(
Computes the Gauss error function of `x` element-wise.
)doc");

REGISTER_OP("Erfc")
    .UNARY_REAL()
    .Doc(R"doc(
Computes the complementary error function of `x` element-wise.
)doc");

REGISTER_OP("Sigmoid")
    .UNARY_COMPLEX()
    .Doc(R"doc(
Computes sigmoid of `x` element-wise.

Specifically, `y = 1 / (1 + exp(-x))`.
)doc");

REGISTER_OP("SigmoidGrad").UNARY_GRADIENT_COMPLEX().Doc(R"doc(
Computes the gradient of the sigmoid of `x` wrt its input.

Specifically, `grad = dy * y * (1 - y)`, where `y = sigmoid(x)`, and
`dy` is the corresponding input gradient.
)doc");

REGISTER_OP("Sin")
    .UNARY_COMPLEX()
    .Doc(R"doc(
Computes sin of x element-wise.
)doc");

REGISTER_OP("Cos")
    .UNARY_COMPLEX()
    .Doc(R"doc(
Computes cos of x element-wise.
)doc");

REGISTER_OP("Tan")
    .UNARY()
    .Doc(R"doc(
Computes tan of x element-wise.
)doc");

REGISTER_OP("Asin")
    .UNARY()
    .Doc(R"doc(
Computes asin of x element-wise.
)doc");

REGISTER_OP("Acos")
    .UNARY()
    .Doc(R"doc(
Computes acos of x element-wise.
)doc");

REGISTER_OP("Atan")
    .UNARY()
    .Doc(R"doc(
Computes atan of x element-wise.
)doc");

#undef UNARY
#undef UNARY_REAL
#undef UNARY_COMPLEX

REGISTER_OP("IsNan")
    .Input("x: T")
    .Output("y: bool")
    .Attr("T: {half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Returns which elements of x are NaN.
)doc");

REGISTER_OP("IsInf")
    .Input("x: T")
    .Output("y: bool")
    .Attr("T: {half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Returns which elements of x are Inf.
)doc");

REGISTER_OP("IsFinite")
    .Input("x: T")
    .Output("y: bool")
    .Attr("T: {half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Returns which elements of x are finite.
)doc");

REGISTER_OP("Sign")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {half, float, double, int32, int64, complex64, complex128}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Returns an element-wise indication of the sign of a number.

`y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.

For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.
)doc");

REGISTER_OP("Floor")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Returns element-wise largest integer not greater than x.
)doc");

REGISTER_OP("Ceil")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Returns element-wise smallest integer in not less than x.
)doc");

// Declares cwise binary operations signature: 't, 't -> 't.

#define BINARY_MORE()                              \
  Input("x: T").Input("y: T").Output("z: T").Attr( \
      "T: {half, float, double, uint8, int8, int16, int32, int64, complex64, complex128}")

#define BINARY_FEWER()                             \
  Input("x: T").Input("y: T").Output("z: T").Attr( \
      "T: {half, float, double, int32, int64, complex64, complex128}")

// TODO(mrry): Restore `SetIsCommutative()` for non-string types.
REGISTER_OP("Add")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr(
        "T: {half, float, double, uint8, int8, int16, int32, int64, complex64, "
        "complex128, string}")
    .SetShapeFn(BroadcastBinaryOpShapeFn)
    .Doc(R"doc(
Returns x + y element-wise.

*NOTE*: `Add` supports broadcasting. `AddN` does not. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

REGISTER_OP("Sub")
    .BINARY_FEWER()
    .SetShapeFn(BroadcastBinaryOpShapeFn)
    .Doc(R"doc(
Returns x - y element-wise.

*NOTE*: `Sub` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

REGISTER_OP("Mul")
    .BINARY_MORE()
    .SetIsCommutative()
    .SetShapeFn(BroadcastBinaryOpShapeFn)
    .Doc(R"doc(
Returns x * y element-wise.

*NOTE*: `Mul` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

REGISTER_OP("Div").BINARY_MORE().SetShapeFn(BroadcastBinaryOpShapeFn).Doc(R"doc(
Returns x / y element-wise.

*NOTE*: `Div` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

REGISTER_OP("SquaredDifference")
    .BINARY_FEWER()
    .SetIsCommutative()
    .SetShapeFn(BroadcastBinaryOpShapeFn)
    .Doc(R"doc(
Returns (x - y)(x - y) element-wise.

*NOTE*: `SquaredDifference` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

#undef BINARY_FEWER
#undef BINARY_MORE

REGISTER_OP("Maximum")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {half, float, double, int32, int64}")
    .SetIsCommutative()
    .SetShapeFn(BroadcastBinaryOpShapeFn)
    .Doc(R"doc(
Returns the max of x and y (i.e. x > y ? x : y) element-wise.

*NOTE*: `Maximum` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

REGISTER_OP("Minimum")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {half, float, double, int32, int64}")
    .SetIsCommutative()
    .SetShapeFn(BroadcastBinaryOpShapeFn)
    .Doc(R"doc(
Returns the min of x and y (i.e. x < y ? x : y) element-wise.

*NOTE*: `Minimum` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

REGISTER_OP("Mod")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {int32, int64, float, double}")
    .SetShapeFn(BroadcastBinaryOpShapeFn)
    .Doc(R"doc(
Returns element-wise remainder of division.

*NOTE*: `Mod` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

REGISTER_OP("Pow")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {half, float, double, int32, int64, complex64, complex128}")
    .SetShapeFn(BroadcastBinaryOpShapeFn)
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

REGISTER_OP("Igammac")
    .Input("a: T")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {float, double}")
    .SetShapeFn(BroadcastBinaryOpShapeFn)
    .Doc(R"doc(
Compute the upper regularized incomplete Gamma function `Q(a, x)`.

The upper regularized incomplete Gamma function is defined as:

```
Q(a, x) = Gamma(a, x) / Gamma(x) = 1 - P(a, x)
```
where
```
Gamma(a, x) = int_{x}^{\infty} t^{a-1} exp(-t) dt
```
is the upper incomplete Gama function.

Note, above `P(a, x)` (`Igamma`) is the lower regularized complete
Gamma function.
)doc");

REGISTER_OP("Igamma")
    .Input("a: T")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {float, double}")
    .SetShapeFn(BroadcastBinaryOpShapeFn)
    .Doc(R"doc(
Compute the lower regularized incomplete Gamma function `Q(a, x)`.

The lower regularized incomplete Gamma function is defined as:

```
P(a, x) = gamma(a, x) / Gamma(x) = 1 - Q(a, x)
```
where
```
gamma(a, x) = int_{0}^{x} t^{a-1} exp(-t) dt
```
is the lower incomplete Gamma function.

Note, above `Q(a, x)` (`Igammac`) is the upper regularized complete
Gamma function.
)doc");

REGISTER_OP("Zeta")
    .Input("x: T")
    .Input("q: T")
    .Output("z: T")
    .Attr("T: {float, double}")
    .SetShapeFn(BroadcastBinaryOpShapeFn)
    .Doc(R"doc(
Compute the Hurwitz zeta function \\(\zeta(x, q)\\).

The Hurwitz zeta function is defined as:

```
\zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}
```
)doc");

REGISTER_OP("Polygamma")
    .Input("a: T")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {float, double}")
    .SetShapeFn(BroadcastBinaryOpShapeFn)
    .Doc(R"doc(
Compute the polygamma function \\(\psi^{(n)}(x)\\).

The polygamma function is defined as:

```
\psi^{(n)}(x) = \frac{d^n}{dx^n} \psi(x)
```
where \\(\psi(x)\\) is the digamma function.
)doc");

// --------------------------------------------------------------------------

// Declares cwise binary comparison operations signature: 't, 't -> bool,
// where 't has a natural total order.
#define COMPARISON()             \
  Input("x: T")                  \
      .Input("y: T")             \
      .Output("z: bool")         \
      .Attr("T: realnumbertype") \
      .SetShapeFn(BroadcastBinaryOpShapeFn)

REGISTER_OP("Less")
    .COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x < y) element-wise.

*NOTE*: `Less` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

REGISTER_OP("LessEqual")
    .COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x <= y) element-wise.

*NOTE*: `LessEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

REGISTER_OP("Greater")
    .COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x > y) element-wise.

*NOTE*: `Greater` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

REGISTER_OP("GreaterEqual")
    .COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x >= y) element-wise.

*NOTE*: `GreaterEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

#undef COMPARISON

// --------------------------------------------------------------------------

#define EQUALITY_COMPARISON()                                           \
  Input("x: T")                                                         \
      .Input("y: T")                                                    \
      .Output("z: bool")                                                \
      .SetIsCommutative()                                               \
      .Attr(                                                            \
          "T: {half, float, double, uint8, int8, int16, int32, int64, " \
          "complex64, "                                                 \
          "quint8, qint8, qint32, string, bool, complex128}")           \
      .SetShapeFn(BroadcastBinaryOpShapeFn)

REGISTER_OP("Equal")
    .EQUALITY_COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x == y) element-wise.

*NOTE*: `Equal` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

REGISTER_OP("NotEqual")
    .EQUALITY_COMPARISON()
    .Doc(R"doc(
Returns the truth value of (x != y) element-wise.

*NOTE*: `NotEqual` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

#undef EQUALITY_COMPARISON

// --------------------------------------------------------------------------

REGISTER_OP("LogicalNot")
    .Input("x: bool")
    .Output("y: bool")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Returns the truth value of NOT x element-wise.
)doc");

#define BINARY_LOGICAL()  \
  Input("x: bool")        \
      .Input("y: bool")   \
      .Output("z: bool")  \
      .SetIsCommutative() \
      .SetShapeFn(BroadcastBinaryOpShapeFn)

REGISTER_OP("LogicalAnd")
    .BINARY_LOGICAL()
    .Doc(R"doc(
Returns the truth value of x AND y element-wise.

*NOTE*: `LogicalAnd` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

REGISTER_OP("LogicalOr")
    .BINARY_LOGICAL()
    .Doc(R"doc(
Returns the truth value of x OR y element-wise.

*NOTE*: `LogicalOr` supports broadcasting. More about broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
)doc");

#undef BINARY_LOGICAL

// --------------------------------------------------------------------------

REGISTER_OP("Select")
    .Input("condition: bool")
    .Input("t: T")
    .Input("e: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* cond = c->input(0);
      const Shape* data = c->input(1);
      TF_RETURN_IF_ERROR(c->Merge(data, c->input(2), &data));

      // Validate condition's shape if possible.
      if (c->RankKnown(data)) {
        const int32 data_rank = c->Rank(data);
        if (data_rank == 0 || data_rank == 1) {
          // Cond must match inputs since they are scalar or vector.
          TF_RETURN_IF_ERROR(c->Merge(data, cond, &data));
        } else {
          // cond must be the shape [data.dim[0]] or data.
          if (c->RankKnown(cond)) {
            if (c->Rank(cond) == 1) {
              // Must be a vector whose first dimension matches first dimension
              // of the data vectors.
              const Dimension* merged_dim;
              TF_RETURN_IF_ERROR(
                  c->Merge(c->Dim(data, 0), c->Dim(cond, 0), &merged_dim));
              if (merged_dim != c->Dim(data, 0)) {
                // Merging used the cond dim.  Update data to refer to it.
                std::vector<const Dimension*> dims{merged_dim};
                for (int i = 1; i < data_rank; ++i) {
                  dims.push_back(c->Dim(data, i));
                }
                data = c->MakeShape(dims);
              }
            } else {
              // Must be the same as the data vectors.
              TF_RETURN_IF_ERROR(c->Merge(data, cond, &data));
            }
          } else {
            // We want to express  that it's either [data.dim[0]] or data.
            // - rank(cond) must be 1 or rank(data).
            // - cond.dim[0] is same as data.dim[0]
            // But neither of these are expressible with unknown rank cond.
            // TODO(cwhipkey): improve this case.
          }
        }
      } else if (c->RankKnown(cond)) {
        if (c->Rank(cond) == 1) {
          // cond is vector, so data is either the same shape, or a shape with
          // higher rank or the same first dimension.
          // TODO(cwhipkey): make the call to WithRankAtLeast do something when
          // <data> is known.  Then we could assert the first dimensions are the
          // same.
          // TF_RETURN_IF_ERROR(c->WithRankAtLeast(data, 1, &data));
        } else {
          // If cond is a non-vector, it must be the same shape as data.
          TF_RETURN_IF_ERROR(c->Merge(data, cond, &data));
        }
      }

      c->set_output(0, data);
      return Status::OK();
    })
    .Doc(R"doc(
Selects elements from `t` or `e`, depending on `condition`.

The `t`, and `e` tensors must all have the same shape,
and the output will also have that shape.  The `condition` tensor
must be a scalar if `t` and `e` are scalars.  If `t` and `e` are vectors
or higher rank, then `condition` must be either a vector with size
matching the first dimension of `t`, or must have the same shape as `t`.

The `condition` tensor acts as a mask that chooses, based on the value at each
element, whether the corresponding element / row in the output should be
taken from `t` (if true) or `e` (if false).

If `condition` is a vector and `t` and `e` are higher rank matrices, then
it chooses which row (outer dimension) to copy from `t` and `e`.
If `condition` has the same shape as `t` and `e`, then it chooses which
element to copy from `t` and `e`.

For example:

```prettyprint
# 'condition' tensor is [[True,  False]
#                        [False, True]]
# 't' is [[1, 2],
#         [3, 4]]
# 'e' is [[5, 6],
#         [7, 8]]
select(condition, t, e) ==> [[1, 6],
                             [7, 4]]


# 'condition' tensor is [True, False]
# 't' is [[1, 2],
#         [3, 4]]
# 'e' is [[5, 6],
#         [7, 8]]
select(condition, t, e) ==> [[1, 2],
                             [7, 8]]

```

t:= A `Tensor` which may have the same shape as `condition`.
    If `condition` is rank 1, `t` may have higher rank,
    but its first dimension must match the size of `condition`.
e:= A `Tensor` with the same type and shape as `t`.
output:= A `Tensor` with the same type and shape as `t` and `e`.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("MatMul")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {half, float, double, int32, complex64, complex128}")
    .SetShapeFn(shape_inference::MatMulShape)
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
    .Input("a: Ta")
    .Input("b: Tb")
    .Output("product: float")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("a_is_sparse: bool = false")
    .Attr("b_is_sparse: bool = false")
    .Attr("Ta: {float, bfloat16} = DT_FLOAT")
    .Attr("Tb: {float, bfloat16} = DT_FLOAT")
    .SetShapeFn(shape_inference::MatMulShape)
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

namespace {

Status SegmentReductionShapeFn(InferenceContext* c) {
  const Shape* data_shape;
  const Shape* segment_ids_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &data_shape));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &segment_ids_shape));

  const Shape* subshape;
  TF_RETURN_IF_ERROR(c->Subshape(data_shape, 1, &subshape));

  const Shape* out;
  TF_RETURN_IF_ERROR(
      c->Concatenate(c->Vector(InferenceContext::kUnknownDim), subshape, &out));
  c->set_output(0, out);
  return Status::OK();
}

}  // namespace

REGISTER_OP("SegmentSum")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(SegmentReductionShapeFn)
    .Doc(R"doc(
Computes the sum along segments of a tensor.

Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \sum_j data_j\\) where sum is over `j` such
that `segment_ids[j] == i`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/SegmentSum.png" alt>
</div>

segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
first dimension.  Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension 0 which
  has size `k`, the number of segments.
)doc");

REGISTER_OP("SegmentMean")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(SegmentReductionShapeFn)
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
<img style="width:100%" src="../../images/SegmentMean.png" alt>
</div>

segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
first dimension.  Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension 0 which
  has size `k`, the number of segments.
)doc");

REGISTER_OP("SegmentProd")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(SegmentReductionShapeFn)
    .Doc(R"doc(
Computes the product along segments of a tensor.

Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Computes a tensor such that
\\(output_i = \prod_j data_j\\) where the product is over `j` such
that `segment_ids[j] == i`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/SegmentProd.png" alt>
</div>

segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
first dimension.  Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension 0 which
  has size `k`, the number of segments.
)doc");

REGISTER_OP("SegmentMin")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(SegmentReductionShapeFn)
    .Doc(R"doc(
Computes the minimum along segments of a tensor.

Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Computes a tensor such that
\\(output_i = \min_j(data_j)\\) where `min` is over `j` such
that `segment_ids[j] == i`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/SegmentMin.png" alt>
</div>

segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
first dimension.  Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension 0 which
  has size `k`, the number of segments.
)doc");

REGISTER_OP("SegmentMax")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(SegmentReductionShapeFn)
    .Doc(R"doc(
Computes the maximum along segments of a tensor.

Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \max_j(data_j)\\) where `max` is over `j` such
that `segment_ids[j] == i`.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/SegmentMax.png" alt>
</div>

segment_ids: A 1-D tensor whose rank is equal to the rank of `data`'s
first dimension.  Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension 0 which
  has size `k`, the number of segments.
)doc");

REGISTER_OP("UnsortedSegmentSum")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Input("num_segments: int32")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn([](InferenceContext* c) {
      const Shape* s_data = c->input(0);
      const Shape* s_segment_ids = c->input(1);
      const Shape* s_num_segments = c->input(2);
      TF_RETURN_IF_ERROR(c->WithRank(s_num_segments, 0, &s_num_segments));

      const Shape* out;

      // Leading dimensions of data must be compatible with dimensions of
      // <s_segment_ids>.
      if (c->RankKnown(s_segment_ids)) {
        TF_RETURN_IF_ERROR(
            c->MergePrefix(s_data, s_segment_ids, &s_data, &s_segment_ids));

        // Get the value of the num_segments input tensor.
        const Dimension* num_segments_dim;
        TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(2, &num_segments_dim));

        // Output is {segment_id_rank} + s_data[segment_id_rank:].
        const Shape* s_data_suffix;
        TF_RETURN_IF_ERROR(
            c->Subshape(s_data, c->Rank(s_segment_ids), &s_data_suffix));
        TF_RETURN_IF_ERROR(
            c->Concatenate(c->Vector(num_segments_dim), s_data_suffix, &out));
      } else {
        out = c->UnknownShape();
      }
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Computes the sum along segments of a tensor.

Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

Computes a tensor such that
`(output[i] = sum_{j...} data[j...]` where the sum is over tuples `j...` such
that `segment_ids[j...] == i`.  Unlike `SegmentSum`, `segment_ids`
need not be sorted and need not cover all values in the full
range of valid values.

If the sum is empty for a given segment ID `i`, `output[i] = 0`.

`num_segments` should equal the number of distinct segment IDs.

<div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
<img style="width:100%" src="../../images/UnsortedSegmentSum.png" alt>
</div>

segment_ids: A tensor whose shape is a prefix of `data.shape`.

output: Has same shape as data, except for the first `segment_ids.rank`
  dimensions, which are replaced with a single dimension which has size
  `num_segments`.

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
dimension, selecting a subset of dimension 0, specified by `indices`.

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

output: Has same shape as data, except for dimension 0 which
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
dimension, selecting a subset of dimension 0, specified by `indices`.

indices: A 1-D tensor. Has same rank as `segment_ids`.

segment_ids: A 1-D tensor. Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension 0 which
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

Returns tensor "output" with same shape as grad, except for dimension 0 whose
value is output_dim0.

grad: gradient propagated to the SparseSegmentMean op.
indices: indices passed to the corresponding SparseSegmentMean op.
segment_ids: segment_ids passed to the corresponding SparseSegmentMean op.
output_dim0: dimension 0 of "data" passed to SparseSegmentMean op.
)doc");

REGISTER_OP("SparseSegmentSqrtN")
    .Input("data: T")
    .Input("indices: int32")
    .Input("segment_ids: int32")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Computes the sum along sparse segments of a tensor divided by the sqrt of N.

N is the size of the segment being reduced.

Read [the section on
Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
of segments.

indices: A 1-D tensor. Has same rank as `segment_ids`.

segment_ids: A 1-D tensor. Values should be sorted and can be repeated.

output: Has same shape as data, except for dimension 0 which
  has size `k`, the number of segments.

)doc");

REGISTER_OP("SparseSegmentSqrtNGrad")
    .Input("grad: T")
    .Input("indices: int32")
    .Input("segment_ids: int32")
    .Input("output_dim0: int32")
    .Output("output: T")
    .Attr("T: {float, double}")
    .Doc(R"doc(
Computes gradients for SparseSegmentSqrtN.

Returns tensor "output" with same shape as grad, except for dimension 0 whose
value is output_dim0.

grad: gradient propagated to the SparseSegmentSqrtN op.
indices: indices passed to the corresponding SparseSegmentSqrtN op.
segment_ids: segment_ids passed to the corresponding SparseSegmentSqrtN op.
output_dim0: dimension 0 of "data" passed to SparseSegmentSqrtN op.
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
    .SetShapeFn([](InferenceContext* c) {
      const Shape* unused;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(0), 0, &unused),
                                      " for 'start'");
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(1), 0, &unused),
                                      " for 'limit'");
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(2), 0, &unused),
                                      " for 'delta'");
      const Tensor* start_t = c->input_tensor(0);
      const Tensor* limit_t = c->input_tensor(1);
      const Tensor* delta_t = c->input_tensor(2);
      if (start_t == nullptr || limit_t == nullptr || delta_t == nullptr) {
        c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
        return Status::OK();
      }
      const int32 start = start_t->scalar<int32>()();
      const int32 limit = limit_t->scalar<int32>()();
      const int32 delta = delta_t->scalar<int32>()();
      if (start > limit) {
        return errors::InvalidArgument("Requires start <= limit: ", start, "/",
                                       limit);
      }
      if (delta <= 0) {
        return errors::InvalidArgument("Requires delta > 0: ", delta);
      }
      const int32 size = (limit - start + delta - 1) / delta;
      c->set_output(0, c->Vector(size));
      return Status::OK();
    })
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
    .SetShapeFn([](InferenceContext* c) {
      const Shape* unused;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(0), 0, &unused),
                                      " for 'start'");
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(1), 0, &unused),
                                      " for 'stop'");
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(2), 0, &unused),
                                      " for 'num'");
      const Tensor* num_t = c->input_tensor(2);
      if (num_t == nullptr) {
        c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
        return Status::OK();
      }
      const int64 num = num_t->scalar<int32>()();
      if (num <= 0) return errors::InvalidArgument("Requires num > 0: ", num);
      c->set_output(0, c->Vector(num));
      return Status::OK();
    })
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
    .Input("real: T")
    .Input("imag: T")
    .Output("out: Tout")
    .Attr("T: {float, double} = DT_FLOAT")
    .Attr("Tout: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn(BroadcastBinaryOpShapeFn)
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
    .Input("input: T")
    .Output("output: Tout")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .Attr("Tout: {float, double} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Returns the real part of a complex number.

Given a tensor `input` of complex numbers, this operation returns a tensor of
type `float` that is the real part of each element in `input`. All elements in
`input` must be complex numbers of the form \\(a + bj\\), where *a* is the real
 part returned by this operation and *b* is the imaginary part.

For example:

```
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.real(input) ==> [-2.25, 3.25]
```
)doc");

REGISTER_OP("Imag")
    .Input("input: T")
    .Output("output: Tout")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .Attr("Tout: {float, double} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Returns the imaginary part of a complex number.

Given a tensor `input` of complex numbers, this operation returns a tensor of
type `float` that is the imaginary part of each element in `input`. All
elements in `input` must be complex numbers of the form \\(a + bj\\), where *a*
is the real part and *b* is the imaginary part returned by this operation.

For example:

```
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.imag(input) ==> [4.75, 5.75]
```
)doc");

REGISTER_OP("Conj")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Returns the complex conjugate of a complex number.

Given a tensor `input` of complex numbers, this operation returns a tensor of
complex numbers that are the complex conjugate of each element in `input`. The
complex numbers in `input` must be of the form \\(a + bj\\), where *a* is the
real part and *b* is the imaginary part.

The complex conjugate returned by this operation is of the form \\(a - bj\\).

For example:

```
# tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
```
)doc");

REGISTER_OP("FFT")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRank(c, 1);
    })
    .Doc(R"doc(
Compute the 1-dimensional discrete Fourier Transform.

input: A complex64 vector.
output: The 1D Fourier Transform of `input`.
)doc");

REGISTER_OP("IFFT")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRank(c, 1);
    })
    .Doc(R"doc(
    .Doc(R"doc(
Compute the inverse 1-dimensional discrete Fourier Transform.

input: A complex64 vector.
output: The inverse 1D Fourier Transform of `input`.
)doc");

REGISTER_OP("FFT2D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRank(c, 2);
    })
    .Doc(R"doc(
Compute the 2-dimensional discrete Fourier Transform.

input: A complex64 matrix.
output: The 2D Fourier Transform of `input`.
)doc");

REGISTER_OP("IFFT2D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRank(c, 2);
    })
    .Doc(R"doc(
Compute the inverse 2-dimensional discrete Fourier Transform.

input: A complex64 matrix.
output: The inverse 2D Fourier Transform of `input`.
)doc");

REGISTER_OP("FFT3D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRank(c, 3);
    })
    .Doc(R"doc(
Compute the 3-dimensional discrete Fourier Transform.

input: A complex64 3-D tensor.
output: The 3D Fourier Transform of `input`.
)doc");

REGISTER_OP("IFFT3D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRank(c, 3);
    })
    .Doc(R"doc(
Compute the inverse 3-dimensional discrete Fourier Transform.

input: A complex64 3-D tensor.
output: The inverse 3D Fourier Transform of `input`.
)doc");

REGISTER_OP("BatchFFT")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    })
    .Doc(R"doc(
Compute the 1-dimensional discrete Fourier Transform over the inner-most
dimension of `input`.

input: A complex64 tensor.
output: A complex64 tensor of the same shape as `input`. The inner-most
  dimension of `input` is replaced with its 1D Fourier Transform.
)doc");

REGISTER_OP("BatchIFFT")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 1);
    })
    .Doc(R"doc(
Compute the inverse 1-dimensional discrete Fourier Transform over the inner-most
dimension of `input`.

input: A complex64 tensor.
output: A complex64 tensor of the same shape as `input`. The inner-most
  dimension of `input` is replaced with its inverse 1D Fourier Transform.
)doc");

REGISTER_OP("BatchFFT2D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 2);
    })
    .Doc(R"doc(
Compute the 2-dimensional discrete Fourier Transform over the inner-most
2 dimensions of `input`.

input: A complex64 tensor.
output: A complex64 tensor of the same shape as `input`. The inner-most 2
  dimensions of `input` are replaced with their 2D Fourier Transform.
)doc");

REGISTER_OP("BatchIFFT2D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 2);
    })
    .Doc(R"doc(
Compute the inverse 2-dimensional discrete Fourier Transform over the inner-most
2 dimensions of `input`.

input: A complex64 tensor.
output: A complex64 tensor of the same shape as `input`. The inner-most 2
  dimensions of `input` are replaced with their inverse 2D Fourier Transform.
)doc");

REGISTER_OP("BatchFFT3D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    })
    .Doc(R"doc(
Compute the 3-dimensional discrete Fourier Transform over the inner-most 3
dimensions of `input`.

input: A complex64 tensor.
output: A complex64 tensor of the same shape as `input`. The inner-most 3
  dimensions of `input` are replaced with their 3D Fourier Transform.
)doc");

REGISTER_OP("BatchIFFT3D")
    .Input("input: complex64")
    .Output("output: complex64")
    .SetShapeFn([](InferenceContext* c) {
      return shape_inference::UnchangedShapeWithRankAtLeast(c, 3);
    })
    .Doc(R"doc(
Compute the inverse 3-dimensional discrete Fourier Transform over the inner-most
3 dimensions of `input`.

input: A complex64 tensor.
output: A complex64 tensor of the same shape as `input`. The inner-most 3
  dimensions of `input` are replaced with their inverse 3D Fourier Transform.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("Cross")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("T: realnumbertype")
    // TODO(cwhipkey): implement these shape inference constraints here:
    // * Both inputs have the same shape.
    // * Input rank >= 1.
    // * input_shape[-1] == 3.
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Compute the pairwise cross product.

`a` and `b` must be the same shape; they can either be simple 3-element vectors,
or any shape where the innermost dimension is 3. In the latter case, each pair
of corresponding 3-element vectors is cross-multiplied independently.

a: A tensor containing 3-element vectors.
b: Another tensor, of same type and shape as `a`.
product: Pairwise cross product of the vectors in `a` and `b`.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("Cumsum")
    .Input("x: T")
    .Input("axis: int32")
    .Attr("exclusive: bool = false")
    .Attr("reverse: bool = false")
    .Output("out: T")
    .Attr("T: numbertype")
    .Doc(R"doc(
Compute the cumulative sum of the tensor `x` along `axis`.

By default, this op performs an inclusive cumsum, which means that the first
element of the input is identical to the first element of the output:
```prettyprint
tf.cumsum([a, b, c]) ==> [a, a + b, a + b + c]
```

By setting the `exclusive` kwarg to `True`, an exclusive cumsum is
performed instead:
```prettyprint
tf.cumsum([a, b, c], exclusive=True) ==> [0, a, a + b]
```

By setting the `reverse` kwarg to `True`, the cumsum is performed in the
opposite direction:
```prettyprint
tf.cumsum([a, b, c], reverse=True) ==> [a + b + c, b + c, c]
```
This is more efficient than using separate `tf.reverse` ops.

The `reverse` and `exclusive` kwargs can also be combined:
```prettyprint
tf.cumsum([a, b, c], exclusive=True, reverse=True) ==> [b + c, c, 0]
```
)doc");

REGISTER_OP("Cumprod")
    .Input("x: T")
    .Input("axis: int32")
    .Attr("exclusive: bool = false")
    .Attr("reverse: bool = false")
    .Output("out: T")
    .Attr("T: numbertype")
    .Doc(R"doc(
Compute the cumulative product of the tensor `x` along `axis`.

By default, this op performs an inclusive cumprod, which means that the first
element of the input is identical to the first element of the output:
```prettyprint
tf.cumprod([a, b, c]) ==> [a, a * b, a * b * c]
```

By setting the `exclusive` kwarg to `True`, an exclusive cumprod is
performed instead:
```prettyprint
tf.cumprod([a, b, c], exclusive=True) ==> [0, a, a * b]
```

By setting the `reverse` kwarg to `True`, the cumprod is performed in the
opposite direction:
```prettyprint
tf.cumprod([a, b, c], reverse=True) ==> [a * b * c, b * c, c]
```
This is more efficient than using separate `tf.reverse` ops.

The `reverse` and `exclusive` kwargs can also be combined:
```prettyprint
tf.cumprod([a, b, c], exclusive=True, reverse=True) ==> [b * c, c, 0]
```
)doc");

}  // namespace tensorflow
