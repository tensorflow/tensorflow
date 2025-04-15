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
#include <limits>
#include <vector>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

// TODO(intel-tf): Move all MKL ops in this file to a separate file,
// mkl_math_ops.cc.

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("AddN")
    .Input("inputs: N * T")
    .Output("sum: T")
    .Attr("N: int >= 1")
    .Attr("T: {numbertype, variant}")
    .SetIsCommutative()
    .SetIsAggregate()
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle cur = c->input(c->num_inputs() - 1);
      for (int i = c->num_inputs() - 2; i >= 0; --i) {
        TF_RETURN_WITH_CONTEXT_IF_ERROR(c->Merge(c->input(i), cur, &cur),
                                        "From merging shape ", i,
                                        " with other shapes.");
      }
      c->set_output(0, cur);

      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("T", &dtype));

      if (dtype != DT_VARIANT) {
        // Exit early if not DT_VARIANT.
        return absl::OkStatus();
      } else {
        // DT_VARIANT shape handle shape inference.  All sizes and dtypes must
        // be the same; all shapes must be compatible via Merge.
        std::vector<shape_inference::ShapeAndType> cur_shapes_and_types;
        auto* shapes_and_types =
            c->input_handle_shapes_and_types(c->num_inputs() - 1);
        if (shapes_and_types) {
          cur_shapes_and_types = *shapes_and_types;
        }

        for (int i = c->num_inputs() - 2; i >= 0; --i) {
          auto shapes_and_types_i = c->input_handle_shapes_and_types(i);
          if (!shapes_and_types && shapes_and_types_i) {
            // TODO(ebrevdo): Find cases where this happens and fix their shape
            // inference.  If we are calling AddN on variant types, they should
            // all have consistent shape_and_type info.
            shapes_and_types = shapes_and_types_i;
          } else if (shapes_and_types && shapes_and_types_i) {
            if (shapes_and_types_i->size() != shapes_and_types->size()) {
              return errors::InvalidArgument(
                  "shapes_and_types[", i,
                  "].size() == ", shapes_and_types_i->size(),
                  " != shapes_and_types[0].size() == ",
                  shapes_and_types->size());
            }
            for (int j = 0; j < shapes_and_types->size(); ++j) {
              if (shapes_and_types->at(j).dtype !=
                  shapes_and_types_i->at(j).dtype) {
                return errors::InvalidArgument(
                    "shapes_and_types[", i, "][", j, "].dtype() == ",
                    DataTypeString(shapes_and_types_i->at(j).dtype),
                    " != shapes_and_types[0][", j, "].dtype == ",
                    DataTypeString(shapes_and_types->at(j).dtype));
              }
              TF_RETURN_WITH_CONTEXT_IF_ERROR(
                  c->Merge(shapes_and_types_i->at(j).shape,
                           cur_shapes_and_types.at(j).shape,
                           &cur_shapes_and_types.at(j).shape),
                  "From merging shapes_and_types[", i, "][", j, "].shape with ",
                  "shapes_and_types[0][", j, "].shape");
            }
          }
        }
        if (shapes_and_types) {
          c->set_output_handle_shapes_and_types(0, cur_shapes_and_types);
        }
        return absl::OkStatus();
      }
    });

// --------------------------------------------------------------------------

// Note that the following operator is just a placeholder and has no
// associated kernel. The code in accumulate_n_optimizer.cc replaces
// this placeholder with a graph of operators that do have kernels.
// The Python code that generates instances of this op is currently in
// contrib/framework/python/ops/accumulate_n_v2.py
REGISTER_OP("AccumulateNV2")
    .Input("inputs: N * T")
    .Output("sum: T")
    .Attr("N: int >= 1")
    .Attr("T: numbertype")
    .Attr("shape: shape")
    .SetIsCommutative()
    .SetIsAggregate()
    .SetShapeFn(shape_inference::ExplicitShape);

// --------------------------------------------------------------------------

REGISTER_OP("BatchMatMul")
    .Input("x: T")
    .Input("y: T")
    .Output("output: T")
    .Attr(
        "T: {bfloat16, half, float, double, int32, int64, complex64, "
        "complex128}")
    .Attr("adj_x: bool = false")
    .Attr("adj_y: bool = false")
    .Attr("grad_x: bool = false")
    .Attr("grad_y: bool = false")
    .SetShapeFn(shape_inference::BatchMatMulShape);

REGISTER_OP("BatchMatMulV2")
    .Input("x: T")
    .Input("y: T")
    .Output("output: T")
    .Attr(
        "T: {bfloat16, half, float, double, int16, int32, int64, uint8, "
        "uint16, uint32, uint64, complex64, complex128}")
    .Attr("adj_x: bool = false")
    .Attr("adj_y: bool = false")
    .Attr("grad_x: bool = false")
    .Attr("grad_y: bool = false")
    .SetShapeFn(shape_inference::BatchMatMulV2Shape);

REGISTER_OP("BatchMatMulV3")
    .Input("x: Ta")
    .Input("y: Tb")
    .Output("output: Tout")
    .Attr(
        "Ta: {bfloat16, half, float, double, uint8, int8, int16, int32, int64, "
        "complex64, complex128}")
    .Attr(
        "Tb: {bfloat16, half, float, double, uint8, int8, int16, int32, int64, "
        "complex64, complex128}")
    .Attr(
        "Tout: {bfloat16, half, float, double, int16, int32, int64, complex64, "
        "complex128}")
    .Attr("adj_x: bool = false")
    .Attr("adj_y: bool = false")
    .Attr("grad_x: bool = false")
    .Attr("grad_y: bool = false")
    .SetShapeFn(shape_inference::BatchMatMulV2Shape);

#ifdef INTEL_MKL
REGISTER_OP("_MklBatchMatMul")
    .Input("x: T")
    .Input("y: T")
    .Output("output: T")
    .Attr("T: {bfloat16, float, half}")
    .Attr("adj_x: bool = false")
    .Attr("adj_y: bool = false")
    .Attr("grad_x: bool = false")
    .Attr("grad_y: bool = false")
    .SetShapeFn(shape_inference::BatchMatMulShape);

REGISTER_OP("_MklBatchMatMulV2")
    .Input("x: T")
    .Input("y: T")
    .Output("output: T")
    .Attr("T: {bfloat16, float, half}")
    .Attr("adj_x: bool = false")
    .Attr("adj_y: bool = false")
    .Attr("grad_x: bool = false")
    .Attr("grad_y: bool = false")
    .SetShapeFn(shape_inference::BatchMatMulV2Shape);
#endif  // INTEL_MKL

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
    .Attr("Truncate: bool = false")
    .SetTypeConstructor(full_type::NoOp())
    .SetForwardTypeFn(full_type::KeepExisting())
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("_HostCast")
    .Input("x: SrcT")
    .Output("y: DstT")
    .Attr("SrcT: type")
    .Attr("DstT: type")
    .Attr("Truncate: bool = false")
    .SetTypeConstructor(full_type::NoOp())
    .SetForwardTypeFn(full_type::KeepExisting())
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Cast x of type SrcT to y of DstT.

_HostCast requires its input and produces its output in host memory.
)doc");

// --------------------------------------------------------------------------

REGISTER_OP("Abs")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double, int8, int16, int32, int64}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("ComplexAbs")
    .Input("x: T")
    .Output("y: Tout")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .Attr("Tout: {float, double} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape);

// Declares cwise unary operations signature: 't -> 't
#define UNARY()                                                            \
  Input("x: T")                                                            \
      .Output("y: T")                                                      \
      .Attr(                                                               \
          "T: {bfloat16, half, float, double, int8, int16, int32, int64, " \
          "complex64, complex128}")                                        \
      .SetShapeFn(shape_inference::UnchangedShape)

#define UNARY_UNSIGNED()                                                   \
  Input("x: T")                                                            \
      .Output("y: T")                                                      \
      .Attr(                                                               \
          "T: {bfloat16, half, float, double, int8, int16, int32, int64, " \
          "uint8, uint16, uint32, uint64, complex64, complex128}")         \
      .SetShapeFn(shape_inference::UnchangedShape)

#define UNARY_REAL()                              \
  Input("x: T")                                   \
      .Output("y: T")                             \
      .Attr("T: {bfloat16, half, float, double}") \
      .SetShapeFn(shape_inference::UnchangedShape)

#define UNARY_COMPLEX()                                                  \
  Input("x: T")                                                          \
      .Output("y: T")                                                    \
      .Attr("T: {bfloat16, half, float, double, complex64, complex128}") \
      .SetShapeFn(shape_inference::UnchangedShape)

#define UNARY_GRADIENT_COMPLEX()                                         \
  Input("y: T")                                                          \
      .Input("dy: T")                                                    \
      .Output("z: T")                                                    \
      .Attr("T: {bfloat16, half, float, double, complex64, complex128}") \
      .SetShapeFn(shape_inference::UnchangedShape)

REGISTER_OP("Neg").UNARY();

REGISTER_OP("Inv").UNARY();

REGISTER_OP("InvGrad").UNARY_GRADIENT_COMPLEX();

REGISTER_OP("Reciprocal").UNARY();

REGISTER_OP("ReciprocalGrad").UNARY_GRADIENT_COMPLEX();

REGISTER_OP("Square").UNARY_UNSIGNED();

REGISTER_OP("Sqrt").UNARY_COMPLEX();

REGISTER_OP("SqrtGrad").UNARY_GRADIENT_COMPLEX();

REGISTER_OP("Rsqrt").UNARY_COMPLEX();

REGISTER_OP("Round").UNARY();

REGISTER_OP("RsqrtGrad").UNARY_GRADIENT_COMPLEX();

REGISTER_OP("Exp").UNARY_COMPLEX();

REGISTER_OP("Expm1").UNARY_COMPLEX();

REGISTER_OP("Log").UNARY_COMPLEX();

REGISTER_OP("Log1p").UNARY_COMPLEX();

REGISTER_OP("Sinh").UNARY_COMPLEX();

REGISTER_OP("Cosh").UNARY_COMPLEX();

REGISTER_OP("Tanh").UNARY_COMPLEX();

REGISTER_OP("Asinh").UNARY_COMPLEX();

REGISTER_OP("Acosh").UNARY_COMPLEX();

REGISTER_OP("Atanh").UNARY_COMPLEX();

REGISTER_OP("TanhGrad").UNARY_GRADIENT_COMPLEX();

REGISTER_OP("Lgamma").UNARY_REAL();

REGISTER_OP("Digamma").UNARY_REAL();

REGISTER_OP("Erf").UNARY_REAL();
REGISTER_OP("Erfinv").UNARY_REAL();
REGISTER_OP("Ndtri").UNARY_REAL();
REGISTER_OP("Erfc").UNARY_REAL();

REGISTER_OP("Sigmoid").UNARY_COMPLEX();

REGISTER_OP("SigmoidGrad").UNARY_GRADIENT_COMPLEX();

REGISTER_OP("Sin").UNARY_COMPLEX();

REGISTER_OP("Cos").UNARY_COMPLEX();

REGISTER_OP("Tan").UNARY_COMPLEX();

REGISTER_OP("Asin").UNARY_COMPLEX();

REGISTER_OP("Acos").UNARY_COMPLEX();

REGISTER_OP("Atan").UNARY_COMPLEX();

REGISTER_OP("_UnaryOpsComposition")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float, half, double}")
    .Attr("op_names: list(string)")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to create these operators.
)doc");

#undef UNARY
#undef UNARY_REAL
#undef UNARY_COMPLEX

REGISTER_OP("IsNan")
    .Input("x: T")
    .Output("y: bool")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("IsInf")
    .Input("x: T")
    .Output("y: bool")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("IsFinite")
    .Input("x: T")
    .Output("y: bool")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Sign")
    .Input("x: T")
    .Output("y: T")
    .Attr(
        "T: {bfloat16, half, float, double, int8, int16, int32, int64, "
        "complex64, complex128}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Floor")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Ceil")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Rint")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::UnchangedShape);

// Declares cwise binary operations signature: 't, 't -> 't.

#define BINARY_MORE()                                                          \
  Input("x: T").Input("y: T").Output("z: T").Attr(                             \
      "T: {bfloat16, half, float, double, uint8, int8, uint16, int16, int32, " \
      "uint32, uint64, int64, complex64, complex128}")

#define BINARY_FEWER()                                               \
  Input("x: T").Input("y: T").Output("z: T").Attr(                   \
      "T: {bfloat16, half, float, double, int32, int64, complex64, " \
      "complex128}")

REGISTER_OP("Add")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr(
        "T: {bfloat16, half, float, double, uint8, int8, int16, int32, int64, "
        "complex64, complex128, string}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("AddV2")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr(
        "T: {bfloat16, half, float, double, uint8, uint16, uint32, uint64, "
        "int8, int16, int32, int64, complex64, complex128}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)
    .SetIsAggregate()
    .SetIsCommutative();

REGISTER_OP("Sub")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr(
        "T: {bfloat16, half, float, double, uint8, int8, uint16, int16, int32, "
        "int64, complex64, complex128, uint32, uint64}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Mul").BINARY_MORE().SetIsCommutative().SetShapeFn(
    shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("MulNoNan")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double, complex64, complex128}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Div").BINARY_MORE().SetShapeFn(
    shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("DivNoNan")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {half, float, bfloat16, double, complex64, complex128}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("FloorDiv")
    .BINARY_MORE()
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("TruncateDiv")
    .BINARY_MORE()
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("RealDiv").BINARY_MORE().SetShapeFn(
    shape_inference::BroadcastBinaryOpShapeFn);

// Note SquaredDifference implements conj(x - y)*(x - y).
REGISTER_OP("SquaredDifference")
    .BINARY_FEWER()
    .SetIsCommutative()
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Xlogy")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {half, bfloat16, float, double, complex64, complex128}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Xlog1py")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {half, bfloat16, float, double, complex64, complex128}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Xdivy")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {half, bfloat16, float, double, complex64, complex128}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

#undef BINARY_FEWER
#undef BINARY_MORE

REGISTER_OP("Maximum")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr(
        "T: {bfloat16, half, float, double, int8, uint8, int16, uint16, "
        "int32, uint32, int64, uint64}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Minimum")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr(
        "T: {bfloat16, half, float, double, int8, uint8, int16, uint16, "
        "int32, uint32, int64, uint64}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Mod")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {int32, int64, float16, half, bfloat16, float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("FloorMod")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr(
        "T: {int8, int16, int32, int64, uint8, uint16, uint32, uint64, "
        "bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("TruncateMod")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr("T: {int32, int64, bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Pow")
    .Input("x: T")
    .Input("y: T")
    .Output("z: T")
    .Attr(
        "T: {bfloat16, float, half, double, int8, int16, int32, int64, "
        "complex64, complex128}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Igammac")
    .Input("a: T")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Igamma")
    .Input("a: T")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("IgammaGradA")
    .Input("a: T")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Zeta")
    .Input("x: T")
    .Input("q: T")
    .Output("z: T")
    .Attr("T: {float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Polygamma")
    .Input("a: T")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Atan2")
    .Input("y: T")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {bfloat16, half, float, double}")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Betainc")
    .Input("a: T")
    .Input("b: T")
    .Input("x: T")
    .Output("z: T")
    .Attr("T: {float, double}")
    .SetShapeFn([](InferenceContext* c) {
      const int num_inputs = 3;
      ShapeHandle output = c->UnknownShape();
      int num_scalars = 0;
      ShapeHandle some_non_scalar;
      for (int i = 0; i < num_inputs; ++i) {
        ShapeHandle in = c->input(i);
        if (!c->RankKnown(in)) {
          some_non_scalar = in;
          // An input with unknown rank could be either a scalar (to be
          // broadcast) or some other shape.
        } else if (c->Rank(in) == 0) {
          // Input is a scalar, it will be broadcast to the output shape.
          ++num_scalars;
        } else {
          TF_RETURN_IF_ERROR(c->Merge(output, in, &output));
          some_non_scalar = output;
        }
      }

      if (num_scalars == num_inputs - 1) {
        // If all but one input is known to be a scalar, then output is the
        // remaining input.
        output = some_non_scalar;
      } else if (num_scalars == num_inputs) {
        // If all are scalars, output is scalar; pick the first one arbitrarily.
        output = c->input(0);
      }

      c->set_output(0, output);
      return absl::OkStatus();
    });

// --------------------------------------------------------------------------

// Declares cwise binary comparison operations signature: 't, 't -> bool,
// where 't has a natural total order.
#define COMPARISON()             \
  Input("x: T")                  \
      .Input("y: T")             \
      .Output("z: bool")         \
      .Attr("T: realnumbertype") \
      .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)

REGISTER_OP("Less").COMPARISON();

REGISTER_OP("LessEqual").COMPARISON();

REGISTER_OP("Greater").COMPARISON();

REGISTER_OP("GreaterEqual").COMPARISON();

#undef COMPARISON

// --------------------------------------------------------------------------

#define EQUALITY_COMPARISON()                                      \
  Input("x: T")                                                    \
      .Input("y: T")                                               \
      .Output("z: bool")                                           \
      .SetIsCommutative()                                          \
      .Attr("T: type")                                             \
      .Attr("incompatible_shape_error: bool = true")               \
      .SetShapeFn([](InferenceContext* c) {                        \
        ShapeHandle x = c->input(0);                               \
        ShapeHandle y = c->input(1);                               \
        ShapeHandle output;                                        \
        bool incompatible_shape_error;                             \
        TF_RETURN_IF_ERROR(c->GetAttr("incompatible_shape_error",  \
                                      &incompatible_shape_error)); \
        TF_RETURN_IF_ERROR(BroadcastBinaryOpOutputShapeFnHelper(   \
            c, x, y, incompatible_shape_error, &output));          \
        c->set_output(0, output);                                  \
        return OkStatus();                                         \
      })

REGISTER_OP("Equal").EQUALITY_COMPARISON();

REGISTER_OP("NotEqual").EQUALITY_COMPARISON();

#undef EQUALITY_COMPARISON

REGISTER_OP("ApproximateEqual")
    .Input("x: T")
    .Input("y: T")
    .Output("z: bool")
    .SetIsCommutative()
    .Attr("T: numbertype")
    .Attr("tolerance: float = 0.00001")
    .SetShapeFn([](InferenceContext* c) {
      // The inputs 'x' and 'y' must have the same shape.
      ShapeHandle data_x = c->input(0);
      ShapeHandle data_y = c->input(1);
      TF_RETURN_IF_ERROR(c->Merge(data_x, data_y, &data_x));
      return shape_inference::UnchangedShape(c);
    });

// --------------------------------------------------------------------------

REGISTER_OP("LogicalNot")
    .Input("x: bool")
    .Output("y: bool")
    .SetShapeFn(shape_inference::UnchangedShape);

#define BINARY_LOGICAL()  \
  Input("x: bool")        \
      .Input("y: bool")   \
      .Output("z: bool")  \
      .SetIsCommutative() \
      .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn)

REGISTER_OP("LogicalAnd").BINARY_LOGICAL();

REGISTER_OP("LogicalOr").BINARY_LOGICAL();

#undef BINARY_LOGICAL

// --------------------------------------------------------------------------

REGISTER_OP("Select")
    .Input("condition: bool")
    .Input("t: T")
    .Input("e: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      auto* handle_data_1 = c->input_handle_shapes_and_types(1);
      auto* handle_data_2 = c->input_handle_shapes_and_types(2);
      // Merge handle shape and dtype if applicable.
      if (handle_data_1 != nullptr && handle_data_2 != nullptr) {
        const auto size = handle_data_1->size();
        std::vector<shape_inference::ShapeAndType> merged_handle_data(size);
        if (size != handle_data_2->size()) {
          return errors::InvalidArgument(
              "Trying to merge handles pointing to different numbers of "
              "tensors.");
        }

        for (int i = 0; i < size; ++i) {
          const shape_inference::ShapeAndType& s1 = (*handle_data_1)[i];
          const shape_inference::ShapeAndType& s2 = (*handle_data_2)[i];
          if (s1.dtype != s2.dtype) {
            // TODO(apassos) resolve this in the manner of b/32476923
            return errors::InvalidArgument(
                "Trying to merge handles pointing to different dtypes.");
          }
          merged_handle_data[i].dtype = s1.dtype;
          TF_RETURN_IF_ERROR(
              c->Merge(s1.shape, s2.shape, &merged_handle_data[i].shape));
        }

        c->set_output_handle_shapes_and_types(0, merged_handle_data);
      }

      // The inputs 'then' and 'else' must have the same shape.
      ShapeHandle data = c->input(1);
      ShapeHandle other = c->input(2);
      TF_RETURN_IF_ERROR(c->Merge(data, other, &data));

      // The input 'cond' must either have the same shape as 'then' and
      // 'else', or be a vector if 'then' and 'else' are at least vectors.
      ShapeHandle cond = c->input(0);

      if (!c->RankKnown(cond) || !c->RankKnown(data)) {
        c->set_output(0, data);
        return absl::OkStatus();
      }

      // rank of shape and data is known.

      const int32_t cond_rank = c->Rank(cond);
      const int32_t data_rank = c->Rank(data);

      if (cond_rank == 0) {
        // The rank of 'cond' is a scalar.
        // t and e can have any shape.
        c->set_output(0, data);
        return absl::OkStatus();
      }

      if (cond_rank != 1) {
        // If 'cond' is not a vector, and not a scalar,
        // then shape must match 'then' and 'else'
        TF_RETURN_IF_ERROR(c->Merge(data, cond, &data));
        c->set_output(0, data);
        return absl::OkStatus();
      }

      if (data_rank == 0) {
        // if 'then' and 'else' are scalar also the cond must be
        TF_RETURN_IF_ERROR(c->Merge(data, cond, &data));
        c->set_output(0, data);
        return absl::OkStatus();
      }

      if (cond_rank == 1) {
        // if the cond is a vector and the 'then' is not a scalar,
        // the first dimension of 'then' and 'else'
        TF_RETURN_IF_ERROR(c->Merge(cond, c->Vector(c->Dim(data, 0)), &cond));
        c->set_output(0, data);
        return absl::OkStatus();
      }

      c->set_output(0, data);

      return absl::OkStatus();
    });

REGISTER_OP("SelectV2")
    .Input("condition: bool")
    .Input("t: T")
    .Input("e: T")
    .Output("output: T")
    .Attr("T: type")
    .SetShapeFn([](InferenceContext* c) {
      auto* handle_data_1 = c->input_handle_shapes_and_types(1);
      auto* handle_data_2 = c->input_handle_shapes_and_types(2);
      // Merge handle shape and dtype if applicable.
      if (handle_data_1 != nullptr && handle_data_2 != nullptr) {
        const auto size = handle_data_1->size();
        std::vector<shape_inference::ShapeAndType> merged_handle_data(size);
        if (size != handle_data_2->size()) {
          return errors::InvalidArgument(
              "Trying to merge handles pointing to different numbers of "
              "tensors.");
        }

        for (int i = 0; i < size; ++i) {
          const shape_inference::ShapeAndType& s1 = (*handle_data_1)[i];
          const shape_inference::ShapeAndType& s2 = (*handle_data_2)[i];
          if (s1.dtype != s2.dtype) {
            // TODO(apassos) resolve this in the manner of b/32476923
            return errors::InvalidArgument(
                "Trying to merge handles pointing to different dtypes.");
          }
          merged_handle_data[i].dtype = s1.dtype;
          TF_RETURN_IF_ERROR(
              c->Merge(s1.shape, s2.shape, &merged_handle_data[i].shape));
        }

        c->set_output_handle_shapes_and_types(0, merged_handle_data);
      }

      // The inputs 'cond', 'then', and 'else' must be broadcastable.
      // TODO (yongtang): Consolidate 3-ary broadcast instead of
      // multiple 2-ary broadcast.
      ShapeHandle cond = c->input(0);
      ShapeHandle then = c->input(1);
      ShapeHandle else_ = c->input(2);
      ShapeHandle other;
      TF_RETURN_IF_ERROR(
          BroadcastBinaryOpOutputShapeFnHelper(c, then, else_, true, &other));
      ShapeHandle output;
      TF_RETURN_IF_ERROR(
          BroadcastBinaryOpOutputShapeFnHelper(c, cond, other, true, &output));
      c->set_output(0, output);
      return absl::OkStatus();
    });

// --------------------------------------------------------------------------

REGISTER_OP("MatMul")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr(
        "T: {bfloat16, half, float, double, int32, int64, uint8, "
        "uint16, uint32, uint64, complex64, complex128}")
    .Attr("grad_a: bool = false")
    .Attr("grad_b: bool = false")
    .SetShapeFn(shape_inference::MatMulShape);

#ifdef INTEL_MKL
REGISTER_OP("_MklMatMul")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {bfloat16, float, half}")
    .Attr("grad_a: bool = false")
    .Attr("grad_b: bool = false")
    .SetShapeFn(shape_inference::MatMulShape);
#endif  // INTEL_MKL

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
    .SetShapeFn(shape_inference::MatMulShape);

REGISTER_OP("_FusedMatMul")
    .Input("a: T")
    .Input("b: T")
    .Input("args: num_args * T")
    .Output("product: T")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("T: {bfloat16, half, float}")
    .Attr("num_args: int >= 0")
    .Attr("fused_ops: list(string) = []")
    // Attributes for the FusedBatchNorm ----------- //
    .Attr("epsilon: float = 0.0001")
    // Attributes for the LeakyRelu ---------------- //
    .Attr("leakyrelu_alpha: float = 0.2")
    // --------------------------------------------- //
    .SetShapeFn(shape_inference::MatMulShape)
    .Doc(R"doc(
Performs a MatMul followed by a specified series of operations.

The inputs to the MatMul are specified by `a` and `b`. The series of operations
that follows is specified by the `fused_ops` attribute, which is a list of TF op
names specified as strings (e.g. "Relu"). They are performed in order, where the
(first) input to each op is the output of the preceding op. The first input and
the output of each fused_op must be of type T.

Currently supported fused_op combinations are: ["BiasAdd"] and ["BiasAdd",A],
where A is one of {"Elu","Relu","Relu6"}.

* The first input to BiasAdd is the MatMul result, and the additional BiasAdd
input is specified by `args`.
* If there is an op A specified, the output of the BiasAdd is the input to op A,
and op A produces the _FusedConv2D output. Otherwise, the BiasAdd produces the
_FusedConv2D output.

*NOTE*: Do not invoke this operator directly in Python. Grappler is
expected to create these operators.
)doc");

// --------------------------------------------------------------------------

// For operations where the output is a reduction function along some
// dimensions of the input.
REGISTER_OP("Sum")
    .Input("input: T")
    .Input("reduction_indices: Tidx")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);

REGISTER_OP("EuclideanNorm")
    .Input("input: T")
    .Input("reduction_indices: Tidx")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);

REGISTER_OP("Mean")
    .Input("input: T")
    .Input("reduction_indices: Tidx")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);

REGISTER_OP("Prod")
    .Input("input: T")
    .Input("reduction_indices: Tidx")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);

REGISTER_OP("Min")
    .Input("input: T")
    .Input("reduction_indices: Tidx")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: {realnumbertype, quantizedtype}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);

REGISTER_OP("Max")
    .Input("input: T")
    .Input("reduction_indices: Tidx")
    .Output("output: T")
    .Attr("keep_dims: bool = false")
    .Attr("T: {realnumbertype, quantizedtype}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);

namespace {

absl::Status ArgOpShape(shape_inference::InferenceContext* c) {
  ShapeHandle dimension_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &dimension_shape));

  ShapeHandle input_shape = c->input(0);
  if (!c->RankKnown(input_shape)) {
    return shape_inference::UnknownShape(c);
  }

  const int32_t input_rank = c->Rank(input_shape);
  if (input_rank <= 1) {
    // Reducing a scalar/vector must return a scalar.
    return shape_inference::ScalarShape(c);
  }

  const Tensor* dim_t = c->input_tensor(1);
  if (dim_t == nullptr) {
    // We don't know the value of the dimension, but we
    // know the rank of the input, so return the correct
    // rank with unknown dimensions.
    std::vector<DimensionHandle> dims(input_rank - 1);
    for (int i = 0; i < dims.size(); ++i) {
      dims[i] = c->UnknownDim();
    }

    c->set_output(0, c->MakeShape(dims));
    return absl::OkStatus();
  }

  int64_t dimension_val;
  if (dim_t->dtype() == DT_INT32) {
    dimension_val = dim_t->scalar<int32>()();
  } else {
    dimension_val = dim_t->scalar<int64_t>()();
  }

  int64_t axis = dimension_val < 0 ? dimension_val + input_rank : dimension_val;
  if (axis < 0 || axis >= input_rank) {
    return errors::InvalidArgument(
        "Dimension (", dimension_val, ") must be in the range [", -input_rank,
        ", ", input_rank, "), where ", input_rank,
        " is the number of dimensions in the input.");
  }

  // Return the input shape without the dimension being reduced.
  std::vector<DimensionHandle> dims;
  for (int i = 0; i < input_rank; ++i) {
    if (axis != i) {
      dims.emplace_back(c->Dim(input_shape, i));
    }
  }
  c->set_output(0, c->MakeShape(dims));
  return absl::OkStatus();
}

}  // namespace

REGISTER_OP("ArgMax")
    .Input("input: T")
    .Input("dimension: Tidx")
    .Output("output: output_type")
    .Attr("T: {realnumbertype, quantizedtype, bool}")
    .Attr("Tidx: {int16, int32, int64} = DT_INT32")
    .Attr("output_type: {int16, uint16, int32, int64} = DT_INT64")
    .SetShapeFn(ArgOpShape);

REGISTER_OP("ArgMin")
    .Input("input: T")
    .Input("dimension: Tidx")
    .Output("output: output_type")
    .Attr("T: {realnumbertype, quantizedtype, bool}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("output_type: {int32, int64} = DT_INT64")
    .SetShapeFn(ArgOpShape);

namespace {

absl::Status SegmentReductionShapeFn(InferenceContext* c) {
  ShapeHandle data_shape;
  ShapeHandle segment_ids_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &data_shape));
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &segment_ids_shape));

  ShapeHandle subshape;
  TF_RETURN_IF_ERROR(c->Subshape(data_shape, 1, &subshape));

  ShapeHandle out;
  TF_RETURN_IF_ERROR(
      c->Concatenate(c->Vector(InferenceContext::kUnknownDim), subshape, &out));
  c->set_output(0, out);
  return absl::OkStatus();
}

absl::Status SparseSegmentReductionShapeFn(InferenceContext* c) {
  ShapeHandle data_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &data_shape));

  ShapeHandle indices_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &indices_shape));

  ShapeHandle segment_ids_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &segment_ids_shape));

  // indices and segment_ids should merge cleanly.
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(indices_shape, segment_ids_shape, &unused));

  ShapeHandle subshape;
  TF_RETURN_IF_ERROR(c->Subshape(data_shape, 1, &subshape));

  ShapeHandle out;
  TF_RETURN_IF_ERROR(
      c->Concatenate(c->Vector(InferenceContext::kUnknownDim), subshape, &out));
  c->set_output(0, out);
  return absl::OkStatus();
}

absl::Status SparseSegmentReductionGradShapeFnImpl(
    InferenceContext* c, bool outputs_unique_indices) {
  ShapeHandle data_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &data_shape));

  ShapeHandle indices_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &indices_shape));

  // indices and segment_ids should merge cleanly.
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(c->input(2), indices_shape, &unused));

  // output_dim0 should be a scalar
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));

  ShapeHandle subshape;
  TF_RETURN_IF_ERROR(c->Subshape(data_shape, 1, &subshape));

  // For the V2 version of the op, dim0 is the number of unique indices, which
  // is always unknown at inference time.
  const Tensor* dim0 = outputs_unique_indices ? nullptr : c->input_tensor(3);
  ShapeHandle dim0_shape;
  if (dim0 == nullptr) {
    // We don't have the value at inference time, so the output
    // shape is unknown.
    dim0_shape = c->Vector(InferenceContext::kUnknownDim);
  } else {
    auto dim0_value = dim0->scalar<int32>()();
    if (dim0_value < 0) {
      return errors::InvalidArgument(
          "Cannot specify a negative value for output_dim0");
    }
    dim0_shape = c->Vector(dim0_value);
  }

  ShapeHandle out;
  TF_RETURN_IF_ERROR(c->Concatenate(dim0_shape, subshape, &out));
  c->set_output(0, out);
  if (outputs_unique_indices) {
    c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
  }
  return absl::OkStatus();
}

absl::Status SparseSegmentReductionGradShapeFn(InferenceContext* c) {
  return SparseSegmentReductionGradShapeFnImpl(
      c,
      /*outputs_unique_indices=*/false);
}

absl::Status SparseSegmentReductionGradV2ShapeFn(InferenceContext* c) {
  return SparseSegmentReductionGradShapeFnImpl(c,
                                               /*outputs_unique_indices=*/true);
}

absl::Status SparseSegmentReductionWithNumSegmentsShapeFn(InferenceContext* c) {
  ShapeHandle data_shape;
  TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &data_shape));

  ShapeHandle indices_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &indices_shape));

  ShapeHandle segment_ids_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &segment_ids_shape));

  ShapeHandle num_segments_shape;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &num_segments_shape));

  // indices and segment_ids should merge cleanly.
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->Merge(indices_shape, segment_ids_shape, &unused));

  ShapeHandle subshape;
  TF_RETURN_IF_ERROR(c->Subshape(data_shape, 1, &subshape));

  ShapeHandle out;
  const Tensor* dim0 = c->input_tensor(3);
  if (dim0 == nullptr) {
    // We don't have the value at inference time, so the output
    // shape is unknown.
    TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(InferenceContext::kUnknownDim),
                                      subshape, &out));
  } else {
    auto dim0_value = dim0->scalar<int32>()();
    if (dim0_value < 0) {
      return errors::InvalidArgument(
          "Cannot specify a negative value for num_segments");
    }
    TF_RETURN_IF_ERROR(c->Concatenate(c->Vector(dim0_value), subshape, &out));
  }
  c->set_output(0, out);
  return absl::OkStatus();
}
}  // namespace

REGISTER_OP("SegmentSum")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(SegmentReductionShapeFn);

// TODO(hinsu): Introduce Segment{Min,Max}V2 ops, similarly.
REGISTER_OP("SegmentSumV2")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .SetShapeFn(shape_inference::SegmentReductionWithNumSegmentsShapeFn);

REGISTER_OP("SegmentMean")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(SegmentReductionShapeFn);

REGISTER_OP("SegmentProd")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(SegmentReductionShapeFn);

REGISTER_OP("SegmentProdV2")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .SetShapeFn(shape_inference::SegmentReductionWithNumSegmentsShapeFn);

REGISTER_OP("SegmentMin")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(SegmentReductionShapeFn);

REGISTER_OP("SegmentMinV2")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .SetShapeFn(shape_inference::SegmentReductionWithNumSegmentsShapeFn);

REGISTER_OP("SegmentMax")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .SetShapeFn(SegmentReductionShapeFn);

REGISTER_OP("SegmentMaxV2")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .SetShapeFn(shape_inference::SegmentReductionWithNumSegmentsShapeFn);

REGISTER_OP("UnsortedSegmentSum")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int16,int32,int64}")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .SetShapeFn(shape_inference::SegmentReductionWithNumSegmentsShapeFn);

REGISTER_OP("UnsortedSegmentMax")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .SetShapeFn(shape_inference::SegmentReductionWithNumSegmentsShapeFn);

REGISTER_OP("UnsortedSegmentMin")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tindices: {int32,int64}")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .SetShapeFn(shape_inference::SegmentReductionWithNumSegmentsShapeFn);

REGISTER_OP("UnsortedSegmentProd")
    .Input("data: T")
    .Input("segment_ids: Tindices")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: numbertype")
    .Attr("Tindices: {int32,int64}")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .SetShapeFn(shape_inference::SegmentReductionWithNumSegmentsShapeFn);

REGISTER_OP("SparseSegmentSum")
    .Input("data: T")
    .Input("indices: Tidx")
    .Input("segment_ids: Tsegmentids")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("Tsegmentids: {int32, int64} = DT_INT32")
    .Attr("sparse_gradient: bool = false")
    .SetShapeFn(SparseSegmentReductionShapeFn);

REGISTER_OP("SparseSegmentSumWithNumSegments")
    .Input("data: T")
    .Input("indices: Tidx")
    .Input("segment_ids: Tsegmentids")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: realnumbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .Attr("Tsegmentids: {int32, int64} = DT_INT32")
    .Attr("sparse_gradient: bool = false")
    .SetShapeFn(SparseSegmentReductionWithNumSegmentsShapeFn);

REGISTER_OP("SparseSegmentSumGrad")
    .Input("grad: T")
    .Input("indices: Tidx")
    .Input("segment_ids: Tsegmentids")
    .Input("output_dim0: int32")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("Tsegmentids: {int32, int64} = DT_INT32")
    .SetShapeFn(SparseSegmentReductionGradShapeFn);

// V2 returns a sparse gradient instead of dense. Its (sparse) output dim0 is
// equal to the number of unique indices.
REGISTER_OP("SparseSegmentSumGradV2")
    .Input("grad: T")
    .Input("indices: Tidx")
    .Input("segment_ids: Tsegmentids")
    .Input("dense_output_dim0: int32")
    .Output("output: T")
    .Output("sorted_unique_indices: Tidx")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("Tsegmentids: {int32, int64} = DT_INT32")
    .SetShapeFn(SparseSegmentReductionGradV2ShapeFn);

REGISTER_OP("SparseSegmentMean")
    .Input("data: T")
    .Input("indices: Tidx")
    .Input("segment_ids: Tsegmentids")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("Tsegmentids: {int32, int64} = DT_INT32")
    .Attr("sparse_gradient: bool = false")
    .SetShapeFn(SparseSegmentReductionShapeFn);

REGISTER_OP("SparseSegmentMeanWithNumSegments")
    .Input("data: T")
    .Input("indices: Tidx")
    .Input("segment_ids: Tsegmentids")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .Attr("Tsegmentids: {int32, int64} = DT_INT32")
    .Attr("sparse_gradient: bool = false")
    .SetShapeFn(SparseSegmentReductionWithNumSegmentsShapeFn);

REGISTER_OP("SparseSegmentMeanGrad")
    .Input("grad: T")
    .Input("indices: Tidx")
    .Input("segment_ids: Tsegmentids")
    .Input("output_dim0: int32")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("Tsegmentids: {int32, int64} = DT_INT32")
    .SetShapeFn(SparseSegmentReductionGradShapeFn);

REGISTER_OP("SparseSegmentMeanGradV2")
    .Input("grad: T")
    .Input("indices: Tidx")
    .Input("segment_ids: Tsegmentids")
    .Input("dense_output_dim0: int32")
    .Output("output: T")
    .Output("sorted_unique_indices: Tidx")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("Tsegmentids: {int32, int64} = DT_INT32")
    .SetShapeFn(SparseSegmentReductionGradV2ShapeFn);

REGISTER_OP("SparseSegmentSqrtN")
    .Input("data: T")
    .Input("indices: Tidx")
    .Input("segment_ids: Tsegmentids")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("Tsegmentids: {int32, int64} = DT_INT32")
    .Attr("sparse_gradient: bool = false")
    .SetShapeFn(SparseSegmentReductionShapeFn);

REGISTER_OP("SparseSegmentSqrtNWithNumSegments")
    .Input("data: T")
    .Input("indices: Tidx")
    .Input("segment_ids: Tsegmentids")
    .Input("num_segments: Tnumsegments")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("Tnumsegments: {int32,int64} = DT_INT32")
    .Attr("Tsegmentids: {int32, int64} = DT_INT32")
    .Attr("sparse_gradient: bool = false")
    .SetShapeFn(SparseSegmentReductionWithNumSegmentsShapeFn);

REGISTER_OP("SparseSegmentSqrtNGrad")
    .Input("grad: T")
    .Input("indices: Tidx")
    .Input("segment_ids: Tsegmentids")
    .Input("output_dim0: int32")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("Tsegmentids: {int32, int64} = DT_INT32")
    .SetShapeFn(SparseSegmentReductionGradShapeFn);

REGISTER_OP("SparseSegmentSqrtNGradV2")
    .Input("grad: T")
    .Input("indices: Tidx")
    .Input("segment_ids: Tsegmentids")
    .Input("dense_output_dim0: int32")
    .Output("output: T")
    .Output("sorted_unique_indices: Tidx")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .Attr("Tsegmentids: {int32, int64} = DT_INT32")
    .SetShapeFn(SparseSegmentReductionGradV2ShapeFn);

REGISTER_OP("All")
    .Input("input: bool")
    .Input("reduction_indices: Tidx")
    .Output("output: bool")
    .Attr("keep_dims: bool = false")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);

REGISTER_OP("Any")
    .Input("input: bool")
    .Input("reduction_indices: Tidx")
    .Attr("keep_dims: bool = false")
    .Output("output: bool")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::ReductionShape);

// --------------------------------------------------------------------------

namespace {

template <typename T>
absl::Status RangeSize(const Tensor* start_t, const Tensor* limit_t,
                       const Tensor* delta_t, InferenceContext* const c) {
  T start = start_t->scalar<T>()();
  T limit = limit_t->scalar<T>()();
  T delta = delta_t->scalar<T>()();
  if (start > limit && delta > T(0)) {
    return errors::InvalidArgument(
        "Requires start <= limit when delta > 0: ", start, "/", limit);
  }
  if (start < limit && delta < T(0)) {
    return errors::InvalidArgument(
        "Requires start >= limit when delta < 0: ", start, "/", limit);
  }
  if (delta == T(0)) {
    return errors::InvalidArgument("Requires delta != 0");
  }

  int64_t size;
  if (std::is_integral<T>::value) {
    uint64_t range;
    if ((limit > 0 && start < 0) || (limit < 0 && start > 0)) {
      range = static_cast<uint64_t>(Eigen::numext::abs(limit)) +
              static_cast<uint64_t>(Eigen::numext::abs(start));
    } else {
      range = static_cast<uint64_t>(Eigen::numext::abs(limit - start));
    }

    uint64_t size_unsigned =
        Eigen::divup(range, static_cast<uint64_t>(Eigen::numext::abs(delta)));
    if (size_unsigned > std::numeric_limits<int64_t>::max()) {
      return errors::InvalidArgument("Requires ((limit - start) / delta) <= ",
                                     std::numeric_limits<int64_t>::max());
    }
    size = static_cast<int64_t>(size_unsigned);
  } else {
    auto size_auto =
        Eigen::numext::ceil(Eigen::numext::abs((limit - start) / delta));
    if (size_auto > std::numeric_limits<int64_t>::max()) {
      return errors::InvalidArgument("Requires ((limit - start) / delta) <= ",
                                     std::numeric_limits<int64_t>::max());
    }
    size = static_cast<int64_t>(size_auto);
  }

  c->set_output(0, c->Vector(static_cast<int64_t>(size)));
  return absl::OkStatus();
}

}  // namespace

REGISTER_OP("Range")
    .Input("start: Tidx")
    .Input("limit: Tidx")
    .Input("delta: Tidx")
    .Output("output: Tidx")
    .Attr(
        "Tidx: "
        "{bfloat16, half, float, double, int8, int16, int32, int64, uint16, "
        "uint32} = "
        "DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(0), 0, &unused),
                                      " for 'start'");
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(1), 0, &unused),
                                      " for 'limit'");
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(2), 0, &unused),
                                      " for 'delta'");
      const Tensor* start_t = c->input_tensor(0);
      const Tensor* limit_t = c->input_tensor(1);
      const Tensor* delta_t = c->input_tensor(2);
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("Tidx", &dtype));
      if (start_t == nullptr || limit_t == nullptr || delta_t == nullptr) {
        c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
        return absl::OkStatus();
      }
      if (dtype == DT_INT32) {
        return RangeSize<int32>(start_t, limit_t, delta_t, c);
      } else if (dtype == DT_INT16) {
        return RangeSize<int16>(start_t, limit_t, delta_t, c);
      } else if (dtype == DT_INT8) {
        return RangeSize<int8>(start_t, limit_t, delta_t, c);
      } else if (dtype == DT_INT64) {
        return RangeSize<int64_t>(start_t, limit_t, delta_t, c);
      } else if (dtype == DT_UINT16) {
        return RangeSize<uint16>(start_t, limit_t, delta_t, c);
      } else if (dtype == DT_UINT32) {
        return RangeSize<uint32>(start_t, limit_t, delta_t, c);
      } else if (dtype == DT_FLOAT) {
        return RangeSize<float>(start_t, limit_t, delta_t, c);
      } else if (dtype == DT_DOUBLE) {
        return RangeSize<double>(start_t, limit_t, delta_t, c);
      } else if (dtype == DT_BFLOAT16) {
        return RangeSize<bfloat16>(start_t, limit_t, delta_t, c);
      } else {
        return errors::InvalidArgument("Unsupported dtype", dtype);
      }
      return absl::OkStatus();
    });

REGISTER_OP("LinSpace")
    .Input("start: T")
    .Input("stop: T")
    .Input("num: Tidx")
    .Output("output: T")
    .Attr("T: {bfloat16, half, float, double}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(0), 0, &unused),
                                      " for 'start'");
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(1), 0, &unused),
                                      " for 'stop'");
      TF_RETURN_WITH_CONTEXT_IF_ERROR(c->WithRank(c->input(2), 0, &unused),
                                      " for 'num'");
      const Tensor* num_t = c->input_tensor(2);
      if (num_t == nullptr) {
        c->set_output(0, c->Vector(InferenceContext::kUnknownDim));
        return absl::OkStatus();
      }

      int64_t num;
      if (num_t->dtype() == DT_INT32) {
        num = num_t->scalar<int32>()();
      } else {
        num = num_t->scalar<int64_t>()();
      }
      if (num <= 0) return errors::InvalidArgument("Requires num > 0: ", num);
      c->set_output(0, c->Vector(num));
      return absl::OkStatus();
    });

REGISTER_OP("Complex")
    .Input("real: T")
    .Input("imag: T")
    .Output("out: Tout")
    .Attr("T: {float, double} = DT_FLOAT")
    .Attr("Tout: {complex64, complex128} = DT_COMPLEX64")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("Real")
    .Input("input: T")
    .Output("output: Tout")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .Attr("Tout: {float, double} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Imag")
    .Input("input: T")
    .Output("output: Tout")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .Attr("Tout: {float, double} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Angle")
    .Input("input: T")
    .Output("output: Tout")
    .Attr("T: {complex64, complex128} = DT_COMPLEX64")
    .Attr("Tout: {float, double} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Conj")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: {complex64, complex128, variant} = DT_COMPLEX64")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->input(0));
      auto* handle_data = c->input_handle_shapes_and_types(0);
      if (handle_data != nullptr) {
        c->set_output_handle_shapes_and_types(0, *handle_data);
      }
      return absl::OkStatus();
    });

// --------------------------------------------------------------------------

REGISTER_OP("Cross")
    .Input("a: T")
    .Input("b: T")
    .Output("product: T")
    .Attr("T: realnumbertype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle a_shape;
      ShapeHandle b_shape;
      // * Input rank >= 1.
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(0), 1, &a_shape));
      TF_RETURN_IF_ERROR(c->WithRankAtLeast(c->input(1), 1, &b_shape));

      // * Both inputs have the same shape.
      TF_RETURN_IF_ERROR(c->Merge(a_shape, b_shape, &a_shape));

      // * input_shape[-1] == 3.
      if (c->RankKnown(a_shape)) {
        int rank = c->Rank(a_shape);
        auto dim = c->Dim(a_shape, rank - 1);
        TF_RETURN_IF_ERROR(c->WithValue(dim, 3, &dim));
      }
      c->set_output(0, a_shape);
      return absl::OkStatus();
    });

// --------------------------------------------------------------------------

REGISTER_OP("HistogramFixedWidth")
    .Input("values: T")
    .Input("value_range: T")
    .Input("nbins: int32")
    .Output("out: dtype")
    .Attr("T: {int32, int64, float32, float64}")
    .Attr("dtype: {int32, int64} = DT_INT32")
    .SetShapeFn([](InferenceContext* c) {
      // value_range should be a vector.
      ShapeHandle value_range_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &value_range_shape));
      // value_range should have two elements.
      DimensionHandle unused;
      TF_RETURN_IF_ERROR(
          c->WithValue(c->Dim(value_range_shape, 0), 2, &unused));
      // nbins should be a scalar.
      ShapeHandle nbins_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &nbins_shape));

      // If nbins is available, set the shape from nbins.
      const Tensor* nbins_input = c->input_tensor(2);
      if (nbins_input != nullptr) {
        int64_t nbins;
        TF_RETURN_IF_ERROR(c->GetScalarFromTensor(nbins_input, &nbins));
        // nbins has to be positive.
        if (nbins <= 0) {
          return errors::InvalidArgument("Requires nbins > 0: ", nbins);
        }
        c->set_output(0, c->Vector(nbins));
      } else {
        c->set_output(0, c->UnknownShapeOfRank(1));
      }
      return absl::OkStatus();
    });

REGISTER_OP("Bincount")
    .Input("arr: int32")
    .Input("size: int32")
    .Input("weights: T")
    .Attr("T: {int32, int64, float32, float64}")
    .Output("bins: T")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      // The input `size` must be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));

      const Tensor* size_tensor = c->input_tensor(1);
      if (size_tensor == nullptr) {
        // Return unknown shape if size is not known.
        c->set_output(0, c->UnknownShapeOfRank(1));
        return absl::OkStatus();
      }

      if (size_tensor->dims() != 0) {
        return errors::InvalidArgument("Shape must be rank 0 but is rank ",
                                       size_tensor->dims());
      }

      // Return `[size]` shape if size is known.
      int32_t size_val = size_tensor->scalar<int32>()();
      if (size_val < 0) {
        return errors::InvalidArgument("size (", size_val,
                                       ") must be non-negative");
      }
      c->set_output(0, c->MakeShape({size_val}));
      return absl::OkStatus();
    });

REGISTER_OP("DenseBincount")
    .Input("input: Tidx")
    .Input("size: Tidx")
    .Input("weights: T")
    .Attr("Tidx: {int32, int64}")
    .Attr("T: {int32, int64, float32, float64}")
    .Attr("binary_output: bool = false")
    .Output("output: T")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      // The input `input` must be at most matrix.
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 2, &unused));
      // The input `size` must be a scalar.
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));

      const Tensor* size_tensor = c->input_tensor(1);
      if (size_tensor == nullptr) {
        // Return "vector of unknown size", "matrix of unknown size" or
        // "unknown shape" if size is unknown, based on whether the rank of the
        // input is 1, 2 or unknown respectively.
        c->set_output(0, c->UnknownShapeOfRank(c->Rank(c->input(0))));
        return absl::OkStatus();
      }
      if (size_tensor->dims() != 0) {
        return errors::InvalidArgument("Shape must be rank 0 but is rank ",
                                       size_tensor->dims());
      }

      int64_t size_val;
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("Tidx", &dtype));
      if (dtype == DT_INT32) {
        size_val = static_cast<int64_t>(size_tensor->scalar<int32>()());
      } else if (dtype == DT_INT64) {
        size_val = size_tensor->scalar<int64_t>()();
      } else {
        return errors::InvalidArgument("size dtype must be int32 or int64");
      }
      // Return `[size]` shape if size is known.
      if (size_val < 0) {
        return errors::InvalidArgument("size (", size_val,
                                       ") must be non-negative");
      }
      if (c->Rank(c->input(0)) == 1 || c->Rank(c->input(0)) == 0) {
        c->set_output(0, c->MakeShape({size_val}));
      } else if (c->Rank(c->input(0)) == 2) {
        c->set_output(0, c->MakeShape({c->Dim(c->input(0), 0), size_val}));
      }
      return absl::OkStatus();
    });

REGISTER_OP("SparseBincount")
    .Input("indices: int64")
    .Input("values: Tidx")
    .Input("dense_shape: int64")
    .Input("size: Tidx")
    .Input("weights: T")
    .Attr("Tidx: {int32, int64}")
    .Attr("T: {int32, int64, float32, float64}")
    .Attr("binary_output: bool = false")
    .Output("output: T")
    .SetShapeFn([](InferenceContext* c) {
      const Tensor* size_tensor = c->input_tensor(3);
      if (size_tensor == nullptr) {
        // Return unknown shape if size is not known.
        c->set_output(0, c->UnknownShape());
        return absl::OkStatus();
      }
      if (size_tensor->dims() != 0) {
        return errors::InvalidArgument("Shape must be rank 0 but is rank ",
                                       size_tensor->dims());
      }

      int64_t size_val;
      DataType dtype;
      TF_RETURN_IF_ERROR(c->GetAttr("Tidx", &dtype));
      if (dtype == DT_INT32) {
        size_val = static_cast<int64_t>(size_tensor->scalar<int32>()());
      } else if (dtype == DT_INT64) {
        size_val = size_tensor->scalar<int64_t>()();
      } else {
        return errors::InvalidArgument("size dtype must be int32 or int64");
      }
      // Return `[size]` shape if size is known.
      if (size_val < 0) {
        return errors::InvalidArgument("size (", size_val,
                                       ") must be non-negative");
      }

      const Tensor* shape_tensor = c->input_tensor(2);
      if (shape_tensor == nullptr) {
        // Return unknown shape if size is not known.
        c->set_output(0, c->UnknownShape());
        return absl::OkStatus();
      }
      if (shape_tensor->NumElements() == 1) {
        c->set_output(0, c->MakeShape({size_val}));
      } else if (shape_tensor->NumElements() == 2) {
        c->set_output(
            0, c->MakeShape({shape_tensor->flat<int64_t>()(0), size_val}));
      } else {
        return errors::InvalidArgument("Input must be less than rank 2");
      }
      return absl::OkStatus();
    });

REGISTER_OP("RaggedBincount")
    .Input("splits: int64")
    .Input("values: Tidx")
    .Input("size: Tidx")
    .Input("weights: T")
    .Attr("Tidx: {int32, int64}")
    .Attr("T: {int32, int64, float32, float64}")
    .Attr("binary_output: bool = false")
    .Output("output: T")
    .SetShapeFn([](InferenceContext* c) {
      c->set_output(0, c->UnknownShape());
      return absl::OkStatus();
    });

REGISTER_OP("Cumsum")
    .Input("x: T")
    .Input("axis: Tidx")
    .Attr("exclusive: bool = false")
    .Attr("reverse: bool = false")
    .Output("out: T")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("Cumprod")
    .Input("x: T")
    .Input("axis: Tidx")
    .Attr("exclusive: bool = false")
    .Attr("reverse: bool = false")
    .Output("out: T")
    .Attr("T: numbertype")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("CumulativeLogsumexp")
    .Input("x : T")
    .Input("axis: Tidx")
    .Attr("exclusive: bool = false")
    .Attr("reverse: bool = false")
    .Output("out: T")
    .Attr("T: {bfloat16, float16, float32, float64}")
    .Attr("Tidx: {int32, int64} = DT_INT32")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("QuantizedMatMul")
    .Input("a: T1")
    .Input("b: T2")
    .Input("min_a: float")
    .Input("max_a: float")
    .Input("min_b: float")
    .Input("max_b: float")
    .Output("out: Toutput")
    .Output("min_out: float")
    .Output("max_out: float")
    .Attr("T1: quantizedtype")
    .Attr("T2: quantizedtype")
    .Attr("Toutput: quantizedtype = DT_QINT32")
    .Attr("transpose_a: bool = false")
    .Attr("transpose_b: bool = false")
    .Attr("Tactivation: quantizedtype = DT_QUINT8")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::MatMulShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));

      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return absl::OkStatus();
    });

// Note: This op is not commutative w.r.t. to all its inputs.
REGISTER_OP("QuantizedMul")
    .Input("x: T1")
    .Input("y: T2")
    .Input("min_x: float")
    .Input("max_x: float")
    .Input("min_y: float")
    .Input("max_y: float")
    .Output("z: Toutput")
    .Output("min_z: float")
    .Output("max_z: float")
    .Attr("T1: quantizedtype")
    .Attr("T2: quantizedtype")
    .Attr("Toutput: quantizedtype = DT_QINT32")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::BroadcastBinaryOpShapeFn(c));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return absl::OkStatus();
    });

// Note: This op is not commutative w.r.t. to all its inputs.
REGISTER_OP("QuantizedAdd")
    .Input("x: T1")
    .Input("y: T2")
    .Input("min_x: float")
    .Input("max_x: float")
    .Input("min_y: float")
    .Input("max_y: float")
    .Output("z: Toutput")
    .Output("min_z: float")
    .Output("max_z: float")
    .Attr("T1: quantizedtype")
    .Attr("T2: quantizedtype")
    .Attr("Toutput: quantizedtype = DT_QINT32")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::BroadcastBinaryOpShapeFn(c));
      // min_x, max_x, min_y, max_y should be scalar.
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));

      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return absl::OkStatus();
    });

REGISTER_OP("QuantizeDownAndShrinkRange")
    .Input("input: Tinput")
    .Input("input_min: float")
    .Input("input_max: float")
    .Output("output: out_type")
    .Output("output_min: float")
    .Output("output_max: float")
    .Attr("Tinput: quantizedtype")
    .Attr("out_type: quantizedtype")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::UnchangedShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return absl::OkStatus();
    });

REGISTER_OP("Requantize")
    .Input("input: Tinput")
    .Input("input_min: float")
    .Input("input_max: float")
    .Input("requested_output_min: float")
    .Input("requested_output_max: float")
    .Output("output: out_type")
    .Output("output_min: float")
    .Output("output_max: float")
    .Attr("Tinput: quantizedtype")
    .Attr("out_type: quantizedtype")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::UnchangedShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return absl::OkStatus();
    });

REGISTER_OP("RequantizationRange")
    .Input("input: Tinput")
    .Input("input_min: float")
    .Input("input_max: float")
    .Output("output_min: float")
    .Output("output_max: float")
    .Attr("Tinput: quantizedtype")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      return absl::OkStatus();
    });

// --------------------------------------------------------------------------

REGISTER_OP("Bucketize")
    .Input("input: T")
    .Output("output: int32")
    .Attr("T: {int32, int64, float, double}")
    .Attr("boundaries: list(float)")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("ClipByValue")
    .Input("t: T")
    .Input("clip_value_min: T")
    .Input("clip_value_max: T")
    .Output("output: T")
    .Attr("T: numbertype")
    .SetShapeFn(shape_inference::UnchangedShape);

REGISTER_OP("RequantizePerChannel")
    .Input("input: T")
    .Input("input_min: float")
    .Input("input_max: float")
    .Input("requested_output_min: float")
    .Input("requested_output_max: float")
    .Output("output: out_type")
    .Output("output_min: float")
    .Output("output_max: float")
    .Attr("T: quantizedtype = DT_QINT32")
    .Attr("out_type: quantizedtype = DT_QUINT8")
    .SetShapeFn([](InferenceContext* c) {
      TF_RETURN_IF_ERROR(shape_inference::UnchangedShape(c));
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      c->set_output(1, c->Scalar());
      c->set_output(2, c->Scalar());
      return absl::OkStatus();
    });
REGISTER_OP("RequantizationRangePerChannel")
    .Input("input: T")
    .Input("input_min: float")
    .Input("input_max: float")
    .Output("output_min: float")
    .Output("output_max: float")
    .Attr("T: quantizedtype = DT_QINT32")
    .Attr("clip_value_max: float")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &unused));
      c->set_output(0, c->Scalar());
      c->set_output(1, c->Scalar());
      return absl::OkStatus();
    });

REGISTER_OP("NextAfter")
    .Attr("T: {float64, float32} = DT_FLOAT")
    .Input("x1: T")
    .Input("x2: T")
    .Output("output: T")
    .SetShapeFn(shape_inference::BroadcastBinaryOpShapeFn);

REGISTER_OP("SobolSample")
    .Input("dim: int32")
    .Input("num_results: int32")
    .Input("skip: int32")
    .Attr("dtype: {float, double} = DT_FLOAT")
    .Output("samples: dtype")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      ShapeHandle unused;

      // inputs must be scalars
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));

      const Tensor* dim_t = c->input_tensor(0);
      const Tensor* num_results_t = c->input_tensor(1);

      int32_t dim = dim_t == nullptr ? InferenceContext::kUnknownDim
                                     : dim_t->scalar<int32>()();

      int32_t num_results = num_results_t == nullptr
                                ? InferenceContext::kUnknownDim
                                : num_results_t->scalar<int32>()();

      c->set_output(0, c->Matrix(num_results, dim));
      return absl::OkStatus();
    });

}  // namespace tensorflow
