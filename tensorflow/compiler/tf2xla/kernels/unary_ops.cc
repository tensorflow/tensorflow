/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Native XLA implementations of simple unary Ops

#include <cmath>

#include "absl/status/statusor.h"
#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/lib/math.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

#define XLAJIT_MAKE_UNARY(NAME, COMPUTATION)                           \
  class NAME##Op : public XlaOpKernel {                                \
   public:                                                             \
    explicit NAME##Op(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {} \
    void Compile(XlaOpKernelContext* ctx) {                            \
      xla::XlaBuilder* b = ctx->builder();                             \
      (void)b;                                                         \
      xla::XlaOp x = ctx->Input(0);                                    \
      xla::XlaOp y = COMPUTATION;                                      \
      ctx->SetOutput(0, y);                                            \
    }                                                                  \
  };                                                                   \
  REGISTER_XLA_OP(Name(#NAME), NAME##Op);

XLAJIT_MAKE_UNARY(ComplexAbs, xla::Abs(x));

XLAJIT_MAKE_UNARY(Angle, xla::Atan2(xla::Imag(x), xla::Real(x)));

XLAJIT_MAKE_UNARY(Conj, xla::Conj(x));

// Return x if x>0, otherwise -x.
REGISTER_XLA_OP(Name("Abs"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Acos"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Acosh"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Asin"), MlirXlaOpKernel);
XLAJIT_MAKE_UNARY(Asinh, xla::Asinh(x));
REGISTER_XLA_OP(Name("Atan"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Atanh"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Ceil"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Cos"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Cosh"), MlirXlaOpKernel);
XLAJIT_MAKE_UNARY(Sin, xla::Sin(x));
XLAJIT_MAKE_UNARY(Tan, xla::Tan(x));
REGISTER_XLA_OP(Name("Exp"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Expm1"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Floor"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("IsFinite"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("IsInf"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("IsNan"), MlirXlaOpKernel);
// Return 1/x
XLAJIT_MAKE_UNARY(Inv, xla::ScalarLike(x, 1.0) / x);
REGISTER_XLA_OP(Name("Reciprocal"), MlirXlaOpKernel);
XLAJIT_MAKE_UNARY(Log, xla::Log(x));
REGISTER_XLA_OP(Name("Log1p"), MlirXlaOpKernel);

XLAJIT_MAKE_UNARY(Invert, xla::Not(x));
XLAJIT_MAKE_UNARY(LogicalNot, xla::Not(x));
XLAJIT_MAKE_UNARY(PopulationCount,
                  xla::ConvertElementType(xla::PopulationCount(x), xla::U8));
XLAJIT_MAKE_UNARY(Neg, -x);

XLAJIT_MAKE_UNARY(Rint, xla::RoundToEven(x));
XLAJIT_MAKE_UNARY(Round, xla::RoundToEven(x));

REGISTER_XLA_OP(Name("Rsqrt"), MlirXlaOpKernel);

REGISTER_XLA_OP(Name("Sigmoid"), MlirXlaOpKernel);

// Returns NaN if x is NaN, 0 if x is 0, -1 if x < 0 and 1 if x > 0.
REGISTER_XLA_OP(Name("Sign"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Sinh"), MlirXlaOpKernel);

static xla::XlaOp Softplus(xla::XlaBuilder* b, xla::XlaOp features) {
  return b->ReportErrorOrReturn([&]() -> absl::StatusOr<xla::XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, b->GetShape(features));
    xla::XlaOp threshold =
        Log(xla::Epsilon(b, shape.element_type())) + ScalarLike(features, 2.0);
    // Value above which exp(x) may overflow, but softplus(x) == x
    // is within machine epsilon.
    xla::XlaOp too_large = Gt(features, -threshold);
    // Value below which exp(x) may underflow, but softplus(x) == exp(x)
    // is within machine epsilon.
    xla::XlaOp too_small = Lt(features, threshold);
    xla::XlaOp features_exp = Exp(features);
    xla::XlaOp output =
        Select(too_large, features,
               Select(too_small, features_exp, Log1p(features_exp)));
    return output;
  });
}
XLAJIT_MAKE_UNARY(Softplus, Softplus(b, x));

// softsign(x) = x / (abs(x) + 1)
XLAJIT_MAKE_UNARY(Softsign, x / (xla::Abs(x) + xla::ScalarLike(x, 1.0)));
REGISTER_XLA_OP(Name("Sqrt"), MlirXlaOpKernel);
XLAJIT_MAKE_UNARY(Square, x* x);
REGISTER_XLA_OP(Name("Tanh"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Real"), MlirXlaOpKernel);
REGISTER_XLA_OP(Name("Imag"), MlirXlaOpKernel);
XLAJIT_MAKE_UNARY(Erf, xla::Erf(x));
XLAJIT_MAKE_UNARY(Erfc, xla::Erfc(x));
XLAJIT_MAKE_UNARY(Erfinv, xla::ErfInv(x));
// ndtri = sqrt(2) * erfinv(2 * x - 1)
XLAJIT_MAKE_UNARY(Ndtri, xla::ScalarLike(x, std::sqrt(2.0)) *
                             xla::ErfInv(xla::ScalarLike(x, 2.0) * x -
                                         xla::ScalarLike(x, 1.0)));
REGISTER_XLA_OP(Name("Lgamma"), MlirXlaOpKernel);
XLAJIT_MAKE_UNARY(Digamma, xla::Digamma(x));
XLAJIT_MAKE_UNARY(BesselI0e, xla::BesselI0e(x));
XLAJIT_MAKE_UNARY(BesselI1e, xla::BesselI1e(x));

}  // namespace
}  // namespace tensorflow
