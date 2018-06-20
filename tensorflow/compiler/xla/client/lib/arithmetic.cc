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

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"

#include <string>

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_computation.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {
namespace {

using XlaOpGenerator = XlaOp (*)(XlaBuilder*, const XlaOp&, const XlaOp&);

XlaComputation CreateScalarComputation(const string& name, PrimitiveType type,
                                       XlaBuilder* builder,
                                       XlaOpGenerator generator) {
  std::unique_ptr<XlaBuilder> b;
  if (type == PRED) {
    b = builder->CreateSubBuilder(name);
  } else {
    b = builder->CreateSubBuilder(
        tensorflow::strings::StrCat(name, "_", PrimitiveType_Name(type)));
  }

  const Shape scalar = ShapeUtil::MakeShape(type, {});
  auto lhs = b->Parameter(0, scalar, "lhs");
  auto rhs = b->Parameter(1, scalar, "rhs");
  generator(b.get(), lhs, rhs);
  return b->BuildAndNoteError();
}

}  // namespace

XlaComputation CreateScalarAddComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
  return CreateScalarComputation(
      "add", type, builder,
      [](XlaBuilder* b, const XlaOp& lhs, const XlaOp& rhs) {
        return b->Add(lhs, rhs);
      });
}

XlaComputation CreateScalarMultiplyComputation(PrimitiveType type,
                                               XlaBuilder* builder) {
  return CreateScalarComputation(
      "mul", type, builder,
      [](XlaBuilder* b, const XlaOp& lhs, const XlaOp& rhs) {
        return b->Mul(lhs, rhs);
      });
}

XlaComputation CreateScalarGeComputation(PrimitiveType type,
                                         XlaBuilder* builder) {
  return CreateScalarComputation(
      "ge", type, builder,
      [](XlaBuilder* b, const XlaOp& lhs, const XlaOp& rhs) {
        return b->Ge(lhs, rhs);
      });
}

XlaComputation CreateScalarMaxComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
  return CreateScalarComputation(
      "max", type, builder,
      [](XlaBuilder* b, const XlaOp& lhs, const XlaOp& rhs) {
        return b->Max(lhs, rhs);
      });
}

XlaComputation CreateScalarMinComputation(PrimitiveType type,
                                          XlaBuilder* builder) {
  return CreateScalarComputation(
      "min", type, builder,
      [](XlaBuilder* b, const XlaOp& lhs, const XlaOp& rhs) {
        return b->Min(lhs, rhs);
      });
}

XlaComputation CreateScalarAndComputation(XlaBuilder* builder) {
  return CreateScalarComputation(
      "and", PRED, builder,
      [](XlaBuilder* b, const XlaOp& lhs, const XlaOp& rhs) {
        return b->And(lhs, rhs);
      });
}

XlaComputation CreateScalarOrComputation(XlaBuilder* builder) {
  return CreateScalarComputation(
      "or", PRED, builder,
      [](XlaBuilder* b, const XlaOp& lhs, const XlaOp& rhs) {
        return b->Or(lhs, rhs);
      });
}

StatusOr<XlaOp> Any(const XlaOp& predicates, XlaBuilder* builder) {
  auto f = builder->ConstantR0<bool>(false);
  XlaComputation logical_or = CreateScalarOrComputation(builder);
  TF_ASSIGN_OR_RETURN(const Shape& predicates_shape,
                      builder->GetShape(predicates));
  std::vector<int64> all_dimensions(ShapeUtil::Rank(predicates_shape));
  std::iota(all_dimensions.begin(), all_dimensions.end(), 0);
  return builder->Reduce(predicates, f, logical_or, all_dimensions);
}

namespace {
xla::XlaOp FloatLiteral(xla::XlaBuilder* b, PrimitiveType data_type,
                        float value) {
  return b->ConvertElementType(b->ConstantR0(value), data_type);
}

// Polynomials for computing erf/erfc.  Originally from cephes.
// Note we use float for compatibility across devices, at the cost of some
// precision for 64 bit computations.
//
// Coefficients are in descending order.
std::array<float, 9> kErfcPCoefficient = {
    2.46196981473530512524E-10, 5.64189564831068821977E-1,
    7.46321056442269912687E0,   4.86371970985681366614E1,
    1.96520832956077098242E2,   5.26445194995477358631E2,
    9.34528527171957607540E2,   1.02755188689515710272E3,
    5.57535335369399327526E2};
std::array<float, 9> kErfcQCoefficient = {
    1.00000000000000000000E0, 1.32281951154744992508E1,
    8.67072140885989742329E1, 3.54937778887819891062E2,
    9.75708501743205489753E2, 1.82390916687909736289E3,
    2.24633760818710981792E3, 1.65666309194161350182E3,
    5.57535340817727675546E2};
std::array<float, 6> kErfcRCoefficient = {
    5.64189583547755073984E-1, 1.27536670759978104416E0,
    5.01905042251180477414E0,  6.16021097993053585195E0,
    7.40974269950448939160E0,  2.97886665372100240670E0};
std::array<float, 7> kErfcSCoefficient = {
    1.00000000000000000000E0, 2.26052863220117276590E0,
    9.39603524938001434673E0, 1.20489539808096656605E1,
    1.70814450747565897222E1, 9.60896809063285878198E0,
    3.36907645100081516050E0};
std::array<float, 5> kErfTCoefficient = {
    9.60497373987051638749E0, 9.00260197203842689217E1,
    2.23200534594684319226E3, 7.00332514112805075473E3,
    5.55923013010394962768E4};
std::array<float, 6> kErfUCoefficient = {
    1.00000000000000000000E0, 3.35617141647503099647E1,
    5.21357949780152679795E2, 4.59432382970980127987E3,
    2.26290000613890934246E4, 4.92673942608635921086E4};
}  // namespace

// Evaluate the polynomial given coefficients and `x`.
// N.B. Coefficients should be supplied in decreasing order.
xla::XlaOp EvaluatePolynomial(xla::XlaBuilder* b, const xla::XlaOp& x,
                              tensorflow::gtl::ArraySlice<float> coefficients,
                              PrimitiveType data_type) {
  xla::XlaOp poly = FloatLiteral(b, data_type, 0.0);
  for (float c : coefficients) {
    poly = b->Add(b->Mul(poly, x), FloatLiteral(b, data_type, c));
  }
  return poly;
}

// Compute an approximation of the error function complement (1 - erf(x)).
xla::XlaOp ComputeErfc(xla::XlaBuilder* b, const xla::XlaOp& x,
                       PrimitiveType data_type) {
  xla::XlaOp zero = FloatLiteral(b, data_type, 0.0);
  xla::XlaOp two = FloatLiteral(b, data_type, 2.0);
  xla::XlaOp eight = FloatLiteral(b, data_type, 8.0);

  xla::XlaOp abs_x = b->Abs(x);
  xla::XlaOp z = b->Exp(b->Mul(b->Neg(x), x));

  xla::XlaOp pp = EvaluatePolynomial(b, abs_x, kErfcPCoefficient, data_type);
  xla::XlaOp pq = EvaluatePolynomial(b, abs_x, kErfcQCoefficient, data_type);
  xla::XlaOp pr = EvaluatePolynomial(b, abs_x, kErfcRCoefficient, data_type);
  xla::XlaOp ps = EvaluatePolynomial(b, abs_x, kErfcSCoefficient, data_type);

  xla::XlaOp y = b->Select(b->Lt(abs_x, eight), b->Div(b->Mul(z, pp), pq),
                           b->Div(b->Mul(z, pr), ps));

  return b->Select(b->Lt(x, zero), b->Sub(two, y), y);
}

// Compute a polynomial approximation of the error function.
xla::XlaOp ComputeErf(xla::XlaBuilder* b, const xla::XlaOp& x,
                      PrimitiveType data_type) {
  xla::XlaOp z = b->Mul(x, x);
  xla::XlaOp pt = EvaluatePolynomial(b, z, kErfTCoefficient, data_type);
  xla::XlaOp pu = EvaluatePolynomial(b, z, kErfUCoefficient, data_type);
  return b->Div(b->Mul(x, pt), pu);
}

}  // namespace xla
