/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/lib/math.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>

#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/loops.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"

namespace xla {
namespace {

// Evaluate the polynomial given `x` and coefficients in decreasing order.
template <typename FP>
XlaOp EvaluatePolynomial(XlaOp x, absl::Span<const FP> coefficients) {
  static_assert(std::is_floating_point<FP>::value,
                "Template-argument 'FP' must be a floating-point type");
  if (coefficients.empty()) {
    return ScalarLike(x, FP(0.0));
  }
  XlaOp poly = ScalarLike(x, coefficients[0]);
  for (int i = 1; i < coefficients.size(); ++i) {
    FP c = coefficients[i];
    poly = poly * x + ScalarLike(x, c);
  }
  return poly;
}

// Evaluate the chebyshev polynomial given `x` and coefficients in decreasing
// order.
template <typename FP>
XlaOp EvaluateChebyshevPolynomial(XlaOp x, absl::Span<const FP> coefficients) {
  static_assert(std::is_floating_point<FP>::value,
                "Template-argument 'FP' must be a floating-point type");
  XlaOp b0 = ScalarLike(x, 0.0);
  XlaOp b1 = ScalarLike(x, 0.0);
  XlaOp b2 = ScalarLike(x, 0.0);
  for (FP c : coefficients) {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + ScalarLike(x, c);
  }
  return ScalarLike(x, 0.5) * (b0 - b2);
}

}  // namespace

// Returns operation(operand), except if `operand` is one of the types in
// upcast_types, in which case first converts it to F32, and then converts the
// result down to the original type.
static XlaOp DoWithUpcastToF32(XlaOp operand,
                               absl::Span<const PrimitiveType> upcast_types,
                               const std::function<XlaOp(XlaOp)>& operation) {
  auto& b = *operand.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, b.GetShape(operand));
    PrimitiveType elem_ty = shape.element_type();
    bool needs_upcast = absl::c_linear_search(upcast_types, elem_ty);

    if (needs_upcast) {
      operand = ConvertElementType(operand, F32);
    }
    XlaOp result = operation(operand);
    if (needs_upcast) {
      result = ConvertElementType(result, elem_ty);
    }
    return result;
  });
}

// TODO(jlebar): Use this function in more places in this file to restrict the
// domain of other functions.
static Status EnsureOperandIsRealFp(absl::string_view op_name, XlaOp operand) {
  auto& b = *operand.builder();
  TF_ASSIGN_OR_RETURN(auto shape, b.GetShape(operand));
  auto elem_ty = shape.element_type();
  if (!primitive_util::IsFloatingPointType(elem_ty)) {
    return InvalidArgument(
        "Operands to %s must be real-valued floating-point, but got %s",
        op_name, PrimitiveType_Name(elem_ty));
  }
  return OkStatus();
}

XlaOp IsPosInf(XlaOp operand) {
  auto& b = *operand.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("IsPosInf", operand));
    TF_ASSIGN_OR_RETURN(auto shape, b.GetShape(operand));
    // Note that this is only correct for floating-point types.  If we wanted it
    // to be correct for all types, we'd need to Gt(MaxFiniteValue).
    return Eq(operand, MaxValue(&b, shape.element_type()));
  });
}

XlaOp IsNegInf(XlaOp operand) {
  auto& b = *operand.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("IsNegInf", operand));
    TF_ASSIGN_OR_RETURN(auto shape, b.GetShape(operand));
    // Note that this is only correct for floating-point types.  If we wanted it
    // to be correct for all types, we'd need to Lt(MinFiniteValue).
    return Eq(operand, MinValue(&b, shape.element_type()));
  });
}

XlaOp IsInf(XlaOp operand) {
  auto& b = *operand.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("IsInf", operand));
    return IsPosInf(Abs(operand));
  });
}

XlaOp IsNan(XlaOp operand) {
  auto& b = *operand.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("IsNan", operand));
    return Ne(operand, operand);
  });
}

XlaOp IsNegZero(XlaOp operand) {
  auto& b = *operand.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("IsNegZero", operand));
    TF_ASSIGN_OR_RETURN(auto shape, b.GetShape(operand));

    // The bitwise representation of -0 in bfloat16 and IEEE 754 is 0x80...0
    // (sign bit on, all other bits off).
    switch (shape.element_type()) {
      case F64:
        return Eq(BitcastConvertType(operand, U64),
                  ConstantR0WithType(&b, U64, uint64_t{1} << 63));
      case F32:
        return Eq(BitcastConvertType(operand, U32),
                  ConstantR0WithType(&b, U32, uint32_t{1} << 31));
      case F8E5M2:
      case F8E4M3FN:
      case F8E4M3B11FNUZ:
      case F16:
      case BF16:
        // Not all XLA backends handle U16 well, so we convert to F32/U32.
        // TODO(jlebar): It would be nice if we could stay in (B)F16/U16 for
        // backends that *do* support it.
        return Eq(BitcastConvertType(ConvertElementType(operand, F32), U32),
                  ConstantR0WithType(&b, U32, uint32_t{1} << 31));
      default:
        LOG(FATAL) << "Expected real fp type.";
    }
  });
}

XlaOp Square(XlaOp operand) { return operand * operand; }

XlaOp Reciprocal(XlaOp operand) { return ScalarLike(operand, 1.0) / operand; }

// Computes an approximation of the error function complement (1 - erf(x)).
//
// Precondition: abs(x) >= 1.  Otherwise, use ErfImpl.
//
// This follows Cephes's f32 implementation of erfc.
static XlaOp ErfcImpl32(XlaOp x) {
  // Coefficients for erfc(f32), from Cephes.
  const double kMaxlog = 88.72283905206835;
  // erfc(x) = exp(-x^2) P(1/x^2), 1 < x < 2
  static const std::array<float, 9> kErfcPCoefficient{
      +2.326819970068386E-2, -1.387039388740657E-1, +3.687424674597105E-1,
      -5.824733027278666E-1, +6.210004621745983E-1, -4.944515323274145E-1,
      +3.404879937665872E-1, -2.741127028184656E-1, +5.638259427386472E-1,
  };
  // erfc(x) = exp(-x^2) R(1/x^2), 2 <= x < kMaxlog
  static const std::array<float, 8> kErfcRCoefficient{
      -1.047766399936249E+1, +1.297719955372516E+1, -7.495518717768503E+0,
      +2.921019019210786E+0, -1.015265279202700E+0, +4.218463358204948E-1,
      -2.820767439740514E-1, +5.641895067754075E-1,
  };
  XlaOp abs_x = Abs(x);
  XlaOp z = Exp(-x * x);
  XlaOp q = ScalarLike(x, 1) / abs_x;
  XlaOp y = q * q;
  XlaOp p = Select(Lt(abs_x, ScalarLike(x, 2.0)),
                   EvaluatePolynomial<float>(y, kErfcPCoefficient),
                   EvaluatePolynomial<float>(y, kErfcRCoefficient));
  y = z * q * p;
  XlaOp y_clamp = Select(Lt(z, ScalarLike(x, -kMaxlog)), ScalarLike(x, 0), y);
  return Select(Lt(x, ScalarLike(x, 0)), ScalarLike(x, 2.0) - y_clamp, y_clamp);
}

// Compute a polynomial approximation of the error function.
//
// Precondition: abs(x) <= 1.  Otherwise, use ErfcImpl.
//
// This follows Cephes's f32 implementation of erf.
static XlaOp ErfImpl32Cephes(XlaOp x) {
  // Coefficients for by erf(f32), from Cephes.
  //
  // erf(x) = x P(x^2), 0 < x < 1
  static const std::array<float, 7> kErfTCoefficient{
      +7.853861353153693E-5, -8.010193625184903E-4, +5.188327685732524E-3,
      -2.685381193529856E-2, +1.128358514861418E-1, -3.761262582423300E-1,
      +1.128379165726710E+0,
  };
  return x * EvaluatePolynomial<float>(x * x, kErfTCoefficient);
}

static XlaOp ErfcImpl64(XlaOp x) {
  // Coefficients for erfc(f64), from Cephes.
  const double kMaxlog = 7.09782712893383996843E2;
  // erfc(x) = exp(-x^2) P(|x|) / Q(|x|), 1 < x < 8
  static const std::array<double, 9> kErfcPCoefficient{
      2.46196981473530512524E-10, 5.64189564831068821977E-1,
      7.46321056442269912687E0,   4.86371970985681366614E1,
      1.96520832956077098242E2,   5.26445194995477358631E2,
      9.34528527171957607540E2,   1.02755188689515710272E3,
      5.57535335369399327526E2};
  static const std::array<double, 9> kErfcQCoefficient{
      1.00000000000000000000E0, 1.32281951154744992508E1,
      8.67072140885989742329E1, 3.54937778887819891062E2,
      9.75708501743205489753E2, 1.82390916687909736289E3,
      2.24633760818710981792E3, 1.65666309194161350182E3,
      5.57535340817727675546E2};

  // erfc(x) = exp(-x^2) R(|x|) / S(|x|), 8 <= x < kMaxlog
  static const std::array<double, 6> kErfcRCoefficient{
      5.64189583547755073984E-1, 1.27536670759978104416E0,
      5.01905042251180477414E0,  6.16021097993053585195E0,
      7.40974269950448939160E0,  2.97886665372100240670E0};
  static const std::array<double, 7> kErfcSCoefficient{
      1.00000000000000000000E0, 2.26052863220117276590E0,
      9.39603524938001434673E0, 1.20489539808096656605E1,
      1.70814450747565897222E1, 9.60896809063285878198E0,
      3.36907645100081516050E0};

  XlaOp z = -x * x;
  XlaOp abs_x = Abs(x);
  XlaOp y =
      Select(Lt(abs_x, ScalarLike(x, 8.0)),
             Exp(z) * EvaluatePolynomial<double>(abs_x, kErfcPCoefficient) /
                 EvaluatePolynomial<double>(abs_x, kErfcQCoefficient),
             Exp(z) * EvaluatePolynomial<double>(abs_x, kErfcRCoefficient) /
                 EvaluatePolynomial<double>(abs_x, kErfcSCoefficient));
  XlaOp y_clamp = Select(Lt(z, ScalarLike(x, -kMaxlog)), ScalarLike(x, 0), y);
  return Select(Lt(x, ScalarLike(x, 0)), ScalarLike(x, 2.0) - y_clamp, y_clamp);
}

// Compute a polynomial approximation of the error function.
//
// Precondition: abs(x) <= 1.  Otherwise, use ErfcImpl.
static XlaOp ErfImpl64(XlaOp x) {
  // Coefficients for by erf(f64), from Cephes.
  //
  // erf(x) = x T(x^2) / U(x^2), 0 < x < 1
  static std::array<double, 5> kErfTCoefficient{
      9.60497373987051638749E0, 9.00260197203842689217E1,
      2.23200534594684319226E3, 7.00332514112805075473E3,
      5.55923013010394962768E4};
  static std::array<double, 6> kErfUCoefficient{
      1.00000000000000000000E0, 3.35617141647503099647E1,
      5.21357949780152679795E2, 4.59432382970980127987E3,
      2.26290000613890934246E4, 4.92673942608635921086E4};
  XlaOp z = x * x;
  return x * EvaluatePolynomial<double>(z, kErfTCoefficient) /
         EvaluatePolynomial<double>(z, kErfUCoefficient);
}

XlaOp Erfc(XlaOp x) {
  auto& b = *x.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("Erfc", x));
    TF_ASSIGN_OR_RETURN(auto shape, b.GetShape(x));
    // erfc(x) =
    //   erfc_impl(x)           if x > 1
    //   1 - erf_impl(x)        otherwise
    if (shape.element_type() == F64) {
      return Select(Gt(Abs(x), ScalarLike(x, 1)), ErfcImpl64(x),
                    ScalarLike(x, 1) - ErfImpl64(x));
    }
    // Erf(c)Impl don't have enough precision when run with bf16 intermediates
    // (not surprising!), so upcast to f32 in this case.
    return DoWithUpcastToF32(
        x, {BF16, F16, F8E5M2, F8E4M3FN, F8E4M3B11FNUZ}, [](XlaOp x) {
          return Select(Gt(Abs(x), ScalarLike(x, 1)), ErfcImpl32(x),
                        ScalarLike(x, 1) - ErfImpl32Cephes(x));
        });
  });
}

// Compute a rational approximation of the error function.
static XlaOp ErfImpl32(XlaOp x) {
  static const std::array<float, 5> kAlpha{
      0.00022905065861350646f, 0.0034082910107109506f, 0.050955695062380861f,
      0.18520832239976145f, 1.128379143519084f};

  static const std::array<float, 7> kBeta{-1.1791602954361697e-7,
                                          0.000023547966471313185f,
                                          0.0010179625278914885f,
                                          0.014070470171167667f,
                                          0.11098505178285362f,
                                          0.49746925110067538f,
                                          1.0f};

  // We clamp x to be within [-c;c] where c = erfinv(1-2^-23), outside of
  // which x should be +/-1.
  constexpr float kErfInvOneMinusHalfULP = 3.7439211627767994f;
  x = Clamp(ScalarLike(x, -kErfInvOneMinusHalfULP), x,
            ScalarLike(x, kErfInvOneMinusHalfULP));
  auto x2 = x * x;
  return (x * EvaluatePolynomial<float>(x2, kAlpha)) /
         EvaluatePolynomial<float>(x2, kBeta);
}

XlaOp Erf(XlaOp x) {
  auto& b = *x.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("Erf", x));
    TF_ASSIGN_OR_RETURN(auto shape, b.GetShape(x));
    // erf(x) =
    //   erf_impl(x)            if x < 1
    //   1 - erfc_impl(x)       otherwise
    if (shape.element_type() == F64) {
      return Select(Lt(Abs(x), ScalarLike(x, 1)), ErfImpl64(x),
                    ScalarLike(x, 1) - ErfcImpl64(x));
    }
    // Erf(c)Impl don't have enough precision when run with bf16 intermediates
    // (not surprising!), so upcast to f32 in this case.
    return DoWithUpcastToF32(x, {BF16, F16, F8E5M2, F8E4M3FN, F8E4M3B11FNUZ},
                             [](XlaOp x) { return ErfImpl32(x); });
  });
}

namespace {

// Approximation for the inverse error function from
//   Giles, M., "Approximating the erfinv function".
// The approximation has the form:
//   w = -log((1 - x) * (1 + x))
//   if ( w < 5 ) {
//     w = w - 2.5
//     p = sum_{i=1}^n lq[i]*w^i
//   } else {
//     w = sqrt(w) - 3
//     p = sum_{i=1}^n gq[i]*w^i
//   }
//   return p*x
XlaOp ErfInv32(XlaOp x) {
  constexpr int kDegree = 9;
  constexpr std::array<float, 9> w_less_than_5_constants = {
      2.81022636e-08f,  3.43273939e-07f, -3.5233877e-06f,
      -4.39150654e-06f, 0.00021858087f,  -0.00125372503f,
      -0.00417768164f,  0.246640727f,    1.50140941f};
  constexpr std::array<float, 9> w_greater_than_5_constants = {
      -0.000200214257f, 0.000100950558f, 0.00134934322f,
      -0.00367342844f,  0.00573950773f,  -0.0076224613f,
      0.00943887047f,   1.00167406f,     2.83297682f};

  // Compute logarithm of (1+arg) using log1p(arg) which is more precise than
  // log(1+arg) when arg is close to zero. For more details, see
  // https://en.cppreference.com/w/cpp/numeric/math/log1p
  auto w = -Log1p(-x * x);

  auto lt = Lt(w, ScalarLike(x, 5.0));
  auto coefficient = [&](int i) {
    return Select(lt, FullLike(x, w_less_than_5_constants[i]),
                  FullLike(x, w_greater_than_5_constants[i]));
  };
  w = Select(lt, w - ScalarLike(x, 2.5), Sqrt(w) - ScalarLike(x, 3.0));
  auto p = coefficient(0);
  for (int i = 1; i < kDegree; ++i) {
    p = coefficient(i) + p * w;
  }

  // Result modulo edge cases.
  XlaOp result = p * x;

  // Handle edge cases, namely erfinv(+/-1) = +/-inf.  (The above computation is
  // indeterminate, and can give nan or -/+inf.)
  auto& b = *x.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, b.GetShape(x));
    return Select(Eq(Abs(x), ScalarLike(x, 1)),
                  x * MaxValue(&b, shape.element_type()), result);
  });
}

XlaOp ErfInv64(XlaOp x) {
  constexpr std::array<double, 23> w_less_than_6_25_constants = {
      -3.6444120640178196996e-21, -1.685059138182016589e-19,
      1.2858480715256400167e-18,  1.115787767802518096e-17,
      -1.333171662854620906e-16,  2.0972767875968561637e-17,
      6.6376381343583238325e-15,  -4.0545662729752068639e-14,
      -8.1519341976054721522e-14, 2.6335093153082322977e-12,
      -1.2975133253453532498e-11, -5.4154120542946279317e-11,
      1.051212273321532285e-09,   -4.1126339803469836976e-09,
      -2.9070369957882005086e-08, 4.2347877827932403518e-07,
      -1.3654692000834678645e-06, -1.3882523362786468719e-05,
      0.0001867342080340571352,   -0.00074070253416626697512,
      -0.0060336708714301490533,  0.24015818242558961693,
      1.6536545626831027356};
  constexpr std::array<double, 19> w_less_than_16_constants = {
      2.2137376921775787049e-09,  9.0756561938885390979e-08,
      -2.7517406297064545428e-07, 1.8239629214389227755e-08,
      1.5027403968909827627e-06,  -4.013867526981545969e-06,
      2.9234449089955446044e-06,  1.2475304481671778723e-05,
      -4.7318229009055733981e-05, 6.8284851459573175448e-05,
      2.4031110387097893999e-05,  -0.0003550375203628474796,
      0.00095328937973738049703,  -0.0016882755560235047313,
      0.0024914420961078508066,   -0.0037512085075692412107,
      0.005370914553590063617,    1.0052589676941592334,
      3.0838856104922207635,
  };
  constexpr std::array<double, 17> w_greater_than_16_constants = {
      -2.7109920616438573243e-11, -2.5556418169965252055e-10,
      1.5076572693500548083e-09,  -3.7894654401267369937e-09,
      7.6157012080783393804e-09,  -1.4960026627149240478e-08,
      2.9147953450901080826e-08,  -6.7711997758452339498e-08,
      2.2900482228026654717e-07,  -9.9298272942317002539e-07,
      4.5260625972231537039e-06,  -1.9681778105531670567e-05,
      7.5995277030017761139e-05,  -0.00021503011930044477347,
      -0.00013871931833623122026, 1.0103004648645343977,
      4.8499064014085844221,
  };
  // Compute logarithm of (1+arg) using log1p(arg) which is more precise than
  // log(1+arg) when arg is close to zero. For more details, see
  // https://en.cppreference.com/w/cpp/numeric/math/log1p
  auto w = -Log1p(-x * x);

  auto lt_6_25 = Lt(w, ScalarLike(x, 6.25));
  auto lt_16 = Lt(w, ScalarLike(x, 16));
  auto coefficient = [&](int i) {
    auto c = FullLike(x, w_less_than_6_25_constants[i]);
    if (i < 19) {
      c = Select(lt_6_25, c, FullLike(x, w_less_than_16_constants[i]));
    }
    if (i < 17) {
      c = Select(lt_16, c, FullLike(x, w_greater_than_16_constants[i]));
    }
    return c;
  };
  auto sqrt_w = Sqrt(w);
  w = Select(lt_6_25, w - ScalarLike(x, 3.125),
             sqrt_w - Select(lt_16, ScalarLike(x, 3.25), ScalarLike(x, 5.0)));
  auto p = coefficient(0);
  for (int i = 1; i < 17; ++i) {
    p = coefficient(i) + p * w;
  }
  for (int i = 17; i < 19; ++i) {
    p = Select(lt_16, coefficient(i) + p * w, p);
  }
  for (int i = 19; i < 23; ++i) {
    p = Select(lt_6_25, coefficient(i) + p * w, p);
  }
  // Result modulo edge cases.
  XlaOp result = p * x;

  // Handle edge cases, namely erfinv(+/-1) = +/-inf.  (The above computation is
  // indeterminate, and can give nan or -/+inf.)
  auto& b = *x.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, b.GetShape(x));
    return Select(Eq(Abs(x), ScalarLike(x, 1)),
                  x * MaxValue(&b, shape.element_type()), result);
  });
}

}  // namespace

XlaOp ErfInv(XlaOp x) {
  auto& b = *x.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("ErfInv", x));
    TF_ASSIGN_OR_RETURN(auto shape, b.GetShape(x));
    if (shape.element_type() == F64) {
      return ErfInv64(x);
    }
    return DoWithUpcastToF32(x, {BF16, F16, F8E5M2, F8E4M3FN, F8E4M3B11FNUZ},
                             [](XlaOp x) { return ErfInv32(x); });
  });
}

namespace {
// Coefficients for the Lanczos approximation of the gamma function. The
// coefficients are uniquely determined by the choice of g and n (kLanczosGamma
// and kLanczosCoefficients.size() + 1). The coefficients below correspond to
// [7, 9]. [5, 7], [7, 9], [9, 10], and [607/128.0, 15] were evaluated and [7,
// 9] seemed to be the least sensitive to the quality of the log function. In
// particular, [5, 7] is the only choice where -1.5e-5 <= lgamma(2) <= 1.5e-5
// for a particularly inaccurate log function.
static constexpr double kLanczosGamma = 7;  // aka g
static constexpr double kBaseLanczosCoeff = 0.99999999999980993227684700473478;
static constexpr std::array<double, 8> kLanczosCoefficients = {
    676.520368121885098567009190444019, -1259.13921672240287047156078755283,
    771.3234287776530788486528258894,   -176.61502916214059906584551354,
    12.507343278686904814458936853,     -0.13857109526572011689554707,
    9.984369578019570859563e-6,         1.50563273514931155834e-7};
}  // namespace

// Compute the Lgamma function using Lanczos' approximation from "A Precision
// Approximation of the Gamma Function". SIAM Journal on Numerical Analysis
// series B. Vol. 1:
// lgamma(z + 1) = (log(2) + log(pi)) / 2 + (z + 1/2) * log(t(z)) - t(z) + A(z)
// t(z) = z + kLanczosGamma + 1/2
// A(z) = kBaseLanczosCoeff + sigma(k = 1, n, kLanczosCoefficients[i] / (z + k))
XlaOp Lgamma(XlaOp input) {
  auto do_it = [](XlaOp input) {
    XlaOp one_half = ScalarLike(input, 0.5);
    XlaOp one = ScalarLike(input, 1);

    XlaOp pi = ScalarLike(input, M_PI);
    XlaOp log_pi = ScalarLike(input, std::log(M_PI));
    XlaOp log_sqrt_two_pi =
        ScalarLike(input, (std::log(2) + std::log(M_PI)) / 2);

    XlaOp lanczos_gamma_plus_one_half = ScalarLike(input, kLanczosGamma + 0.5);
    XlaOp log_lanczos_gamma_plus_one_half =
        ScalarLike(input, std::log(kLanczosGamma + 0.5));

    XlaOp base_lanczos_coeff = ScalarLike(input, kBaseLanczosCoeff);

    // If the input is less than 0.5 use Euler's reflection formula:
    // gamma(x) = pi / (sin(pi * x) * gamma(1 - x))
    XlaOp need_to_reflect = Lt(input, one_half);
    XlaOp z = Select(need_to_reflect, -input, input - one);

    XlaOp x = base_lanczos_coeff;
    for (int i = 0, end = kLanczosCoefficients.size(); i < end; ++i) {
      XlaOp lanczos_coefficient = ScalarLike(input, kLanczosCoefficients[i]);
      XlaOp index = ScalarLike(input, i);
      x = x + lanczos_coefficient / (z + index + one);
    }

    // To improve accuracy on platforms with less-precise log implementations,
    // compute log(lanczos_gamma_plus_one_half) at compile time and use log1p on
    // the device.
    // log(t) = log(kLanczosGamma + 0.5 + z)
    //        = log(kLanczosGamma + 0.5) + log1p(z / (kLanczosGamma + 0.5))
    XlaOp t = lanczos_gamma_plus_one_half + z;
    XlaOp log_t = log_lanczos_gamma_plus_one_half +
                  Log1p(z / lanczos_gamma_plus_one_half);

    // Compute the final result (modulo reflection).  t(z) may be large, and we
    // need to be careful not to overflow to infinity in the first term of
    //
    //   (z + 1/2) * log(t(z)) - t(z).
    //
    // Therefore we compute this as
    //
    //   (z + 1/2 - t(z) / log(t(z))) * log(t(z)).
    //
    XlaOp log_y = log_sqrt_two_pi + (z + one_half - t / log_t) * log_t + Log(x);

    // Compute the reflected value, used when x < 0.5:
    //
    //   lgamma(x) = log(pi) - lgamma(1-x) - log(abs(sin(pi * x))).
    //
    // (The abs is because lgamma is the log of the absolute value of the gamma
    // function.)
    //
    // We have to be careful when computing the final term above. gamma(x) goes
    // to +/-inf at every integer x < 0, and this is controlled by the
    // sin(pi * x) term.  The slope is large, so precision is particularly
    // important.
    //
    // Because abs(sin(pi * x)) has period 1, we can equivalently use
    // abs(sin(pi * frac(x))), where frac(x) is the fractional part of x.  This
    // is more numerically accurate: It doesn't overflow to inf like pi * x can,
    // and if x is an integer, it evaluates to 0 exactly, which is significant
    // because we then take the log of this value, and log(0) is inf.
    //
    // We don't have a frac(x) primitive in XLA and computing it is tricky, but
    // because abs(sin(pi * x)) = abs(sin(pi * abs(x))), it's good enough for
    // our purposes to use abs(frac(x)) = abs(x) - floor(abs(x)).
    //
    // Furthermore, pi * abs(frac(x)) loses precision when abs(frac(x)) is close
    // to 1.  To remedy this, we can use the fact that sin(pi * x) in the domain
    // [0, 1] is symmetric across the line Y=0.5.
    //
    XlaOp abs_input = Abs(input);
    XlaOp abs_frac_input = abs_input - Floor(abs_input);
    // Convert values of abs_frac_input > 0.5 to (1 - frac_input) to improve
    // precision of pi * abs_frac_input for values of abs_frac_input close to 1.
    XlaOp reduced_frac_input =
        Select(Gt(abs_frac_input, ScalarLike(abs_frac_input, 0.5)),
               ScalarLike(abs_frac_input, 1) - abs_frac_input, abs_frac_input);
    XlaOp reflection_denom = Log(Sin(pi * reduced_frac_input));

    // Avoid computing -inf - inf, which is nan.  If reflection_denom is +/-inf,
    // then it "wins" and the result is +/-inf.
    XlaOp reflection =
        Select(IsFinite(reflection_denom), log_pi - reflection_denom - log_y,
               -reflection_denom);
    XlaOp result = Select(need_to_reflect, reflection, log_y);

    // lgamma(+/-inf) = +inf.
    XlaOp inf_bcast = FullLike(input, std::numeric_limits<float>::infinity());
    return Select(IsInf(input), inf_bcast, result);
  };

  auto& b = *input.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("Lgamma", input));
    // F16 and BF16 don't provide sufficient precision for intermediate results
    // here (although it's better than you might expect!), so do the
    // computations in F32.
    return DoWithUpcastToF32(
        input, {BF16, F16, F8E5M2, F8E4M3FN, F8E4M3B11FNUZ}, do_it);
  });
}

// Computes an approximation of the lbeta function which is equivalent to
// log(abs(Beta(a, b))) but avoids overflow by computing it with lgamma.
static XlaOp Lbeta(XlaOp a, XlaOp b) {
  // Beta(a, b) can be computed using Gamma as per
  // http://dlmf.nist.gov/5.12.E1 as follows:
  //   Beta(a, b) = (Gamma(a) * Gamma(b)) / Gamma(a + b)
  //
  // To avoid overflow, we compute in the log domain.
  //
  // As per http://dlmf.nist.gov/4.8.E2 we can transform:
  //   Log(a * b)
  // into:
  //   Log(a) + Log(b)
  //
  // Likewise, per https://dlmf.nist.gov/4.8.E4, we can turn:
  //   Log(a - b)
  // into:
  //   Log(a) - Log(b)
  //
  // This means that we can compute Log(Beta(a, b)) by:
  //   Log(Gamma(a)) + Log(Gamma(b)) - Log(Gamma(a + b))
  return Lgamma(a) + Lgamma(b) - Lgamma(a + b);
}

// Compute the Digamma function using Lanczos' approximation from "A Precision
// Approximation of the Gamma Function". SIAM Journal on Numerical Analysis
// series B. Vol. 1:
// digamma(z + 1) = log(t(z)) + A'(z) / A(z) - kLanczosGamma / t(z)
// t(z) = z + kLanczosGamma + 1/2
// A(z) = kBaseLanczosCoeff + sigma(k = 1, n, kLanczosCoefficients[i] / (z + k))
// A'(z) = sigma(k = 1, n, kLanczosCoefficients[i] / (z + k) / (z + k))
XlaOp Digamma(XlaOp input) {
  auto do_it = [](XlaOp input) {
    XlaOp zero = ScalarLike(input, 0);
    XlaOp one_half = ScalarLike(input, 0.5);
    XlaOp one = ScalarLike(input, 1);

    XlaOp pi = ScalarLike(input, M_PI);

    XlaOp lanczos_gamma = ScalarLike(input, kLanczosGamma);
    XlaOp lanczos_gamma_plus_one_half = ScalarLike(input, kLanczosGamma + 0.5);
    XlaOp log_lanczos_gamma_plus_one_half =
        ScalarLike(input, std::log(kLanczosGamma + 0.5));

    XlaOp base_lanczos_coeff = ScalarLike(input, kBaseLanczosCoeff);

    // If the input is less than 0.5 use Euler's reflection formula:
    // digamma(x) = digamma(1 - x) - pi * cot(pi * x)
    XlaOp need_to_reflect = Lt(input, one_half);
    XlaOp z = Select(need_to_reflect, -input, input - one);

    XlaOp num = zero;
    XlaOp denom = base_lanczos_coeff;
    for (int i = 0, end = kLanczosCoefficients.size(); i < end; ++i) {
      XlaOp lanczos_coefficient = ScalarLike(input, kLanczosCoefficients[i]);
      XlaOp index = ScalarLike(input, i);
      num = num - lanczos_coefficient / ((z + index + one) * (z + index + one));
      denom = denom + lanczos_coefficient / (z + index + one);
    }

    // To improve accuracy on platforms with less-precise log implementations,
    // compute log(lanczos_gamma_plus_one_half) at compile time and use log1p on
    // the device.
    // log(t) = log(kLanczosGamma + 0.5 + z)
    //        = log(kLanczosGamma + 0.5) + log1p(z / (kLanczosGamma + 0.5))
    XlaOp t = lanczos_gamma_plus_one_half + z;
    XlaOp log_t = log_lanczos_gamma_plus_one_half +
                  Log1p(z / lanczos_gamma_plus_one_half);

    XlaOp y = log_t + num / denom - lanczos_gamma / t;

    // We need to be careful how we compute cot(pi * input) below: For
    // near-integral values of `input`, pi * input can lose precision.
    //
    // Input is already known to be less than 0.5 (otherwise we don't have to
    // reflect).  We shift values smaller than -0.5 into the range [-.5, .5] to
    // increase precision of pi * input and the resulting cotangent.
    XlaOp reduced_input = input + Abs(Floor(input + ScalarLike(input, 0.5)));
    XlaOp reflection =
        y - pi * Cos(pi * reduced_input) / Sin(pi * reduced_input);
    XlaOp real_result = Select(need_to_reflect, reflection, y);

    // Digamma has poles at negative integers and zero; return nan for those.
    return Select(And(Le(input, zero), Eq(input, Floor(input))),
                  FullLike(input, std::numeric_limits<float>::quiet_NaN()),
                  real_result);
  };

  auto& b = *input.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("Digamma", input));
    return DoWithUpcastToF32(
        input, {BF16, F16, F8E5M2, F8E4M3FN, F8E4M3B11FNUZ}, do_it);
  });
}

// Incomplete gamma functions

namespace {

enum kIgammaMode { VALUE, DERIVATIVE, SAMPLE_DERIVATIVE };

// Helper function for computing Igamma using a power series.
template <kIgammaMode mode>
XlaOp IgammaSeries(XlaOp ax, XlaOp x, XlaOp a, XlaOp enabled,
                   xla::PrimitiveType type) {
  // vals: (enabled, r, c, ans, x)
  // 'enabled' is a predication mask that says for which elements we should
  // execute the loop body. Disabled elements have no effect in the loop body.
  // TODO(phawkins): in general this isn't an optimal implementation on any
  // backend. For example, on GPU, we should probably vectorize to the warp
  // size, and then run independent loops for each warp's worth of
  // data.
  auto cond = [&](absl::Span<const XlaOp> vals,
                  XlaBuilder* builder) -> StatusOr<XlaOp> {
    XlaOp enabled = vals[0];
    return Any(enabled);
  };
  auto body = [&](absl::Span<const XlaOp> vals,
                  XlaBuilder* builder) -> StatusOr<std::vector<XlaOp>> {
    XlaOp enabled = vals[0];
    XlaOp r = vals[1];
    XlaOp c = vals[2];
    XlaOp ans = vals[3];
    XlaOp x = vals[4];
    XlaOp dc_da = vals[5];
    XlaOp dans_da = vals[6];

    r = r + ScalarLike(r, 1);
    dc_da = dc_da * (x / r) + (ScalarLike(r, -1) * c * x) / (r * r);
    dans_da = dans_da + dc_da;
    c = c * (x / r);
    ans = ans + c;
    XlaOp conditional;
    if (mode == VALUE) {
      conditional = And(enabled, Gt(c / ans, Epsilon(builder, type)));
    } else {
      conditional =
          And(enabled, Gt(Abs(dc_da / dans_da), Epsilon(builder, type)));
    }

    return std::vector<XlaOp>{
        conditional,
        Select(enabled, r, vals[1]),
        Select(enabled, c, vals[2]),
        Select(enabled, ans, vals[3]),
        Select(enabled, x, vals[4]),
        Select(enabled, dc_da, vals[5]),
        Select(enabled, dans_da, vals[6]),
    };
  };
  auto& b = *ax.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    std::vector<XlaOp> vals = {
        enabled,        a, FullLike(a, 1), FullLike(a, 1), x, FullLike(a, 0),
        FullLike(a, 0),
    };

    TF_ASSIGN_OR_RETURN(vals, WhileLoopHelper(cond, body, vals, "igamma", &b));
    XlaOp ans = vals[3];
    XlaOp dans_da = vals[6];
    if (mode == VALUE) {
      return (ans * ax) / a;
    }

    XlaOp dlogax_da = Log(x) - Digamma(a + ScalarLike(a, 1));

    switch (mode) {
      case DERIVATIVE:
        return ax * (ans * dlogax_da + dans_da) / a;
      case SAMPLE_DERIVATIVE:
      default:
        return -(dans_da + ans * dlogax_da) * x / a;
    }
  });
}

// Helper function for computing Igammac using a continued fraction.
template <kIgammaMode mode>
XlaOp IgammacContinuedFraction(XlaOp ax, XlaOp x, XlaOp a, XlaOp enabled,
                               xla::PrimitiveType type) {
  // vals: enabled, ans, t, y, z, c, pkm1, qkm1, pkm2, qkm2
  auto cond = [&](absl::Span<const XlaOp> vals,
                  XlaBuilder* builder) -> StatusOr<XlaOp> {
    XlaOp enabled = vals[0];
    XlaOp c = vals[5];
    return And(Lt(c, ScalarLike(c, 2000)), Any(enabled));
  };
  auto body = [&](absl::Span<const XlaOp> vals,
                  XlaBuilder* builder) -> StatusOr<std::vector<XlaOp>> {
    XlaOp enabled = vals[0];
    XlaOp ans = vals[1];
    XlaOp t = vals[2];
    XlaOp y = vals[3];
    XlaOp z = vals[4];
    XlaOp c = vals[5];
    XlaOp pkm1 = vals[6];
    XlaOp qkm1 = vals[7];
    XlaOp pkm2 = vals[8];
    XlaOp qkm2 = vals[9];

    XlaOp dpkm2_da = vals[10];
    XlaOp dqkm2_da = vals[11];
    XlaOp dpkm1_da = vals[12];
    XlaOp dqkm1_da = vals[13];
    XlaOp dans_da = vals[14];

    c = c + ScalarLike(c, 1);
    y = y + ScalarLike(y, 1);
    z = z + ScalarLike(z, 2);
    XlaOp yc = y * c;
    XlaOp pk = pkm1 * z - pkm2 * yc;
    XlaOp qk = qkm1 * z - qkm2 * yc;
    XlaOp qk_is_nonzero = Ne(qk, ScalarLike(qk, 0));
    XlaOp r = pk / qk;

    t = Select(qk_is_nonzero, Abs((ans - r) / r), FullLike(t, 1));
    ans = Select(qk_is_nonzero, r, ans);

    XlaOp dpk_da = dpkm1_da * z - pkm1 - dpkm2_da * yc + pkm2 * c;
    XlaOp dqk_da = dqkm1_da * z - qkm1 - dqkm2_da * yc + qkm2 * c;
    XlaOp dans_da_new =
        Select(qk_is_nonzero, (dpk_da - ans * dqk_da) / qk, dans_da);
    XlaOp grad_conditional =
        Select(qk_is_nonzero, Abs(dans_da_new - dans_da), FullLike(dans_da, 1));

    pkm2 = pkm1;
    pkm1 = pk;
    qkm2 = qkm1;
    qkm1 = qk;

    dpkm2_da = dpkm1_da;
    dqkm2_da = dqkm1_da;
    dpkm1_da = dpk_da;
    dqkm1_da = dqk_da;

    XlaOp rescale = Gt(Abs(pk), Reciprocal(Epsilon(builder, type)));
    pkm2 = Select(rescale, pkm2 * Epsilon(builder, type), pkm2);
    pkm1 = Select(rescale, pkm1 * Epsilon(builder, type), pkm1);
    qkm2 = Select(rescale, qkm2 * Epsilon(builder, type), qkm2);
    qkm1 = Select(rescale, qkm1 * Epsilon(builder, type), qkm1);

    dpkm2_da = Select(rescale, dpkm2_da * Epsilon(builder, type), dpkm2_da);
    dqkm2_da = Select(rescale, dqkm2_da * Epsilon(builder, type), dqkm2_da);
    dpkm1_da = Select(rescale, dpkm1_da * Epsilon(builder, type), dpkm1_da);
    dqkm1_da = Select(rescale, dqkm1_da * Epsilon(builder, type), dqkm1_da);

    XlaOp conditional;
    if (mode == VALUE) {
      conditional = And(enabled, Gt(t, Epsilon(builder, type)));
    } else {
      conditional = And(enabled, Gt(grad_conditional, Epsilon(builder, type)));
    }

    return std::vector<XlaOp>{conditional,
                              Select(enabled, ans, vals[1]),
                              Select(enabled, t, vals[2]),
                              Select(enabled, y, vals[3]),
                              Select(enabled, z, vals[4]),
                              c,
                              Select(enabled, pkm1, vals[6]),
                              Select(enabled, qkm1, vals[7]),
                              Select(enabled, pkm2, vals[8]),
                              Select(enabled, qkm2, vals[9]),
                              Select(enabled, dpkm2_da, vals[10]),
                              Select(enabled, dqkm2_da, vals[11]),
                              Select(enabled, dpkm1_da, vals[12]),
                              Select(enabled, dqkm1_da, vals[13]),
                              Select(enabled, dans_da_new, vals[14])};
  };

  auto& b = *ax.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    XlaOp y = ScalarLike(a, 1) - a;
    XlaOp z = x + y + ScalarLike(x, 1);
    XlaOp c = ScalarLike(x, 0);
    XlaOp pkm2 = FullLike(x, 1);
    XlaOp qkm2 = x;
    XlaOp pkm1 = x + ScalarLike(x, 1);
    XlaOp qkm1 = z * x;
    XlaOp ans = pkm1 / qkm1;
    XlaOp t = FullLike(x, 1);
    XlaOp dpkm2_da = FullLike(x, 0);
    XlaOp dqkm2_da = FullLike(x, 0);
    XlaOp dpkm1_da = FullLike(x, 0);
    XlaOp dqkm1_da = -x;
    XlaOp dans_da = (dpkm1_da - ans * dqkm1_da) / qkm1;
    std::vector<XlaOp> vals = {enabled,  ans,      t,        y,        z,
                               c,        pkm1,     qkm1,     pkm2,     qkm2,
                               dpkm2_da, dqkm2_da, dpkm1_da, dqkm1_da, dans_da};

    TF_ASSIGN_OR_RETURN(vals, WhileLoopHelper(cond, body, vals, "igammac", &b));
    ans = vals[1];
    if (mode == VALUE) {
      return ans * ax;
    }

    dans_da = vals[14];
    XlaOp dlogax_da = Log(x) - Digamma(a);

    switch (mode) {
      case DERIVATIVE:
        return ax * (ans * dlogax_da + dans_da);
      case SAMPLE_DERIVATIVE:
      default:
        return -(dans_da + ans * dlogax_da) * x;
    }
  });
}

}  // namespace

XlaOp Igamma(XlaOp a, XlaOp x) {
  auto& b = *a.builder();
  auto doit = [&b](XlaOp a, XlaOp x, PrimitiveType type) -> XlaOp {
    XlaOp is_nan = Or(IsNan(a), IsNan(x));
    XlaOp x_is_zero = Eq(x, ScalarLike(x, 0));
    XlaOp x_is_infinity =
        Eq(x, ScalarLike(x, std::numeric_limits<float>::infinity()));
    XlaOp domain_error = Or(Lt(x, ScalarLike(x, 0)), Le(a, ScalarLike(a, 0)));
    XlaOp use_igammac = And(Gt(x, ScalarLike(x, 1)), Gt(x, a));
    XlaOp ax = a * Log(x) - x - Lgamma(a);
    XlaOp underflow = Lt(ax, -Log(MaxFiniteValue(&b, type)));
    ax = Exp(ax);
    XlaOp enabled = Not(Or(Or(Or(x_is_zero, domain_error), underflow), is_nan));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    XlaOp output = Select(
        use_igammac,
        ScalarLike(a, 1) - IgammacContinuedFraction<VALUE>(
                               ax, x, a, And(enabled, use_igammac), type),
        IgammaSeries<VALUE>(ax, x, a, And(enabled, Not(use_igammac)), type));
    output = Select(x_is_zero, ZerosLike(output), output);
    output = Select(x_is_infinity, FullLike(output, 1), output);
    output = Select(Or(domain_error, is_nan), FullLike(a, nan), output);
    return output;
  };
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto a_shape, b.GetShape(a));
    TF_ASSIGN_OR_RETURN(auto x_shape, b.GetShape(x));
    if (a_shape != x_shape) {
      return InvalidArgument(
          "Arguments to Igamma must have equal shapes and types; got %s and %s",
          a_shape.ToString(), x_shape.ToString());
    }
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("Igamma", a));
    PrimitiveType a_x_type = a_shape.element_type();
    bool needs_upcast = false;
    for (PrimitiveType type : {BF16, F16, F8E5M2, F8E4M3FN, F8E4M3B11FNUZ}) {
      if (a_shape.element_type() == type) {
        needs_upcast = true;
        break;
      }
    }

    if (needs_upcast) {
      a = ConvertElementType(a, F32);
      x = ConvertElementType(x, F32);
      a_x_type = F32;
    }
    XlaOp result = doit(a, x, a_x_type);
    if (needs_upcast) {
      result = ConvertElementType(result, a_shape.element_type());
    }
    return result;
  });
}

XlaOp IgammaGradA(XlaOp a, XlaOp x) {
  auto& b = *a.builder();
  auto doit = [&b](XlaOp a, XlaOp x, PrimitiveType type) -> XlaOp {
    XlaOp is_nan = Or(IsNan(a), IsNan(x));
    XlaOp x_is_zero = Eq(x, ScalarLike(x, 0));
    XlaOp domain_error = Or(Lt(x, ScalarLike(x, 0)), Le(a, ScalarLike(a, 0)));
    XlaOp use_igammac = And(Gt(x, ScalarLike(x, 1)), Gt(x, a));
    XlaOp ax = a * Log(x) - x - Lgamma(a);
    XlaOp underflow = Lt(ax, -Log(MaxFiniteValue(&b, type)));
    ax = Exp(ax);
    XlaOp enabled = Not(Or(Or(Or(x_is_zero, domain_error), underflow), is_nan));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    XlaOp output = Select(use_igammac,
                          -IgammacContinuedFraction<DERIVATIVE>(
                              ax, x, a, And(enabled, use_igammac), type),
                          IgammaSeries<DERIVATIVE>(
                              ax, x, a, And(enabled, Not(use_igammac)), type));
    output = Select(x_is_zero, ZerosLike(output), output);
    output = Select(Or(domain_error, is_nan), FullLike(a, nan), output);
    return output;
  };
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto a_shape, b.GetShape(a));
    TF_ASSIGN_OR_RETURN(auto x_shape, b.GetShape(x));
    if (a_shape != x_shape) {
      return InvalidArgument(
          "Arguments to IgammaGradA must have equal shapes and types; got %s "
          "and %s",
          a_shape.ToString(), x_shape.ToString());
    }
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("IgammaGradA", a));
    bool needs_upcast = false;
    for (PrimitiveType type : {BF16, F16, F8E5M2, F8E4M3FN, F8E4M3B11FNUZ}) {
      if (a_shape.element_type() == type) {
        needs_upcast = true;
        break;
      }
    }

    if (needs_upcast) {
      a = ConvertElementType(a, F32);
      x = ConvertElementType(x, F32);
    }
    XlaOp result = doit(a, x, a_shape.element_type());
    if (needs_upcast) {
      result = ConvertElementType(result, a_shape.element_type());
    }
    return result;
  });
}

// Gradient of Gamma sample from Gamma(a, 1) with respect to `a`.
XlaOp RandomGammaGrad(XlaOp a, XlaOp x) {
  auto& b = *a.builder();
  auto doit = [&b](XlaOp a, XlaOp x, PrimitiveType type) -> XlaOp {
    XlaOp is_nan = Or(IsNan(a), IsNan(x));
    XlaOp x_is_zero = Eq(x, ScalarLike(x, 0));
    XlaOp domain_error = Or(Lt(x, ScalarLike(x, 0)), Le(a, ScalarLike(a, 0)));
    XlaOp use_igammac = And(Gt(x, ScalarLike(x, 1)), Gt(x, a));
    XlaOp ax = a * Log(x) - x - Lgamma(a);
    XlaOp underflow = Lt(ax, -Log(MaxFiniteValue(&b, type)));
    ax = Exp(ax);
    XlaOp enabled = Not(Or(Or(Or(x_is_zero, domain_error), underflow), is_nan));
    const double nan = std::numeric_limits<double>::quiet_NaN();
    XlaOp output = Select(use_igammac,
                          -IgammacContinuedFraction<SAMPLE_DERIVATIVE>(
                              ax, x, a, And(enabled, use_igammac), type),
                          IgammaSeries<SAMPLE_DERIVATIVE>(
                              ax, x, a, And(enabled, Not(use_igammac)), type));
    output = Select(x_is_zero, ZerosLike(output), output);
    output = Select(Or(domain_error, is_nan), FullLike(a, nan), output);
    return output;
  };
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto a_shape, b.GetShape(a));
    TF_ASSIGN_OR_RETURN(auto x_shape, b.GetShape(x));
    if (a_shape != x_shape) {
      return InvalidArgument(
          "Arguments to RandomGammaGrad must have equal shapes and types; got "
          "%s and %s",
          a_shape.ToString(), x_shape.ToString());
    }
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("RandomGammaGrad", a));
    bool needs_upcast =
        a_shape.element_type() == F16 || a_shape.element_type() == BF16;

    if (needs_upcast) {
      a = ConvertElementType(a, F32);
      x = ConvertElementType(x, F32);
    }
    XlaOp result = doit(a, x, a_shape.element_type());
    if (needs_upcast) {
      result = ConvertElementType(result, a_shape.element_type());
    }
    return result;
  });
}

XlaOp Igammac(XlaOp a, XlaOp x) {
  auto& b = *a.builder();
  auto doit = [&b](XlaOp a, XlaOp x, PrimitiveType type) -> XlaOp {
    XlaOp out_of_range = Or(Le(x, ScalarLike(x, 0)), Le(a, ScalarLike(a, 0)));
    XlaOp use_igamma = Or(Lt(x, ScalarLike(x, 1)), Lt(x, a));
    XlaOp ax = a * Log(x) - x - Lgamma(a);
    XlaOp underflow = Lt(ax, -Log(MaxFiniteValue(&b, type)));
    XlaOp enabled = Not(Or(out_of_range, underflow));
    ax = Exp(ax);
    XlaOp result =
        Select(use_igamma,
               ScalarLike(a, 1) - IgammaSeries<VALUE>(
                                      ax, x, a, And(enabled, use_igamma), type),
               IgammacContinuedFraction<VALUE>(
                   ax, x, a, And(enabled, Not(use_igamma)), type));
    XlaOp x_is_infinity =
        Eq(x, ScalarLike(x, std::numeric_limits<float>::infinity()));
    result = Select(x_is_infinity, ZerosLike(result), result);
    return Select(out_of_range, FullLike(a, 1), result);
  };
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto a_shape, b.GetShape(a));
    TF_ASSIGN_OR_RETURN(auto x_shape, b.GetShape(x));
    if (a_shape != x_shape) {
      return InvalidArgument(
          "Arguments to Igammac must have equal shapes and types; "
          "got %s and %s",
          a_shape.ToString(), x_shape.ToString());
    }
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("Igammac", a));
    PrimitiveType a_x_type = a_shape.element_type();
    bool needs_upcast =
        a_shape.element_type() == F16 || a_shape.element_type() == BF16;

    if (needs_upcast) {
      a = ConvertElementType(a, F32);
      x = ConvertElementType(x, F32);
      a_x_type = F32;
    }
    XlaOp result = doit(a, x, a_x_type);
    if (needs_upcast) {
      result = ConvertElementType(result, a_shape.element_type());
    }
    return result;
  });
}

// Implements Banker's rounding: numbers that are equidistant between two
// integers are rounded towards even.
XlaOp RoundToEven(XlaOp x) {
  auto& b = *x.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    // Reject non-real non-fp inputs (What does it even mean to round a complex
    // number?  Do you round each component equally?  In that case, you should
    // just ask for that explicitly.)
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("RoundToEven", x));

    return RoundNearestEven(x);
  });
}

// Trigonometric functions.

// acos(x) = 2 * atan(sqrt(1 - x^2) / (1 + x)) if x != -1
//           pi                                if x == -1
// For complex:
// acos(x) = -(i * log(x + i * sqrt((1 + x) * (1 - x))))
XlaOp Acos(XlaOp x) {
  XlaBuilder* b = x.builder();
  return b->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, b->GetShape(x));

    if (primitive_util::IsComplexType(shape.element_type())) {
      auto one = ScalarLike(x, 1);
      auto imag_one = Complex(
          Zero(b, primitive_util::ComplexComponentType(shape.element_type())),
          One(b, primitive_util::ComplexComponentType(shape.element_type())));

      auto result =
          Neg(imag_one * Log(x + imag_one * Sqrt((one + x) * (one - x))));
      return result;
    }
    return Select(Ne(x, FullLike(x, -1)),
                  ScalarLike(x, 2.0) * Atan2(Sqrt(ScalarLike(x, 1.0) - x * x),
                                             ScalarLike(x, 1.0) + x),
                  FullLike(x, M_PI));
  });
}

// asin(x) = 2 * atan(x / (1 + sqrt(1 - x^2)))
XlaOp Asin(XlaOp x) {
  return ScalarLike(x, 2.0) *
         Atan2(x, ScalarLike(x, 1.0) + Sqrt(ScalarLike(x, 1.0) - x * x));
}

XlaOp Atan(XlaOp x) { return Atan2(x, ScalarLike(x, 1.0)); }

// Hyperbolic trigonometric functions.

// acosh(x) = log(x + sqrt(x^2 - 1))      if x >= -1
//          = log(x + sqrt((x+1)*(x-1)))
// acosh(x) = nan                         if x < -1
//
// If x^2 will overflow, we approximate sqrt(x^2 - 1) == x and compute as
// log(2*x) = log(2) + log(x).  (Note this works because negative x never
// overflows; x < -1 simply yields nan.  This is quite different than asinh!)
XlaOp Acosh(XlaOp x) {
  XlaBuilder* b = x.builder();
  return b->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, b->GetShape(x));

    auto one = ScalarLike(x, 1);
    auto neg_one = ScalarLike(x, -1);
    auto nan = FullLike(x, std::numeric_limits<float>::quiet_NaN());

    // return
    //
    //   nan                        if x < -1
    //   log(x) + log(2)            if x >= sqrt_max_value
    //   log(x + sqrt((x+1)*(x-1))) otherwise
    //
    // TODO(jlebar): For now, we ignore the question of overflow if x is a
    // complex type, because we don't yet have exhaustive tests for complex trig
    // functions.
    auto naive_result = Log(x + Sqrt((x + one) * (x - one)));
    if (primitive_util::IsComplexType(shape.element_type())) {
      return naive_result;
    }
    auto overflow_result = Log(x) + Log(ScalarLike(x, 2));

    auto sqrt_max_value = Sqrt(MaxFiniteValue(b, shape.element_type()));
    return Select(Lt(x, neg_one), nan,
                  Select(Ge(x, sqrt_max_value), overflow_result, naive_result));
  });
}

// asinh(x) = log(x + sqrt(x^2 + 1))
//
// If x^2 will overflow and x is positive, we can approximate x + sqrt(x^2 + 1)
// as 2*x and return log(2) + log(x).
//
// If x is negative, the above would give us some trouble; we can't approximate
// the result as x + abs(x) = 0!  But we're saved by the fact that asinh(-x) =
// -asinh(x).
XlaOp Asinh(XlaOp x) {
  XlaBuilder* b = x.builder();
  auto do_it = [&](XlaOp x) -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, b->GetShape(x));
    auto one = ScalarLike(x, 1);

    // Let a = abs(x).  Compute
    //
    //   y = log(a + sqrt(a*a + 1))  if a < sqrt_max_value, or
    //   y = log(a) + log(2)         otherwise
    //
    // and then return
    //
    //   y * sign(x).
    //
    // TODO(jlebar): For now, we ignore the question of overflow if x is a
    // complex type, because we don't yet have exhaustive tests for complex trig
    // functions.
    if (primitive_util::IsComplexType(shape.element_type())) {
      return Log(x + Sqrt(x * x + one));
    }
    // For small x, sqrt(x**2 + 1) will evaluate to 1 due to floating point
    // arithmetic. However, we would like to retain the low order term of this,
    // which is around 0.5 * x**2 using a binomial expansion.
    // Let z = sqrt(a**2 + 1)
    // log(a + sqrt(a**2 + 1)) =
    // log((a + sqrt(a**2 + 1)) * (1 + sqrt(a**2 + 1)) / (1 + sqrt(a**2 + 1))) =
    // log((a + a**2 + 1 + a * z + z) / (1 + z)) =
    // log(1 + a + a**2 / (1 + z)) =
    // log(1 + a + a ** 2 / (1 + sqrt(a**2 + 1)))
    // This rewrite retains the lower order term.
    auto a = Abs(x);
    auto small_result = Log1p(a + a * a / (one + Sqrt(a * a + one)));
    auto naive_result = Log(a + Sqrt(a * a + one));
    auto overflow_result = Log(Abs(a)) + Log(ScalarLike(a, 2));
    auto sqrt_max_value = Sqrt(MaxFiniteValue(b, shape.element_type()));
    return Sign(x) * Select(Ge(a, sqrt_max_value), overflow_result,
                            Select(Le(a, one), small_result, naive_result));
  };
  // These upcasts are not strictly necessary on all platforms to get within our
  // error tolerances, so we could relax this if it ever mattered.
  return DoWithUpcastToF32(x, {BF16, F16}, [&](XlaOp x) {
    return b->ReportErrorOrReturn(do_it(x));
  });
}

// atanh(x) = 0.5 * log((1 + x) / (1 - x)) if abs(x) <= 1
// atanh(x) = nan                          otherwise
XlaOp Atanh(XlaOp x) {
  XlaBuilder* b = x.builder();
  auto do_it = [&](XlaOp x) -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, b->GetShape(x));
    auto naive_result = (Log1p(x) - Log1p(-x)) * ScalarLike(x, 0.5);

    // TODO(jlebar): For now, we ignore the nan edge case for complex inputs,
    // because we don't yet have exhaustive tests for complex trig functions.
    if (primitive_util::IsComplexType(shape.element_type())) {
      return naive_result;
    }

    auto nan = FullLike(x, std::numeric_limits<float>::quiet_NaN());
    return Select(Gt(Abs(x), ScalarLike(x, 1)), nan, naive_result);
  };
  return DoWithUpcastToF32(x, {BF16}, [&](XlaOp x) {  //
    return b->ReportErrorOrReturn(do_it(x));
  });
}

// Cosh(x) = (e^x + e^-x) / 2
//         = e^(x + log(1/2)) + e^(-x + log(1/2)).
//
// The second formulation avoids overflowing when e^x = inf but (e^x)/2 is not
// inf.
//
// This incorrectly overflows to inf for two f32 input values, namely
// +/-89.4159851, due to rounding error when computing x +/- log(1/2).  The
// correct answer of 3.40281961e+38 (0x7f7fffec) is very close to max-float, so
// we deem this acceptable.
XlaOp Cosh(XlaOp x) {
  return DoWithUpcastToF32(x, {BF16, F16}, [](XlaOp x) {
    auto log_one_half = Log(ScalarLike(x, 0.5));
    return Exp(x + log_one_half) + Exp(-x + log_one_half);
  });
}

// Sinh(x) = (e^x - e^-x) / 2
//         = e^(x + log(1/2)) - e^(-x + log(1/2)).
//
// The second formulation avoids overflowing when e^x = inf but (e^x)/2 is not
// inf.
//
// This incorrectly overflows to +/-inf for two f32 input values, namely
// +/-89.4159851, due to rounding error when computing x +/- log(1/2).  The
// correct answer of 3.40281961e+38 (0x7f7fffec) is very close to max-float, so
// we deem this acceptable.
XlaOp Sinh(XlaOp x) {
  XlaBuilder* b = x.builder();
  auto do_it = [&](XlaOp x) -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, b->GetShape(x));
    auto one_half = ScalarLike(x, 0.5);
    auto log_one_half = Log(ScalarLike(x, 0.5));
    auto large_sinh_result = Exp(x + log_one_half) - Exp(-x + log_one_half);

    if (primitive_util::IsComplexType(shape.element_type())) {
      return large_sinh_result;
    }

    // Here we use e^x = e^(x / 2) * e^(x / 2). This avoids overflow for large
    // values of x.

    // For smaller x, we get unwanted cancellations of e^x - e^-x, resulting in
    // 0.
    // Rewrite this to avoid that. We use expm1(x) because that preserves the
    // first order term of the taylor series of e^x.
    // (e^(x) - e^(-x)) / 2. =
    // (e^(x) - 1 + 1 - e^(-x)) / 2.
    // (expm1(x) + (e^(x) - 1) / e^x) / 2.
    // (expm1(x) + expm1(x) / (expm1(x) + 1)) / 2.
    auto expm1 = Expm1(x);
    auto one = ScalarLike(x, 1.);
    auto small_sinh_result = one_half * (expm1 + expm1 / (expm1 + one));
    return Select(Lt(Abs(x), one), small_sinh_result, large_sinh_result);
  };
  return DoWithUpcastToF32(x, {BF16, F16}, [&](XlaOp x) {
    return b->ReportErrorOrReturn(do_it(x));
  });
}

XlaOp MaybeConjugate(XlaOp x, bool conjugate) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder->GetShape(x));
    auto perform_conj =
        primitive_util::IsComplexType(shape.element_type()) && conjugate;
    return perform_conj ? Conj(x) : x;
  });
}

XlaOp NextAfter(XlaOp from, XlaOp to) {
  auto builder = from.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, builder->GetShape(from));
    int bitwidth = primitive_util::BitWidth(shape.element_type());
    auto int_type = primitive_util::UnsignedIntegralTypeForBitWidth(bitwidth);
    auto from_as_int = BitcastConvertType(from, int_type);
    auto to_as_int = BitcastConvertType(to, int_type);

    // The result is NaN if either "from" or "to" are NaN.
    auto from_is_nan = Ne(from, from);
    auto to_is_nan = Ne(to, to);
    auto nan_input = Or(from_is_nan, to_is_nan);
    auto result_for_nan =
        Broadcast(ScalarLike(from, std::numeric_limits<double>::quiet_NaN()),
                  shape.dimensions());
    result_for_nan = BitcastConvertType(result_for_nan, int_type);

    // The sign bit is the MSB.
    const int64_t sign_mask = int64_t{1} << (bitwidth - 1);
    // Discard the sign bit to make the result non-negative.
    auto from_abs = And(from_as_int, ScalarLike(from_as_int, ~sign_mask));
    auto to_abs = And(to_as_int, ScalarLike(to_as_int, ~sign_mask));

    // When both "from" and "to" are equal, the result is "to".
    // N.B. It would not make a difference if we chose the result to be "from".
    auto from_and_to_are_equal = Eq(from_as_int, to_as_int);
    auto result_for_equal = to_as_int;

    // When both "from" and "to" are both 0, the result is "to". This ensures we
    // get a zero signed like "to".
    auto from_is_zero = Eq(from_abs, ZerosLike(from_abs));
    auto to_is_zero = Eq(to_abs, ZerosLike(to_abs));
    auto result_for_both_zero = to_as_int;

    auto from_sign = And(from_as_int, ScalarLike(from_as_int, sign_mask));
    auto to_sign = And(to_as_int, ScalarLike(to_as_int, sign_mask));

    // If from == 0 && to != 0, we need to return the smallest subnormal number
    // signed like "to".
    auto result_for_from_zero_to_non_zero =
        Or(to_sign, ScalarLike(from_as_int, 1));

    // If the sign of "from" and "to" disagree:
    // - we need to make the magnitude of "from" smaller so that it is closer to
    //   zero.
    //
    // Otherwise the signs agree:
    // - "from" with a magnitude larger than "to" means we need to make the
    //   magnitude smaller.
    // - "from" with a magnitude smaller than "to" means we need to make the
    //   magnitude larger.
    // - "from" with the same magnitude and sign as "to" has already been
    //   handled.
    auto signs_disagree = Ne(from_sign, to_sign);
    auto from_magnitude_larger_than_to = Gt(from_abs, to_abs);
    auto result_has_smaller_magnitude =
        Or(from_magnitude_larger_than_to, signs_disagree);
    auto magnitude_adjustment =
        Select(result_has_smaller_magnitude,
               Broadcast(ScalarLike(from_as_int, -1), shape.dimensions()),
               Broadcast(ScalarLike(from_as_int, 1), shape.dimensions()));
    auto result = Add(from_as_int, magnitude_adjustment);
    // Handle from == ±0.
    result = Select(from_is_zero,
                    Select(to_is_zero, result_for_both_zero,
                           result_for_from_zero_to_non_zero),
                    result);
    // Handle from == to.
    result = Select(from_and_to_are_equal, result_for_equal, result);
    // Handle isnan(from) || isnan(to).
    result = Select(nan_input, result_for_nan, result);

    // Cast back to the original type.
    return BitcastConvertType(result, shape.element_type());
  });
}

// Computes an approximation to the modified Bessel function of the first kind,
// zeroth order.
// The following implementation follows Cephes' F32 and F64 implementation of
// i0e.
static XlaOp I0eImpl32(XlaOp x) {
  static const std::array<float, 18> kI0eCoeffsA{
      -1.30002500998624804212E-8f, 6.04699502254191894932E-8f,
      -2.67079385394061173391E-7f, 1.11738753912010371815E-6f,
      -4.41673835845875056359E-6f, 1.64484480707288970893E-5f,
      -5.75419501008210370398E-5f, 1.88502885095841655729E-4f,
      -5.76375574538582365885E-4f, 1.63947561694133579842E-3f,
      -4.32430999505057594430E-3f, 1.05464603945949983183E-2f,
      -2.37374148058994688156E-2f, 4.93052842396707084878E-2f,
      -9.49010970480476444210E-2f, 1.71620901522208775349E-1f,
      -3.04682672343198398683E-1f, 6.76795274409476084995E-1f};

  static const std::array<float, 7> kI0eCoeffsB{
      3.39623202570838634515E-9f, 2.26666899049817806459E-8f,
      2.04891858946906374183E-7f, 2.89137052083475648297E-6f,
      6.88975834691682398426E-5f, 3.36911647825569408990E-3f,
      8.04490411014108831608E-1f};

  x = Abs(x);
  auto half = xla::ScalarLike(x, 0.5);
  auto two = xla::ScalarLike(x, 2.0);
  auto thirty_two = xla::ScalarLike(x, 32.0);
  auto result_le_8 =
      EvaluateChebyshevPolynomial<float>(half * x - two, kI0eCoeffsA);
  auto result_gt_8 =
      EvaluateChebyshevPolynomial<float>(thirty_two / x - two, kI0eCoeffsB) /
      Sqrt(x);
  return Select(Le(x, xla::ScalarLike(x, 8.0)), result_le_8, result_gt_8);
}

static XlaOp I0eImpl64(XlaOp x) {
  static const std::array<double, 30> kI0eCoeffsA{
      -4.41534164647933937950E-18, 3.33079451882223809783E-17,
      -2.43127984654795469359E-16, 1.71539128555513303061E-15,
      -1.16853328779934516808E-14, 7.67618549860493561688E-14,
      -4.85644678311192946090E-13, 2.95505266312963983461E-12,
      -1.72682629144155570723E-11, 9.67580903537323691224E-11,
      -5.18979560163526290666E-10, 2.65982372468238665035E-9,
      -1.30002500998624804212E-8,  6.04699502254191894932E-8,
      -2.67079385394061173391E-7,  1.11738753912010371815E-6,
      -4.41673835845875056359E-6,  1.64484480707288970893E-5,
      -5.75419501008210370398E-5,  1.88502885095841655729E-4,
      -5.76375574538582365885E-4,  1.63947561694133579842E-3,
      -4.32430999505057594430E-3,  1.05464603945949983183E-2,
      -2.37374148058994688156E-2,  4.93052842396707084878E-2,
      -9.49010970480476444210E-2,  1.71620901522208775349E-1,
      -3.04682672343198398683E-1,  6.76795274409476084995E-1};

  static const std::array<double, 25> kI0eCoeffsB{
      -7.23318048787475395456E-18, -4.83050448594418207126E-18,
      4.46562142029675999901E-17,  3.46122286769746109310E-17,
      -2.82762398051658348494E-16, -3.42548561967721913462E-16,
      1.77256013305652638360E-15,  3.81168066935262242075E-15,
      -9.55484669882830764870E-15, -4.15056934728722208663E-14,
      1.54008621752140982691E-14,  3.85277838274214270114E-13,
      7.18012445138366623367E-13,  -1.79417853150680611778E-12,
      -1.32158118404477131188E-11, -3.14991652796324136454E-11,
      1.18891471078464383424E-11,  4.94060238822496958910E-10,
      3.39623202570838634515E-9,   2.26666899049817806459E-8,
      2.04891858946906374183E-7,   2.89137052083475648297E-6,
      6.88975834691682398426E-5,   3.36911647825569408990E-3,
      8.04490411014108831608E-1};

  x = Abs(x);
  auto half = xla::ScalarLike(x, 0.5);
  auto two = xla::ScalarLike(x, 2.0);
  auto thirty_two = xla::ScalarLike(x, 32.0);
  auto result_le_8 =
      EvaluateChebyshevPolynomial<double>(half * x - two, kI0eCoeffsA);
  auto result_gt_8 =
      EvaluateChebyshevPolynomial<double>(thirty_two / x - two, kI0eCoeffsB) /
      Sqrt(x);
  return Select(Le(x, xla::ScalarLike(x, 8.0)), result_le_8, result_gt_8);
}

XlaOp BesselI0e(XlaOp x) {
  auto& b = *x.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("BesselI0e", x));
    TF_ASSIGN_OR_RETURN(auto shape, b.GetShape(x));
    if (shape.element_type() == F64) {
      return I0eImpl64(x);
    }
    // I0eF32Impl don't have enough precision when run with bf16 intermediates
    // (not surprising!), so upcast to f32 in this case.
    return DoWithUpcastToF32(x, {BF16, F16},
                             [](XlaOp x) { return I0eImpl32(x); });
  });
}

// Computes an approximation to the modified Bessel function of the first kind,
// first order.
// The following implementation follows Cephes' F32 and F64 implementation of
// i1e.

static XlaOp I1eImpl32(XlaOp x) {
  static const std::array<float, 17> kI1eCoeffsA{
      9.38153738649577178388E-9f, -4.44505912879632808065E-8f,
      2.00329475355213526229E-7f, -8.56872026469545474066E-7f,
      3.47025130813767847674E-6f, -1.32731636560394358279E-5f,
      4.78156510755005422638E-5f, -1.61760815825896745588E-4f,
      5.12285956168575772895E-4f, -1.51357245063125314899E-3f,
      4.15642294431288815669E-3f, -1.05640848946261981558E-2f,
      2.47264490306265168283E-2f, -5.29459812080949914269E-2f,
      1.02643658689847095384E-1f, -1.76416518357834055153E-1f,
      2.52587186443633654823E-1f};

  static const std::array<float, 7> kI1eCoeffsB{
      -3.83538038596423702205E-9f, -2.63146884688951950684E-8f,
      -2.51223623787020892529E-7f, -3.88256480887769039346E-6f,
      -1.10588938762623716291E-4f, -9.76109749136146840777E-3f,
      7.78576235018280120474E-1f};
  XlaOp z = Abs(x);
  auto half = xla::ScalarLike(x, 0.5);
  auto two = xla::ScalarLike(x, 2.0);
  auto thirty_two = xla::ScalarLike(x, 32.0);
  auto result_le_8 =
      z * EvaluateChebyshevPolynomial<float>(half * z - two, kI1eCoeffsA);
  auto result_gt_8 =
      EvaluateChebyshevPolynomial<float>(thirty_two / z - two, kI1eCoeffsB) /
      Sqrt(z);
  return Sign(x) *
         Select(Le(z, xla::ScalarLike(x, 8.0)), result_le_8, result_gt_8);
}

static XlaOp I1eImpl64(XlaOp x) {
  static const std::array<double, 29> kI1eCoeffsA{
      2.77791411276104639959E-18, -2.11142121435816608115E-17,
      1.55363195773620046921E-16, -1.10559694773538630805E-15,
      7.60068429473540693410E-15, -5.04218550472791168711E-14,
      3.22379336594557470981E-13, -1.98397439776494371520E-12,
      1.17361862988909016308E-11, -6.66348972350202774223E-11,
      3.62559028155211703701E-10, -1.88724975172282928790E-9,
      9.38153738649577178388E-9,  -4.44505912879632808065E-8,
      2.00329475355213526229E-7,  -8.56872026469545474066E-7,
      3.47025130813767847674E-6,  -1.32731636560394358279E-5,
      4.78156510755005422638E-5,  -1.61760815825896745588E-4,
      5.12285956168575772895E-4,  -1.51357245063125314899E-3,
      4.15642294431288815669E-3,  -1.05640848946261981558E-2,
      2.47264490306265168283E-2,  -5.29459812080949914269E-2,
      1.02643658689847095384E-1,  -1.76416518357834055153E-1,
      2.52587186443633654823E-1};

  static const std::array<double, 25> kI1eCoeffsB{
      7.51729631084210481353E-18,  4.41434832307170791151E-18,
      -4.65030536848935832153E-17, -3.20952592199342395980E-17,
      2.96262899764595013876E-16,  3.30820231092092828324E-16,
      -1.88035477551078244854E-15, -3.81440307243700780478E-15,
      1.04202769841288027642E-14,  4.27244001671195135429E-14,
      -2.10154184277266431302E-14, -4.08355111109219731823E-13,
      -7.19855177624590851209E-13, 2.03562854414708950722E-12,
      1.41258074366137813316E-11,  3.25260358301548823856E-11,
      -1.89749581235054123450E-11, -5.58974346219658380687E-10,
      -3.83538038596423702205E-9,  -2.63146884688951950684E-8,
      -2.51223623787020892529E-7,  -3.88256480887769039346E-6,
      -1.10588938762623716291E-4,  -9.76109749136146840777E-3,
      7.78576235018280120474E-1};

  XlaOp z = Abs(x);
  auto half = xla::ScalarLike(x, 0.5);
  auto two = xla::ScalarLike(x, 2.0);
  auto thirty_two = xla::ScalarLike(x, 32.0);
  auto result_le_8 =
      z * EvaluateChebyshevPolynomial<double>(half * z - two, kI1eCoeffsA);
  auto result_gt_8 =
      EvaluateChebyshevPolynomial<double>(thirty_two / z - two, kI1eCoeffsB) /
      Sqrt(z);
  return Sign(x) *
         Select(Le(z, xla::ScalarLike(x, 8.0)), result_le_8, result_gt_8);
}

XlaOp BesselI1e(XlaOp x) {
  auto& b = *x.builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("BesselI1e", x));
    TF_ASSIGN_OR_RETURN(auto shape, b.GetShape(x));
    if (shape.element_type() == F64) {
      return I1eImpl64(x);
    }
    // I1eF32Impl don't have enough precision when run with bf16 intermediates
    // (not surprising!), so upcast to f32 in this case.
    return DoWithUpcastToF32(x, {BF16, F16},
                             [](XlaOp x) { return I1eImpl32(x); });
  });
}

// I J Thompson and A R Barnett. 1986. Coulomb and Bessel functions of complex
// arguments and order. J. Comput. Phys. 64, 2 (June 1986), 490-509.
// DOI=http://dx.doi.org/10.1016/0021-9991(86)90046-X
static XlaOp LentzThompsonBarnettAlgorithm(
    int64_t num_iterations, double small, double threshold,
    const ForEachIndexBodyFunction& nth_partial_numerator,
    const ForEachIndexBodyFunction& nth_partial_denominator,
    absl::Span<const XlaOp> inputs, absl::string_view name) {
  auto& b = *inputs.front().builder();
  return b.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_RET_CHECK(num_iterations < INT32_MAX);

    enum {
      // Position in the evaluation.
      kIterationIdx,
      // Whether or not we have reached the desired tolerance.
      kValuesUnconvergedIdx,
      // Ratio between nth canonical numerator and the nth-1 canonical
      // numerator.
      kCIdx,
      // Ratio between nth-1 canonical denominator and the nth canonical
      // denominator.
      kDIdx,
      // Computed approximant in the evaluation.
      kHIdx,
      // Inputs follow all of the other state.
      kFirstInputIdx,
    };
    auto while_cond_fn = [num_iterations](
                             absl::Span<const XlaOp> values,
                             XlaBuilder* cond_builder) -> StatusOr<XlaOp> {
      auto iteration = values[kIterationIdx];
      auto iterations_remain_cond =
          Lt(iteration, ScalarLike(iteration, num_iterations));
      auto values_unconverged_cond = values[kValuesUnconvergedIdx];
      return And(iterations_remain_cond, values_unconverged_cond);
    };

    auto while_body_fn =
        [small, threshold, &nth_partial_numerator, &nth_partial_denominator](
            absl::Span<const XlaOp> values,
            XlaBuilder* body_builder) -> StatusOr<std::vector<XlaOp>> {
      XlaOp iteration = values[kIterationIdx];

      TF_ASSIGN_OR_RETURN(
          std::vector<XlaOp> partial_numerator,
          nth_partial_numerator(iteration, values.subspan(kFirstInputIdx),
                                body_builder));
      TF_RET_CHECK(partial_numerator.size() == 1);

      TF_ASSIGN_OR_RETURN(
          std::vector<XlaOp> partial_denominator,
          nth_partial_denominator(iteration, values.subspan(kFirstInputIdx),
                                  body_builder));
      TF_RET_CHECK(partial_denominator.size() == 1);

      auto c = partial_denominator[0] + partial_numerator[0] / values[kCIdx];
      auto small_constant = FullLike(c, small);
      c = Select(Lt(Abs(c), small_constant), small_constant, c);

      auto d = partial_denominator[0] + partial_numerator[0] * values[kDIdx];
      d = Select(Lt(Abs(d), small_constant), small_constant, d);

      d = Reciprocal(d);

      auto delta = c * d;
      auto h = values[kHIdx] * delta;

      std::vector<XlaOp> updated_values(values.size());
      updated_values[kIterationIdx] = Add(iteration, ScalarLike(iteration, 1));
      updated_values[kCIdx] = c;
      updated_values[kDIdx] = d;
      updated_values[kHIdx] = h;
      std::copy(values.begin() + kFirstInputIdx, values.end(),
                updated_values.begin() + kFirstInputIdx);

      // If any values are greater than the tolerance, we have not converged.
      auto tolerance_comparison =
          Ge(Abs(Sub(delta, FullLike(delta, 1.0))), FullLike(delta, threshold));
      updated_values[kValuesUnconvergedIdx] =
          ReduceAll(tolerance_comparison, ConstantR0<bool>(body_builder, false),
                    CreateScalarOrComputation(PRED, body_builder));
      return updated_values;
    };

    TF_ASSIGN_OR_RETURN(std::vector<XlaOp> partial_denominator,
                        nth_partial_denominator(Zero(&b, U32), inputs, &b));
    TF_RET_CHECK(partial_denominator.size() == 1);
    auto h = partial_denominator[0];
    auto small_constant = FullLike(h, small);
    h = Select(Lt(Abs(h), small_constant), small_constant, h);

    std::vector<XlaOp> values(kFirstInputIdx + inputs.size());
    values[kIterationIdx] = One(&b, U32);
    values[kValuesUnconvergedIdx] = ConstantR0<bool>(&b, true);
    values[kCIdx] = h;
    values[kDIdx] = FullLike(h, 0.0);
    values[kHIdx] = h;
    std::copy(inputs.begin(), inputs.end(), values.begin() + kFirstInputIdx);
    TF_ASSIGN_OR_RETURN(values, WhileLoopHelper(while_cond_fn, while_body_fn,
                                                values, name, &b));
    return values[kHIdx];
  });
}

XlaOp RegularizedIncompleteBeta(XlaOp a, XlaOp b, XlaOp x) {
  auto& builder = *x.builder();
  return builder.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(Shape shape, builder.GetShape(a));
    TF_ASSIGN_OR_RETURN(Shape b_shape, builder.GetShape(b));
    TF_ASSIGN_OR_RETURN(Shape x_shape, builder.GetShape(x));
    if (b_shape.element_type() != shape.element_type() ||
        x_shape.element_type() != shape.element_type()) {
      return InvalidArgument(
          "Operands to RegularizedIncompleteBeta must have identical types, "
          "got shapes %s, %s, and %s",
          shape.ToString(), b_shape.ToString(), x_shape.ToString());
    }
    if (!primitive_util::IsFloatingPointType(shape.element_type())) {
      return InvalidArgument(
          "Operands to RegularizedIncompleteBeta must be real-valued "
          "floating-point, but got %s",
          PrimitiveType_Name(shape.element_type()));
    }
    PrimitiveType element_type = shape.element_type();
    if (element_type == F16 || element_type == BF16) {
      element_type = F32;
      a = ConvertElementType(a, F32);
      b = ConvertElementType(b, F32);
      x = ConvertElementType(x, F32);
    }

    // The partial numerator for the incomplete beta function is given
    // here: http://dlmf.nist.gov/8.17.E23 Note that there is a special
    // case: the partial numerator for the first iteration is one.
    auto NthPartialBetaincNumerator =
        [&](XlaOp iteration, absl::Span<const XlaOp> inputs,
            XlaBuilder* builder) -> StatusOr<std::vector<XlaOp>> {
      auto a = inputs[0];
      auto b = inputs[1];
      auto x = inputs[2];
      auto iteration_bcast = Broadcast(iteration, shape.dimensions());
      auto iteration_is_even =
          Eq(iteration_bcast % FullLike(iteration_bcast, 2),
             FullLike(iteration_bcast, 0));
      auto iteration_is_one = Eq(iteration_bcast, FullLike(iteration_bcast, 1));
      auto iteration_minus_one = iteration_bcast - FullLike(iteration_bcast, 1);
      auto m = iteration_minus_one / FullLike(iteration_minus_one, 2);
      m = ConvertElementType(m, element_type);
      auto one = FullLike(a, 1.0);
      auto two = FullLike(a, 2.0);
      // Partial numerator terms.
      auto even_numerator =
          -(a + m) * (a + b + m) * x / ((a + two * m) * (a + two * m + one));
      auto odd_numerator =
          m * (b - m) * x / ((a + two * m - one) * (a + two * m));
      auto one_numerator = ScalarLike(x, 1.0);
      auto numerator = Select(iteration_is_even, even_numerator, odd_numerator);
      return std::vector<XlaOp>{
          Select(iteration_is_one, one_numerator, numerator)};
    };

    auto NthPartialBetaincDenominator =
        [&shape](XlaOp iteration, absl::Span<const XlaOp> inputs,
                 XlaBuilder* builder) -> StatusOr<std::vector<XlaOp>> {
      auto x = inputs[2];
      auto iteration_bcast = Broadcast(iteration, shape.dimensions());
      return std::vector<XlaOp>{
          Select(Eq(iteration_bcast, ScalarLike(iteration_bcast, 0)),
                 ScalarLike(x, 0.0), ScalarLike(x, 1.0))};
    };

    // Determine if the inputs are out of range.
    auto result_is_nan =
        Or(Or(Or(Le(a, ScalarLike(a, 0.0)), Le(b, ScalarLike(b, 0.0))),
              Lt(x, ScalarLike(x, 0.0))),
           Gt(x, ScalarLike(x, 1.0)));

    // The continued fraction will converge rapidly when x < (a+1)/(a+b+2)
    // as per: http://dlmf.nist.gov/8.17.E23
    //
    // Otherwise, we can rewrite using the symmetry relation as per:
    // http://dlmf.nist.gov/8.17.E4
    auto converges_rapidly =
        Lt(x, (a + FullLike(a, 1.0)) / (a + b + FullLike(b, 2.0)));
    auto a_orig = a;
    a = Select(converges_rapidly, a, b);
    b = Select(converges_rapidly, b, a_orig);
    x = Select(converges_rapidly, x, Sub(FullLike(x, 1.0), x));

    XlaOp continued_fraction;

    // Thresholds and iteration counts taken from Cephes.
    if (element_type == F32) {
      continued_fraction = LentzThompsonBarnettAlgorithm(
          /*num_iterations=*/200,
          /*small=*/std::numeric_limits<float>::epsilon() / 2.0f,
          /*threshold=*/std::numeric_limits<float>::epsilon() / 2.0f,
          /*nth_partial_numerator=*/NthPartialBetaincNumerator,
          /*nth_partial_denominator=*/NthPartialBetaincDenominator, {a, b, x},
          "Betainc");
    } else {
      TF_RET_CHECK(element_type == F64);
      continued_fraction = LentzThompsonBarnettAlgorithm(
          /*num_iterations=*/600,
          /*small=*/std::numeric_limits<double>::epsilon() / 2.0f,
          /*threshold=*/std::numeric_limits<double>::epsilon() / 2.0f,
          /*nth_partial_numerator=*/NthPartialBetaincNumerator,
          /*nth_partial_denominator=*/NthPartialBetaincDenominator, {a, b, x},
          "Betainc");
    }

    // We want to compute the regularized complete beta function so we need to
    // combine the continued fraction with a few more terms as well as dividing
    // it by Beta(a, b). To avoid overflow, we compute in the log domain.
    // See http://dlmf.nist.gov/8.17.E22 for an easier to read version of this
    // formula.
    auto lbeta = Lbeta(a, b);
    auto result =
        continued_fraction * Exp(Log(x) * a + Log1p(-x) * b - lbeta) / a;
    result = Select(result_is_nan, NanValue(&builder, element_type), result);

    // We have an additional fixup to do if we are taking advantage of the
    // symmetry relation.
    auto out =
        Select(converges_rapidly, result, Sub(FullLike(result, 1.0), result));
    return shape.element_type() == element_type
               ? out
               : ConvertElementType(out, shape.element_type());
  });
}

XlaOp Polygamma(XlaOp n, XlaOp x) {
  auto& builder = *x.builder();
  auto doit = [](XlaOp n, XlaOp x, PrimitiveType type) -> XlaOp {
    XlaOp n_plus_one = n + ScalarLike(n, 1.);
    XlaOp sign =
        (ScalarLike(n, 2.) * Rem(n, ScalarLike(n, 2.)) - ScalarLike(n, 1.));

    const double nan = std::numeric_limits<double>::quiet_NaN();

    XlaOp output = Select(Eq(n, ScalarLike(n, 0.)), Digamma(x),
                          sign * Exp(Lgamma(n_plus_one)) * Zeta(n_plus_one, x));
    // Check that n is a natural number.
    output = Select(Or(Ne(n, Floor(n)), Lt(n, ScalarLike(n, 0.))),
                    ScalarLike(n, nan), output);
    return output;
  };
  return builder.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto n_shape, builder.GetShape(n));
    TF_ASSIGN_OR_RETURN(auto x_shape, builder.GetShape(x));
    if (n_shape != x_shape) {
      return InvalidArgument(
          "Arguments to Polygamma must have equal shapes and types; "
          "got %s and %s",
          n_shape.ToString(), x_shape.ToString());
    }
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("Zeta", x));
    bool needs_upcast =
        n_shape.element_type() == F16 || x_shape.element_type() == BF16;

    if (needs_upcast) {
      n = ConvertElementType(n, F32);
      x = ConvertElementType(x, F32);
    }
    XlaOp result = doit(n, x, n_shape.element_type());
    if (needs_upcast) {
      result = ConvertElementType(result, n_shape.element_type());
    }
    return result;
  });
}

XlaOp Zeta(XlaOp x, XlaOp q) {
  auto& builder = *x.builder();
  auto doit = [&builder](XlaOp x, XlaOp q, PrimitiveType type) -> XlaOp {
    // (2k) ! / B_{2k}, where B_{2k} are the Bernoulli numbers.
    // These are ordered in reverse.
    static const std::array<double, 12> kZetaCoeffs{
        -7.1661652561756670113e18,
        1.8152105401943546773e17,
        -4.5979787224074726105e15,
        1.1646782814350067249e14,
        -2.950130727918164224e12,
        7.47242496e10,
        -1.8924375803183791606e9,
        47900160.0,
        -1209600.0,
        30240.0,
        -720.0,
        12.0,
    };

    // For speed we'll always use 9 iterations for the initial series estimate,
    // and a 12 term expansion for the Euler-Maclaurin formula.

    XlaOp a = q;
    XlaOp neg_power = ScalarLike(a, 0.);
    XlaOp initial_sum = Pow(q, Neg(x));
    for (int i = 0; i < 9; ++i) {
      a = a + ScalarLike(a, 1.);
      neg_power = Pow(a, Neg(x));
      initial_sum = initial_sum + neg_power;
    }
    a = a + ScalarLike(a, 1.);
    neg_power = Pow(a, Neg(x));
    XlaOp s = initial_sum + neg_power * a / (x - ScalarLike(a, 1.));
    XlaOp a_inverse_square = Reciprocal(Square(a));
    XlaOp horner_sum = ScalarLike(a, 0.);
    XlaOp factor = ScalarLike(a, 1.);
    // Use Horner's rule for this.
    // Note this differs from Cephes which does a 'naive' polynomial evaluation.
    // Using Horner's rule allows to avoid some NaN's and Infs from happening,
    // resulting in more numerically stable code.
    for (int i = 0; i < 11; ++i) {
      factor =
          (x - ScalarLike(x, 22 - 2 * i)) * (x - ScalarLike(x, 21 - 2 * i));
      horner_sum = factor * a_inverse_square *
                   (horner_sum + ScalarLike(a, 1. / kZetaCoeffs[i]));
    }
    s = s + neg_power *
                (ScalarLike(neg_power, 0.5) +
                 x / a * (ScalarLike(a, 1. / kZetaCoeffs[11]) + horner_sum));

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    // Use the initial zeta sum without the correction term coming
    // from Euler-Maclaurin if it is accurate enough.
    XlaOp output =
        Select(Lt(Abs(neg_power), Abs(initial_sum) * Epsilon(&builder, type)),
               initial_sum, s);

    // This is the harmonic series.
    output = Select(Eq(x, ScalarLike(x, 1.)), ScalarLike(x, inf), output);

    // Function is not defined for x < 1.
    output = Select(Lt(x, ScalarLike(x, 1.)), ScalarLike(x, nan), output);

    // For q <= 0, x must be an integer.
    XlaOp x_domain_error = And(Le(q, ScalarLike(x, 0.)), Ne(x, Floor(x)));
    output = Select(x_domain_error, ScalarLike(x, nan), output);

    // For all integer q <= 0, zeta has a pole. The limit is only defined as
    // +inf if x is and even integer.
    XlaOp at_pole = And(Le(q, ScalarLike(x, 0.)), Eq(q, Floor(q)));
    XlaOp x_is_even_int =
        And(Eq(Rem(x, ScalarLike(x, 2.)), ScalarLike(x, 0.)), Eq(x, Floor(x)));
    output = Select(
        at_pole, Select(x_is_even_int, ScalarLike(x, inf), ScalarLike(x, nan)),
        output);

    return output;
  };
  return builder.ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto x_shape, builder.GetShape(x));
    TF_ASSIGN_OR_RETURN(auto q_shape, builder.GetShape(q));
    if (x_shape != q_shape) {
      return InvalidArgument(
          "Arguments to Zeta must have equal shapes and types; got %s and %s",
          x_shape.ToString(), q_shape.ToString());
    }
    TF_RETURN_IF_ERROR(EnsureOperandIsRealFp("Zeta", x));
    bool needs_upcast =
        x_shape.element_type() == F16 || x_shape.element_type() == BF16;

    if (needs_upcast) {
      x = ConvertElementType(x, F32);
      q = ConvertElementType(q, F32);
    }
    XlaOp result = doit(x, q, x_shape.element_type());
    if (needs_upcast) {
      result = ConvertElementType(result, x_shape.element_type());
    }
    return result;
  });
}

}  // namespace xla
