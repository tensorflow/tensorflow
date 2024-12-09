/* Copyright 2024 The OpenXLA Authors.

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

template <typename T, size_t N>
T EvaluatePolynomial(T x, const std::array<T, N>& coeffs) {
  // Evaluate the polynomial as accurately as we can using double precision and
  // FMA.
  double result = 0;
  double x_d = static_cast<double>(x);
  for (T c : coeffs) {
    result = std::fma(result, x_d, static_cast<double>(c));
  }
  return static_cast<T>(result);
}

// There's no std::erfinv, so we have to implement it ourselves.  This follows
// Wichura 1998, https://www.jstor.org/stable/2347330 which, notably, is a
// different implementation from that in math.cc.
template <typename NativeRefT>
NativeRefT HostErfInv(NativeRefT x) {
  const std::array<double, 8> poly_a = {
      8.8709406962545514830200e2, 1.1819493347062294404278e4,
      2.3782041382114385731252e4, 1.6235862515167575384252e4,
      4.8548868893843886794648e3, 6.9706266534389598238465e2,
      4.7072688112383978012285e1, 1.1975323115670912564578e0,
  };
  const std::array<double, 8> poly_b = {
      5.2264952788528545610e3, 2.8729085735721942674e4, 3.9307895800092710610e4,
      2.1213794301586595867e4, 5.3941960214247511077e3, 6.8718700749205790830e2,
      4.2313330701600911252e1, 1.0000000000000000000e0,
  };
  const std::array<double, 8> poly_c = {
      7.74545014278341407640e-4, 2.27238449892691845833e-2,
      2.41780725177450611770e-1, 1.27045825245236838258e0,
      3.64784832476320460504e0,  5.76949722146069140550e0,
      4.63033784615654529590e0,  1.42343711074968357734e0,
  };
  const std::array<double, 8> poly_d = {
      1.4859850019840355905497876e-9, 7.7441459065157709165577218e-4,
      2.1494160384252876777097297e-2, 2.0945065210512749128288442e-1,
      9.7547832001787427186894837e-1, 2.3707661626024532365971225e0,
      2.9036514445419946173133295e0,  1.4142135623730950488016887e0,
  };
  const std::array<double, 8> poly_e = {
      2.01033439929228813265e-7, 2.71155556874348757815e-5,
      1.24266094738807843860e-3, 2.65321895265761230930e-2,
      2.96560571828504891230e-1, 1.78482653991729133580e0,
      5.46378491116411436990e0,  6.65790464350110377720e0,
  };
  const std::array<double, 8> poly_f = {
      2.891024605872965461538222e-15, 2.010321207683943062279931e-7,
      2.611088405080593625138020e-5,  1.112800997078859844711555e-3,
      2.103693768272068968719679e-2,  1.936480946950659106176712e-1,
      8.482908416595164588112026e-1,  1.414213562373095048801689e0,
  };

  if (std::abs(x) > 1 || std::isnan(x)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (std::abs(x) == 1) {
    return static_cast<NativeRefT>(
        std::copysign(std::numeric_limits<double>::infinity(), x));
  }

  double unsigned_result = [&] {
    double y = std::abs(x);
    if (y <= 0.85) {
      double r = 0.180625 - 0.25 * y * y;
      return (y * EvaluatePolynomial(r, poly_a)) /
             EvaluatePolynomial(r, poly_b);
    }

    double r = std::sqrt(std::log(2.0) - std::log1p(-y));
    if (r <= 5.0) {
      r -= 1.6;
      return EvaluatePolynomial(r, poly_c) / EvaluatePolynomial(r, poly_d);
    }

    r -= 5;
    return EvaluatePolynomial(r, poly_e) / EvaluatePolynomial(r, poly_f);
  }();
  return static_cast<NativeRefT>(std::copysign(unsigned_result, x));
}

// Digamma implementation using a polynomial from Cephes.  Notably this is a
// different implementation from the one in math.cc.
template <typename NativeRefT>
NativeRefT HostDigamma(NativeRefT x) {
  // Euler-Mascheroni constant
  const double gamma_constant = 0.57721566490153286061;

  const std::array<double, 4> poly = {
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  double reflection = 0;
  if (x <= 0) {
    double floor = std::floor(x);
    if (x == floor) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    // Compute reflection term, pi * cot(pi * x).
    reflection = x - floor;
    if (reflection == 0.5) {
      reflection = 0;
    } else {
      if (reflection > 0.5) {
        reflection = x - (floor + 1.0f);
      }
      reflection = M_PI / std::tan(M_PI * reflection);
    }
    x = 1 - x;
  }

  double result = 0;
  if (x <= 10 && x == std::floor(x)) {
    // Special case for integers <= 10.
    for (size_t i = 1; i < static_cast<size_t>(std::floor(x)); ++i) {
      result += 1.0 / static_cast<double>(i);
    }
    result -= gamma_constant;
  } else {
    double w = 0;
    while (x < 10) {
      w += 1.0 / x;
      ++x;
    }
    if (x < 1e8) {
      double z = 1.0 / (x * x);
      result = z * EvaluatePolynomial(z, poly);
    }
    result = std::log(x) - 0.5 / x - result - w;
  }

  // Compute the final, reflected value.
  return static_cast<NativeRefT>(result - reflection);
}

#define DEFINE_UNARY_TEST_OP(NAME, ENQUEUE, EVALUATE)        \
  template <PrimitiveType T>                                 \
  class NAME final : public UnaryTestOp<T> {                 \
   public:                                                   \
    using Traits = UnaryTestOp<T>::Traits;                   \
    using Test = UnaryTestOp<T>::Test;                       \
                                                             \
    explicit NAME(Test* test) : UnaryTestOp<T>(test) {}      \
    ~NAME() override {}                                      \
                                                             \
    Traits::EnqueueOp EnqueueOp() const override ENQUEUE;    \
                                                             \
    Traits::EvaluateOp EvaluateOp() const override EVALUATE; \
  };                                                         \
  static_assert(true, "")

DEFINE_UNARY_TEST_OP(LogOp, { return Log; }, { return std::log; });
DEFINE_UNARY_TEST_OP(Log1pOp, { return Log1p; }, { return std::log1p; });
DEFINE_UNARY_TEST_OP(
    ExpOp, { return [](XlaOp x) { return Exp(x); }; }, { return std::exp; });
DEFINE_UNARY_TEST_OP(Expm1Op, { return Expm1; }, { return std::expm1; });
DEFINE_UNARY_TEST_OP(
    LogisticOp, { return Logistic; },
    {
      return +[](Traits::NativeRefT x) { return 1.0f / (1.0f + std::exp(-x)); };
    });
DEFINE_UNARY_TEST_OP(
    PowOneHalfOp,
    { return [](XlaOp x) { return Pow(x, ScalarLike(x, 0.5)); }; },
    { return +[](Traits::NativeRefT x) { return std::pow(x, 0.5f); }; });
DEFINE_UNARY_TEST_OP(
    Exp2Op,
    { return [](XlaOp x) { return Pow(ScalarLike(x, 2.0f), x); }; },
    { return +[](Traits::NativeRefT x) { return std::exp2(x); }; });
DEFINE_UNARY_TEST_OP(
    RsqrtOp, { return Rsqrt; },
    { return +[](Traits::NativeRefT x) { return 1 / std::sqrt(x); }; });
DEFINE_UNARY_TEST_OP(SqrtOp, { return Sqrt; }, { return std::sqrt; });
DEFINE_UNARY_TEST_OP(CbrtOp, { return Cbrt; }, { return std::cbrt; });
DEFINE_UNARY_TEST_OP(AcoshOp, { return Acosh; }, { return std::acosh; });
DEFINE_UNARY_TEST_OP(AsinhOp, { return Asinh; }, { return std::asinh; });
DEFINE_UNARY_TEST_OP(AtanhOp, { return Atanh; }, { return std::atanh; });
DEFINE_UNARY_TEST_OP(AcosOp, { return Acos; }, { return std::acos; });
DEFINE_UNARY_TEST_OP(AsinOp, { return Asin; }, { return std::asin; });
DEFINE_UNARY_TEST_OP(AtanOp, { return Atan; }, { return std::atan; });
DEFINE_UNARY_TEST_OP(CoshOp, { return Cosh; }, { return std::cosh; });
DEFINE_UNARY_TEST_OP(SinhOp, { return Sinh; }, { return std::sinh; });
DEFINE_UNARY_TEST_OP(TanhOp, { return Tanh; }, { return std::tanh; });
DEFINE_UNARY_TEST_OP(CosOp, { return Cos; }, { return std::cos; });
DEFINE_UNARY_TEST_OP(SinOp, { return Sin; }, { return std::sin; });
DEFINE_UNARY_TEST_OP(TanOp, { return Tan; }, { return std::tan; });
DEFINE_UNARY_TEST_OP(ErfOp, { return Erf; }, { return std::erf; });
DEFINE_UNARY_TEST_OP(ErfcOp, { return Erfc; }, { return std::erfc; });
DEFINE_UNARY_TEST_OP(
    ErfInvOp, { return ErfInv; },
    { return HostErfInv<typename Traits::NativeRefT>; });
DEFINE_UNARY_TEST_OP(
    DigammaOp, { return Digamma; },
    { return HostDigamma<typename Traits::NativeRefT>; });
DEFINE_UNARY_TEST_OP(LgammaOp, { return Lgamma; }, { return std::lgamma; });
DEFINE_UNARY_TEST_OP(RoundOp, { return Round; }, { return std::round; });
DEFINE_UNARY_TEST_OP(
    RoundNearestEvenOp, { return RoundNearestEven; },
    { return std::nearbyint; });
DEFINE_UNARY_TEST_OP(
    ReciprocalOp, { return Reciprocal; },
    { return +[](Traits::NativeRefT x) { return 1 / x; }; });

#undef DEFINE_UNARY_TEST_OP
