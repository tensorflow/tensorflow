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

#include <algorithm>  // IWYU pragma: keep, exhaustive_unary_test_ops.inc
#include <array>      // IWYU pragma: keep, exhaustive_unary_test_ops.inc
#include <cfenv>  // NOLINT
#include <cmath>
#include <cstddef>  // IWYU pragma: keep, exhaustive_unary_test_ops.inc
#include <limits>
#include <type_traits>

#include "xla/hlo/builder/lib/constants.h"  // IWYU pragma: keep, exhaustive_unary_test_ops.inc
#include "xla/hlo/builder/lib/math.h"  // IWYU pragma: keep, exhaustive_unary_test_ops.inc
#include "xla/hlo/builder/xla_builder.h"  // IWYU pragma: keep, exhaustive_unary_test_ops.inc
#include "xla/tests/exhaustive/error_spec.h"
#include "xla/tests/exhaustive/exhaustive_op_test.h"  // IWYU pragma: keep, exhaustive_unary_test_ops.inc
#include "xla/tests/exhaustive/exhaustive_op_test_utils.h"
#include "xla/tests/exhaustive/exhaustive_unary_test_definitions.h"
#include "xla/tests/exhaustive/test_op.h"  // IWYU pragma: keep, exhaustive_unary_test_ops.inc
#include "xla/types.h"

#ifdef __FAST_MATH__
#error "Can't be compiled with fast math on"
#endif

namespace xla {
namespace exhaustive_op_test {
namespace {

#include "xla/tests/exhaustive/exhaustive_unary_test_ops.inc"

UNARY_TEST(Log, { LogOp<kT>(this).Error(GetDefaultSpecGenerator()).Run(); })
UNARY_TEST(Log1p, {
  Log1pOp<kT>(this)
      .CpuError(+[](NativeT x) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e5m2>) {
          return ErrorSpec::Builder().distance_err(1).build();
        }
        return GetDefaultSpecGenerator()(x);
      })
      .GpuError(+[](NativeT x) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e5m2>) {
          return ErrorSpec::Builder().distance_err(1).build();
        }
        return GetDefaultSpecGenerator()(x);
      })
      .Run();
})

UNARY_TEST(Exp, {
  ExpOp<kT>(this)
      .CpuError(+[](NativeT x) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e5m2>) {
          return ErrorSpec::Builder().distance_err(1).build();
        }
        return GetDefaultSpecGenerator()(x);
      })
      .GpuError(+[](NativeT x) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e5m2>) {
          return ErrorSpec::Builder().distance_err(1).build();
        }
        return GetDefaultSpecGenerator()(x);
      })
      .Run();
})
UNARY_TEST(Expm1, { Expm1Op<kT>(this).Error(GetDefaultSpecGenerator()).Run(); })

UNARY_TEST(Logistic, {
  LogisticOp<kT>(this)
      .OutputRangeCheck(+[](NativeInputs in, NativeT out) {
        if (std::isnan(in[0])) {
          return std::isnan(out);
        }
        return std::abs(out) <= 1.0f;
      })
      .CpuError(+[](NativeT x) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e4m3fn>) {
          return ErrorSpec::Builder().distance_err(1).build();
        }
        // FIXME(rmlarsen): Break into region around zero and everything else.
        return GetDefaultSpecGenerator()(x);
      })
      .GpuError(+[](NativeT x) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e4m3fn>) {
          return ErrorSpec::Builder().distance_err(1).build();
        }
        // FIXME(rmlarsen): Break into region around zero and everything else.
        return GetDefaultSpecGenerator()(x);
      })
      .Run();
})

// It feels a little overkill to exhaustively test sqrt and pow(x, 0.5), but
// this *did* find a bug, namely that some backends were assuming sqrt(x) ==
// pow(x, 0.5), but this is not true for x == -inf.
UNARY_TEST(PowOneHalf,
           { PowOneHalfOp<kT>(this).Error(GetDefaultSpecGenerator()).Run(); })
UNARY_TEST(Rsqrt, {
  RsqrtOp<kT>(this)
      .Error(+[](NativeT x) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .abs_err(0)
            .rel_err(2 * eps)
            .strict_signed_zeros()
            .build();
      })
      .Run();
})
UNARY_TEST(Sqrt, {
  SqrtOp<kT>(this)
      .Error(+[](NativeT x) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .abs_err(0)
            .rel_err(2 * eps)
            .strict_signed_zeros()
            .build();
      })
      .Run();
})
UNARY_TEST(Cbrt, {
  CbrtOp<kT>(this)
      .Error(+[](NativeT x) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .abs_err(0)
            .rel_err(16 * eps)
            .strict_signed_zeros()
            .build();
      })
      .CpuError(+[](NativeT x) {
        // While GPUs flush subnormal inputs to zero, CPU returns a relatively
        // inaccurate approximation for such inputs. Therefore we allow a small
        // absolute error (e.g. ~9e-16 for F32). This corresponds to a 0.5%
        // relative error for the smallest normalized floating point values,
        // increasing gradually to 100% for the smallest subnormal value.
        NativeT denorm_min = std::numeric_limits<NativeT>::denorm_min();
        double abs_err = std::cbrt(denorm_min);

        if constexpr (std::is_same_v<NativeT, double>) {
          NativeT eps = std::numeric_limits<NativeT>::epsilon();
          return ErrorSpec::Builder()
              .abs_err(abs_err)
              .rel_err(70 * eps)
              .strict_signed_zeros()
              .build();
        } else {
          NativeT eps = std::numeric_limits<NativeT>::epsilon();
          return ErrorSpec::Builder()
              .abs_err(abs_err)
              .rel_err(10 * eps)
              .strict_signed_zeros()
              .build();
        }
      })
      .Run();
})

// Tests for inverse hyperbolic functions.
UNARY_TEST(Acosh, {
  AcoshOp<kT>(this)
      .Error(+[](NativeT x) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder().abs_err(1e-7).rel_err(50 * eps).build();
      })
      .Run();
})
UNARY_TEST(Asinh, {
  AsinhOp<kT>(this)
      .CpuError(+[](NativeT x) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e4m3fn> ||
                      std::is_same_v<NativeT, tsl::float8_e5m2>) {
          return ErrorSpec::Builder().distance_err(1).build();
        }
        return GetDefaultSpecGenerator()(x);
      })
      .GpuError(+[](NativeT x) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e4m3fn> ||
                      std::is_same_v<NativeT, tsl::float8_e5m2>) {
          return ErrorSpec::Builder().distance_err(1).build();
        }
        return GetDefaultSpecGenerator()(x);
      })
      .Run();
})
UNARY_TEST(Atanh, {
  AtanhOp<kT>(this)
      .Error(GetDefaultSpecGenerator())
      .CpuError(+[](NativeT x) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e4m3fn>) {
          return ErrorSpec::Builder().distance_err(1).build();
        }
        return GetDefaultSpecGenerator()(x);
      })
      .Run();
})

// Tests for inverse trigonometric functions.
UNARY_TEST(Acos, {
  AcosOp<kT>(this)
      .CpuError(+[](NativeT x) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e4m3fn> ||
                      std::is_same_v<NativeT, tsl::float8_e5m2>) {
          return ErrorSpec::Builder().distance_err(1).build();
        }
        return GetDefaultSpecGenerator()(x);
      })
      .GpuError(+[](NativeT x) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder().abs_err(1e-6).rel_err(10 * eps).build();
      })
      .Run();
})
UNARY_TEST(Asin, {
  AsinOp<kT>(this)
      .Error(+[](NativeT x) {
        NativeT min = std::numeric_limits<NativeT>::min();
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .abs_err(2.0f * min)
            .rel_err(10 * eps)
            .build();
      })
      .Run();
})
UNARY_TEST(Atan, {
  AtanOp<kT>(this)
      .Error(+[](NativeT x) {
        NativeT min = std::numeric_limits<NativeT>::min();
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .abs_err(2.0f * min)
            .rel_err(20 * eps)
            .build();
      })
      .Run();
})

UNARY_TEST(Cosh, {
  CoshOp<kT>(this)
      .CpuError(+[](NativeT x) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e4m3fn>) {
          return ErrorSpec::Builder().distance_err(3).build();
        } else if constexpr (std::is_same_v<NativeT, tsl::float8_e5m2>) {
          return ErrorSpec::Builder().distance_err(4).build();
        }
        return GetDefaultSpecGenerator()(x);
      })
      .GpuError(+[](NativeT x) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e4m3fn> ||
                      std::is_same_v<NativeT, tsl::float8_e5m2>) {
          return ErrorSpec::Builder().distance_err(1).build();
        }
        return GetDefaultSpecGenerator()(x);
      })
      .OutputRangeCheck(
          +[](NativeInputs in, NativeT actual) { return !(actual < 1); })
      .Run();
})
UNARY_TEST(Sinh, {
  SinhOp<kT>(this)
      .Error(GetDefaultSpecGenerator())
      .CpuError(+[](NativeT x) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e4m3fn>) {
          return ErrorSpec::Builder().distance_err(3).build();
        } else if constexpr (std::is_same_v<NativeT, tsl::float8_e5m2>) {
          return ErrorSpec::Builder().distance_err(4).build();
        }
        return GetDefaultSpecGenerator()(x);
      })
      .GpuError(+[](NativeT x) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e5m2>) {
          return ErrorSpec::Builder().distance_err(1).build();
        }
        return GetDefaultSpecGenerator()(x);
      })
      .Run();
})
UNARY_TEST(Tanh, {
  TanhOp<kT>(this)
      .Error(GetDefaultSpecGenerator())
      .OutputRangeCheck([](NativeInputs in, NativeT out) -> bool {
        if (std::isnan(in[0])) {
          return std::isnan(out);
        }
        return std::abs(out) <= 1.0f;
      })
      .Run();
})

UNARY_TEST(Cos, {
  CosOp<kT>(this)
      .Error(+[](NativeT) {
        // This error spec corresponds to a maximum relative error of 2 ULP.
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder().abs_err(0).rel_err(2 * eps).build();
      })
      .OutputRangeCheck(
          +[](NativeInputs in, NativeT out) { return !(out < -1 || out > 1); })
      .Run();
})
UNARY_TEST(Sin, {
  SinOp<kT>(this)
      .Error(+[](NativeT) {
        // This error spec corresponds to a maximum relative error of 2 ULP.
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder().abs_err(0).rel_err(2 * eps).build();
      })
      .CpuArmError(+[](NativeT val) {
        // Flushes subnormals and minimum positive output to 0.
        NativeT output = static_cast<NativeT>(std::sin(val));
        // TODO(b/365622116): Understand why ARM flushes these but x86 doesn't.
        if (IsSubnormalOrMinNormal(output)) {
          return ErrorSpec::Builder()
              .abs_err(std::numeric_limits<NativeT>::min())
              .build();
        }

        // This error spec corresponds to a maximum relative error of 2 ULP.
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder().abs_err(0).rel_err(2 * eps).build();
      })
      .OutputRangeCheck(
          +[](NativeInputs in, NativeT out) { return !(out < -1 || out > 1); })
      .Run();
})
UNARY_TEST(Tan, {
  TanOp<kT>(this)
      .Error(+[](NativeT) {
        // This error spec corresponds to a maximum relative error of 4 ULP.
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder().abs_err(0).rel_err(4 * eps).build();
      })
      .CpuArmError(+[](NativeT val) {
        // Flushes positive subnormals and minimum positive output to 0.
        NativeT output = static_cast<NativeT>(std::tan(val));
        // TODO(b/365622116): Understand why ARM flushes these but x86 doesn't.
        if (IsSubnormalOrMinNormal(output)) {
          return ErrorSpec::Builder()
              .abs_err(std::numeric_limits<NativeT>::min())
              .build();
        }

        // This error spec corresponds to a maximum relative error of 4 ULP.
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder().abs_err(0).rel_err(4 * eps).build();
      })
      .Run();
})

UNARY_TEST(Erf, { ErfOp<kT>(this).Error(GetDefaultSpecGenerator()).Run(); })
UNARY_TEST(Erfc, {
  ErfcOp<kT>(this)
      .Error(+[](NativeT x) {
        NativeT min = std::numeric_limits<NativeT>::min();
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder().abs_err(min).rel_err(35 * eps).build();
      })
      .Run();
})
UNARY_TEST(ErfInv, {
  ErfInvOp<kT>(this)
      .Error(+[](NativeT x) {
        NativeT min = std::numeric_limits<NativeT>::min();
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder().abs_err(2 * min).rel_err(50 * eps).build();
      })
      .Run();
})

UNARY_TEST(Digamma, {
  DigammaOp<kT>(this)
      .CpuError(+[](NativeT x) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e5m2>) {
          return ErrorSpec::Builder().distance_err(1).build();
        }
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder().abs_err(2e-5).rel_err(10 * eps).build();
      })
      .GpuError(+[](NativeT x) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder().abs_err(2e-5).rel_err(10 * eps).build();
      })
      .Run();
})

UNARY_TEST(Lgamma, {
  LgammaOp<kT>(this)
      .Error(+[](NativeT x) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder().abs_err(1e-5).rel_err(150 * eps).build();
      })
      .GpuError(+[](NativeT x) {
        if constexpr (std::is_same_v<NativeT, double>) {
          // Very large error on the smallest subnormal input.
          if (static_cast<double>(std::abs(x)) == 4.9406564584124654e-324) {
            return ErrorSpec::Builder().abs_err(0.05).build();
          } else {
            return ErrorSpec::Builder().distance_err(2).build();
          }
        } else {
          NativeT eps = std::numeric_limits<NativeT>::epsilon();
          return ErrorSpec::Builder().abs_err(1e-5).rel_err(5000 * eps).build();
        }
      })
      .Run();
})

UNARY_TEST(Round, { RoundOp<kT>(this).Error(GetDefaultSpecGenerator()).Run(); })
UNARY_TEST(RoundNearestEven, {
  int curr_direction = fegetround();
  fesetround(FE_TONEAREST);
  RoundNearestEvenOp<kT>(this).Run();
  fesetround(curr_direction);
})

UNARY_TEST(Reciprocal, {
  // Can be thought of as an absolute error of `<=
  // |std::numeric_limits<Native>::min()|`.
  auto* abs_err = +[](NativeT val) -> double {
    NativeT output = static_cast<NativeT>(1.0) / val;
    if (IsSubnormal(output)) {
      return std::numeric_limits<NativeT>::min();
    }
    return 0.0;
  };

  ReciprocalOp<kT>(this)
      .CpuError([&](NativeT val) {
        return ErrorSpec::Builder()
            .abs_err(abs_err(val))
            .strict_signed_zeros()
            .build();
      })
      .GpuError([&](NativeT val) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .abs_err(abs_err(val))
            .rel_err(eps)
            .strict_signed_zeros()
            .build();
      })
      .Run();
})

}  // namespace
}  // namespace exhaustive_op_test
}  // namespace xla
