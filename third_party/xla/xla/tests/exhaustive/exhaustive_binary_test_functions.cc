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

#include <cmath>    // IWYU pragma: keep, exhaustive_binary_test_ops.inc
#include <complex>  // IWYU pragma: keep, exhaustive_binary_test_ops.inc
#include <limits>
#include <type_traits>

#include "xla/hlo/builder/xla_builder.h"  // IWYU pragma: keep, exhaustive_binary_test_ops.inc
#include "xla/tests/exhaustive/error_spec.h"
#include "xla/tests/exhaustive/exhaustive_binary_test_definitions.h"
#include "xla/tests/exhaustive/exhaustive_op_test_utils.h"
#include "xla/tests/exhaustive/test_op.h"  // IWYU pragma: keep, exhaustive_binary_test_ops.inc
#include "xla/types.h"

#ifdef __FAST_MATH__
#error("Can't be compiled with fast math on");
#endif

namespace xla {
namespace exhaustive_op_test {
namespace {

#include "xla/tests/exhaustive/exhaustive_binary_test_ops.inc"

// Can be thought of as an absolute error of
// `<= |std::numeric_limits::<float>::min()|`.
template <typename NativeT, typename NativeRefT>
double AddCpuAbsErr(NativeT left, NativeT right) {
  NativeRefT output =
      static_cast<NativeRefT>(left) + static_cast<NativeRefT>(right);
  // Hardware flushes subnormal outputs to 0.
  if (IsSubnormal(output)) {
    return std::numeric_limits<NativeRefT>::min();
  }
  return 0.0;
}

BINARY_TEST(Add, {
  AddOp<kT>(this)
      .Error(+[](NativeT, NativeT) {
        return ErrorSpec::Builder().strict_signed_zeros().build();
      })
      .CpuError(+[](NativeT left, NativeT right) {
        if constexpr (std::is_same_v<NativeT, xla::bfloat16> ||
                      std::is_same_v<NativeT, float> ||
                      std::is_same_v<NativeT, double>) {
          return ErrorSpec::Builder()
              .abs_err(AddCpuAbsErr<NativeT, NativeRefT>(left, right))
              .strict_signed_zeros()
              .build();
        }
        return ErrorSpec::Builder().strict_signed_zeros().build();
      })
      .Run();
})

// Can be thought of as an absolute error of
// `<= |std::numeric_limits::<float>::min()|`.
template <typename NativeT, typename NativeRefT>
double SubCpuAbsErr(NativeT left, NativeT right) {
  NativeRefT output =
      static_cast<NativeRefT>(left) - static_cast<NativeRefT>(right);
  // Hardware flushes subnormal outputs to 0.
  if (IsSubnormal(output)) {
    return std::numeric_limits<NativeRefT>::min();
  }
  return 0.0;
}

BINARY_TEST(Sub, {
  SubOp<kT>(this)
      .Error(+[](NativeT, NativeT) {
        return ErrorSpec::Builder().strict_signed_zeros().build();
      })
      .CpuError(+[](NativeT left, NativeT right) {
        if constexpr (std::is_same_v<NativeT, xla::bfloat16> ||
                      std::is_same_v<NativeT, float> ||
                      std::is_same_v<NativeT, double>) {
          return ErrorSpec::Builder()
              .abs_err(SubCpuAbsErr<NativeT, NativeRefT>(left, right))
              .strict_signed_zeros()
              .build();
        }
        return ErrorSpec::Builder().strict_signed_zeros().build();
      })
      .Run();
})

// Can be thought of as an absolute error of
// `<= |std::numeric_limits::<float>::min()|`.
template <typename NativeT, typename NativeRefT>
double MulCpuAbsErr(NativeT left, NativeT right) {
  NativeRefT output =
      static_cast<NativeRefT>(left) * static_cast<NativeRefT>(right);
  // CPU BF16 flush subnormals to 0.
  auto output_is_subnormal = IsSubnormal(output);
  if (output_is_subnormal) {
    return std::numeric_limits<NativeRefT>::min();
  }
  return 0.0;
}

bool MulCpuBf16Skip(xla::bfloat16 left, xla::bfloat16 right) {
  // For CPU BF16, multiplying a subnormal by infinity will lead to
  // calculating 0 multiplied by infinity due to subnormal flushing, which is
  // defined to be NaN. However, the calculation in higher precision does not
  // flush the subnormal value to 0, leading to a result of infinity.
  return (IsSubnormal(left) && std::isinf(right)) ||
         (std::isinf(left) && IsSubnormal(right));
}

BINARY_TEST(Mul, {
  MulOp<kT>(this)
      .Error(+[](NativeT left, NativeT right) {
        return ErrorSpec::Builder().strict_signed_zeros().build();
      })
      .CpuError(+[](NativeT left, NativeT right) {
        if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
          return ErrorSpec::Builder()
              .abs_err(MulCpuAbsErr<NativeT, NativeRefT>(left, right))
              .strict_signed_zeros()
              .skip_comparison(
                  MulCpuBf16Skip(static_cast<xla::bfloat16>(left),
                                 static_cast<xla::bfloat16>(right)))
              .build();
        }
        if constexpr (std::is_same_v<NativeT, float>) {
          return ErrorSpec::Builder()
              .abs_err(MulCpuAbsErr<NativeT, NativeRefT>(left, right))
              .strict_signed_zeros()
              .build();
        }
        return ErrorSpec::Builder().strict_signed_zeros().build();
      })
      .Run();
})

// Can be thought of as an absolute error of
// `<= |std::numeric_limits::<float>::min()|`.
template <typename NativeT, typename NativeRefT>
double DivCpuAbsErr(NativeT left, NativeT right) {
  NativeRefT output =
      static_cast<NativeRefT>(left) / static_cast<NativeRefT>(right);
  // Subnormals are flushed to 0 so we add a absolute error margin that is
  // larger than any subnormal.
  if (IsSubnormal(output)) {
    return std::numeric_limits<NativeRefT>::min();
  }
  return 0.0;
}

BINARY_TEST(Div, {
  DivOp<kT>(this)
      .CpuError(+[](NativeT left, NativeT right) {
        if constexpr (std::is_same_v<NativeT, xla::bfloat16> ||
                      std::is_same_v<NativeT, float> ||
                      std::is_same_v<NativeT, double>) {
          return ErrorSpec::Builder()
              .abs_err(DivCpuAbsErr<NativeT, NativeRefT>(left, right))
              .strict_signed_zeros()
              .build();
        }
        return ErrorSpec::Builder().strict_signed_zeros().build();
      })
      .GpuError(+[](NativeT, NativeT) {
        if constexpr (std::is_same_v<NativeT, xla::half>) {
          return ErrorSpec::Builder()
              .distance_err(1)
              .strict_signed_zeros()
              .build();
        } else if constexpr (std::is_same_v<NativeT, float>) {
          return ErrorSpec::Builder()
              .distance_err(2)
              .strict_signed_zeros()
              .build();
        }
        return ErrorSpec::Builder().strict_signed_zeros().build();
      })
      .Run();
})

// Can be thought of as an absolute error of
// `<= |std::numeric_limits::<float>::min()|`.
template <typename NativeT>
double MaxMinCpuAbsErr(NativeT left, NativeT right) {
  // Subnormals are treated as 0 and max returns the first if all are
  // 0-equivalent.
  if (IsSubnormal(left) && (right == 0.0 || IsSubnormal(right))) {
    return std::abs(left);
  }
  return 0.0;
}

BINARY_TEST(Max, {
  MaxOp<kT>(this)
      .CpuError(+[](NativeT left, NativeT right) {
        if ((std::is_same_v<NativeT, xla::bfloat16> ||
             std::is_same_v<NativeT, float> ||
             std::is_same_v<NativeT, double>)) {
          return ErrorSpec::Builder()
              .abs_err(MaxMinCpuAbsErr(left, right))
              .strict_signed_zeros()
              .build();
        }
        return ErrorSpec::Builder().strict_signed_zeros().build();
      })
      .GpuError(+[](NativeT, NativeT) {
        // A100 and H100 return -0 for max(-0,0).
        return ErrorSpec::Builder().strict_signed_zeros(false).build();
      })
      .Run();
})

BINARY_TEST(Min, {
  MinOp<kT>(this)
      .CpuError(+[](NativeT left, NativeT right) {
        if (std::is_same_v<NativeT, xla::bfloat16> ||
            std::is_same_v<NativeT, float> || std::is_same_v<NativeT, double>) {
          return ErrorSpec::Builder()
              .abs_err(MaxMinCpuAbsErr(left, right))
              .strict_signed_zeros()
              .build();
        }
        return ErrorSpec::Builder().strict_signed_zeros().build();
      })
      .GpuError(+[](NativeT, NativeT) {
        // A100 and H100 return 0 for min(0,-0).
        return ErrorSpec::Builder().strict_signed_zeros(false).build();
      })
      .Run();
})

template <typename NativeT, typename NativeRefT>
double PowCpuBf16F32AbsErr(NativeT left, NativeT right) {
  NativeRefT output =
      std::pow(static_cast<NativeRefT>(left), static_cast<NativeRefT>(right));

  // Output is flushed to 0 if subnormal.
  if (IsSubnormal(output)) {
    return std::numeric_limits<NativeRefT>::min();
  }

  // TODO(b/359325328): pow computation for subnormal bases is different from
  // std::pow.
  //
  // If the base is subnormal, the output computation selects a different base.
  // The minimum value ever chosen is slightly greater than the 1e-91 used
  // below. We return an absolute error from this value to the "real" output.
  //
  // Because the exponent (right) can be any floating point value, this allows
  // an arbitrary absolute error for subnormal values.
  if (IsSubnormal(left)) {
    NativeT output_as_bf16 = static_cast<NativeT>(output);
    auto expected = std::pow(1e-91, static_cast<double>(right));
    auto err = std::abs(expected - output_as_bf16);
    if (!std::isnan(err)) {
      return err;
    }
  }

  return 0.0;
}

bool PowCpuF64Skip(double left, double right) {
  // Hardware returns 0 if right is positive and inf otherwise.
  if ((IsSubnormal(left) || std::isinf(left) || left == 0) &&
      IsSubnormal(right)) {
    return true;
  }
  return false;
}

template <typename NativeT>
bool PowCpuGpuF16Skip(NativeT left, NativeT right) {
  // Hardware always returns 1 if right is 0, no matter if left is NaN.
  if (std::isnan(left) && right == 0.0f) {
    return true;
  }
  // Hardware always returns 1 if left is 1, no matter if right is NaN.
  if (left == 1.0f && std::isnan(right)) {
    return true;
  }
  return false;
}

BINARY_TEST(Pow, {
  PowOp<kT>(this)
      .CpuError(+[](NativeT left, NativeT right) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e4m3fn> ||
                      std::is_same_v<NativeT, tsl::float8_e5m2>) {
          return ErrorSpec::Builder()
              .distance_err(1)
              .strict_signed_zeros()
              .build();
        } else if constexpr (std::is_same_v<NativeT, xla::half>) {
          return ErrorSpec::Builder()
              .strict_signed_zeros()
              .skip_comparison(PowCpuGpuF16Skip(left, right))
              .build();
        } else if constexpr (std::is_same_v<NativeT, xla::bfloat16> ||
                             std::is_same_v<NativeT, float>) {
          return ErrorSpec::Builder()
              .abs_err(PowCpuBf16F32AbsErr<NativeT, NativeRefT>(left, right))
              .strict_signed_zeros()
              .build();
        } else if constexpr (std::is_same_v<NativeT, double>) {
          return ErrorSpec::Builder()
              .strict_signed_zeros()
              .skip_comparison(PowCpuF64Skip(static_cast<double>(left),
                                             static_cast<double>(right)))
              .build();
        }
        return ErrorSpec::Builder().strict_signed_zeros().build();
      })
      .GpuError(+[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .distance_err(1)
            .strict_signed_zeros()
            .skip_comparison(PowCpuGpuF16Skip(left, right))
            .build();
      })
      .Run();
})

// Can be thought of as an absolute error of
// `<= |std::numeric_limits::<float>::min()|`.
template <typename NativeT, typename NativeRefT>
double Atan2CpuBf16F32F64AbsErr(NativeT left, NativeT right) {
  NativeRefT output =
      std::atan2(static_cast<NativeRefT>(left), static_cast<NativeRefT>(right));
  // If the output would be a subnormal float, we allow some error to account
  // for BF16 implementation flushing subnormals to zero.
  if (IsSubnormal(output)) {
    return std::numeric_limits<NativeRefT>::min();
  }
  return 0.0;
}

template <typename NativeT>
bool Atan2CpuBf16F32Skip(NativeT left, NativeT right) {
  // Subnormals are flushed to 0, but 0/0 returns NaN instead of
  // <subnormal>/<subnormal> which returns some positive number. We cannot set
  // an error to compare against NaN.
  if (IsSubnormal(left) && IsSubnormal(right)) {
    return true;
  }
  return false;
}

BINARY_TEST(Atan2, {
  Atan2Op<kT>(this)
      .CpuError([](NativeT left, NativeT right) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e4m3fn> ||
                      std::is_same_v<NativeT, tsl::float8_e5m2>) {
          return ErrorSpec::Builder()
              .distance_err(1)
              .strict_signed_zeros()
              .build();

        } else if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
          return ErrorSpec::Builder()
              .abs_err(
                  Atan2CpuBf16F32F64AbsErr<NativeT, NativeRefT>(left, right))
              .strict_signed_zeros()
              .skip_comparison(Atan2CpuBf16F32Skip(left, right))
              .build();
        } else if constexpr (std::is_same_v<NativeT, float>) {
          return ErrorSpec::Builder()
              .abs_err(
                  Atan2CpuBf16F32F64AbsErr<NativeT, NativeRefT>(left, right))
              // Only used when right is subnormal.
              .distance_err(2)
              .strict_signed_zeros()
              .skip_comparison(Atan2CpuBf16F32Skip(left, right))
              .build();
        } else if constexpr (std::is_same_v<NativeT, double>) {
          return ErrorSpec::Builder()
              .abs_err(
                  Atan2CpuBf16F32F64AbsErr<NativeT, NativeRefT>(left, right))
              .strict_signed_zeros()
              .build();
        }
        return ErrorSpec::Builder().strict_signed_zeros().build();
      })
      .GpuError(+[](NativeT, NativeT) {
        if constexpr (std::is_same_v<NativeT, tsl::float8_e4m3fn> ||
                      std::is_same_v<NativeT, tsl::float8_e5m2>) {
          return ErrorSpec::Builder()
              .distance_err(1)
              .strict_signed_zeros()
              .build();
        }
        if constexpr (std::is_same_v<NativeT, xla::half> ||
                      std::is_same_v<NativeT, xla::bfloat16>) {
          return ErrorSpec::Builder()
              .distance_err(1)
              .strict_signed_zeros()
              .build();
        } else if constexpr (std::is_same_v<NativeT, float>) {
          return ErrorSpec::Builder()
              .distance_err(3)
              .strict_signed_zeros()
              .build();
        } else if constexpr (std::is_same_v<NativeT, double>) {
          return ErrorSpec::Builder()
              .distance_err(2)
              .strict_signed_zeros()
              .build();
        }
        return ErrorSpec::Builder().strict_signed_zeros().build();
      })
      .Run();
})

// Can be thought of as an absolute error of
// `<= |std::numeric_limits::<float>::min()|`.
template <typename NativeRefT>
double AbsComplexCpuAbsErr(NativeRefT real, NativeRefT imag) {
  // absolute value (distance) short circuits if the first component is
  // subnormal.
  if (!std::isnan(real) && IsSubnormal(real)) {
    return std::abs(real);
  }
  return 0.0;
}

template <typename NativeRefT>
bool AbsComplexSkip(NativeRefT real, NativeRefT imag) {
  // TODO(timshen): see b/162664705.
  return std::isnan(real) || std::isnan(imag);
}

// It is more convenient to implement Abs(complex) as a binary op than a unary
// op, as the operations we currently support all have the same data type for
// the source operands and the results.
// TODO(bixia): May want to move this test to unary test if we will be able to
// implement Abs(complex) as unary conveniently.
BINARY_TEST_COMPLEX(AbsComplex, {
  AbsComplexOp<kT>(this)
      .CpuError(+[](NativeRefT real, NativeRefT imag) {
        if constexpr (std::is_same_v<NativeT, float> ||
                      std::is_same_v<NativeT, double>) {
          return ErrorSpec::Builder()
              .abs_err(AbsComplexCpuAbsErr(real, imag))
              .distance_err(2)
              .skip_comparison(AbsComplexSkip(real, imag))
              .build();
        }
        return ErrorSpec::Builder().strict_signed_zeros().build();
      })
      .GpuError(+[](NativeRefT real, NativeRefT imag) {
        if constexpr (std::is_same_v<NativeT, float>) {
          return ErrorSpec::Builder()
              .distance_err(3)
              .skip_comparison(AbsComplexSkip(real, imag))
              .build();
        } else if constexpr (std::is_same_v<NativeT, double>) {
          return ErrorSpec::Builder()
              .distance_err(2)
              .skip_comparison(AbsComplexSkip(real, imag))
              .build();
        }
        return ErrorSpec::Builder().strict_signed_zeros().build();
      })
      .Run();
})

}  // namespace
}  // namespace exhaustive_op_test
}  // namespace xla
