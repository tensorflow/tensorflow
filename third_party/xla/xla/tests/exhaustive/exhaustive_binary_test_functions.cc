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

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>
#include <type_traits>

#include "xla/client/xla_builder.h"
#include "xla/tests/exhaustive/exhaustive_binary_test_definitions.h"
#include "xla/tests/exhaustive/exhaustive_op_test_utils.h"
#include "xla/types.h"

#ifdef __FAST_MATH__
#error("Can't be compiled with fast math on");
#endif

namespace xla {
namespace exhaustive_op_test {
namespace {

// Can be thought of as an absolute error of
// `<= |std::numeric_limits::<float>::min()|`.
template <typename NativeT, typename NativeRefT>
double AddCpuTpuAbsErr(NativeT left, NativeT right) {
  NativeRefT output =
      static_cast<NativeRefT>(left) + static_cast<NativeRefT>(right);

  // Hardware flushes subnormal outputs to 0.
  if (IsSubnormal(output)) {
    return std::numeric_limits<NativeRefT>::min();
  }

  return 0.0;
}

BINARY_TEST(Add, {
  ErrorSpecGen error_spec_gen = +[](NativeT, NativeT) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };

  if ((IsCpu(platform_) || IsTpu(platform_))) {
    if (std::is_same_v<NativeT, xla::bfloat16> ||
        std::is_same_v<NativeT, float> || std::is_same_v<NativeT, double>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(AddCpuTpuAbsErr<NativeT, NativeRefT>(left, right))
            .strict_signed_zeros()
            .build();
      };
    }
  }

  Run(
      AddEmptyBroadcastDimension(Add),
      [](NativeRefT x, NativeRefT y) { return x + y; }, error_spec_gen);
})

// Can be thought of as an absolute error of
// `<= |std::numeric_limits::<float>::min()|`.
template <typename NativeT, typename NativeRefT>
double SubCpuTpuAbsErr(NativeT left, NativeT right) {
  NativeRefT output =
      static_cast<NativeRefT>(left) - static_cast<NativeRefT>(right);

  // Hardware flushes subnormal outputs to 0.
  if (IsSubnormal(output)) {
    return std::numeric_limits<NativeRefT>::min();
  }

  return 0.0;
}

BINARY_TEST(Sub, {
  ErrorSpecGen error_spec_gen = +[](NativeT, NativeT) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };

  if (IsCpu(platform_) || IsTpu(platform_)) {
    if (std::is_same_v<NativeT, xla::bfloat16> ||
        std::is_same_v<NativeT, float> || std::is_same_v<NativeT, double>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(SubCpuTpuAbsErr<NativeT, NativeRefT>(left, right))
            .strict_signed_zeros()
            .build();
      };
    }
  }

  Run(
      AddEmptyBroadcastDimension(Sub),
      [](NativeRefT x, NativeRefT y) { return x - y; }, error_spec_gen);
})

// Can be thought of as an absolute error of
// `<= |std::numeric_limits::<float>::min()|`.
template <typename NativeT, typename NativeRefT>
double MulCpuTpuAbsErr(NativeT left, NativeT right) {
  NativeRefT output =
      static_cast<NativeRefT>(left) * static_cast<NativeRefT>(right);

  // CPU BF16 and TPU (all types) flush subnormals to 0.
  auto output_is_subnormal = IsSubnormal(output);
  if (output_is_subnormal) {
    return std::numeric_limits<NativeRefT>::min();
  }

  return 0.0;
}

bool MulCpuTpuBf16Skip(xla::bfloat16 left, xla::bfloat16 right) {
  // For CPU and TPU BF16, multiplying a subnormal by infinity will lead to
  // calculating 0 multiplied by infinity due to subnormal flushing, which is
  // defined to be NaN. However, the calculation in higher precision does not
  // flush the subnormal value to 0, leading to a result of infinity.
  if ((IsSubnormal(left) && std::isinf(right)) ||
      (std::isinf(left) && IsSubnormal(right))) {
    return true;
  }
  return false;
}

BINARY_TEST(Mul, {
  ErrorSpecGen error_spec_gen = +[](NativeT left, NativeT right) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };

  if (IsCpu(platform_) || IsTpu(platform_)) {
    if (std::is_same_v<NativeT, xla::bfloat16>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(MulCpuTpuAbsErr<NativeT, NativeRefT>(left, right))
            .strict_signed_zeros()
            .skip_comparison(
                MulCpuTpuBf16Skip(static_cast<xla::bfloat16>(left),
                                  static_cast<xla::bfloat16>(right)))
            .build();
      };
    }
    if (std::is_same_v<NativeT, float>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(MulCpuTpuAbsErr<NativeT, NativeRefT>(left, right))
            .strict_signed_zeros()
            .build();
      };
    }
  }

  Run(
      AddEmptyBroadcastDimension(Mul),
      [](NativeRefT x, NativeRefT y) { return x * y; }, error_spec_gen);
})

// Can be thought of as an absolute error of
// `<= |std::numeric_limits::<float>::min()|`.
template <typename NativeT, typename NativeRefT>
double DivCpuTpuAbsErr(NativeT left, NativeT right) {
  NativeRefT output =
      static_cast<NativeRefT>(left) / static_cast<NativeRefT>(right);

  // Subnormals are flushed to 0 so we add a absolute error margin that is
  // larger than any subnormal.
  if (IsSubnormal(output)) {
    return std::numeric_limits<NativeRefT>::min();
  }

  return 0.0;
}

template <typename NativeT, typename NativeRefT>
double DivTpuAbsErr(NativeT left, NativeT right) {
  NativeRefT reciprocal = 1.0f / static_cast<NativeRefT>(right);
  NativeT output = left / right;
  NativeRefT output_as_native_ref_t =
      static_cast<NativeRefT>(left) / static_cast<NativeRefT>(right);

  // If we calculate NaN, we don't need to adjust tolerances.
  if (std::isnan(output_as_native_ref_t)) {
    return 0.0;
  }

  // TPUs perform `left * (1 / right)`, where `left` and `1 / right` are
  // flushed to `0` if they are subnormal. Also applies to if reciprocal is min
  // normal.
  if (IsSubnormal(left) || IsSubnormal(reciprocal)) {
    // Subnormals can have a larger value in BF16 than float due to rounding to
    // the nearest BF16 value during conversion while having less representation
    // bits. For normals, the float value is usually always bigger due to
    // greater precision.
    return std::max(std::abs(output), std::abs(output_as_native_ref_t));
  }

  // For subnormals, we need to set absolute error to the smallest positive
  // representable value due to hardware implementations that truncate
  // subnormals to zero.
  if (IsSubnormal(output)) {
    return std::numeric_limits<NativeT>::min();
  }

  return 0.0;
}

template <typename NativeT, typename NativeRefT>
double DivTpuBf16F32AbsErr(NativeT left, NativeT right) {
  NativeRefT reciprocal = 1.0f / static_cast<NativeRefT>(right);
  NativeT output = left / right;
  NativeRefT output_as_native_ref_t =
      static_cast<NativeRefT>(left) / static_cast<NativeRefT>(right);

  // If we calculate NaN, we don't need to adjust tolerances.
  if (std::isnan(output_as_native_ref_t)) {
    return 0.0;
  }

  // TPUs perform `left * (1 / right)`, where `left` and `1 / right` are
  // flushed to `0` if they are subnormal. Also applies to if reciprocal is min
  // normal.
  if (IsSubnormal(left) || IsSubnormalOrMinNormal(reciprocal)) {
    // Subnormals can have a larger value in BF16 than float due to rounding to
    // the nearest BF16 value during conversion while having less representation
    // bits. For normals, the float value is usually always bigger due to
    // greater precision.
    return std::max(std::abs(output), std::abs(output_as_native_ref_t));
  }

  // For subnormals, we need to set absolute error to the smallest positive
  // representable value due to hardware implementations that truncate
  // subnormals to zero.
  if (IsSubnormalOrMinNormal(output)) {
    return std::numeric_limits<NativeT>::min();
  }

  return 0.0;
}

template <typename NativeT, typename NativeRefT>
bool DivTpuBf16F32Skip(NativeT left, NativeT right) {
  NativeRefT reciprocal = 1.0f / right;

  // TPU calculates `left * (1 / right)` and flushed `(1 / right)` to `0` when
  // it is subnormal or min normal. It also follows the IEEE multiplication spec
  // that inf * 0 is NaN. However, IEEE division of infinity by a subnormal is
  // infinity, so we must skip comparison.
  if (std::isinf(left) && IsSubnormalOrMinNormal(reciprocal)) {
    return true;
  }

  return false;
}

BINARY_TEST(Div, {
  ErrorSpecGen error_spec_gen = +[](NativeT, NativeT) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };

  if (IsCpu(platform_) &&
      (std::is_same_v<NativeT, xla::bfloat16> ||
       std::is_same_v<NativeT, float> || std::is_same_v<NativeT, double>)) {
    error_spec_gen = +[](NativeT left, NativeT right) {
      return ErrorSpec::Builder()
          .abs_err(DivCpuTpuAbsErr<NativeT, NativeRefT>(left, right))
          .strict_signed_zeros()
          .build();
    };
  }

  if (IsGpu(platform_)) {
    if (std::is_same_v<NativeT, xla::half>) {
      error_spec_gen = +[](NativeT, NativeT) {
        return ErrorSpec::Builder()
            .distance_err(1)
            .strict_signed_zeros()
            .build();
      };
    } else if (std::is_same_v<NativeT, float>) {
      error_spec_gen = +[](NativeT, NativeT) {
        return ErrorSpec::Builder()
            .distance_err(2)
            .strict_signed_zeros()
            .build();
      };
    }
  }

  if (IsTpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(DivTpuBf16F32AbsErr<NativeT, NativeRefT>(left, right))
            .strict_signed_zeros()
            .skip_comparison(
                DivTpuBf16F32Skip<NativeT, NativeRefT>(left, right))
            .build();
      };
    } else if constexpr (std::is_same_v<NativeT, xla::half>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            // This is basically distance_err(1), but is tighter because it
            // guarantees this only happens when the abs_err is less than min
            // normal.
            .abs_err(std::numeric_limits<NativeT>::min())
            .strict_signed_zeros()
            .build();
      };
    } else if constexpr (std::is_same_v<NativeT, float>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(DivTpuAbsErr<NativeT, NativeRefT>(left, right))
            .distance_err(2)
            .strict_signed_zeros()
            .skip_comparison(
                DivTpuBf16F32Skip<NativeT, NativeRefT>(left, right))
            .build();
      };
    }
  }
  if (IsPreV6Tpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, float>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .abs_err(DivTpuAbsErr<NativeT, NativeRefT>(left, right))
            .rel_err(34 * eps)
            .strict_signed_zeros()
            .skip_comparison(
                DivTpuBf16F32Skip<NativeT, NativeRefT>(left, right))
            .build();
      };
    }
  }
  if (IsPreV5Tpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .abs_err(DivTpuBf16F32AbsErr<NativeT, NativeRefT>(left, right))
            .rel_err(eps)
            .strict_signed_zeros()
            .skip_comparison(
                DivTpuBf16F32Skip<NativeT, NativeRefT>(left, right))
            .build();
      };
    } else if constexpr (std::is_same_v<NativeT, float>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .abs_err(DivTpuAbsErr<NativeT, NativeRefT>(left, right))
            .rel_err(136 * eps)
            .strict_signed_zeros()
            .skip_comparison(
                DivTpuBf16F32Skip<NativeT, NativeRefT>(left, right))
            .build();
      };
    }
  }

  Run(
      AddEmptyBroadcastDimension(Div),
      [](NativeRefT x, NativeRefT y) { return x / y; }, error_spec_gen);
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
  ErrorSpecGen error_spec_gen = +[](NativeT, NativeT) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };

  if (IsCpu(platform_) &&
      (std::is_same_v<NativeT, xla::bfloat16> ||
       std::is_same_v<NativeT, float> || std::is_same_v<NativeT, double>)) {
    error_spec_gen = +[](NativeT left, NativeT right) {
      return ErrorSpec::Builder()
          .abs_err(MaxMinCpuAbsErr(left, right))
          .strict_signed_zeros()
          .build();
    };
  }

  if (IsGpu(platform_) || IsTpu(platform_)) {
    error_spec_gen = +[](NativeT, NativeT) {
      // A100 and H100 return -0 for max(-0,0).
      //
      // TPUs return -0 for max(0,-0) and 0 for max(-0,0).
      return ErrorSpec::Builder().strict_signed_zeros(false).build();
    };
  }

  Run(AddEmptyBroadcastDimension(Max), ReferenceMax<NativeRefT>,
      error_spec_gen);
})

BINARY_TEST(Min, {
  ErrorSpecGen error_spec_gen = +[](NativeT, NativeT) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };

  if (IsCpu(platform_) &&
      (std::is_same_v<NativeT, xla::bfloat16> ||
       std::is_same_v<NativeT, float> || std::is_same_v<NativeT, double>)) {
    error_spec_gen = +[](NativeT left, NativeT right) {
      return ErrorSpec::Builder()
          .abs_err(MaxMinCpuAbsErr(left, right))
          .strict_signed_zeros()
          .build();
    };
  }

  if (IsGpu(platform_) || IsTpu(platform_)) {
    error_spec_gen = +[](NativeT, NativeT) {
      // A100 and H100 return 0 for min(0,-0).
      //
      // TPUs return 0 for min(-0,0) and -0 for min(0,-0).
      return ErrorSpec::Builder().strict_signed_zeros(false).build();
    };
  }

  Run(AddEmptyBroadcastDimension(Min), ReferenceMin<NativeRefT>,
      error_spec_gen);
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

double PowTpuBf16AbsErr(xla::bfloat16 left, xla::bfloat16 right) {
  float output = std::pow(static_cast<float>(left), static_cast<float>(right));

  // Output is flushed to 0 if subnormal.
  if (IsSubnormal(output)) {
    return std::numeric_limits<float>::min();
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

template <typename NativeT>
bool PowTpuSkip(NativeT left, NativeT right) {
  // Hardware always returns 1 if right is 0 (or subnormal due to
  // flushing subnormals to zero before the operation), no matter if left is
  // NaN.
  if (std::isnan(left) && (right == 0.0f || IsSubnormal(right))) {
    return true;
  }
  // Hardware always returns 1 if left is 1, no matter if right is NaN.
  if (left == 1.0f && std::isnan(right)) {
    return true;
  }

  return false;
}

BINARY_TEST(Pow, {
  ErrorSpecGen error_spec_gen = +[](NativeT, NativeT) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };

  if (IsCpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, xla::half>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .strict_signed_zeros()
            .skip_comparison(PowCpuGpuF16Skip(left, right))
            .build();
      };
    } else if constexpr (std::is_same_v<NativeT, xla::bfloat16> ||
                         std::is_same_v<NativeT, float>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(PowCpuBf16F32AbsErr<NativeT, NativeRefT>(left, right))
            .strict_signed_zeros()
            .build();
      };
    } else if constexpr (std::is_same_v<NativeT, double>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .strict_signed_zeros()
            .skip_comparison(PowCpuF64Skip(static_cast<double>(left),
                                           static_cast<double>(right)))
            .build();
      };
    }
  }

  if (IsGpu(platform_)) {
    error_spec_gen = +[](NativeT left, NativeT right) {
      return ErrorSpec::Builder()
          .distance_err(1)
          .strict_signed_zeros()
          .skip_comparison(PowCpuGpuF16Skip(left, right))
          .build();
    };
  }

  if (IsTpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(PowTpuBf16AbsErr(static_cast<xla::bfloat16>(left),
                                      static_cast<xla::bfloat16>(right)))
            .distance_err(1)
            .strict_signed_zeros()
            .skip_comparison(PowTpuSkip(left, right))
            .build();
      };
    } else if constexpr (std::is_same_v<NativeT, xla::half>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .distance_err(1)
            .strict_signed_zeros()
            .skip_comparison(PowTpuSkip(left, right))
            .build();
      };
    } else if constexpr (std::is_same_v<NativeT, float>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .distance_err(8)
            .strict_signed_zeros()
            .skip_comparison(PowTpuSkip(left, right))
            .build();
      };
    }
  }
  if (IsPreV6Tpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, float>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .rel_err(41 * eps)
            .strict_signed_zeros()
            .skip_comparison(PowTpuSkip(left, right))
            .build();
      };
    }
  }
  if (IsPreV5Tpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, float>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .rel_err(44 * eps)
            .strict_signed_zeros()
            .skip_comparison(PowTpuSkip(left, right))
            .build();
      };
    }
  }

  Run(AddEmptyBroadcastDimension(Pow), std::pow, error_spec_gen);
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

template <typename NativeT, typename NativeRefT>
double Atan2TpuBf16F32AbsErr(NativeT left, NativeT right) {
  NativeT output = static_cast<NativeT>(std::atan2(left, right));
  NativeRefT output_as_float =
      std::atan2(static_cast<NativeRefT>(left), static_cast<NativeRefT>(right));

  // If the output would be a subnormal float, we allow some error to account
  // for BF16 implementation flushing subnormals to zero. TPUs also seem to
  // flush the minimum value to 0 along with subnormals.
  if (IsSubnormalOrMinNormal(output_as_float)) {
    return std::numeric_limits<xla::bfloat16>::min();
  }

  // Implementation of Atan2 on TPUs is that they take the reciprocal of the
  // larger of left or right. If this is subnormal or the minimum value, the TPU
  // flushes it to 0 before using it in multiplication. When this happens, the
  // error is the output calculation, either in BF16 or float, or PI/2,
  // depending on which of the three is bigger.
  NativeRefT reciprocal_as_float =
      1.0f / std::max(std::abs(static_cast<NativeRefT>(left)),
                      std::abs(static_cast<NativeRefT>(right)));
  if (!std::isnan(output_as_float) && IsSubnormal(reciprocal_as_float)) {
    return std::max({std::abs(output_as_float), std::abs(output),
                     static_cast<NativeRefT>(M_PI_2)});
  }

  return 0.0;
}

BINARY_TEST(Atan2, {
  auto error_spec_gen = +[](NativeT, NativeT) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };

  if (IsCpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(Atan2CpuBf16F32F64AbsErr<NativeT, NativeRefT>(left, right))
            .strict_signed_zeros()
            .skip_comparison(Atan2CpuBf16F32Skip(left, right))
            .build();
      };
    } else if constexpr (std::is_same_v<NativeT, float>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(Atan2CpuBf16F32F64AbsErr<NativeT, NativeRefT>(left, right))
            // Only used when right is subnormal.
            .distance_err(2)
            .strict_signed_zeros()
            .skip_comparison(Atan2CpuBf16F32Skip(left, right))
            .build();
      };
    } else if constexpr (std::is_same_v<NativeT, double>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(Atan2CpuBf16F32F64AbsErr<NativeT, NativeRefT>(left, right))
            .strict_signed_zeros()
            .build();
      };
    }
  }

  if (IsGpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, xla::half> ||
                  std::is_same_v<NativeT, xla::bfloat16>) {
      error_spec_gen = +[](NativeT, NativeT) {
        return ErrorSpec::Builder()
            .distance_err(1)
            .strict_signed_zeros()
            .build();
      };
    } else if constexpr (std::is_same_v<NativeT, float>) {
      error_spec_gen = +[](NativeT, NativeT) {
        return ErrorSpec::Builder()
            .distance_err(3)
            .strict_signed_zeros()
            .build();
      };
    } else if constexpr (std::is_same_v<NativeT, double>) {
      error_spec_gen = +[](NativeT, NativeT) {
        return ErrorSpec::Builder()
            .distance_err(2)
            .strict_signed_zeros()
            .build();
      };
    }
  }

  if (IsTpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, xla::bfloat16>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(Atan2TpuBf16F32AbsErr<NativeT, NativeRefT>(left, right))
            .distance_err(1)
            .strict_signed_zeros()
            .build();
      };
    } else if constexpr (std::is_same_v<NativeT, xla::half>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .distance_err(1)
            .strict_signed_zeros()
            .build();
      };
    } else if constexpr (std::is_same_v<NativeT, float>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        return ErrorSpec::Builder()
            .abs_err(Atan2TpuBf16F32AbsErr<NativeT, NativeRefT>(left, right))
            .distance_err(3)
            .strict_signed_zeros()
            .build();
      };
    }
  }
  if (IsPreV6Tpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, float>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .abs_err(Atan2TpuBf16F32AbsErr<NativeT, NativeRefT>(left, right))
            .rel_err(28 * eps)
            .strict_signed_zeros()
            .build();
      };
    }
  }
  if (IsPreV5Tpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, float>) {
      error_spec_gen = +[](NativeT left, NativeT right) {
        NativeT eps = std::numeric_limits<NativeT>::epsilon();
        return ErrorSpec::Builder()
            .abs_err(Atan2TpuBf16F32AbsErr<NativeT, NativeRefT>(left, right))
            .rel_err(133 * eps)
            .strict_signed_zeros()
            .build();
      };
    }
  }

  Run(AddEmptyBroadcastDimension(Atan2), std::atan2, error_spec_gen);
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
  if (std::isnan(real) || std::isnan(imag)) {
    return true;
  }
  return false;
}

template <typename NativeRefT>
double AbsComplexTpuRelErr(NativeRefT real, NativeRefT imag) {
  NativeRefT abs_max = std::max(std::abs(real), std::abs(imag));
  NativeRefT kOne(1);
  NativeRefT reciprocal = kOne / abs_max;
  if (IsSubnormal(reciprocal)) {
    // In this case, the reciprocal erroneously returns zero, and
    // we get max(|real|, |imag|) instead of sqrt(real^2 + imag^2),
    // so the relative error can be as large as (sqrt(2)-1)/sqrt(2) ~= 0.293,
    // when using the typical hypot implementation hypot(max, min) = max *
    // sqrt(1 + min / max).
    return 0.293;
  }
  return 0.0;
}

// It is more convenient to implement Abs(complex) as a binary op than a unary
// op, as the operations we currently support all have the same data type for
// the source operands and the results.
// TODO(bixia): May want to move this test to unary test if we will be able to
// implement Abs(complex) as unary conveniently.
BINARY_TEST_COMPLEX(AbsComplex, {
  ErrorSpecGen error_spec_gen = +[](NativeRefT, NativeRefT) {
    return ErrorSpec::Builder().strict_signed_zeros().build();
  };

  if (IsCpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, float> ||
                  std::is_same_v<NativeT, double>) {
      error_spec_gen = +[](NativeRefT real, NativeRefT imag) {
        return ErrorSpec::Builder()
            .abs_err(AbsComplexCpuAbsErr(real, imag))
            .distance_err(2)
            .skip_comparison(AbsComplexSkip(real, imag))
            .build();
      };
    }
  }

  if (IsGpu(platform_)) {
    if constexpr (std::is_same_v<NativeT, float>) {
      error_spec_gen = +[](NativeRefT real, NativeRefT imag) {
        return ErrorSpec::Builder()
            .distance_err(3)
            .skip_comparison(AbsComplexSkip(real, imag))
            .build();
      };
    } else if constexpr (std::is_same_v<NativeT, double>) {
      error_spec_gen = +[](NativeRefT real, NativeRefT imag) {
        return ErrorSpec::Builder()
            .distance_err(2)
            .skip_comparison(AbsComplexSkip(real, imag))
            .build();
      };
    }
  }

  if (IsTpu(platform_)) {
    error_spec_gen = +[](NativeRefT real, NativeRefT imag) {
      return ErrorSpec::Builder()
          .rel_err(AbsComplexTpuRelErr(real, imag))
          .distance_err(3)
          .skip_comparison(AbsComplexSkip(real, imag))
          .build();
    };
  }
  if (IsPreV6Tpu(platform_)) {
    error_spec_gen = +[](NativeRefT real, NativeRefT imag) {
      return ErrorSpec::Builder()
          .rel_err(AbsComplexTpuRelErr(real, imag))
          .distance_err(125)
          .skip_comparison(AbsComplexSkip(real, imag))
          .build();
    };
  }

  Run([](XlaOp x, XlaOp y) { return Abs(Complex(x, y)); },
      [](NativeRefT x, NativeRefT y) {
        return std::abs(std::complex<NativeRefT>(x, y));
      },
      error_spec_gen);
})

}  // namespace
}  // namespace exhaustive_op_test
}  // namespace xla
