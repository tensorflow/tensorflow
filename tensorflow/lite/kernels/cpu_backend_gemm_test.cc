/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/cpu_backend_gemm.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "ruy/matrix.h"  // from @ruy
#include "ruy/reference_mul.h"  // from @ruy
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_params.h"
#include "tensorflow/lite/kernels/cpu_backend_gemm_ruy.h"

namespace tflite {

namespace {

using cpu_backend_gemm::Gemm;
using cpu_backend_gemm::GemmParams;
using cpu_backend_gemm::MatrixParams;
using cpu_backend_gemm::QuantizationFlavor;

template <typename Scalar>
std::string ToString(const std::vector<Scalar>& vector) {
  std::stringstream s;
  if (vector.empty()) {
    s << "{}";
  } else {
    s << "{ " << static_cast<double>(vector[0]);
    for (int i = 1; i < vector.size(); i++) {
      s << ", " << static_cast<double>(vector[i]);
    }
    s << "}";
  }
  return s.str();
}

template <typename Scalar>
void MakeDeterministicPseudoRandomVector(int size,
                                         std::vector<Scalar>* vector) {
  // Intentionally create a new local random_engine in each invocation,
  // so pseudorandom values don't depend on invocation order.
  // Otherwise, test results would be affecting by e.g. filtering.
  std::default_random_engine random_engine;
  (void)random_engine();
  // Do not use std::uniform*_distribution: the values that it
  // generates are implementation-defined.
  const double random_min = static_cast<double>(random_engine.min());
  const double random_max = static_cast<double>(random_engine.max());
  const double result_min =
      std::is_floating_point<Scalar>::value
          ? -1.0
          : std::max(-256., static_cast<double>(
                                std::numeric_limits<Scalar>::lowest()));
  const double result_max =
      std::is_floating_point<Scalar>::value
          ? 1.0
          : std::min(256.,
                     static_cast<double>(std::numeric_limits<Scalar>::max()));
  const double random_scale =
      (result_max - result_min) / (random_max - random_min);

  vector->resize(size);
  for (int i = 0; i < size; i++) {
    double val = random_scale * (random_engine() - random_min);
    val = std::max(val,
                   static_cast<double>(std::numeric_limits<Scalar>::lowest()));
    val =
        std::min(val, static_cast<double>(std::numeric_limits<Scalar>::max()));
    (*vector)[i] = static_cast<Scalar>(val);
  }
}

template <typename Scalar>
void MakeVectorFilledWithConsecutiveInts(int size,
                                         std::vector<Scalar>* vector) {
  vector->resize(size);
  EXPECT_LE(size, std::numeric_limits<Scalar>::max());
  for (int i = 0; i < size; i++) {
    (*vector)[i] = static_cast<Scalar>(i + 1);
  }
}

template <typename Scalar>
Scalar Median(const std::vector<Scalar>& vector) {
  EXPECT_GT(vector.size(), 0);
  std::vector<Scalar> vector_copy = vector;
  std::sort(std::begin(vector_copy), std::end(vector_copy));
  return vector_copy[vector_copy.size() / 2];
}

template <typename Scalar>
double MedianAbs(const std::vector<Scalar>& vector) {
  EXPECT_GT(vector.size(), 0);
  std::vector<double> vector_abs;
  vector_abs.resize(vector.size());
  for (int i = 0; i < vector.size(); i++) {
    vector_abs[i] = std::abs(static_cast<double>(vector[i]));
  }
  std::sort(std::begin(vector_abs), std::end(vector_abs));
  return vector_abs[vector_abs.size() / 2];
}

template <typename Scalar>
void Clamp(const std::vector<Scalar>& src, Scalar clamp_min, Scalar clamp_max,
           std::vector<Scalar>* dst) {
  dst->resize(src.size());
  for (int i = 0; i < src.size(); i++) {
    (*dst)[i] = std::max(std::min(src[i], clamp_max), clamp_min);
  }
}

template <typename AccumScalar, typename DstScalar,
          QuantizationFlavor quantization_flavor>
void Clamp(const GemmParams<AccumScalar, DstScalar, quantization_flavor>& src,
           DstScalar clamp_min, DstScalar clamp_max,
           GemmParams<AccumScalar, DstScalar, quantization_flavor>* dst) {
  *dst = src;
  dst->clamp_min = clamp_min;
  dst->clamp_max = clamp_max;
}

struct ErrorStats {
  int size;
  double scale_factor;
  double max_abs_diff;
  double mean_abs_diff;
  double abs_mean_diff;
};

template <typename Scalar>
void ComputeErrorStats(const std::vector<Scalar>& actual,
                       const std::vector<Scalar>& expected,
                       ErrorStats* error_stats) {
  double max_abs_diff = 0;
  double sum_abs_diff = 0;
  double sum_diff = 0;
  double max_abs_expected = 0;
  EXPECT_EQ(actual.size(), expected.size());
  for (int i = 0; i < actual.size(); i++) {
    double actual_val = static_cast<double>(actual[i]);
    double expected_val = static_cast<double>(expected[i]);
    double diff = actual_val - expected_val;
    max_abs_expected = std::max(max_abs_expected, std::abs(expected_val));
    sum_diff += diff;
    sum_abs_diff += std::abs(diff);
    max_abs_diff = std::max(max_abs_diff, std::abs(diff));
  }
  error_stats->scale_factor = max_abs_expected;
  error_stats->max_abs_diff = max_abs_diff;
  error_stats->mean_abs_diff = sum_abs_diff / actual.size();
  error_stats->abs_mean_diff = std::abs(sum_diff / actual.size());
  error_stats->size = actual.size();
}

template <typename AccumScalar, typename DstScalar>
bool CheckErrorStats(const ErrorStats& error_stats, int accumulation_depth) {
  double tolerated_relative_max_abs_diff = 0;
  double tolerated_relative_mean_abs_diff = 0;
  double tolerated_relative_abs_mean_diff = 0;

  double inverse_size = 1. / error_stats.size;

  if (std::is_floating_point<AccumScalar>::value) {
    // Somewhat naive requirement: the worst case should be epsilons
    // adding up towards the same direction, on values of same magnitude.
    tolerated_relative_max_abs_diff =
        accumulation_depth * std::numeric_limits<DstScalar>::epsilon();
    // Naive interpretation of the Central Limit Theorem is the rationale
    // for the sqrt here. We haven't even worked out the correct scale factor,
    // or how applicable that theorem is here (the random variables being added
    // might not be mutually independent).
    tolerated_relative_mean_abs_diff =
        std::sqrt(static_cast<double>(accumulation_depth)) *
        std::numeric_limits<DstScalar>::epsilon();
    // Unbiasing requirement: we require the bias, abs_mean_diff, to be much
    // smaller than the mean_abs_diff, except when there are very few values.
    tolerated_relative_abs_mean_diff =
        tolerated_relative_mean_abs_diff * std::sqrt(inverse_size);
  } else {
    // In quantized arithmetic, tolerate minor rounding differences, resulting
    // in off-by-one errors (tolerated_relative_max_abs_diff = 1), as long
    // as they are rare (tolerated_relative_mean_abs_diff) and unbiased
    // (tolerated_relative_abs_mean_diff).
    tolerated_relative_max_abs_diff = 1;
    // Naively require mean_abs_diff and abs_mean_diff to converge to zero
    // as size gets large. We don't know at all how quick that convergence
    // should be: this is just based on trial-and-error and striking a
    // compromise between something that works and something that's simple
    // enough code that doesn't feel too ad-hoc. As above in the float path,
    // abs_mean_diff is subject to a stricter requirement as it is a bias.
    tolerated_relative_mean_abs_diff = std::sqrt(inverse_size) * 0.5;
    tolerated_relative_abs_mean_diff = inverse_size * 2.;
  }

  double tolerated_max_abs_diff =
      tolerated_relative_max_abs_diff * error_stats.scale_factor;
  double tolerated_mean_abs_diff =
      tolerated_relative_mean_abs_diff * error_stats.scale_factor;
  double tolerated_abs_mean_diff =
      tolerated_relative_abs_mean_diff * error_stats.scale_factor;

  EXPECT_LE(error_stats.max_abs_diff, tolerated_max_abs_diff);
  EXPECT_LE(error_stats.mean_abs_diff, tolerated_mean_abs_diff);
  EXPECT_LE(error_stats.abs_mean_diff, tolerated_abs_mean_diff);

  return error_stats.max_abs_diff <= tolerated_max_abs_diff &&
         error_stats.mean_abs_diff <= tolerated_mean_abs_diff &&
         error_stats.abs_mean_diff <= tolerated_abs_mean_diff;
}

template <typename AccumScalar, typename DstScalar>
void CheckErrorForAccumulation(int accumulation_depth,
                               const std::vector<DstScalar>& actual,
                               const std::vector<DstScalar>& expected) {
  ErrorStats error_stats;
  ComputeErrorStats(actual, expected, &error_stats);
  bool success =
      CheckErrorStats<AccumScalar, DstScalar>(error_stats, accumulation_depth);
  EXPECT_TRUE(success) << "Actual vector\n"
                       << ToString(actual) << "\ndiffers from expected vector\n"
                       << ToString(expected) << "\n";
}

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
void PerformGemmThenCompareResultsThenAgainWithClamping(
    const MatrixParams<LhsScalar>& lhs_params,
    const std::vector<LhsScalar>& lhs_data,
    const MatrixParams<RhsScalar>& rhs_params,
    const std::vector<RhsScalar>& rhs_data,
    const MatrixParams<DstScalar>& dst_params, std::vector<DstScalar>* dst_data,
    const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
    const std::vector<DstScalar>& expected,
    CpuBackendContext* cpu_backend_context) {
  const int accumulation_depth = lhs_params.cols;
  Gemm(lhs_params, lhs_data.data(), rhs_params, rhs_data.data(), dst_params,
       dst_data->data(), params, cpu_backend_context);
  CheckErrorForAccumulation<AccumScalar>(accumulation_depth, *dst_data,
                                         expected);
  DstScalar expected_median = Median(expected);
  std::vector<DstScalar> expected_with_clamp;
  GemmParams<AccumScalar, DstScalar, quantization_flavor> params_with_clamp;
  DstScalar clamp_min, clamp_max;

  clamp_min = std::numeric_limits<DstScalar>::lowest();
  clamp_max = expected_median;
  Clamp(expected, clamp_min, clamp_max, &expected_with_clamp);
  Clamp(params, clamp_min, clamp_max, &params_with_clamp);
  Gemm(lhs_params, lhs_data.data(), rhs_params, rhs_data.data(), dst_params,
       dst_data->data(), params_with_clamp, cpu_backend_context);
  CheckErrorForAccumulation<AccumScalar>(accumulation_depth, *dst_data,
                                         expected_with_clamp);

  clamp_min = expected_median;
  clamp_max = std::numeric_limits<DstScalar>::max();
  Clamp(expected, clamp_min, clamp_max, &expected_with_clamp);
  Clamp(params, clamp_min, clamp_max, &params_with_clamp);
  Gemm(lhs_params, lhs_data.data(), rhs_params, rhs_data.data(), dst_params,
       dst_data->data(), params_with_clamp, cpu_backend_context);
  CheckErrorForAccumulation<AccumScalar>(accumulation_depth, *dst_data,
                                         expected_with_clamp);
}

// When generating testcases for a quantized GEMM, it's not trivial to
// pick multiplier exponents: a too low value will result in too many zeros,
// a too high value will result in too many large clamped values, in both
// cases testing coverage is harmed. Therefore to ensure good testing coverage
// we must find a multiplier exponent that's just right.  It would be possible
// to do so by analysis of the random distribution of values in the result
// matrix. That however would require some mathematical work that we haven't
// done so far. Until that is done, the best that we can do is to search for
// a good exponent value by trial-and-error. This is expensive, as each try
// requires computing a whole GEMM. This is thus probably a major contribution
// to the overall latency of this tesat. To partially mitigate that,
// we use a bisection to reduce the required number of tries.
//
// This function is recursive. The bisect_min and bisect_max arguments
// are the current bisection bounds. It performs a Gemm with the mid-point,
// named bisect_mid, as the multiplier exponent. Based on whether the values
// in the resulting matrix are rather too low or too large in absolute
// value, it then recurses into the corresponding half of the bisection range.
template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar>
int BisectReasonableMultiplierExponent(
    int bisect_min, int bisect_max, const MatrixParams<LhsScalar>& lhs_params,
    const std::vector<LhsScalar>& lhs_data,
    const MatrixParams<RhsScalar>& rhs_params,
    const std::vector<RhsScalar>& rhs_data,
    const MatrixParams<DstScalar>& dst_params, std::vector<DstScalar>* dst_data,
    const GemmParams<AccumScalar, DstScalar>& params,
    CpuBackendContext* cpu_backend_context) {
  if (bisect_min == bisect_max) {
    return bisect_min;
  }
  // Compute the midpoint as the floor of the average of bisect_min and
  // bisect_max. As C++ integer division is rounding towards zero and our values
  // may be of any sign, it is not trivial to implement this using only integer
  // arithmetic.
  int bisect_mid =
      static_cast<int>(std::floor(0.5 * (bisect_min + bisect_max)));
  GemmParams<AccumScalar, DstScalar> params_copy(params);
  params_copy.multiplier_exponent = bisect_mid;
  double clamp_abs = std::max(std::abs(static_cast<double>(params.clamp_min)),
                              std::abs(static_cast<double>(params.clamp_max)));
  Gemm(lhs_params, lhs_data.data(), rhs_params, rhs_data.data(), dst_params,
       dst_data->data(), params_copy, cpu_backend_context);
  double median_abs = MedianAbs(*dst_data);
  if (median_abs < 0.25 * clamp_abs) {
    return BisectReasonableMultiplierExponent(
        bisect_mid + 1, bisect_max, lhs_params, lhs_data, rhs_params, rhs_data,
        dst_params, dst_data, params_copy, cpu_backend_context);
  } else {
    return BisectReasonableMultiplierExponent(
        bisect_min, bisect_mid, lhs_params, lhs_data, rhs_params, rhs_data,
        dst_params, dst_data, params_copy, cpu_backend_context);
  }
}

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
void ReferenceGemm(
    const MatrixParams<LhsScalar>& lhs_params, const LhsScalar* lhs_data,
    const MatrixParams<RhsScalar>& rhs_params, const RhsScalar* rhs_data,
    const MatrixParams<DstScalar>& dst_params, DstScalar* dst_data,
    const GemmParams<AccumScalar, DstScalar, quantization_flavor>& params,
    CpuBackendContext* context) {
  ruy::Matrix<LhsScalar> ruy_lhs;
  ruy::Matrix<RhsScalar> ruy_rhs;
  ruy::Matrix<DstScalar> ruy_dst;
  cpu_backend_gemm::detail::MakeRuyMatrix(lhs_params, lhs_data, &ruy_lhs);
  cpu_backend_gemm::detail::MakeRuyMatrix(rhs_params, rhs_data, &ruy_rhs);
  cpu_backend_gemm::detail::MakeRuyMatrix(dst_params, dst_data, &ruy_dst);

  ruy::MulParams<AccumScalar, DstScalar> ruy_mul_params;
  cpu_backend_gemm::detail::MakeRuyMulParams(params, &ruy_mul_params);

  ruy::ReferenceMul(ruy_lhs, ruy_rhs, ruy_mul_params, &ruy_dst);
}

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar>
void TestSomeGemm(int rows, int depth, int cols,
                  const std::vector<DstScalar>& golden) {
  CpuBackendContext cpu_backend_context;
  std::default_random_engine random_engine;
  cpu_backend_context.SetMaxNumThreads(1 + (random_engine() % 8));
  bool use_caching = static_cast<bool>(random_engine() % 2);
  cpu_backend_context.SetUseCaching(use_caching);
  const bool use_golden = !golden.empty();

  std::vector<LhsScalar> lhs_data;
  std::vector<RhsScalar> rhs_data;
  std::vector<AccumScalar> bias_data;
  std::vector<DstScalar> dst_data;
  if (use_golden) {
    MakeVectorFilledWithConsecutiveInts(rows * depth, &lhs_data);
    MakeVectorFilledWithConsecutiveInts(depth * cols, &rhs_data);
    MakeVectorFilledWithConsecutiveInts(rows, &bias_data);
  } else {
    MakeDeterministicPseudoRandomVector(rows * depth, &lhs_data);
    MakeDeterministicPseudoRandomVector(depth * cols, &rhs_data);
    MakeDeterministicPseudoRandomVector(rows, &bias_data);
  }
  MakeDeterministicPseudoRandomVector(rows * cols, &dst_data);

  MatrixParams<LhsScalar> lhs_params;
  lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  lhs_params.rows = rows;
  lhs_params.cols = depth;
  if (!std::is_floating_point<LhsScalar>::value) {
    lhs_params.zero_point = 1;
    if (!use_golden) {
      lhs_params.zero_point += random_engine() % 8;
    }
  }

  MatrixParams<RhsScalar> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.rows = depth;
  rhs_params.cols = cols;
  if (!std::is_floating_point<RhsScalar>::value) {
    rhs_params.zero_point = 1;
    if (!use_golden) {
      rhs_params.zero_point += random_engine() % 8;
    }
  }

  MatrixParams<DstScalar> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = rows;
  dst_params.cols = cols;
  if (!std::is_floating_point<DstScalar>::value) {
    dst_params.zero_point = 1;
    if (!use_golden) {
      dst_params.zero_point += random_engine() % 8;
    }
  }

  GemmParams<AccumScalar, DstScalar> params;
  if (use_golden || (random_engine() % 2)) {
    // cpu_backend_gemm supports bias=null only in the float path. Test that
    // in 50% of float testcases.
    params.bias = bias_data.data();
  }
  static constexpr std::int32_t kMultiplierFixedpointMin = 1234567890;
  static constexpr std::int32_t kMultiplierFixedpointMax = 1987654321;
  if (!std::is_floating_point<AccumScalar>::value) {
    // some large int32 value. Not being a multiple of a large
    // power of two helps testing rounding behavior.
    params.multiplier_fixedpoint = kMultiplierFixedpointMin;
    // Now find a suitable value for multiplier_exponent.
    // It needs to be low enough for a substantial amount of dst values
    // to avoid getting clamped.
    int bisect_min = -8 * static_cast<int>(sizeof(AccumScalar));
    // We don't increase test coverage by using positive multipliers,
    // and using very large positive multipliers may at the moment
    // result in overflow in some paths.
    // TODO(benoitjacob): fix that.
    int bisect_max = 0;
    params.multiplier_exponent = BisectReasonableMultiplierExponent(
        bisect_min, bisect_max, lhs_params, lhs_data, rhs_params, rhs_data,
        dst_params, &dst_data, params, &cpu_backend_context);
  }

  std::vector<DstScalar> expected;
  if (use_golden) {
    EXPECT_EQ(golden.size(), dst_data.size());
    expected = golden;
  } else {
    expected.resize(dst_data.size());
    ReferenceGemm(lhs_params, lhs_data.data(), rhs_params, rhs_data.data(),
                  dst_params, expected.data(), params, &cpu_backend_context);
  }

  PerformGemmThenCompareResultsThenAgainWithClamping(
      lhs_params, lhs_data, rhs_params, rhs_data, dst_params, &dst_data, params,
      expected, &cpu_backend_context);

  if (!use_golden && !std::is_floating_point<AccumScalar>::value) {
    // Try with per-channel quantized multipliers.
    std::vector<AccumScalar> multiplier_fixedpoint_perchannel(rows);
    std::vector<int> multiplier_exponent_perchannel(rows);
    for (int i = 0; i < rows; i++) {
      multiplier_fixedpoint_perchannel[i] =
          kMultiplierFixedpointMin +
          (random_engine() %
           (kMultiplierFixedpointMax + 1 - kMultiplierFixedpointMin));
      const int exponent_min = params.multiplier_exponent - 2;
      const int exponent_max = params.multiplier_exponent + 2;
      multiplier_exponent_perchannel[i] =
          exponent_min + (random_engine() % (exponent_max + 1 - exponent_min));
    }
    static constexpr QuantizationFlavor perchannel_flavor =
        std::is_floating_point<AccumScalar>::value
            ? QuantizationFlavor::kFloatingPoint
            : QuantizationFlavor::kIntegerWithPerRowMultiplier;
    GemmParams<AccumScalar, DstScalar, perchannel_flavor> params_perchannel;
    params_perchannel.bias = params.bias;
    params_perchannel.clamp_min = params.clamp_min;
    params_perchannel.clamp_max = params.clamp_max;
    params_perchannel.multiplier_fixedpoint_perchannel =
        multiplier_fixedpoint_perchannel.data();
    params_perchannel.multiplier_exponent_perchannel =
        multiplier_exponent_perchannel.data();
    ReferenceGemm(lhs_params, lhs_data.data(), rhs_params, rhs_data.data(),
                  dst_params, expected.data(), params_perchannel,
                  &cpu_backend_context);
    PerformGemmThenCompareResultsThenAgainWithClamping(
        lhs_params, lhs_data, rhs_params, rhs_data, dst_params, &dst_data,
        params_perchannel, expected, &cpu_backend_context);
  }
}

TEST(CpuBackendGemmSimpleTestAgainstGolden, Float) {
  TestSomeGemm<float, float, float, float>(2, 3, 4,
                                           {15, 34, 33, 79, 51, 124, 69, 169});
}

TEST(CpuBackendGemmSimpleTestAgainstGolden, Uint8) {
  TestSomeGemm<std::uint8_t, std::uint8_t, std::int32_t, std::uint8_t>(
      5, 2, 3, {2, 4, 6, 7, 9, 3, 10, 16, 22, 29, 4, 15, 26, 37, 48});
}

TEST(CpuBackendGemmSimpleTestAgainstGolden, Int8) {
  TestSomeGemm<std::int8_t, std::int8_t, std::int32_t, std::int8_t>(
      2, 6, 3, {13, 32, 31, 81, 50, 127});
}

TEST(CpuBackendGemmSimpleTestAgainstGolden, Int8Int16) {
  TestSomeGemm<std::int8_t, std::int8_t, std::int32_t, std::int16_t>(
      3, 5, 4, {19, 48, 77, 48, 149, 250, 76, 249, 422, 105, 350, 595});
}

template <typename tLhsScalar, typename tRhsScalar, typename tAccumScalar,
          typename tDstScalar>
struct TypesTuple {
  using LhsScalar = tLhsScalar;
  using RhsScalar = tRhsScalar;
  using AccumScalar = tAccumScalar;
  using DstScalar = tDstScalar;
};

template <typename TypesTupleType>
void TestRandomGemms(const std::vector<std::tuple<int, int, int>>& shapes) {
  using LhsScalar = typename TypesTupleType::LhsScalar;
  using RhsScalar = typename TypesTupleType::RhsScalar;
  using AccumScalar = typename TypesTupleType::AccumScalar;
  using DstScalar = typename TypesTupleType::DstScalar;
  for (const auto& shape : shapes) {
    int rows = std::get<0>(shape);
    int depth = std::get<1>(shape);
    int cols = std::get<2>(shape);
    TestSomeGemm<LhsScalar, RhsScalar, AccumScalar, DstScalar>(rows, depth,
                                                               cols, {});
  }
}

template <typename TypesTupleType>
class CpuBackendGemmTest : public testing::Test {};

TYPED_TEST_SUITE_P(CpuBackendGemmTest);

typedef ::testing::Types<
    TypesTuple<float, float, float, float>,
    TypesTuple<std::uint8_t, std::uint8_t, std::int32_t, std::uint8_t>,
    TypesTuple<std::int8_t, std::int8_t, std::int32_t, std::int8_t>,
    TypesTuple<std::int8_t, std::int8_t, std::int32_t, std::int16_t>,
    TypesTuple<std::uint8_t, std::uint8_t, std::int32_t, std::int8_t>>
    CpuBackendGemmTestInstantiations;

TYPED_TEST_SUITE(CpuBackendGemmTest, CpuBackendGemmTestInstantiations);

TYPED_TEST(CpuBackendGemmTest, Square) {
  std::vector<std::tuple<int, int, int>> shapes;
  for (int size = 1; size < 50; size++) {
    shapes.push_back(std::make_tuple(size, size, size));
  }
  TestRandomGemms<TypeParam>(shapes);
}

TYPED_TEST(CpuBackendGemmTest, SquarePowerOfTwo) {
  std::vector<std::tuple<int, int, int>> shapes;
  for (int size = 64; size <= 128; size *= 2) {
    shapes.push_back(std::make_tuple(size, size, size));
  }
  TestRandomGemms<TypeParam>(shapes);
}

TYPED_TEST(CpuBackendGemmTest, MatrixTimesVector) {
  std::vector<std::tuple<int, int, int>> shapes;
  for (int size = 1; size < 200; size++) {
    shapes.push_back(std::make_tuple(size, size, 1));
  }
  TestRandomGemms<TypeParam>(shapes);
}

TYPED_TEST(CpuBackendGemmTest, VectorTimesMatrix) {
  std::vector<std::tuple<int, int, int>> shapes;
  for (int size = 1; size < 200; size++) {
    shapes.push_back(std::make_tuple(1, size, size));
  }
  TestRandomGemms<TypeParam>(shapes);
}

TYPED_TEST(CpuBackendGemmTest, MatrixTimesNarrow) {
  std::vector<std::tuple<int, int, int>> shapes;
  for (int size = 1; size < 50; size++) {
    shapes.push_back(std::make_tuple(size, size, 2));
    shapes.push_back(std::make_tuple(size, size, 3));
    shapes.push_back(std::make_tuple(size, size, 4));
    shapes.push_back(std::make_tuple(size, size, 8));
  }
  TestRandomGemms<TypeParam>(shapes);
}

TYPED_TEST(CpuBackendGemmTest, Rectangular) {
  std::vector<std::tuple<int, int, int>> shapes;
  for (int size = 1; size < 50; size++) {
    shapes.push_back(std::make_tuple(size, size + 5, size + 1));
    shapes.push_back(std::make_tuple(size + 10, size + 2, size));
  }
  TestRandomGemms<TypeParam>(shapes);
}

TYPED_TEST(CpuBackendGemmTest, HighlyRectangular) {
  std::vector<std::tuple<int, int, int>> shapes;
  for (int size = 1; size <= 1000; size *= 10) {
    shapes.push_back(std::make_tuple(size, 10, 10));
    shapes.push_back(std::make_tuple(10, size, 10));
    shapes.push_back(std::make_tuple(10, 10, size));
  }
  TestRandomGemms<TypeParam>(shapes);
}

TYPED_TEST(CpuBackendGemmTest, InnerProduct) {
  std::vector<std::tuple<int, int, int>> shapes;
  for (int size = 1; size < 200; size++) {
    shapes.push_back(std::make_tuple(1, size, 1));
  }
  TestRandomGemms<TypeParam>(shapes);
}

TYPED_TEST(CpuBackendGemmTest, OuterProduct) {
  std::vector<std::tuple<int, int, int>> shapes;
  for (int size = 1; size < 100; size++) {
    shapes.push_back(std::make_tuple(size, 1, size));
  }
  TestRandomGemms<TypeParam>(shapes);
}

}  // namespace

}  // namespace tflite

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
