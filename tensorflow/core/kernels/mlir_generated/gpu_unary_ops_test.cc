/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <limits>

#include "tensorflow/core/kernels/mlir_generated/base_ops_test.h"
#include "tensorflow/core/kernels/mlir_generated/base_unary_ops_test.h"

namespace tensorflow {
namespace {

// Test fixture `UnaryOpsTest` that sets the TF device is expected by the TEST
// macros below.
class UnaryOpsTest : public UnaryOpsTestBase {
 protected:
  void SetUp() override {
    std::unique_ptr<tensorflow::Device> device_gpu(
        tensorflow::DeviceFactory::NewDevice("GPU", {},
                                             "/job:a/replica:0/task:0"));
    SetDevice(tensorflow::DEVICE_GPU, std::move(device_gpu));
  }
};

/// Test `tf.Abs`.

template <typename T>
T baseline_abs(T x) {
  return x >= 0 ? x : -x;
}

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Abs, DT_FLOAT, DT_FLOAT, test::NearZeroAndExtremeInput<float>(), std::abs,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Abs, DT_DOUBLE, DT_DOUBLE, test::NearZeroAndExtremeInput<double>(),
    std::abs, test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    Abs, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
    test::NearZeroAndExtremeInput<Eigen::half>(), std::abs,
    test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Abs, DT_INT64, DT_INT64, test::NearZeroAndExtremeInput<int64_t>(),
    baseline_abs, test::OpsTestConfig().ExpectStrictlyEqual())

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TEST(Abs, DT_INT8, DT_INT8, baseline_abs,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(Abs, DT_INT16, DT_INT16, baseline_abs,
                      test::OpsTestConfig().ExpectStrictlyEqual())
#endif

/// Test `tf.Acos`.

// Test only values in the function domain. The otherwise returned nan value
// fails comparison for equality.
GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    Acos, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
    test::DefaultInputBetweenZeroAndOne<Eigen::half>(), std::acos,
    test::OpsTestConfig())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Acos, DT_FLOAT, DT_FLOAT, test::DefaultInputBetweenZeroAndOne<float>(),
    std::acos, test::OpsTestConfig())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Acos, DT_DOUBLE, DT_DOUBLE, test::DefaultInputBetweenZeroAndOne<double>(),
    std::acos, test::OpsTestConfig())

/// Test `tf.Acosh`.

// TODO(herhut): Give this better input once TF testing also supports NaN.
GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    Acosh, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
    test::DefaultInputGreaterEqualOne<Eigen::half>(), std::acosh,
    test::OpsTestConfig())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Acosh, DT_FLOAT, DT_FLOAT, test::DefaultInputGreaterEqualOne<float>(),
    std::acosh, test::OpsTestConfig())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Acosh, DT_DOUBLE, DT_DOUBLE, test::DefaultInputGreaterEqualOne<double>(),
    std::acosh, test::OpsTestConfig())

/// Test `tf.Angle`.

template <typename T>
typename T::value_type baseline_angle(T x) {
  return std::arg(x);
}

GENERATE_DEFAULT_TEST(Angle, DT_COMPLEX64, DT_FLOAT, baseline_angle,
                      test::OpsTestConfig().AddTout().NoBufferReuse())

GENERATE_DEFAULT_TEST(Angle, DT_COMPLEX128, DT_DOUBLE, baseline_angle,
                      test::OpsTestConfig().AddTout().NoBufferReuse())

/// Test `tf.Asin`.

// Test only values in the function domain. The otherwise returned nan value
// fails comparison for equality.
GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    Asin, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
    test::DefaultInputBetweenZeroAndOne<Eigen::half>(), std::asin,
    test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Asin, DT_FLOAT, DT_FLOAT, test::DefaultInputBetweenZeroAndOne<float>(),
    std::asin, test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Asin, DT_DOUBLE, DT_DOUBLE, test::DefaultInputBetweenZeroAndOne<double>(),
    std::asin, test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.Asinh`.

GENERATE_DEFAULT_TEST_2(Asinh, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, std::asinh,
                        test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Asinh, DT_FLOAT, DT_FLOAT, std::asinh,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Asinh, DT_DOUBLE, DT_DOUBLE, std::asinh,
                      test::OpsTestConfig())

/// Test `tf.Atan`.

GENERATE_DEFAULT_TEST_2(Atan, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, std::atan,
                        test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Atan, DT_FLOAT, DT_FLOAT, std::atan,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Atan, DT_DOUBLE, DT_DOUBLE, std::atan,
                      test::OpsTestConfig())

/// Test `tf.Atanh`.

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    Atanh, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
    test::DefaultInputBetweenZeroAndOne<Eigen::half>(), std::atanh,
    test::OpsTestConfig())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Atanh, DT_FLOAT, DT_FLOAT, test::DefaultInputBetweenZeroAndOne<float>(),
    std::atanh, test::OpsTestConfig())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Atanh, DT_DOUBLE, DT_DOUBLE, test::DefaultInputBetweenZeroAndOne<double>(),
    std::atanh, test::OpsTestConfig())

/// Test `tf.Cast`.

template <typename SrcT, typename DstT>
DstT baseline_cast(SrcT x) {
  return static_cast<DstT>(x);
}

template <typename DstT,
          std::enable_if_t<!llvm::is_one_of<DstT, std::complex<float>,
                                            std::complex<double>>::value,
                           bool> = true>
DstT baseline_cast(const std::complex<float>& x) {
  return static_cast<DstT>(x.real());
}

template <typename DstT,
          std::enable_if_t<!llvm::is_one_of<DstT, std::complex<float>,
                                            std::complex<double>>::value,
                           bool> = true>
DstT baseline_cast(const std::complex<double>& x) {
  return static_cast<DstT>(x.real());
}

#define TEST_CAST_FROM_TO(from_type, to_type)                    \
  GENERATE_DEFAULT_TEST(Cast, from_type, to_type, baseline_cast, \
                        test::OpsTestConfig()                    \
                            .AddTout()                           \
                            .NoBufferReuse()                     \
                            .ExpectStrictlyEqual()               \
                            .InputAttribute("SrcT")              \
                            .OutputAttribute("DstT"))

// Casting from floating point types to unsigned integers has undefined behavior
// for negative values <= -1.0
#define TEST_NON_NEGATIVE_VALUES_CAST_FROM_TO(from_type, to_type)       \
  GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(                     \
      Cast, from_type, to_type,                                         \
      test::DefaultInputGreaterOrEqualToZero<NativeT>(), baseline_cast, \
      test::OpsTestConfig()                                             \
          .AddTout()                                                    \
          .NoBufferReuse()                                              \
          .ExpectStrictlyEqual()                                        \
          .InputAttribute("SrcT")                                       \
          .OutputAttribute("DstT"))

#define TEST_CAST_TO_NO_UNSIGNED(from_type)  \
  TEST_CAST_FROM_TO(from_type, DT_BOOL)      \
  TEST_CAST_FROM_TO(from_type, DT_INT8)      \
  TEST_CAST_FROM_TO(from_type, DT_INT16)     \
  TEST_CAST_FROM_TO(from_type, DT_INT32)     \
  TEST_CAST_FROM_TO(from_type, DT_INT64)     \
  TEST_CAST_FROM_TO(from_type, DT_FLOAT)     \
  TEST_CAST_FROM_TO(from_type, DT_DOUBLE)    \
  TEST_CAST_FROM_TO(from_type, DT_COMPLEX64) \
  TEST_CAST_FROM_TO(from_type, DT_COMPLEX128)

#define TEST_CAST_TO_UNSIGNED(from_type)  \
  TEST_CAST_FROM_TO(from_type, DT_UINT8)  \
  TEST_CAST_FROM_TO(from_type, DT_UINT16) \
  TEST_CAST_FROM_TO(from_type, DT_UINT32) \
  TEST_CAST_FROM_TO(from_type, DT_UINT64)

#define TEST_NON_NEGATIVE_VALUES_CAST_TO_UNSIGNED(from_type)  \
  TEST_NON_NEGATIVE_VALUES_CAST_FROM_TO(from_type, DT_UINT8)  \
  TEST_NON_NEGATIVE_VALUES_CAST_FROM_TO(from_type, DT_UINT16) \
  TEST_NON_NEGATIVE_VALUES_CAST_FROM_TO(from_type, DT_UINT32) \
  TEST_NON_NEGATIVE_VALUES_CAST_FROM_TO(from_type, DT_UINT64)

#define TEST_CAST_TO(from_type) \
  TEST_CAST_TO_NO_UNSIGNED(from_type) TEST_CAST_TO_UNSIGNED(from_type)

TEST_CAST_TO(DT_BOOL)
TEST_CAST_TO(DT_INT8)
TEST_CAST_TO(DT_INT16)
TEST_CAST_TO(DT_INT32)
TEST_CAST_TO(DT_INT64)
TEST_CAST_TO(DT_UINT8)
TEST_CAST_TO(DT_UINT16)
TEST_CAST_TO(DT_UINT32)
TEST_CAST_TO(DT_UINT64)
TEST_CAST_TO_NO_UNSIGNED(DT_HALF)
TEST_NON_NEGATIVE_VALUES_CAST_TO_UNSIGNED(DT_HALF)
TEST_CAST_TO_NO_UNSIGNED(DT_FLOAT)
TEST_NON_NEGATIVE_VALUES_CAST_TO_UNSIGNED(DT_FLOAT)
TEST_CAST_TO_NO_UNSIGNED(DT_DOUBLE)
TEST_NON_NEGATIVE_VALUES_CAST_TO_UNSIGNED(DT_DOUBLE)
TEST_CAST_TO_NO_UNSIGNED(DT_COMPLEX64)
TEST_NON_NEGATIVE_VALUES_CAST_TO_UNSIGNED(DT_COMPLEX64)
TEST_CAST_TO_NO_UNSIGNED(DT_COMPLEX128)
TEST_NON_NEGATIVE_VALUES_CAST_TO_UNSIGNED(DT_COMPLEX128)

#undef TEST_CAST_FROM_TO
#undef TEST_NON_NEGATIVE_VALUES_CAST_FROM_TO
#undef TEST_CAST_TO_NO_UNSIGNED
#undef TEST_CAST_TO_UNSIGNED
#undef TEST_NON_NEGATIVE_VALUES_CAST_TO_UNSIGNED
#undef TEST_CAST_TO

/// Test `tf.Ceil`.

GENERATE_DEFAULT_TEST(Ceil, DT_FLOAT, DT_FLOAT, std::ceil,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Ceil, DT_DOUBLE, DT_DOUBLE, std::ceil,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_2(Ceil, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, std::ceil,
                        test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.ComplexAbs`.

template <typename T>
typename T::value_type baseline_complex_abs(T x) {
  return std::abs(x);
}

GENERATE_DEFAULT_TEST(ComplexAbs, DT_COMPLEX64, DT_FLOAT, baseline_complex_abs,
                      test::OpsTestConfig().AddTout().NoBufferReuse())

GENERATE_DEFAULT_TEST(ComplexAbs, DT_COMPLEX128, DT_DOUBLE,
                      baseline_complex_abs,
                      test::OpsTestConfig().AddTout().NoBufferReuse())

/// Test `tf.Conj`.

template <typename T>
T baseline_conj(T x) {
  return std::conj(x);
}

GENERATE_DEFAULT_TEST(Conj, DT_COMPLEX64, DT_COMPLEX64, baseline_conj,
                      test::OpsTestConfig().NoBufferReuse())

GENERATE_DEFAULT_TEST(Conj, DT_COMPLEX128, DT_COMPLEX128, baseline_conj,
                      test::OpsTestConfig().NoBufferReuse())

/// Test `tf.Cos`.

GENERATE_DEFAULT_TEST(Cos, DT_FLOAT, DT_FLOAT, std::cos, test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Cos, DT_DOUBLE, DT_DOUBLE, std::cos,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST_2(Cos, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, std::cos,
                        test::OpsTestConfig())

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)

GENERATE_DEFAULT_TEST(Cos, DT_COMPLEX64, DT_COMPLEX64, std::cos,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Cos, DT_COMPLEX128, DT_COMPLEX128, std::cos,
                      test::OpsTestConfig())

#endif

/// Test `tf.Cosh`.

GENERATE_DEFAULT_TEST_2(Cosh, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, std::cosh,
                        test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Cosh, DT_FLOAT, DT_FLOAT, std::cosh,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Cosh, DT_DOUBLE, DT_DOUBLE, std::cosh,
                      test::OpsTestConfig())

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)

GENERATE_DEFAULT_TEST(Cosh, DT_COMPLEX64, DT_COMPLEX64, std::cosh,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Cosh, DT_COMPLEX128, DT_COMPLEX128, std::cosh,
                      test::OpsTestConfig())

#endif

/// Test `tf.Digamma`.

/// Reference implementation.
double baseline_digamma(double x) {
  constexpr int kN = 100000;
  constexpr double kGammaE = 0.5772156649015328606065120900824024;
  double z = x - 1;
  double sum = -kGammaE;
  for (int i = 1; i <= kN; i++) {
    sum += z / (i * (i + z));
  }
  return sum;
}

// Exclude non-positive integer values as `digamma` is undefined for these and
// the test framework does not suppot NaN comparisons.
constexpr std::initializer_list<double> kDigammaValues = {
    -18.1, -9.2, -0.7, -0.5, -0.3, -0.2, -0.1, -1e-6,
    1e-6,  0.1,  0.2,  0.3,  0.5,  0.7,  0.9,  18.0};

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    Digamma, DT_FLOAT, DT_DOUBLE, DT_FLOAT, DT_DOUBLE,
    test::InputAsVector<float>(kDigammaValues), baseline_digamma,
    test::OpsTestConfig())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Digamma, DT_DOUBLE, DT_DOUBLE, test::InputAsVector<double>(kDigammaValues),
    baseline_digamma, test::OpsTestConfig())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    Digamma, DT_HALF, DT_DOUBLE, DT_HALF, DT_DOUBLE,
    test::InputAsVector<Eigen::half>(kDigammaValues), baseline_digamma,
    test::OpsTestConfig())

/// Test `tf.Elu`.

template <typename T>
T baseline_elu(T x) {
  if (x < 0) return std::exp(x) - 1;
  return x;
}

GENERATE_DEFAULT_TEST(Elu, DT_FLOAT, DT_FLOAT, baseline_elu,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Elu, DT_DOUBLE, DT_DOUBLE, baseline_elu,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST_2(Elu, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, baseline_elu,
                        test::OpsTestConfig())

/// Test `tf.Erf` and `tf.Erfc`.

// Use specific values to cover the different intervals in the f64 erf and f64
// erfc, and f32 erfc approximations.
//   - (-inf, -sqrt(kMaxlog)]
//   - [-sqrt(kMaxlog), -8]
//   - [-8, -1]
//   - [-1, 0]
//   - [0, 1]
//   - [1, 8]
//   - [8, sqrt(kMaxlog)]
//   - [sqrt(kMaxlog), inf)

static constexpr double kSqrtMaxlogF64 = 26.6417;
static constexpr std::initializer_list<double> kErfcF64Values = {
    -1000.0,
    -27.0,
    -kSqrtMaxlogF64 - 0.1,
    -kSqrtMaxlogF64,
    -kSqrtMaxlogF64 + 0.1,
    -16.0,
    -9.0,
    -8.2,
    -8.1,
    -8.0,
    -7.9,
    -6.7,
    -4.5,
    -2.3,
    -1.5,
    -1.2,
    -1.1,
    -1.0,
    -0.9,
    -0.3,
    -0.2,
    -0.1,
    0.0,
    0.1,
    0.2,
    0.3,
    0.9,
    1.0,
    1.1,
    1.2,
    1.5,
    2.3,
    4.5,
    6.7,
    7.9,
    8.0,
    8.1,
    8.2,
    9.0,
    16.0,
    kSqrtMaxlogF64 - 0.1,
    kSqrtMaxlogF64,
    kSqrtMaxlogF64 + 0.1,
    27.0,
    1000.0};

static constexpr float kSqrtMaxlogF32 = 9.41928;
static constexpr std::initializer_list<float> kErfcF32Values = {
    -1000.0,
    -27.0,
    -kSqrtMaxlogF32 - 0.1,
    -kSqrtMaxlogF32,
    -kSqrtMaxlogF32 + 0.1,
    -16.0,
    -9.0,
    -8.2,
    -8.1,
    -8.0,
    -7.9,
    -6.7,
    -4.5,
    -2.3,
    -1.5,
    -1.2,
    -1.1,
    -1.0,
    -0.9,
    -0.3,
    -0.2,
    -0.1,
    0.0,
    0.1,
    0.2,
    0.3,
    0.9,
    1.0,
    1.1,
    1.2,
    1.5,
    2.3,
    4.5,
    6.7,
    7.9,
    8.0,
    8.1,
    8.2,
    9.0,
    16.0,
    kSqrtMaxlogF32 - 0.1,
    kSqrtMaxlogF32,
    kSqrtMaxlogF32 + 0.1,
    27.0,
    1000.0};

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(Erf, DT_DOUBLE, DT_DOUBLE,
                                                 kErfcF64Values, std::erf,
                                                 test::OpsTestConfig())

// Use specific values to cover the different intervals of the f32 erf
// approximation.
//   - (-inf, -4]
//   - [-4, 4]
//   - [4, inf)
GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Erf, DT_FLOAT, DT_FLOAT,
    test::InputAsVector<float>({-100.0, -16.0, -8.9, -6.7, -4.1, -4.0, -3.9,
                                -3.4,   -2.3,  -1.2, -1.1, -1.0, -0.5, -0.2,
                                -0.1,   0.0,   0.1,  0.2,  0.5,  1.0,  1.1,
                                1.2,    2.3,   3.4,  3.9,  4.0,  4.1,  6.7,
                                8.9,    16.0,  100.0}),
    std::erf, test::OpsTestConfig())

GENERATE_DEFAULT_TEST_2(Erf, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, std::erf,
                        test::OpsTestConfig())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Erfc, DT_DOUBLE, DT_DOUBLE, test::InputAsVector<double>(kErfcF64Values),
    std::erfc, test::OpsTestConfig())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Erfc, DT_FLOAT, DT_FLOAT, test::InputAsVector<float>(kErfcF32Values),
    std::erfc, test::OpsTestConfig())

GENERATE_DEFAULT_TEST_2(Erfc, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, std::erfc,
                        test::OpsTestConfig())

/// Test `tf.Exp`.

GENERATE_DEFAULT_TEST(Exp, DT_FLOAT, DT_FLOAT, std::exp, test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Exp, DT_DOUBLE, DT_DOUBLE, std::exp,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST_2(Exp, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, std::exp,
                        test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Exp, DT_COMPLEX64, DT_COMPLEX64, std::exp,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Exp, DT_COMPLEX128, DT_COMPLEX128, std::exp,
                      test::OpsTestConfig())

/// Test `tf.Expm1`.

GENERATE_DEFAULT_TEST(Expm1, DT_FLOAT, DT_FLOAT, std::expm1,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Expm1, DT_DOUBLE, DT_DOUBLE, std::expm1,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST_2(Expm1, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, std::expm1,
                        test::OpsTestConfig())

/// Test `tf.Floor`.

GENERATE_DEFAULT_TEST(Floor, DT_FLOAT, DT_FLOAT, std::floor,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Floor, DT_DOUBLE, DT_DOUBLE, std::floor,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_2(Floor, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, std::floor,
                        test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.Imag`.

template <typename T>
typename T::value_type baseline_imag(T x) {
  return std::imag(x);
}

GENERATE_DEFAULT_TEST(Imag, DT_COMPLEX64, DT_FLOAT, baseline_imag,
                      test::OpsTestConfig().AddTout().NoBufferReuse())

GENERATE_DEFAULT_TEST(Imag, DT_COMPLEX128, DT_DOUBLE, baseline_imag,
                      test::OpsTestConfig().AddTout().NoBufferReuse())

/// Test `tf.Inv`.

template <typename T>
T baseline_inv(T x) {
  return T(1) / x;
}

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Inv, DT_INT64, DT_INT64, test::DefaultInputNonZero<int64_t>(), baseline_inv,
    test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Inv, DT_FLOAT, DT_FLOAT, baseline_inv,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Inv, DT_DOUBLE, DT_DOUBLE, baseline_inv,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST_2(Inv, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, baseline_inv,
                        test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Inv, DT_COMPLEX64, DT_COMPLEX64, baseline_inv,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Inv, DT_COMPLEX128, DT_COMPLEX128, baseline_inv,
                      test::OpsTestConfig())

/// Test `tf.Invert`.

/// Reference implementation.
template <typename T>
T baseline_invert(T x) {
  return ~x;
}

GENERATE_DEFAULT_TEST(Invert, DT_INT8, DT_INT8, baseline_invert,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Invert, DT_UINT8, DT_UINT8, baseline_invert,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Invert, DT_INT16, DT_INT16, baseline_invert,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Invert, DT_UINT16, DT_UINT16, baseline_invert,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Invert, DT_INT32, DT_INT32, baseline_invert,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Invert, DT_UINT32, DT_UINT32, baseline_invert,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Invert, DT_INT64, DT_INT64, baseline_invert,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Invert, DT_UINT64, DT_UINT64, baseline_invert,
                      test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.IsFinite`.

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    IsFinite, DT_FLOAT, DT_FLOAT, DT_BOOL, DT_BOOL,
    test::NearZeroAndExtremeInput<float>(), std::isfinite,
    test::OpsTestConfig().ExpectStrictlyEqual().NoBufferReuse());

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    IsFinite, DT_DOUBLE, DT_DOUBLE, DT_BOOL, DT_BOOL,
    test::NearZeroAndExtremeInput<double>(), std::isfinite,
    test::OpsTestConfig().ExpectStrictlyEqual().NoBufferReuse());

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    IsFinite, DT_HALF, DT_FLOAT, DT_BOOL, DT_BOOL,
    test::NearZeroAndExtremeInput<Eigen::half>(), std::isfinite,
    test::OpsTestConfig().ExpectStrictlyEqual().NoBufferReuse());

/// Test `tf.IsInf`.

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    IsInf, DT_FLOAT, DT_FLOAT, DT_BOOL, DT_BOOL,
    test::NearZeroAndExtremeInput<float>(), std::isinf,
    test::OpsTestConfig().ExpectStrictlyEqual().NoBufferReuse());

// Workaround for gcc bug, it would fail with "unresolved overloaded function
// type" if passing std::isinf with type double. So we use type float for
// comparing expected values.
GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    IsInf, DT_DOUBLE, DT_FLOAT, DT_BOOL, DT_BOOL,
    test::NearZeroAndExtremeInput<double>(), std::isinf,
    test::OpsTestConfig().ExpectStrictlyEqual().NoBufferReuse());

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    IsInf, DT_HALF, DT_FLOAT, DT_BOOL, DT_BOOL,
    test::NearZeroAndExtremeInput<Eigen::half>(), std::isinf,
    test::OpsTestConfig().ExpectStrictlyEqual().NoBufferReuse());

/// Test `tf.IsNan`.

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    IsNan, DT_FLOAT, DT_FLOAT, DT_BOOL, DT_BOOL,
    test::NearZeroInfAndNanInput<float>(), std::isnan,
    test::OpsTestConfig().ExpectStrictlyEqual().NoBufferReuse());

// Workaround for gcc bug, it would fail with "unresolved overloaded function
// type" if passing std::isnan with type double. So we use type float for
// comparing expected values.
GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    IsNan, DT_DOUBLE, DT_FLOAT, DT_BOOL, DT_BOOL,
    test::NearZeroInfAndNanInput<double>(), std::isnan,
    test::OpsTestConfig().ExpectStrictlyEqual().NoBufferReuse());

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    IsNan, DT_HALF, DT_FLOAT, DT_BOOL, DT_BOOL,
    test::NearZeroInfAndNanInput<Eigen::half>(), std::isnan,
    test::OpsTestConfig().ExpectStrictlyEqual().NoBufferReuse());

/// Test `tf.Lgamma`.

static constexpr std::initializer_list<double> kLgammaValues = {
    -std::numeric_limits<double>::infinity(),
    -9.0,
    -8.5,
    -8.3,
    -2.0,
    -1.5,
    -1.3,
    -1.0,
    -0.5,
    -0.3,
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    100.0,
    std::numeric_limits<double>::infinity(),
};

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Lgamma, DT_FLOAT, DT_FLOAT, test::InputAsVector<float>(kLgammaValues),
    std::lgamma, test::OpsTestConfig())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Lgamma, DT_DOUBLE, DT_DOUBLE, test::InputAsVector<double>(kLgammaValues),
    std::lgamma, test::OpsTestConfig())

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES_2(
    Lgamma, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
    test::InputAsVector<Eigen::half>(kLgammaValues), std::lgamma,
    test::OpsTestConfig())

/// Test `tf.Log`.

GENERATE_DEFAULT_TEST(Log, DT_FLOAT, DT_FLOAT, std::log, test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Log, DT_DOUBLE, DT_DOUBLE, std::log,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST_2(Log, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, std::log,
                        test::OpsTestConfig())

/// Test `tf.Log1p`.

GENERATE_DEFAULT_TEST(Log1p, DT_FLOAT, DT_FLOAT, std::log1p,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Log1p, DT_DOUBLE, DT_DOUBLE, std::log1p,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST_2(Log1p, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, std::log1p,
                        test::OpsTestConfig())

/// Test `tf.LogicalNot`

bool baseline_logical_not(bool x) { return !x; }

GENERATE_DEFAULT_TEST(LogicalNot, DT_BOOL, DT_BOOL, baseline_logical_not,
                      test::OpsTestConfig().ExpectStrictlyEqual().NoT())

/// Test `tf.Neg`.

/// Reference implementation.
template <typename T>
T baseline_neg(T x) {
  return -x;
}

GENERATE_DEFAULT_TEST(Neg, DT_FLOAT, DT_FLOAT, baseline_neg,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Neg, DT_DOUBLE, DT_DOUBLE, baseline_neg,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_2(Neg, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, baseline_neg,
                        test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Neg, DT_INT8, DT_INT8, baseline_neg,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Neg, DT_INT16, DT_INT16, baseline_neg,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Neg, DT_INT64, DT_INT64, baseline_neg,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Neg, DT_COMPLEX64, DT_COMPLEX64, baseline_neg,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Neg, DT_COMPLEX128, DT_COMPLEX128, baseline_neg,
                      test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.OnesLike`.

template <typename T>
T baseline_ones_like(T /*inp*/) {
  return T(1);
}

GENERATE_DEFAULT_TEST(OnesLike, DT_BOOL, DT_BOOL, baseline_ones_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(OnesLike, DT_HALF, DT_HALF, baseline_ones_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(OnesLike, DT_FLOAT, DT_FLOAT, baseline_ones_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(OnesLike, DT_DOUBLE, DT_DOUBLE, baseline_ones_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(OnesLike, DT_INT64, DT_INT64, baseline_ones_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(OnesLike, DT_COMPLEX64, DT_COMPLEX64, baseline_ones_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(OnesLike, DT_COMPLEX128, DT_COMPLEX128,
                      baseline_ones_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TEST(OnesLike, DT_INT8, DT_INT8, baseline_ones_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(OnesLike, DT_INT16, DT_INT16, baseline_ones_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(OnesLike, DT_UINT8, DT_UINT8, baseline_ones_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(OnesLike, DT_UINT16, DT_UINT16, baseline_ones_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(OnesLike, DT_UINT32, DT_UINT32, baseline_ones_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(OnesLike, DT_UINT64, DT_UINT64, baseline_ones_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
#endif

/// Test `tf.Real`.

template <typename T>
typename T::value_type baseline_real(T x) {
  return std::real(x);
}

GENERATE_DEFAULT_TEST(Real, DT_COMPLEX64, DT_FLOAT, baseline_real,
                      test::OpsTestConfig().AddTout().NoBufferReuse())

GENERATE_DEFAULT_TEST(Real, DT_COMPLEX128, DT_DOUBLE, baseline_real,
                      test::OpsTestConfig().AddTout().NoBufferReuse())

/// Test `tf.Reciprocal`.

template <typename T>
T baseline_reciprocal(T x) {
  return T(1) / x;
}

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Reciprocal, DT_INT64, DT_INT64, test::DefaultInputNonZero<int64_t>(),
    baseline_reciprocal, test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Reciprocal, DT_FLOAT, DT_FLOAT, baseline_reciprocal,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Reciprocal, DT_DOUBLE, DT_DOUBLE, baseline_reciprocal,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST_2(Reciprocal, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
                        baseline_reciprocal, test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Reciprocal, DT_COMPLEX64, DT_COMPLEX64,
                      baseline_reciprocal, test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Reciprocal, DT_COMPLEX128, DT_COMPLEX128,
                      baseline_reciprocal, test::OpsTestConfig())

/// Test `tf.Relu`.

template <typename T>
T baseline_relu(T x) {
  return std::max(x, static_cast<T>(0.0));
}

GENERATE_DEFAULT_TEST(Relu, DT_FLOAT, DT_FLOAT, baseline_relu,
                      test::OpsTestConfig())
GENERATE_DEFAULT_TEST(Relu, DT_DOUBLE, DT_DOUBLE, baseline_relu,
                      test::OpsTestConfig())
GENERATE_DEFAULT_TEST_2(Relu, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
                        baseline_relu, test::OpsTestConfig())

// Test the JIT-compiled kernels.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TEST(Relu, DT_INT8, DT_INT8, baseline_relu,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(Relu, DT_INT16, DT_INT16, baseline_relu,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(Relu, DT_INT64, DT_INT64, baseline_relu,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(Relu, DT_UINT8, DT_UINT8, baseline_relu,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(Relu, DT_UINT16, DT_UINT16, baseline_relu,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(Relu, DT_UINT32, DT_UINT32, baseline_relu,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(Relu, DT_UINT64, DT_UINT64, baseline_relu,
                      test::OpsTestConfig().ExpectStrictlyEqual())
#endif

/// Test `tf.Rint`.

template <typename T>
T baseline_rint(T x) {
  T y = std::rint(x);
  return y == T(0) ? T(0) : y;
}

// Test the JIT-compiled kernel.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TEST_2(Rint, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
                        baseline_rint,
                        test::OpsTestConfig().ExpectStrictlyEqual())

TEST_F(UnaryOpsTest, RintWithCache) {
  constexpr auto kTFJitCacheDirEnvVar = "TF_JIT_CACHE_DIR";
  // First try to setup a unique directory for the file cache
  auto *env = tensorflow::Env::Default();
  std::string jit_dir;
  auto max_tries = 5;
  do {
    ASSERT_GE(max_tries--, 0);
  } while (!env->LocalTempFilename(&jit_dir));
  char* original_env = getenv(kTFJitCacheDirEnvVar);
  setenv(kTFJitCacheDirEnvVar, jit_dir.c_str(), 1);

  // Run the actual test
  Test<Eigen::half, float, Eigen::half, float>(
      "Rint", test::DefaultInputShape(), test::DefaultInput<Eigen::half>(),
      baseline_rint, test::OpsTestConfig().ExpectStrictlyEqual());

  // Test that the file cache is not empty after compiling
  std::vector<std::string> children;
  TF_ASSERT_OK(env->GetChildren(jit_dir, &children));
  ASSERT_EQ(1, children.size());

  if (original_env != nullptr) setenv(kTFJitCacheDirEnvVar, original_env, 1);
}
#endif

GENERATE_DEFAULT_TEST(Rint, DT_FLOAT, DT_FLOAT, baseline_rint,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Rint, DT_DOUBLE, DT_DOUBLE,
    test::InputAsVector<double>({-1.7, -1.5, -0.2, -0.0, 0.0, 0.2, 0.5000001,
                                 1.5, 1.7, 2.0}),
    baseline_rint, test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.Round`.

/// `tf.Round` is the same as `std::rint` and different from `std::round`. It
/// rounds to the nearest even integer, not towards zero.

GENERATE_DEFAULT_TEST_WITH_SPECIFIC_INPUT_VALUES(
    Round, DT_DOUBLE, DT_DOUBLE,
    test::InputAsVector<double>({-1.7, -1.5, -0.2, -0.0, 0.0, 0.2, 0.5000001,
                                 1.5, 1.7, 2.0}),
    baseline_rint, test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Round, DT_FLOAT, DT_FLOAT, baseline_rint,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST_2(Round, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
                        baseline_rint,
                        test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Round, DT_INT32, DT_INT32, baseline_rint,
                      test::OpsTestConfig().ExpectStrictlyEqual())

GENERATE_DEFAULT_TEST(Round, DT_INT64, DT_INT64, baseline_rint,
                      test::OpsTestConfig().ExpectStrictlyEqual())

/// Test `tf.Rsqrt`.

/// Reference implementation.
template <typename T>
T baseline_rsqrt(T x) {
  return 1.0 / std::sqrt(x);
}

GENERATE_DEFAULT_TEST(Rsqrt, DT_FLOAT, DT_FLOAT, baseline_rsqrt,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Rsqrt, DT_DOUBLE, DT_DOUBLE, baseline_rsqrt,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST_2(Rsqrt, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
                        baseline_rsqrt, test::OpsTestConfig())

/// Test `tf.Selu`.

template <typename T>
T baseline_selu(T x) {
  const double scale = 1.0507009873554804934193349852946;
  if (x > 0) return scale * x;

  const double scaled_alpha = 1.7580993408473768599402175208123;
  return scaled_alpha * (std::exp(x) - 1);
}

GENERATE_DEFAULT_TEST(Selu, DT_FLOAT, DT_FLOAT, baseline_selu,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Selu, DT_DOUBLE, DT_DOUBLE, baseline_selu,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST_2(Selu, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
                        baseline_selu, test::OpsTestConfig())

/// Test `tf.Sigmoid`.

// Reference implementation
template <typename T>
T baseline_sigmoid(T x) {
  return 1.0 / (1 + std::exp(-x));
}

GENERATE_DEFAULT_TEST(Sigmoid, DT_FLOAT, DT_FLOAT, baseline_sigmoid,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Sigmoid, DT_DOUBLE, DT_DOUBLE, baseline_sigmoid,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST_2(Sigmoid, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
                        baseline_sigmoid, test::OpsTestConfig())

/// Test `tf.Sign`.

// Reference implementation
template <typename T>
T baseline_sign(T x) {
  if (std::isnan(x)) return x;
  if (x == 0) return 0;
  if (x < 0) return -1;
  return 1;
}

template <>
std::complex<double> baseline_sign(std::complex<double> x) {
  double abs_x = std::abs(x);
  if (abs_x == 0) return std::complex<double>(0);
  double abs_x_inverse = 1 / abs_x;
  return std::complex<double>(x.real() * abs_x_inverse,
                              x.imag() * abs_x_inverse);
}

GENERATE_DEFAULT_TEST(Sign, DT_FLOAT, DT_FLOAT, baseline_sign,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(Sign, DT_DOUBLE, DT_DOUBLE, baseline_sign,
                      test::OpsTestConfig().ExpectStrictlyEqual())
// TODO(b/162577610): We should actually use ExpectStrictlyEqual()
// here. This requires returning 0.0 for input -0.0.
GENERATE_DEFAULT_TEST_2(Sign, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
                        baseline_sign, test::OpsTestConfig())
GENERATE_DEFAULT_TEST(Sign, DT_INT64, DT_INT64, baseline_sign,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST_2(Sign, DT_COMPLEX64, DT_COMPLEX128, DT_COMPLEX64,
                        DT_COMPLEX128, baseline_sign,
                        test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(Sign, DT_COMPLEX128, DT_COMPLEX128, baseline_sign,
                      test::OpsTestConfig().ExpectStrictlyEqual())

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TEST(Sign, DT_INT8, DT_INT8, baseline_sign,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(Sign, DT_INT16, DT_INT16, baseline_sign,
                      test::OpsTestConfig().ExpectStrictlyEqual())
#endif

/// Test `tf.Sin`.

GENERATE_DEFAULT_TEST(Sin, DT_FLOAT, DT_FLOAT, std::sin, test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Sin, DT_DOUBLE, DT_DOUBLE, std::sin,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST_2(Sin, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, std::sin,
                        test::OpsTestConfig())

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)

GENERATE_DEFAULT_TEST(Sin, DT_COMPLEX64, DT_COMPLEX64, std::sin,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Sin, DT_COMPLEX128, DT_COMPLEX128, std::sin,
                      test::OpsTestConfig())

#endif

/// Test `tf.Sinh`.

GENERATE_DEFAULT_TEST_2(Sinh, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, std::sinh,
                        test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Sinh, DT_FLOAT, DT_FLOAT, std::sinh,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Sinh, DT_DOUBLE, DT_DOUBLE, std::sinh,
                      test::OpsTestConfig())

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)

GENERATE_DEFAULT_TEST(Sinh, DT_COMPLEX64, DT_COMPLEX64, std::sinh,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Sinh, DT_COMPLEX128, DT_COMPLEX128, std::sinh,
                      test::OpsTestConfig())

#endif

/// Test `tf.Softplus`.

// Reference implementation
template <typename T>
T baseline_softplus(T x) {
  T epsilon = std::numeric_limits<T>::epsilon();
  T threshold = 2 + std::log(epsilon);
  if (x > -threshold && x < threshold) {
    return std::exp(x);
  }
  return std::log1p(std::exp(x));
}

GENERATE_DEFAULT_TEST_2(Softplus, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
                        baseline_softplus, test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Softplus, DT_FLOAT, DT_FLOAT, baseline_softplus,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Softplus, DT_DOUBLE, DT_DOUBLE, baseline_softplus,
                      test::OpsTestConfig())

/// Test `tf.Softsign`.

// Reference implementation
template <typename T>
T baseline_softsign(T x) {
  return x / (std::abs(x) + 1);
}

GENERATE_DEFAULT_TEST_2(Softsign, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT,
                        baseline_softsign, test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Softsign, DT_FLOAT, DT_FLOAT, baseline_softsign,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Softsign, DT_DOUBLE, DT_DOUBLE, baseline_softsign,
                      test::OpsTestConfig())

/// Test `tf.Sqrt`.

GENERATE_DEFAULT_TEST(Sqrt, DT_FLOAT, DT_FLOAT, std::sqrt,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Sqrt, DT_DOUBLE, DT_DOUBLE, std::sqrt,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST_2(Sqrt, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, std::sqrt,
                        test::OpsTestConfig())

/// Test `tf.Tan`.

GENERATE_DEFAULT_TEST(Tan, DT_FLOAT, DT_FLOAT, std::tan, test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Tan, DT_DOUBLE, DT_DOUBLE, std::tan,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST_2(Tan, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, std::tan,
                        test::OpsTestConfig())

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)

GENERATE_DEFAULT_TEST(Tan, DT_COMPLEX64, DT_COMPLEX64, std::tan,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Tan, DT_COMPLEX128, DT_COMPLEX128, std::tan,
                      test::OpsTestConfig())

#endif

/// Test `tf.Tanh`.

GENERATE_DEFAULT_TEST(Tanh, DT_FLOAT, DT_FLOAT, std::tanh,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST(Tanh, DT_DOUBLE, DT_DOUBLE, std::tanh,
                      test::OpsTestConfig())

GENERATE_DEFAULT_TEST_2(Tanh, DT_HALF, DT_FLOAT, DT_HALF, DT_FLOAT, std::tanh,
                        test::OpsTestConfig())

// Test small/large input values to be approximated as constant -1/1.
template <typename T>
T baseline_tanh_limits(T x) {
  assert((x < -10 || x > 10) &&
         "baseline_tanh_limits is only applicable to small/large values");
  return x < 0.0 ? -1.0 : 1.0;
}
TEST_F(UnaryOpsTest, TanhSmallAndLarge) {
  Test<float, float, float, float>(
      "Tanh", test::DefaultInputShape(),
      test::InputAsVector<float>({-100.0, -10.5, 12.0, 123.0, 10000.0}),
      baseline_tanh_limits,
      test::OpsTestConfig().ExpectStrictlyEqual().SuppressTolerance());
}

TEST_F(UnaryOpsTest, TanhNaN) {
  Test<float, float, float, float>(
      "Tanh", test::DefaultInputShape(),
      test::InputAsVector<float>({std::numeric_limits<float>::quiet_NaN()}),
      std::tanh, test::OpsTestConfig().ExpectStrictlyEqual());
}

/// Test `tf.Square`.

template <typename T>
T baseline_square(T x) {
  return x * x;
}

GENERATE_DEFAULT_TEST(Square, DT_HALF, DT_HALF, baseline_square,
                      test::OpsTestConfig())
GENERATE_DEFAULT_TEST(Square, DT_FLOAT, DT_FLOAT, baseline_square,
                      test::OpsTestConfig())
GENERATE_DEFAULT_TEST(Square, DT_DOUBLE, DT_DOUBLE, baseline_square,
                      test::OpsTestConfig())
GENERATE_DEFAULT_TEST(Square, DT_INT64, DT_INT64, baseline_square,
                      test::OpsTestConfig().ExpectStrictlyEqual())

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TEST(Square, DT_INT8, DT_INT8, baseline_square,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(Square, DT_INT16, DT_INT16, baseline_square,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(Square, DT_UINT8, DT_UINT8, baseline_square,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(Square, DT_UINT16, DT_UINT16, baseline_square,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(Square, DT_UINT32, DT_UINT32, baseline_square,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(Square, DT_UINT64, DT_UINT64, baseline_square,
                      test::OpsTestConfig().ExpectStrictlyEqual())
#endif

/// Test `tf.ZerosLike`.

template <typename T>
T baseline_zeros_like(T /*inp*/) {
  return T(0);
}

GENERATE_DEFAULT_TEST(ZerosLike, DT_BOOL, DT_BOOL, baseline_zeros_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(ZerosLike, DT_HALF, DT_HALF, baseline_zeros_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(ZerosLike, DT_FLOAT, DT_FLOAT, baseline_zeros_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(ZerosLike, DT_DOUBLE, DT_DOUBLE, baseline_zeros_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(ZerosLike, DT_INT64, DT_INT64, baseline_zeros_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(ZerosLike, DT_COMPLEX64, DT_COMPLEX64,
                      baseline_zeros_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(ZerosLike, DT_COMPLEX128, DT_COMPLEX128,
                      baseline_zeros_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())

// These kernels are JIT-compiled.
#if defined(MLIR_GENERATED_GPU_KERNELS_ENABLED)
GENERATE_DEFAULT_TEST(ZerosLike, DT_INT8, DT_INT8, baseline_zeros_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(ZerosLike, DT_INT16, DT_INT16, baseline_zeros_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(ZerosLike, DT_UINT8, DT_UINT8, baseline_zeros_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(ZerosLike, DT_UINT16, DT_UINT16, baseline_zeros_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(ZerosLike, DT_UINT32, DT_UINT32, baseline_zeros_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
GENERATE_DEFAULT_TEST(ZerosLike, DT_UINT64, DT_UINT64, baseline_zeros_like,
                      test::OpsTestConfig().ExpectStrictlyEqual())
#endif

}  // namespace
}  // namespace tensorflow
