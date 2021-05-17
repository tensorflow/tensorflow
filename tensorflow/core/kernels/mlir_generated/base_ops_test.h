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

#ifndef TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_OPS_TEST_H_
#define TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_OPS_TEST_H_

#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/STLExtras.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace test {

/// Helper functions to create or derive inputs of the right type and size.

template <typename T, typename LiteralT>
absl::InlinedVector<T, 10> InputAsVector(
    std::initializer_list<LiteralT> input) {
  absl::InlinedVector<T, 10> result;
  result.reserve(input.size());
  for (const LiteralT& value : input) {
    result.push_back(static_cast<T>(value));
  }
  return result;
}

template <typename T>
absl::InlinedVector<T, 10> RepeatInputToMatchShape(
    absl::InlinedVector<T, 10> input, int size) {
  absl::InlinedVector<T, 10> result;
  for (int i = 0; i < size; i++) {
    auto value = input[i % input.size()];
    result.push_back(value);
  }
  return result;
}

/// Helper functions to get default input shapes.

TensorShape DefaultInputShape();

/// Helper functions to configure tests.

struct OpsTestConfig {
  bool add_t = true;
  bool add_tout = false;
  // Only used for gpu_unary_ops_test.
  bool expect_buffer_reuse = true;
  bool expect_strictly_equal = false;
  bool supress_tolerance = false;
  // Negative atol/rtol will make ExpectClose use the default.
  double atol = -1;
  double rtol = -1;
  std::string input_attribute = "T";
  std::string output_attribute = "Tout";
  OpsTestConfig ExpectStrictlyEqual() {
    OpsTestConfig config = *this;
    config.expect_strictly_equal = true;
    return config;
  }
  OpsTestConfig SuppressTolerance() {
    OpsTestConfig config = *this;
    config.supress_tolerance = true;
    return config;
  }
  OpsTestConfig NoBufferReuse() {
    OpsTestConfig config = *this;
    config.expect_buffer_reuse = false;
    return config;
  }
  OpsTestConfig AddTout() {
    OpsTestConfig config = *this;
    config.add_tout = true;
    return config;
  }
  OpsTestConfig NoT() {
    OpsTestConfig config = *this;
    config.add_t = false;
    return config;
  }
  OpsTestConfig RTol(double new_rtol) {
    OpsTestConfig config = *this;
    config.rtol = new_rtol;
    return config;
  }
  OpsTestConfig ATol(double new_atol) {
    OpsTestConfig config = *this;
    config.atol = new_atol;
    return config;
  }
  OpsTestConfig InputAttribute(const std::string& attr) {
    OpsTestConfig config = *this;
    config.input_attribute = attr;
    return config;
  }
  OpsTestConfig OutputAttribute(const std::string& attr) {
    OpsTestConfig config = *this;
    config.output_attribute = attr;
    return config;
  }
};

/// Helper functions to get more specific input data.

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
absl::InlinedVector<T, 10> NearZeroAndExtremeInput() {
  return InputAsVector<T, double>({-std::numeric_limits<double>::infinity(),
                                   -0.1, -0.0, 0.0, 0.1,
                                   std::numeric_limits<double>::infinity()});
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, int8, int16, int32, int64>::value,
                           bool> = true>
absl::InlinedVector<T, 10> NearZeroAndExtremeInput() {
  return InputAsVector<T, T>({std::numeric_limits<T>::min(),
                              std::numeric_limits<T>::min() + 1, -1, 0, 1,
                              std::numeric_limits<T>::max()});
}

template <typename T>
absl::InlinedVector<T, 10> NearZeroInfAndNanInput() {
  return InputAsVector<T, double>({-std::numeric_limits<double>::quiet_NaN(),
                                   -std::numeric_limits<double>::infinity(),
                                   -0.1, -0.0, 0.0, 0.1,
                                   std::numeric_limits<double>::infinity(),
                                   std::numeric_limits<double>::quiet_NaN()});
}

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
absl::InlinedVector<T, 10> DefaultInputGreaterEqualOne() {
  return test::InputAsVector<T, double>(
      {18.0, 9.0, 1.0, std::numeric_limits<T>::max(), 42.0, 2.0, 1.0,
       std::sqrt(std::numeric_limits<T>::max()), 9.0, 18.0});
}

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
absl::InlinedVector<T, 10> DefaultInputGreaterThanZero() {
  return test::InputAsVector<T, double>({18.0, 9.0, 1e-6, 1.0, 0.1, 1e-6, 0.1,
                                         0.2, 0.3, 0.5, 0.7, 0.9, 9.0, 18.0});
}

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
absl::InlinedVector<T, 10> DefaultInputGreaterOrEqualToZero() {
  return test::InputAsVector<T, double>({18.0, 9.0, 1e-6, 0.0, 0.1, 1e-6, 0.1,
                                         0.2, 0.3, 0.5, 0.7, 0.9, 9.0, 18.0});
}

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
absl::InlinedVector<T, 10> DefaultInputNonZero() {
  return test::InputAsVector<T, double>({18.0, 9.0, 1e-6, -0.1, 0.1, 1e-6, 0.1,
                                         0.2, 0.3, 0.5, 0.7, 0.9, 9.0, 18.0});
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, int8, int16, int32, int64>::value,
                           bool> = true>
absl::InlinedVector<T, 10> DefaultInputNonZero() {
  return test::InputAsVector<T, double>(
      {-18, -9, -1, 1, 3, 4, 5, 7, 9, 10, 18});
}

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
absl::InlinedVector<T, 10> DefaultInputBetweenZeroAndOne() {
  return test::InputAsVector<T, double>({-0.999, -0.9, -0.8, -0.5, -0.1, -0.001,
                                         -0, 0, 0.001, 0.1, 0.5, 0.8, 0.9,
                                         0.999});
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, int8, int16, int32, int64>::value,
                           bool> = true>
absl::InlinedVector<T, 10> DefaultInputLessThanBitwidth() {
  auto max_shift = sizeof(T) * 8 - 1;
  absl::InlinedVector<T, 10> v;
  for (auto i = 0; i < max_shift; ++i) v.push_back(i);
  return v;
}

/// Helper functions to get default input data.

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, int8, int16, int32, int64>::value,
                           bool> = true>
absl::InlinedVector<T, 10> DefaultInput() {
  return InputAsVector<T, int>({-18, -9, -1, 0, 0, 1, 1, 2, 3, 5, 7, 9, 9, 18});
}

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
absl::InlinedVector<T, 10> DefaultInput() {
  return InputAsVector<T, double>({-18.0, -9.0, -0.7, -0.5, -0.3, -0.2, -0.1,
                                   -1e-6, -0.0, 0.0, 1e-6, 0.1, 0.2, 0.3, 0.5,
                                   0.7, 0.9, 18.0});
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, std::complex<float>,
                                           std::complex<double>>::value,
                           bool> = true>
absl::InlinedVector<T, 10> DefaultInput() {
  using ElementType = typename T::value_type;
  auto input = test::DefaultInput<ElementType>();
  absl::InlinedVector<T, 10> complex_input;
  for (ElementType value : input) {
    complex_input.emplace_back(value, -value);
  }
  return complex_input;
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, std::complex<float>,
                                           std::complex<double>>::value,
                           bool> = true>
absl::InlinedVector<T, 10> ComplexInputFromValues(
    const absl::InlinedVector<typename T::value_type, 10>& real,
    const absl::InlinedVector<typename T::value_type, 10>& imag) {
  using ElementType = typename T::value_type;
  auto input = test::DefaultInput<ElementType>();
  absl::InlinedVector<T, 10> complex_input;
  CHECK_EQ(real.size(), imag.size());
  for (size_t i = 0; i < real.size() && i < imag.size(); ++i) {
    complex_input.emplace_back(real[i], imag[i]);
  }
  return complex_input;
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, std::complex<float>,
                                           std::complex<double>>::value,
                           bool> = true>
absl::InlinedVector<T, 10> DefaultInputNonZero() {
  auto real = test::DefaultInputNonZero<typename T::value_type>();
  auto imag = real;
  std::reverse(imag.begin(), imag.end());
  return test::ComplexInputFromValues<T>(real, imag);
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, bool>::value, bool> = true>
absl::InlinedVector<T, 10> DefaultInput() {
  return InputAsVector<T, bool>({true, false, true, true, false});
}

}  // namespace test
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_BASE_OPS_TEST_H_
