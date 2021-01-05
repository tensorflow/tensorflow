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

#ifndef TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_GPU_OPS_TEST_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_GPU_OPS_TEST_UTIL_H_

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

struct GpuOpsTestConfig {
  bool add_t = true;
  bool add_tout = false;
  // Only used for gpu_unary_ops_test.
  bool expect_buffer_reuse = true;
  bool expect_strictly_equal = false;
  GpuOpsTestConfig ExpectStrictlyEqual() {
    GpuOpsTestConfig config = *this;
    config.expect_strictly_equal = true;
    return config;
  }
  GpuOpsTestConfig NoBufferReuse() {
    GpuOpsTestConfig config = *this;
    config.expect_buffer_reuse = false;
    return config;
  }
  GpuOpsTestConfig AddTout() {
    GpuOpsTestConfig config = *this;
    config.add_tout = true;
    return config;
  }
  GpuOpsTestConfig NoT() {
    GpuOpsTestConfig config = *this;
    config.add_t = false;
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
                                   std::numeric_limits<float>::infinity()});
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, int8, int16, int32, int64>::value,
                           bool> = true>
absl::InlinedVector<T, 10> NearZeroAndExtremeInput() {
  return InputAsVector<T, T>({std::numeric_limits<T>::min(),
                              std::numeric_limits<T>::min() + 1, -1, 0, 1,
                              std::numeric_limits<T>::max()});
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

/// Helper functions to get default input data.

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, int8, int16, int32, int64>::value,
                           bool> = true>
T DefaultScalarInput() {
  return static_cast<T>(3);
}

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
T DefaultScalarInput() {
  return static_cast<T>(2.0);
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, bool>::value, bool> = true>
T DefaultScalarInput() {
  return static_cast<T>(true);
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, int8, int16, int32, int64>::value,
                           bool> = true>
absl::InlinedVector<T, 10> DefaultInput(absl::string_view op_name) {
  if (op_name == "Abs") {
    return NearZeroAndExtremeInput<T>();
  }
  // Only generate values less than the bitwidth of the data type.
  if (op_name == "LeftShift" || op_name == "RightShift") {
    auto max_shift = sizeof(T) * 8 - 1;
    absl::InlinedVector<T, 10> v(max_shift);
    for (auto i = 0; i < max_shift; ++i) v.push_back(i);
    return v;
  }
  return InputAsVector<T, int>({-18, -9, -1, 0, 0, 1, 1, 2, 3, 5, 7, 9, 9, 18});
}

template <typename T, std::enable_if_t<
                          llvm::is_one_of<T, Eigen::half, float, double>::value,
                          bool> = true>
absl::InlinedVector<T, 10> DefaultInput(absl::string_view op_name) {
  if (op_name == "Abs") {
    return NearZeroAndExtremeInput<T>();
  }
  if (op_name == "Log" || op_name == "Rsqrt") {
    return DefaultInputGreaterThanZero<T>();
  }
  if (op_name == "Sqrt") {
    return DefaultInputGreaterOrEqualToZero<T>();
  }
  if (op_name == "FloorDiv") {
    return InputAsVector<T, double>({-18.0, -9.0, -1e-6, -0.1, 0.1, 1e-6, 0.1,
                                     0.2, 0.3, 0.5, 0.7, 0.9, 9.0, 18.0});
  }
  return InputAsVector<T, double>({-18.0, -9.0, -1e-6, -0.0, 0.0, 1e-6, 0.1,
                                   0.2, 0.3, 0.5, 0.7, 0.9, 9.0, 18.0});
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, std::complex<float>,
                                           std::complex<double>>::value,
                           bool> = true>
absl::InlinedVector<T, 10> DefaultInput(absl::string_view op_name) {
  using ElementType = typename T::value_type;
  auto input = test::DefaultInput<ElementType>(op_name);
  absl::InlinedVector<T, 10> complex_input;
  for (ElementType value : input) {
    complex_input.emplace_back(value, -value);
  }
  return complex_input;
}

template <typename T,
          std::enable_if_t<llvm::is_one_of<T, bool>::value, bool> = true>
absl::InlinedVector<T, 10> DefaultInput(absl::string_view /*op_name*/) {
  return InputAsVector<T, bool>({true, false, true, true, false});
}

}  // namespace test
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MLIR_GENERATED_GPU_OPS_TEST_UTIL_H_
