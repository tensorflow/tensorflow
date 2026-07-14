/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <random>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/xnnpack/quantized_variable_ops_tester.h"

namespace tflite {
namespace xnnpack {

TEST(UnsignedQuantizedReadAssignVariable, TwoSubgraphsReadAssign) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  TfLiteDelegate* delegate = xnnpack_delegate.get();
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<uint8_t>::min(),
                                      std::numeric_limits<uint8_t>::max()),
                                  std::ref(rng));

  QuantizedVariableOpsTester()
      .Unsigned(true)
      .ZeroPoint(zero_point_rng())
      .NumInputs(0)
      .NumOutputs(2)
      .NumSubgraphs(2)
      .TestTwoSubgraphsReadAssign(delegate);
}

TEST(UnsignedQuantizedReadAssignVariable, TwoSubgraphsReadAssignOneVarHandle) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  TfLiteDelegate* delegate = xnnpack_delegate.get();
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<uint8_t>::min(),
                                      std::numeric_limits<uint8_t>::max()),
                                  std::ref(rng));

  QuantizedVariableOpsTester()
      .Unsigned(true)
      .ZeroPoint(zero_point_rng())
      .NumInputs(0)
      .NumOutputs(1)
      .NumSubgraphs(2)
      .TestTwoSubgraphsReadAssignOneVarHandle(delegate);
}

TEST(UnsignedQuantizedReadAssignVariable, TwoSubgraphsReadAssignOneVarHandle2) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  TfLiteDelegate* delegate = xnnpack_delegate.get();
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<uint8_t>::min(),
                                      std::numeric_limits<uint8_t>::max()),
                                  std::ref(rng));

  QuantizedVariableOpsTester()
      .Unsigned(true)
      .ZeroPoint(zero_point_rng())
      .NumInputs(0)
      .NumOutputs(1)
      .NumSubgraphs(2)
      .TestTwoSubgraphsReadAssignOneVarHandle2(delegate);
}

}  // namespace xnnpack
}  // namespace tflite
