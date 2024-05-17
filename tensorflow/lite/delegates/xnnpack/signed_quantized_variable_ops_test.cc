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

TEST(SignedQuantizedReadAssignVariable, SimpleAssignThenRead) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  TfLiteDelegate* delegate = xnnpack_delegate.get();
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));

  QuantizedVariableOpsTester()
      .ZeroPoint(zero_point_rng())
      .NumInputs(1)
      .NumOutputs(1)
      .TestAssignThenRead(delegate);
}

TEST(SignedQuantizedReadAssignVariable, AssignTwiceThenRead) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  TfLiteDelegate* delegate = xnnpack_delegate.get();
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));

  QuantizedVariableOpsTester()
      .ZeroPoint(zero_point_rng())
      .NumInputs(2)
      .NumOutputs(1)
      .TestAssignTwiceThenRead(delegate);
}

TEST(SignedQuantizedReadAssignVariable,
     SimpleAssignThenReadUsingAnotherVarHandle) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  TfLiteDelegate* delegate = xnnpack_delegate.get();
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));

  QuantizedVariableOpsTester()
      .ZeroPoint(zero_point_rng())
      .NumInputs(1)
      .NumOutputs(1)
      .TestAssignThenReadUsingAnotherVarHandle(delegate);
}

TEST(SignedQuantizedReadAssignVariable, TwoVarHandlesAssignThenRead) {
  auto xnnpack_delegate = NewXnnPackDelegateSupportingVariableOps();
  TfLiteDelegate* delegate = xnnpack_delegate.get();
  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));

  QuantizedVariableOpsTester()
      .ZeroPoint(zero_point_rng())
      .NumInputs(2)
      .NumOutputs(2)
      .TestTwoVarHandlesAssignThenRead(delegate);
}

}  // namespace xnnpack
}  // namespace tflite
