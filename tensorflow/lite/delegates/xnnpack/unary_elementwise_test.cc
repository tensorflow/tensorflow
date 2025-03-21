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

#include <cstdint>
#include <functional>
#include <memory>
#include <random>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/xnnpack/unary_elementwise_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

ToleranceInfo GetTolerance(BuiltinOperator op) {
  switch (op) {
    case BuiltinOperator_TANH:
    case BuiltinOperator_LOGISTIC:
      return ToleranceInfo{.relative = 1.0e+4f};
    case BuiltinOperator_GELU:
      return ToleranceInfo{.relative = 5.0f, .absolute = 10.0f};
    case BuiltinOperator_COS:
    case BuiltinOperator_SIN:
      return ToleranceInfo{.relative = 5.0f, .absolute = 3.0f};
    default:
      return ToleranceInfo{};
  }
}

class UnaryTest : public testing::TestWithParam<BuiltinOperator> {};

TEST_P(UnaryTest, 4D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  UnaryElementwiseTester()
      .Shape({batch, height, width, channels})
      .Tolerance(GetTolerance(GetParam()))
      .Test(GetParam(), xnnpack_delegate.get());
}

TEST_P(UnaryTest, 3D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  UnaryElementwiseTester()
      .Shape({batch, width, channels})
      .Tolerance(GetTolerance(GetParam()))
      .Test(GetParam(), xnnpack_delegate.get());
}

TEST_P(UnaryTest, 2D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto channels = shape_rng();

  UnaryElementwiseTester()
      .Shape({batch, channels})
      .Tolerance(GetTolerance(GetParam()))
      .Test(GetParam(), xnnpack_delegate.get());
}

TEST_P(UnaryTest, 1D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();

  UnaryElementwiseTester()
      .Shape({batch})
      .Tolerance(GetTolerance(GetParam()))
      .Test(GetParam(), xnnpack_delegate.get());
}

TEST_P(UnaryTest, MultiThreading) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.num_threads = 2;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  UnaryElementwiseTester()
      .Shape({batch, height, width, channels})
      .Tolerance(GetTolerance(GetParam()))
      .Test(GetParam(), xnnpack_delegate.get());
}

BuiltinOperator all_unary_ops[] = {
    BuiltinOperator_ABS,          BuiltinOperator_CEIL,
    BuiltinOperator_COS,          BuiltinOperator_ELU,
    BuiltinOperator_FLOOR,        BuiltinOperator_GELU,
    BuiltinOperator_NEG,          BuiltinOperator_HARD_SWISH,
    BuiltinOperator_RELU,         BuiltinOperator_RELU6,
    BuiltinOperator_RELU_N1_TO_1, BuiltinOperator_ROUND,
    BuiltinOperator_RSQRT,        BuiltinOperator_SIN,
    BuiltinOperator_SQRT,         BuiltinOperator_SQUARE,
    BuiltinOperator_TANH,         BuiltinOperator_LOGISTIC,
};

INSTANTIATE_TEST_SUITE_P(
    UnaryTest, UnaryTest, testing::ValuesIn(all_unary_ops),
    [](const testing::TestParamInfo<UnaryTest::ParamType>& info) {
      return EnumNameBuiltinOperator(info.param);
    });

}  // namespace xnnpack
}  // namespace tflite
