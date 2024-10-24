/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <sstream>
#include <string>
#include <tuple>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/xnnpack/reduce_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

struct TestParam {
  using Tuple = std::tuple<BuiltinOperator, enum ReduceTester::Quantization>;
  explicit TestParam(const Tuple& t)
      : op(std::get<0>(t)), quantization(std::get<1>(t)) {}
  BuiltinOperator op;
  enum ReduceTester::Quantization quantization;
};

class ReduceTest : public testing::TestWithParam<TestParam> {
 public:
  static std::string GetName(const testing::TestParamInfo<TestParam>& i) {
    std::stringstream sstr;
    switch (i.param.op) {
      case BuiltinOperator_MEAN:
        sstr << "mean";
        break;
      case BuiltinOperator_SUM:
        sstr << "sum";
        break;
      default:
        sstr << "unknown";
        break;
    }
    switch (i.param.quantization) {
      case ReduceTester::Quantization::None:
        break;
      case ReduceTester::Quantization::Signed:
        sstr << "_signed_quantized";
        break;
      case ReduceTester::Quantization::Unsigned:
        sstr << "_unsigned_quantized";
        break;
    }
    return sstr.str();
  }
};

INSTANTIATE_TEST_SUITE_P(
    Reduce, ReduceTest,
    testing::ConvertGenerator<TestParam::Tuple>(testing::Combine(
        testing::Values(BuiltinOperator_MEAN, BuiltinOperator_SUM),
        testing::Values(ReduceTester::Quantization::None,
                        ReduceTester::Quantization::Signed,
                        ReduceTester::Quantization::Unsigned))),
    ReduceTest::GetName);

TEST_P(ReduceTest, 4DReduceBatchSqueezeDims) {
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

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, height, width, channels})
      .Axes({0})
      .KeepDims(false)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 4DReduceBatchKeepDims) {
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

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, height, width, channels})
      .Axes({0})
      .KeepDims(true)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 4DReduceHeightSqueezeDims) {
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

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, height, width, channels})
      .Axes({1})
      .KeepDims(false)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 4DReduceHeightKeepDims) {
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

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, height, width, channels})
      .Axes({1})
      .KeepDims(true)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 4DReduceWidthSqueezeDims) {
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

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, height, width, channels})
      .Axes({2})
      .KeepDims(false)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 4DReduceWidthKeepDims) {
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

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, height, width, channels})
      .Axes({2})
      .KeepDims(true)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 4DReduceHeightWidthSqueezeDims) {
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

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, height, width, channels})
      .Axes({1, 2})
      .KeepDims(false)
      .Test(GetParam().op, xnnpack_delegate.get());

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, height, width, channels})
      .Axes({2, 1})
      .KeepDims(false)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 4DReduceHeightWidthKeepDims) {
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

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, height, width, channels})
      .Axes({1, 2})
      .KeepDims(true)
      .Test(GetParam().op, xnnpack_delegate.get());

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, height, width, channels})
      .Axes({2, 1})
      .KeepDims(true)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 4DReduceChannelsSqueezeDims) {
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

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, height, width, channels})
      .Axes({3})
      .KeepDims(false)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 4DReduceChannelsKeepDims) {
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

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, height, width, channels})
      .Axes({3})
      .KeepDims(true)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 3DReduceBatchSqueezeDims) {
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

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, width, channels})
      .Axes({0})
      .KeepDims(false)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 3DReduceBatchKeepDims) {
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

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, width, channels})
      .Axes({0})
      .KeepDims(true)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 3DReduceWidthSqueezeDims) {
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

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, width, channels})
      .Axes({1})
      .KeepDims(false)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 3DReduceWidthKeepDims) {
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

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, width, channels})
      .Axes({1})
      .KeepDims(true)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 3DReduceChannelsSqueezeDims) {
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

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, width, channels})
      .Axes({2})
      .KeepDims(false)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 3DReduceChannelsKeepDims) {
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

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, width, channels})
      .Axes({2})
      .KeepDims(true)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 2DReduceBatchSqueezeDims) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, channels})
      .Axes({0})
      .KeepDims(false)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 2DReduceBatchKeepDims) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, channels})
      .Axes({0})
      .KeepDims(true)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 2DReduceChannelsSqueezeDims) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, channels})
      .Axes({1})
      .KeepDims(false)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 2DReduceChannelsKeepDims) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto channels = shape_rng();

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, channels})
      .Axes({1})
      .KeepDims(true)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 1DSqueezeDims) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch})
      .Axes({0})
      .KeepDims(false)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, 1DKeepDims) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch})
      .Axes({0})
      .KeepDims(true)
      .Test(GetParam().op, xnnpack_delegate.get());
}

TEST_P(ReduceTest, MultiThreading) {
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

  ReduceTester()
      .Quantization(GetParam().quantization)
      .InputShape({batch, height, width, channels})
      .Axes({1, 2})
      .KeepDims(true)
      .Test(GetParam().op, xnnpack_delegate.get());
}

}  // namespace xnnpack
}  // namespace tflite
