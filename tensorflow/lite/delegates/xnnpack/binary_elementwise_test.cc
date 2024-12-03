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
#include <cstdint>
#include <cstddef>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/xnnpack/binary_elementwise_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "absl/strings/str_cat.h"

namespace tflite {
namespace xnnpack {

BuiltinOperator all_binary_ops[] = {
    BuiltinOperator_ADD,
    BuiltinOperator_SUB,
    BuiltinOperator_MUL,
    BuiltinOperator_DIV,
    BuiltinOperator_MAXIMUM,
    BuiltinOperator_MINIMUM,
    BuiltinOperator_SQUARED_DIFFERENCE,
};

ActivationFunctionType all_activations[] = {
    ActivationFunctionType_NONE, ActivationFunctionType_RELU,
    ActivationFunctionType_RELU_N1_TO_1, ActivationFunctionType_RELU6,
    // ActivationFunctionType_SIGN_BIT,
    // ActivationFunctionType_TANH,
};

struct StaticParams {
  bool input1_static;
  bool input2_static;
};

struct ShapeParams {
  int input1_rank;
  int input2_rank;
  int input1_broadcast_mask;
  int input2_broadcast_mask;

  static std::string BroadcastMaskToString(int rank, int mask) {
    std::string result;
    for (int i = 0; i < rank; ++i) {
      if ((mask >> i) & 1) {
        result += "1";
      } else {
        result += "X";
      }
    }
    return result;
  }

  std::string ToString(const StaticParams& static_params) const {
    std::string input1_broadcast, input2_broadcast;
    if (input1_broadcast_mask != 0) {
      input1_broadcast =
          "_" + BroadcastMaskToString(input1_rank, input1_broadcast_mask);
    }
    if (input2_broadcast_mask != 0) {
      input2_broadcast =
          "_" + BroadcastMaskToString(input2_rank, input2_broadcast_mask);
    }
    const char* input1_static = static_params.input1_static ? "_Static" : "";
    const char* input2_static = static_params.input2_static ? "_Static" : "";
    return absl::StrCat(input1_rank, "D", input1_static, input1_broadcast,
                        "_By_", input2_rank, "D", input2_static,
                        input2_broadcast);
  }

  static ShapeParams _4DBy4D() { return {4, 4, 0, 0}; }
  static ShapeParams _4DBy4DBroadcastChannels() { return {4, 4, 0, 0xE}; }
  static ShapeParams _4DBroadcastChannelsBy4D() { return {4, 4, 0xE, 0}; }
  static ShapeParams _4DBy4DBroadcastWidth() { return {4, 4, 0, 0xD}; }
  static ShapeParams _4DBroadcastWidthBy4D() { return {4, 4, 0xD, 0}; }
  static ShapeParams _4DBy4DBroadcastHeight() { return {4, 4, 0, 0xB}; }
  static ShapeParams _4DBroadcastHeightBy4D() { return {4, 4, 0xB, 0}; }
  static ShapeParams _4DBy4DBroadcastBatch() { return {4, 4, 0, 0x7}; }
  static ShapeParams _4DBroadcastBatchBy4D() { return {4, 4, 0x7, 0}; }
  static ShapeParams _4DBy4DBroadcastHeightWidthChannels() {
    return {4, 4, 0, 8};
  }
  static ShapeParams _4DBroadcastHeightWidthChannelsBy4D() {
    return {4, 4, 8, 0};
  }
  static ShapeParams _4DBy3D() { return {4, 3, 0, 0}; }
  static ShapeParams _4DBy2D() { return {4, 2, 0, 0}; }
  static ShapeParams _4DBy1D() { return {4, 1, 0, 0}; }
  static ShapeParams _4DBy0D() { return {4, 0, 0, 0}; }
  static ShapeParams _2DBy2D() { return {2, 2, 0, 0}; }
  static ShapeParams _2DBy1D() { return {2, 1, 0, 0}; }
  static ShapeParams _2DBy0D() { return {2, 0, 0, 0}; }
};

class ShapeTest : public testing::TestWithParam<
                      std::tuple<BuiltinOperator, ShapeParams, StaticParams>> {
};

TEST_P(ShapeTest, Shape) {
  BuiltinOperator op = std::get<0>(GetParam());
  const ShapeParams& shape_params = std::get<1>(GetParam());
  const StaticParams& static_params = std::get<2>(GetParam());

  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  // Make a random result shape.
  std::vector<int> shape;
  for (int i = 0;
       i < std::max(shape_params.input1_rank, shape_params.input2_rank); ++i) {
    shape.push_back(shape_rng());
  }
  std::vector<int> input1_shape(shape);
  std::vector<int> input2_shape(shape);
  for (size_t i = 0; i < input1_shape.size(); ++i) {
    if (((shape_params.input1_broadcast_mask >> i) & 1) != 0) {
      input1_shape[i] = 1;
    }
  }
  for (size_t i = 0; i < input2_shape.size(); ++i) {
    if (((shape_params.input2_broadcast_mask >> i) & 1) != 0) {
      input2_shape[i] = 1;
    }
  }
  while (input1_shape.size() > shape_params.input1_rank) {
    input1_shape.erase(input1_shape.begin());
  }
  while (input2_shape.size() > shape_params.input2_rank) {
    input2_shape.erase(input2_shape.begin());
  }

  BinaryElementwiseTester()
      .Input1Shape(input1_shape)
      .Input2Shape(input2_shape)
      .Input1Static(static_params.input1_static)
      .Input2Static(static_params.input2_static)
      .Test(op, xnnpack_delegate.get());
}

const ShapeParams all_shape_params[] = {
    ShapeParams::_4DBy4D(),
    ShapeParams::_4DBy4DBroadcastChannels(),
    ShapeParams::_4DBroadcastChannelsBy4D(),
    ShapeParams::_4DBy4DBroadcastWidth(),
    ShapeParams::_4DBroadcastWidthBy4D(),
    ShapeParams::_4DBy4DBroadcastHeight(),
    ShapeParams::_4DBroadcastHeightBy4D(),
    ShapeParams::_4DBy4DBroadcastBatch(),
    ShapeParams::_4DBroadcastBatchBy4D(),
    ShapeParams::_4DBy4DBroadcastHeightWidthChannels(),
    ShapeParams::_4DBroadcastHeightWidthChannelsBy4D(),
    ShapeParams::_4DBy3D(),
    ShapeParams::_4DBy2D(),
    ShapeParams::_4DBy1D(),
    ShapeParams::_4DBy0D(),
    ShapeParams::_2DBy2D(),
    ShapeParams::_2DBy1D(),
    ShapeParams::_2DBy0D(),
};

const StaticParams all_static_params[] = {
    {true, false},
    {false, true},
    {false, false},
};

INSTANTIATE_TEST_SUITE_P(
    BroadcastTest, ShapeTest,
    testing::Combine(testing::ValuesIn(all_binary_ops),
                     testing::ValuesIn(all_shape_params),
                     testing::ValuesIn(all_static_params)),
    [](const testing::TestParamInfo<ShapeTest::ParamType>& info) {
      return EnumNameBuiltinOperator(std::get<0>(info.param)) +
             std::string("_") +
             std::get<1>(info.param).ToString(std::get<2>(info.param));
    });

class BinaryTest : public testing::TestWithParam<BuiltinOperator> {};

TEST_P(BinaryTest, FP16Weights) {
  BuiltinOperator op = GetParam();

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

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .FP16Weights()
      .Test(op, xnnpack_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input2Static(true)
      .FP16Weights()
      .Test(op, xnnpack_delegate.get());
}

TEST_P(BinaryTest, INT8Weights) {
  BuiltinOperator op = GetParam();

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

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .INT8Weights()
      .Test(op, xnnpack_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input2Static(true)
      .INT8Weights()
      .Test(op, xnnpack_delegate.get());
}

TEST_P(BinaryTest, INT8ChannelWiseWeights) {
  BuiltinOperator op = GetParam();

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

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .INT8ChannelWiseWeights()
      .Test(op, xnnpack_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input2Static(true)
      .INT8ChannelWiseWeights()
      .Test(op, xnnpack_delegate.get());
}

TEST_P(BinaryTest, SparseWeights) {
  BuiltinOperator op = GetParam();

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

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .SparseWeights()
      .Test(op, xnnpack_delegate.get());

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input2Static(true)
      .SparseWeights()
      .Test(op, xnnpack_delegate.get());
}

class ActivationTest
    : public testing::TestWithParam<
          std::tuple<BuiltinOperator, ActivationFunctionType>> {};

TEST_P(ActivationTest, Activation) {
  BuiltinOperator op = std::get<0>(GetParam());
  ActivationFunctionType activation = std::get<1>(GetParam());

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

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Activation(activation)
      .Test(op, xnnpack_delegate.get());
}

TEST_P(BinaryTest, MultiThreading) {
  BuiltinOperator op = GetParam();

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

  BinaryElementwiseTester()
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Test(op, xnnpack_delegate.get());
}

INSTANTIATE_TEST_SUITE_P(
    BinaryTest, BinaryTest, testing::ValuesIn(all_binary_ops),
    [](const testing::TestParamInfo<BinaryTest::ParamType>& info) {
      return EnumNameBuiltinOperator(info.param);
    });

INSTANTIATE_TEST_SUITE_P(
    ActivationTest, ActivationTest,
    testing::Combine(testing::Values(BuiltinOperator_ADD, BuiltinOperator_SUB,
                                     BuiltinOperator_MUL, BuiltinOperator_DIV),
                     testing::ValuesIn(all_activations)),
    [](const testing::TestParamInfo<ActivationTest::ParamType>& info) {
      return EnumNameBuiltinOperator(std::get<0>(info.param)) +
             std::string("_") +
             EnumNameActivationFunctionType(std::get<1>(info.param));
    });

}  // namespace xnnpack
}  // namespace tflite
