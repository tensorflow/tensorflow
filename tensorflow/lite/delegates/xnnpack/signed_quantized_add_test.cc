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
#include <limits>
#include <memory>
#include <random>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/xnnpack/quantized_binary_elementwise_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

TEST(SignedQuantizedAdd, 4DBy4D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DBy4DBroadcastChannels) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({1, 1, 1, channels})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({1, 1, 1, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DBy4DBroadcastWidth) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({1, 1, width, 1})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({1, 1, width, 1})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DBy4DBroadcastHeight) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({1, height, 1, 1})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({1, height, 1, 1})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DBy4DBroadcastBatch) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, 1, 1, 1})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, 1, 1, 1})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DBy4DBroadcastHeightWidthChannels) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({1, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({1, height, width, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DBy3D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({height, width, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DBy2D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({width, channels})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({width, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DBy1D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({channels})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DBy0D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 2DBy2D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, channels})
      .Input2Shape({batch, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 2DBy1D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({channels})
      .Input2Shape({batch, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, channels})
      .Input2Shape({channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 2DBy0D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({})
      .Input2Shape({batch, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, channels})
      .Input2Shape({})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DByStatic4D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DByStatic4DBroadcastChannels) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({1, 1, 1, channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({1, 1, 1, channels})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DByStatic4DBroadcastWidth) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({1, 1, width, 1})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({1, 1, width, 1})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DByStatic4DBroadcastHeight) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({1, height, 1, 1})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({1, height, 1, 1})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DByStatic4DBroadcastBatch) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, 1, 1, 1})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, 1, 1, 1})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DByStatic4DBroadcastHeightWidthChannels) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({1, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({1, height, width, channels})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DByStatic3D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({height, width, channels})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DByStatic2D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({width, channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({width, channels})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DByStatic1D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({channels})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({channels})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 4DByStatic0D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({})
      .Input2Shape({batch, height, width, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 2DByStatic2D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, channels})
      .Input2Shape({batch, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, channels})
      .Input2Shape({batch, channels})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 2DByStatic1D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({channels})
      .Input2Shape({batch, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, channels})
      .Input2Shape({channels})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, 2DByStatic0D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({})
      .Input2Shape({batch, channels})
      .Input1Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, channels})
      .Input2Shape({})
      .Input2Static(true)
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, ReluActivation) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  // Avoid degenerate situation when
  // output_min == output_max == std::numeric_limits<int8_t>::max()
  auto output_zero_point_rng =
      std::bind(std::uniform_int_distribution<int32_t>(
                    std::numeric_limits<int8_t>::min(),
                    std::numeric_limits<int8_t>::max() - 1),
                std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(output_zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .ReluActivation()
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, Relu6Activation) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  // Avoid degenerate situation when
  // output_min == output_max == std::numeric_limits<int8_t>::max()
  auto output_zero_point_rng =
      std::bind(std::uniform_int_distribution<int32_t>(
                    std::numeric_limits<int8_t>::min(),
                    std::numeric_limits<int8_t>::max() - 1),
                std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(output_zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Relu6Activation()
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, ReluMinus1To1Activation) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .ReluMinus1To1Activation()
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

TEST(SignedQuantizedAdd, MultiThreading) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.num_threads = 2;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto zero_point_rng = std::bind(std::uniform_int_distribution<int32_t>(
                                      std::numeric_limits<int8_t>::min(),
                                      std::numeric_limits<int8_t>::max()),
                                  std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto width = shape_rng();
  const auto channels = shape_rng();

  QuantizedBinaryElementwiseTester()
      .Input1ZeroPoint(zero_point_rng())
      .Input2ZeroPoint(zero_point_rng())
      .OutputZeroPoint(zero_point_rng())
      .Input1Shape({batch, height, width, channels})
      .Input2Shape({batch, height, width, channels})
      .Test(BuiltinOperator_ADD, xnnpack_delegate.get());
}

}  // namespace xnnpack
}  // namespace tflite
