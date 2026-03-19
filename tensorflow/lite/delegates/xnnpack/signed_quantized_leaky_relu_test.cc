/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/xnnpack/quantized_leaky_relu_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

TEST(SignedQuantizedLeakyRelu, 4D) {
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

  QuantizedLeakyReluTester()
      .Shape({batch, height, width, channels})
      .InputZeroPoint(11)
      .InputScale(1.75f)
      .OutputZeroPoint(-7)
      .OutputScale(2.25f)
      .Test(xnnpack_delegate.get());
}

TEST(SignedQuantizedLeakyRelu, 3D) {
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

  QuantizedLeakyReluTester()
      .Shape({batch, width, channels})
      .InputZeroPoint(11)
      .InputScale(1.75f)
      .OutputZeroPoint(-7)
      .OutputScale(2.25f)
      .Test(xnnpack_delegate.get());
}

TEST(SignedQuantizedLeakyRelu, 2D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto channels = shape_rng();

  QuantizedLeakyReluTester()
      .Shape({batch, channels})
      .InputZeroPoint(11)
      .InputScale(1.75f)
      .OutputZeroPoint(-7)
      .OutputScale(2.25f)
      .Test(xnnpack_delegate.get());
}

TEST(SignedQuantizedLeakyRelu, 1D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();

  QuantizedLeakyReluTester()
      .Shape({batch})
      .InputZeroPoint(11)
      .InputScale(1.75f)
      .OutputZeroPoint(-7)
      .OutputScale(2.25f)
      .Test(xnnpack_delegate.get());
}

TEST(SignedQuantizedLeakyRelu, NegativeSlope) {
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

  QuantizedLeakyReluTester()
      .Shape({batch, height, width, channels})
      .InputZeroPoint(11)
      .InputScale(1.75f)
      .OutputZeroPoint(-7)
      .OutputScale(2.25f)
      .NegativeSlope(-0.75f)
      .Test(xnnpack_delegate.get());
}

TEST(SignedQuantizedLeakyRelu, MultiThreading) {
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

  QuantizedLeakyReluTester()
      .Shape({batch, height, width, channels})
      .InputZeroPoint(11)
      .InputScale(1.75f)
      .OutputZeroPoint(-7)
      .OutputScale(2.25f)
      .Test(xnnpack_delegate.get());
}

}  // namespace xnnpack
}  // namespace tflite
