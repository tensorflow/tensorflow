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
#include <memory>
#include <random>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/xnnpack/batch_matrix_multiply_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

TEST(BatchMatrixMultiply, 3D) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto input1_channels = channels_rng();
  const auto output_channels = channels_rng();

  BatchMatrixMultiplyTester()
      .Input1Shape({batch, height, input1_channels})
      .Input2Shape({batch, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
}

TEST(BatchMatrixMultiply, 4D) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto outer_batch = shape_rng();
  const auto inner_batch = shape_rng();
  const auto height = shape_rng();
  const auto input1_channels = channels_rng();
  const auto output_channels = channels_rng();

  BatchMatrixMultiplyTester()
      .Input1Shape({outer_batch, inner_batch, height, input1_channels})
      .Input2Shape({outer_batch, inner_batch, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
}

TEST(BatchMatrixMultiply, 4D_AdjY) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  const auto outer_batch = shape_rng();
  const auto inner_batch = shape_rng();
  const auto height = shape_rng();
  const auto input1_channels = channels_rng();
  const auto output_channels = channels_rng();

  BatchMatrixMultiplyTester()
      .Input1Shape({outer_batch, inner_batch, height, input1_channels})
      .Input2Shape({outer_batch, inner_batch, output_channels, input1_channels})
      .AdjY(true)
      .Test(xnnpack_delegate.get());
}

TEST(BatchMatrixMultiply, MultiThreading) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  delegate_options.num_threads = 2;
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto input1_channels = channels_rng();
  const auto output_channels = channels_rng();

  BatchMatrixMultiplyTester()
      .Input1Shape({batch, height, input1_channels})
      .Input2Shape({batch, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
}

TEST(BatchMatrixMultiply, WeightsCache) {
  TfLiteXNNPackDelegateOptions delegate_options =
      TfLiteXNNPackDelegateOptionsDefault();
  std::unique_ptr<TfLiteXNNPackDelegateWeightsCache,
                  decltype(&TfLiteXNNPackDelegateWeightsCacheDelete)>
      weights_cache(TfLiteXNNPackDelegateWeightsCacheCreate(),
                    TfLiteXNNPackDelegateWeightsCacheDelete);
  delegate_options.weights_cache = weights_cache.get();
  delegate_options.flags |=
      TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(&delegate_options),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto channels_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 9), std::ref(rng));
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto input1_channels = channels_rng();
  const auto output_channels = channels_rng();

  BatchMatrixMultiplyTester()
      .Input1Shape({batch, height, input1_channels})
      .Input2Shape({batch, input1_channels, output_channels})
      .WeightsCache(weights_cache.get())
      .Test(xnnpack_delegate.get());
}

}  // namespace xnnpack
}  // namespace tflite
