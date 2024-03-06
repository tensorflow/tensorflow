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
#include <memory>
#include <random>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/xnnpack/batch_matrix_multiply_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

class BatchMatrixMultiplyTest : public testing::Test {
 public:
  // std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
  auto get_delegate(int num_threads = 1) {
    TfLiteXNNPackDelegateOptions delegate_options =
        TfLiteXNNPackDelegateOptionsDefault();
    delegate_options.flags |=
        TFLITE_XNNPACK_DELEGATE_FLAG_ENABLE_LATEST_OPERATORS;
    delegate_options.num_threads = num_threads;
    return std::unique_ptr<TfLiteDelegate,
                           decltype(&TfLiteXNNPackDelegateDelete)>(
        TfLiteXNNPackDelegateCreate(&delegate_options),
        TfLiteXNNPackDelegateDelete);
  }

  int32_t shape_rng() {
    return std::uniform_int_distribution<int32_t>(2, 5)(rng_);
  }
  int32_t channels_rng() {
    return std::uniform_int_distribution<int32_t>(2, 9)(rng_);
  }

 private:
  std::random_device random_device_;
  std::mt19937 rng_ = std::mt19937(random_device_());
};

TEST_F(BatchMatrixMultiplyTest, 3D) {
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto input1_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .Input1Shape({batch, height, input1_channels})
      .Input2Shape({batch, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, BroadcastOne3D) {
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto input1_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .Input1Shape({batch, height, input1_channels})
      .Input2Shape({1, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());

  BatchMatrixMultiplyTester()
      .Input1Shape({1, height, input1_channels})
      .Input2Shape({batch, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, BroadcastImplicit3D) {
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto input1_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .Input1Shape({batch, height, input1_channels})
      .Input2Shape({input1_channels, output_channels})
      .Test(xnnpack_delegate.get());

  BatchMatrixMultiplyTester()
      .Input1Shape({height, input1_channels})
      .Input2Shape({batch, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, 4D) {
  const auto outer_batch = shape_rng();
  const auto inner_batch = shape_rng();
  const auto height = shape_rng();
  const auto input1_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .Input1Shape({outer_batch, inner_batch, height, input1_channels})
      .Input2Shape({outer_batch, inner_batch, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, BroadcastOne4D) {
  const auto outer_batch = shape_rng();
  const auto inner_batch = shape_rng();
  const auto height = shape_rng();
  const auto input1_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .Input1Shape({1, inner_batch, height, input1_channels})
      .Input2Shape({outer_batch, inner_batch, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
  BatchMatrixMultiplyTester()
      .Input1Shape({outer_batch, inner_batch, height, input1_channels})
      .Input2Shape({1, inner_batch, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
  BatchMatrixMultiplyTester()
      .Input1Shape({outer_batch, 1, height, input1_channels})
      .Input2Shape({outer_batch, inner_batch, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
  BatchMatrixMultiplyTester()
      .Input1Shape({outer_batch, inner_batch, height, input1_channels})
      .Input2Shape({outer_batch, 1, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
  BatchMatrixMultiplyTester()
      .Input1Shape({1, 1, height, input1_channels})
      .Input2Shape({outer_batch, inner_batch, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
  BatchMatrixMultiplyTester()
      .Input1Shape({outer_batch, inner_batch, height, input1_channels})
      .Input2Shape({1, 1, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, BroadcastImplicit4D) {
  const auto outer_batch = shape_rng();
  const auto inner_batch = shape_rng();
  const auto height = shape_rng();
  const auto input1_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .Input1Shape({inner_batch, height, input1_channels})
      .Input2Shape({outer_batch, inner_batch, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
  BatchMatrixMultiplyTester()
      .Input1Shape({outer_batch, inner_batch, height, input1_channels})
      .Input2Shape({inner_batch, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
  BatchMatrixMultiplyTester()
      .Input1Shape({height, input1_channels})
      .Input2Shape({outer_batch, inner_batch, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
  BatchMatrixMultiplyTester()
      .Input1Shape({outer_batch, inner_batch, height, input1_channels})
      .Input2Shape({input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, 4D_AdjY) {
  const auto outer_batch = shape_rng();
  const auto inner_batch = shape_rng();
  const auto height = shape_rng();
  const auto input1_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate();

  BatchMatrixMultiplyTester()
      .Input1Shape({outer_batch, inner_batch, height, input1_channels})
      .Input2Shape({outer_batch, inner_batch, output_channels, input1_channels})
      .AdjY(true)
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, MultiThreading) {
  const auto batch = shape_rng();
  const auto height = shape_rng();
  const auto input1_channels = channels_rng();
  const auto output_channels = channels_rng();
  auto xnnpack_delegate = get_delegate(/*num_threads=*/2);

  BatchMatrixMultiplyTester()
      .Input1Shape({batch, height, input1_channels})
      .Input2Shape({batch, input1_channels, output_channels})
      .Test(xnnpack_delegate.get());
}

TEST_F(BatchMatrixMultiplyTest, WeightsCache) {
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
