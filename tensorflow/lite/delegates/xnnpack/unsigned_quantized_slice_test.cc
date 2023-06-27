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
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/delegates/xnnpack/slice_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
namespace xnnpack {

TEST(UnsignedQuantizedSlice, 1D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  const std::vector<int32_t> input_shape = {shape_rng()};
  const auto offsets = RandomOffsets(rng, input_shape);
  const auto sizes = RandomSizes(rng, input_shape, offsets);

  SliceTester()
      .InputShape(input_shape)
      .Offsets(offsets)
      .Sizes(sizes)
      .UseInt64OffsetsAndSize(true)
      .Test(TensorType_UINT8, xnnpack_delegate.get());
}

TEST(UnsignedQuantizedSlice, 2D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  const std::vector<int32_t> input_shape = {shape_rng(), shape_rng()};
  const auto offsets = RandomOffsets(rng, input_shape);
  const auto sizes = RandomSizes(rng, input_shape, offsets);

  SliceTester()
      .InputShape(input_shape)
      .Offsets(offsets)
      .Sizes(sizes)
      .UseInt64OffsetsAndSize(true)
      .Test(TensorType_UINT8, xnnpack_delegate.get());
}

TEST(UnsignedQuantizedSlice, 3D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  const std::vector<int32_t> input_shape = {shape_rng(), shape_rng(),
                                            shape_rng()};
  const auto offsets = RandomOffsets(rng, input_shape);
  const auto sizes = RandomSizes(rng, input_shape, offsets);

  SliceTester()
      .InputShape(input_shape)
      .Offsets(offsets)
      .Sizes(sizes)
      .UseInt64OffsetsAndSize(true)
      .Test(TensorType_UINT8, xnnpack_delegate.get());
}

TEST(UnsignedQuantizedSlice, 4D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  const std::vector<int32_t> input_shape = {shape_rng(), shape_rng(),
                                            shape_rng(), shape_rng()};
  const auto offsets = RandomOffsets(rng, input_shape);
  const auto sizes = RandomSizes(rng, input_shape, offsets);

  SliceTester()
      .InputShape(input_shape)
      .Offsets(offsets)
      .Sizes(sizes)
      .UseInt64OffsetsAndSize(true)
      .Test(TensorType_UINT8, xnnpack_delegate.get());
}

TEST(UnsignedQuantizedSlice, 5D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  const std::vector<int32_t> input_shape = {
      shape_rng(), shape_rng(), shape_rng(), shape_rng(), shape_rng()};
  const auto offsets = RandomOffsets(rng, input_shape);
  const auto sizes = RandomSizes(rng, input_shape, offsets);

  SliceTester()
      .InputShape(input_shape)
      .Offsets(offsets)
      .Sizes(sizes)
      .UseInt64OffsetsAndSize(true)
      .Test(TensorType_UINT8, xnnpack_delegate.get());
}

}  // namespace xnnpack
}  // namespace tflite
