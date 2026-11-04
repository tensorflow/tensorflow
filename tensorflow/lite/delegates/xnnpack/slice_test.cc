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
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/delegates/xnnpack/slice_tester.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace xnnpack {

TEST(Slice, ToEndTracksReshapedInput) {
  TfLiteXNNPackDelegateOptions options = TfLiteXNNPackDelegateOptionsDefault();
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      delegate(TfLiteXNNPackDelegateCreate(&options),
               TfLiteXNNPackDelegateDelete);
  for (TensorType tensor_type :
       {TensorType_FLOAT32, TensorType_INT8, TensorType_UINT8}) {
    for (bool int64_indices : {false, true}) {
      for (int offset : {0, 1}) {
        SliceTester()
            .InputShape({1, 4, 8})
            .Offsets({0, offset, 2})
            .Sizes({-1, -1, 3})
            .UseInt64OffsetsAndSize(int64_indices)
            .ReshapeInputShapes({{1, 2, 8}, {1, 7, 8}, {1, 3, 8}})
            .RequireDelegation(true)
            .Test(tensor_type, delegate.get());
      }
    }
  }
}

TEST(Slice, 1D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  auto rng = std::mt19937();
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  const std::vector<int32_t> input_shape = {shape_rng()};
  const auto offsets = RandomOffsets(rng, input_shape);
  const auto sizes = RandomSizes(rng, input_shape, offsets);

  SliceTester()
      .InputShape(input_shape)
      .Offsets(offsets)
      .Sizes(sizes)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());

  SliceTester()
      .InputShape(input_shape)
      .Offsets(offsets)
      .Sizes(sizes)
      .UseInt64OffsetsAndSize(true)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());
}

TEST(Slice, 2D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  auto rng = std::mt19937();
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 5), std::ref(rng));

  const std::vector<int32_t> input_shape = {shape_rng(), shape_rng()};
  const auto offsets = RandomOffsets(rng, input_shape);
  const auto sizes = RandomSizes(rng, input_shape, offsets);

  SliceTester()
      .InputShape(input_shape)
      .Offsets(offsets)
      .Sizes(sizes)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());

  SliceTester()
      .InputShape(input_shape)
      .Offsets(offsets)
      .Sizes(sizes)
      .UseInt64OffsetsAndSize(true)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());
}

TEST(Slice, 3D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  auto rng = std::mt19937();
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
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());

  SliceTester()
      .InputShape(input_shape)
      .Offsets(offsets)
      .Sizes(sizes)
      .UseInt64OffsetsAndSize(true)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());
}

TEST(Slice, 4D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  auto rng = std::mt19937();
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
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());

  SliceTester()
      .InputShape(input_shape)
      .Offsets(offsets)
      .Sizes(sizes)
      .UseInt64OffsetsAndSize(true)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());
}

TEST(Slice, 5D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  auto rng = std::mt19937();
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
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());

  SliceTester()
      .InputShape(input_shape)
      .Offsets(offsets)
      .Sizes(sizes)
      .UseInt64OffsetsAndSize(true)
      .Test(TensorType_FLOAT32, xnnpack_delegate.get());
}

TEST(Slice, 6D) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  auto rng = std::mt19937();
  auto shape_rng =
      std::bind(std::uniform_int_distribution<int32_t>(2, 4), std::ref(rng));

  const std::vector<int32_t> input_shape = {shape_rng(), shape_rng(),
                                            shape_rng(), shape_rng(),
                                            shape_rng(), shape_rng()};
  const auto offsets = RandomOffsets(rng, input_shape);
  const auto sizes = RandomSizes(rng, input_shape, offsets);

  for (TensorType tensor_type :
       {TensorType_FLOAT32, TensorType_INT8, TensorType_UINT8}) {
    for (bool int64_indices : {false, true}) {
      SliceTester()
          .InputShape(input_shape)
          .Offsets(offsets)
          .Sizes(sizes)
          .UseInt64OffsetsAndSize(int64_indices)
          .RequireDelegation(true)
          .Test(tensor_type, xnnpack_delegate.get());
    }
  }
}

TEST(Slice, RejectsMismatchedIndexLengthWithDynamicInput) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  for (bool int64_indices : {false, true}) {
    SliceTester()
        .InputShape({2, 3})
        .InputShapeSignature({-1, 3})
        .Offsets({0})
        .Sizes({1})
        .OutputShape({1, 3})
        .UseInt64OffsetsAndSize(int64_indices)
        .TestUnsupported(TensorType_FLOAT32, xnnpack_delegate.get());
  }
}

TEST(Slice, RejectsBeginPlusSizeInt64Overflow) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  SliceTester()
      .InputShape({2})
      .InputShapeSignature({-1})
      .OffsetsInt64({std::numeric_limits<int64_t>::max()})
      .SizesInt64({2})
      .OutputShape({2})
      .UseInt64OffsetsAndSize(true)
      .TestUnsupported(TensorType_FLOAT32, xnnpack_delegate.get());
}

TEST(Slice, RejectsNegativeBeginAndInvalidSize) {
  std::unique_ptr<TfLiteDelegate, decltype(&TfLiteXNNPackDelegateDelete)>
      xnnpack_delegate(TfLiteXNNPackDelegateCreate(nullptr),
                       TfLiteXNNPackDelegateDelete);

  SliceTester()
      .InputShape({4})
      .InputShapeSignature({-1})
      .Offsets({-1})
      .Sizes({2})
      .OutputShape({2})
      .TestUnsupported(TensorType_FLOAT32, xnnpack_delegate.get());

  SliceTester()
      .InputShape({4})
      .InputShapeSignature({-1})
      .Offsets({0})
      .Sizes({-2})
      .OutputShape({2})
      .TestUnsupported(TensorType_FLOAT32, xnnpack_delegate.get());
}

}  // namespace xnnpack
}  // namespace tflite
