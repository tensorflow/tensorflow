
// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/core/model/litert_to_flatbuffer.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_layout.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAreArray;

TEST(LiteRtToFlatbufferTest, MapNoQuantization) {
  Quantization q;
  auto tfl_q = MapQuantization(q);
  ASSERT_TRUE(tfl_q);
  EXPECT_EQ(tfl_q.Value(), nullptr);
}

TEST(LiteRtToFlatbufferTest, MapPerTensorQuantization) {
  static constexpr float kScale = 1.0;
  static constexpr int64_t kZp = 2;

  Quantization q;
  q.first = kLiteRtQuantizationPerTensor;
  q.second.per_tensor.scale = kScale;
  q.second.per_tensor.zero_point = kZp;

  auto tfl_q = MapQuantization(q);
  ASSERT_TRUE(tfl_q);
  EXPECT_THAT(tfl_q->get()->scale, ElementsAreArray({kScale}));
  EXPECT_THAT(tfl_q->get()->zero_point, ElementsAreArray({kZp}));
}

TEST(LiteRtToFlatbufferTest, MapPerChannelQuantization) {
  static constexpr size_t kRank = 2;
  static constexpr size_t kQuantizedDimension = 1;
  static constexpr float kScales[kRank] = {1.0, 2.0};
  static constexpr int64_t kZps[kRank] = {2, 3};

  Quantization q;
  q.first = kLiteRtQuantizationPerChannel;
  q.second.per_channel.scales = const_cast<float*>(kScales);
  q.second.per_channel.zero_points = const_cast<int64_t*>(kZps);
  q.second.per_channel.num_channels = kRank;
  q.second.per_channel.quantized_dimension = kQuantizedDimension;

  auto tfl_q = MapQuantization(q);
  ASSERT_TRUE(tfl_q);
  EXPECT_THAT(tfl_q->get()->scale, ElementsAreArray(kScales));
  EXPECT_THAT(tfl_q->get()->zero_point, ElementsAreArray(kZps));
}

TEST(LiteRtToFlatbufferTest, MapDynamicTensorType) {
  static constexpr int32_t kDims[] = {-1, 2};

  TensorType t;
  t.first = kLiteRtRankedTensorType;
  t.second.ranked_tensor_type.element_type = kLiteRtElementTypeFloat32;
  t.second.ranked_tensor_type.layout = BuildLayout(kDims);

  auto tfl_t = MapTensorType(t);
  ASSERT_TRUE(tfl_t);
  EXPECT_EQ(tfl_t->first, TflElementType::TensorType_FLOAT32);
  EXPECT_TRUE(tfl_t->second.has_rank);
  EXPECT_THAT(tfl_t->second.shape, ElementsAreArray({1, 2}));
  EXPECT_THAT(tfl_t->second.shape_signature, ElementsAreArray(kDims));
}

TEST(LiteRtToFlatbufferTest, MapStaticTensorType) {
  static constexpr int32_t kDims[] = {2, 2};

  TensorType t;
  t.first = kLiteRtRankedTensorType;
  t.second.ranked_tensor_type.element_type = kLiteRtElementTypeFloat32;
  t.second.ranked_tensor_type.layout = BuildLayout(kDims);

  auto tfl_t = MapTensorType(t);
  ASSERT_TRUE(tfl_t);
  EXPECT_EQ(tfl_t->first, TflElementType::TensorType_FLOAT32);
  EXPECT_TRUE(tfl_t->second.has_rank);
  EXPECT_THAT(tfl_t->second.shape, ElementsAreArray({2, 2}));
  EXPECT_TRUE(tfl_t->second.shape_signature.empty());
}

}  // namespace
}  // namespace litert::internal
