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

#include "tensorflow/lite/experimental/litert/core/model/flatbuffer_to_litert.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"

namespace litert::internal {
namespace {

using ::testing::ElementsAreArray;

TEST(FlatbufferToLiteRtTest, MapStaticTensorType) {
  static constexpr int32_t kDims[] = {2, 2};
  static constexpr auto kDimsSpan = absl::MakeConstSpan(kDims);

  auto t = MapTensorType(std::make_pair(TflElementType::TensorType_INT32,
                                        TflShapeInfo(kDimsSpan)));
  ASSERT_TRUE(t);

  ASSERT_EQ(t->first, kLiteRtRankedTensorType);
  auto& ranked = t->second.ranked_tensor_type;
  EXPECT_EQ(ranked.element_type, kLiteRtElementTypeInt32);
  EXPECT_EQ(absl::MakeSpan(ranked.layout.dimensions, ranked.layout.rank),
            kDimsSpan);
}

TEST(FlatbufferToLiteRtTest, MapStaticTensorInt4Type) {
  static constexpr int32_t kDims[] = {2, 2};
  static constexpr auto kDimsSpan = absl::MakeConstSpan(kDims);

  auto t = MapTensorType(
      std::make_pair(TflElementType::TensorType_INT4, TflShapeInfo(kDimsSpan)));
  ASSERT_TRUE(t);

  ASSERT_EQ(t->first, kLiteRtRankedTensorType);
  auto& ranked = t->second.ranked_tensor_type;
  EXPECT_EQ(ranked.element_type, kLiteRtElementTypeInt4);
  EXPECT_EQ(absl::MakeSpan(ranked.layout.dimensions, ranked.layout.rank),
            kDimsSpan);
}

TEST(FlatbufferToLiteRtTest, MapDynamicTensorType) {
  static constexpr int32_t kDims[] = {-1, 2};
  static constexpr auto kDimsSpan = absl::MakeConstSpan(kDims);

  auto t = MapTensorType(std::make_pair(TflElementType::TensorType_INT32,
                                        TflShapeInfo(kDimsSpan)));
  ASSERT_TRUE(t);

  ASSERT_EQ(t->first, kLiteRtRankedTensorType);
  auto& ranked = t->second.ranked_tensor_type;
  EXPECT_EQ(ranked.element_type, kLiteRtElementTypeInt32);
  EXPECT_EQ(absl::MakeSpan(ranked.layout.dimensions, ranked.layout.rank),
            kDimsSpan);
}

TEST(FlatbufferToLiteRtTest, MapNoQuantization) {
  LiteRtTensorT tensor;
  auto q = MapQuantization(nullptr, tensor);
  ASSERT_TRUE(q);
  ASSERT_EQ(q->first, kLiteRtQuantizationNone);
}

TEST(FlatbufferToLiteRtTest, MapPerTensorQuantization) {
  static constexpr float kScale = 1.0;
  static constexpr int64_t kZp = 2;

  TflQuantization tfl_q;
  tfl_q.scale.assign({kScale});
  tfl_q.zero_point.assign({kZp});

  LiteRtTensorT tensor;
  auto q = MapQuantization(&tfl_q, tensor);
  ASSERT_TRUE(q);
  ASSERT_EQ(q->first, kLiteRtQuantizationPerTensor);
  EXPECT_EQ(q->second.per_tensor.scale, kScale);
  EXPECT_EQ(q->second.per_tensor.zero_point, kZp);
}

TEST(FlatbufferToLiteRtTest, MapPerChannelQuantization) {
  static constexpr size_t kRank = 2;
  static constexpr float kScales[kRank] = {1.0, 2.0};
  static constexpr int64_t kZps[kRank] = {2, 3};
  static constexpr size_t kQDim = 1;

  TflQuantization tfl_q;
  tfl_q.scale.assign(kScales, kScales + kRank);
  tfl_q.zero_point.assign(kZps, kZps + kRank);
  tfl_q.quantized_dimension = kQDim;

  LiteRtTensorT tensor;
  auto q = MapQuantization(&tfl_q, tensor);
  ASSERT_TRUE(q);
  ASSERT_EQ(q->first, kLiteRtQuantizationPerChannel);
  EXPECT_THAT(absl::MakeConstSpan(q->second.per_channel.scales, kRank),
              ElementsAreArray(kScales));

  EXPECT_THAT(absl::MakeConstSpan(q->second.per_channel.zero_points, kRank),
              ElementsAreArray(kZps));
  EXPECT_EQ(q->second.per_channel.quantized_dimension, kQDim);
  EXPECT_EQ(q->second.per_channel.num_channels, kRank);
}

}  // namespace
}  // namespace litert::internal
