/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/cuda/tma_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/gpu/tma_metadata.h"

namespace stream_executor::gpu {
namespace {

TEST(TmaUtilTest, GetTensorMapDataTypeReturnsCorrectDataType) {
  EXPECT_THAT(GetTensorMapDataType(1),
              absl_testing::IsOkAndHolds(CU_TENSOR_MAP_DATA_TYPE_UINT8));
  EXPECT_THAT(GetTensorMapDataType(2),
              absl_testing::IsOkAndHolds(CU_TENSOR_MAP_DATA_TYPE_UINT16));
  EXPECT_THAT(GetTensorMapDataType(4),
              absl_testing::IsOkAndHolds(CU_TENSOR_MAP_DATA_TYPE_UINT32));
  EXPECT_THAT(GetTensorMapDataType(8),
              absl_testing::IsOkAndHolds(CU_TENSOR_MAP_DATA_TYPE_UINT64));
}

TEST(TmaUtilTest, GetTensorMapDataTypeFailsGracefully) {
  EXPECT_THAT(GetTensorMapDataType(0),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetTensorMapDataType(16),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(TmaUtilTest, GetTensorMapSwizzleReturnsCorrectSwizzle) {
  EXPECT_EQ(GetTensorMapSwizzle(TmaDescriptor::TmaSwizzle::kNone),
            CU_TENSOR_MAP_SWIZZLE_NONE);
  EXPECT_EQ(GetTensorMapSwizzle(TmaDescriptor::TmaSwizzle::k32B),
            CU_TENSOR_MAP_SWIZZLE_32B);
  EXPECT_EQ(GetTensorMapSwizzle(TmaDescriptor::TmaSwizzle::k64B),
            CU_TENSOR_MAP_SWIZZLE_64B);
  EXPECT_EQ(GetTensorMapSwizzle(TmaDescriptor::TmaSwizzle::k128B),
            CU_TENSOR_MAP_SWIZZLE_128B);
}

TEST(TmaUtilTest, GetTensorMapL2PromotionReturnsCorrectL2Promotion) {
  EXPECT_EQ(GetTensorMapL2Promotion(TmaDescriptor::TmaL2Promotion::kNone),
            CU_TENSOR_MAP_L2_PROMOTION_NONE);
  EXPECT_EQ(GetTensorMapL2Promotion(TmaDescriptor::TmaL2Promotion::k64B),
            CU_TENSOR_MAP_L2_PROMOTION_L2_64B);
  EXPECT_EQ(GetTensorMapL2Promotion(TmaDescriptor::TmaL2Promotion::k128B),
            CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  EXPECT_EQ(GetTensorMapL2Promotion(TmaDescriptor::TmaL2Promotion::k256B),
            CU_TENSOR_MAP_L2_PROMOTION_L2_256B);
}

TEST(TmaUtilTest, GetTensorMapFloatOobFillReturnsCorrectFloatOobFill) {
  EXPECT_EQ(GetTensorMapFloatOOBFill(TmaDescriptor::TmaFloatOobFill::kNone),
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  EXPECT_EQ(GetTensorMapFloatOOBFill(
                TmaDescriptor::TmaFloatOobFill::kNanRequestZeroFma),
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA);
}

TEST(TmaUtilTest, GetTensorMapInterleaveReturnsCorrectInterleave) {
  EXPECT_EQ(GetTensorMapInterleave(TmaDescriptor::TmaInterleave::kNone),
            CU_TENSOR_MAP_INTERLEAVE_NONE);
  EXPECT_EQ(GetTensorMapInterleave(TmaDescriptor::TmaInterleave::k16B),
            CU_TENSOR_MAP_INTERLEAVE_16B);
  EXPECT_EQ(GetTensorMapInterleave(TmaDescriptor::TmaInterleave::k32B),
            CU_TENSOR_MAP_INTERLEAVE_32B);
}

}  // namespace
}  // namespace stream_executor::gpu
