/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/tma_metadata.h"

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "llvm/ADT/ArrayRef.h"

namespace xla::gpu {
namespace {

using absl::StatusCode;

TEST(TmaMetadataTest, CreateValidTmaInfo) {
  EXPECT_TRUE(TmaDescriptor::Create(/*global_dims=*/{500, 360},
                                    /*global_strides=*/{1000, 360000},
                                    /*box_dims=*/{128, 128},
                                    /*element_strides=*/{1, 1},
                                    /*element_byte_width=*/1)
                  .ok());
  EXPECT_TRUE(TmaDescriptor::Create({500}, {1000}, {128}, {1}, 2).ok());
  EXPECT_TRUE(
      TmaDescriptor::Create({100, 280}, {400, 112000}, {64, 64}, {1, 1}, 4)
          .ok());
}

TEST(TmaMetadataTest, CreateInvalidTensorRank) {
  EXPECT_EQ(TmaDescriptor::Create({}, {}, {}, {}, 2).status().code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(
      TmaDescriptor::Create({128, 128, 128, 128, 128, 128},
                            {1000, 1000, 1000, 1000, 1000, 1000},
                            {16, 16, 16, 16, 16, 16}, {1, 1, 1, 1, 1, 1}, 2)
          .status()
          .code(),
      StatusCode::kInvalidArgument);
}

TEST(TmaMetadataTest, CreateMismatchedTensorRanks) {
  EXPECT_EQ(
      TmaDescriptor::Create({128, 128, 128}, {500, 360}, {16, 16}, {1, 1}, 2)
          .status()
          .code(),
      StatusCode::kInvalidArgument);
  EXPECT_EQ(
      TmaDescriptor::Create({128, 128}, {360}, {16}, {1}, 2).status().code(),
      StatusCode::kInvalidArgument);
  EXPECT_EQ(
      TmaDescriptor::Create({128}, {360}, {16, 64}, {1}, 2).status().code(),
      StatusCode::kInvalidArgument);
  EXPECT_EQ(
      TmaDescriptor::Create({128}, {360}, {16}, {1, 1}, 2).status().code(),
      StatusCode::kInvalidArgument);
}

TEST(TmaMetadataTest, CreateInvalidBoxDims) {
  EXPECT_EQ(TmaDescriptor::Create({500, 360}, {1000, 360000},
                                  /*box_dims=*/{512, 128}, {1, 1}, 1)
                .status()
                .code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(TmaDescriptor::Create({500, 360}, {1000, 360000},
                                  /*box_dims=*/{128, 512}, {1, 1}, 1)
                .status()
                .code(),
            StatusCode::kInvalidArgument);
}

TEST(TmaMetadataTest, CreateInvalidElementByteWidth) {
  EXPECT_EQ(TmaDescriptor::Create({500, 360}, {1000, 360000}, {128, 128},
                                  {1, 1}, /*element_byte_width=*/3)
                .status()
                .code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(TmaDescriptor::Create({500, 360}, {1000, 360000}, {128, 128},
                                  {1, 1}, /*element_byte_width=*/0)
                .status()
                .code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(TmaDescriptor::Create({500, 360}, {1000, 360000}, {128, 128},
                                  {1, 1}, /*element_byte_width=*/5)
                .status()
                .code(),
            StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace xla::gpu
