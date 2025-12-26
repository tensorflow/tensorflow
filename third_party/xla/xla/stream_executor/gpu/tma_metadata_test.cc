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

#include "xla/stream_executor/gpu/tma_metadata.h"

#include <cmath>
#include <cstdint>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "xla/stream_executor/gpu/tma_metadata.pb.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/protobuf.h"

namespace stream_executor::gpu {
namespace {

using absl::StatusCode;
using testing::HasSubstr;
using tsl::proto_testing::EqualsProto;

TEST(TmaMetadataTest, CreateValidTmaInfoReturnsOk) {
  EXPECT_THAT(TmaDescriptor::Create(/*global_dims=*/{500, 360},
                                    /*global_strides=*/{1600},
                                    /*box_dims=*/{128, 128},
                                    /*element_strides=*/{1, 1},
                                    /*element_byte_width=*/1),
              absl_testing::IsOk());
  EXPECT_THAT(TmaDescriptor::Create(/*global_dims=*/{500},
                                    /*global_strides=*/{},
                                    /*box_dims=*/{128},
                                    /*element_strides=*/{1},
                                    /*element_byte_width=*/2),
              absl_testing::IsOk());
  EXPECT_THAT(TmaDescriptor::Create(/*global_dims=*/{100, 280},
                                    /*global_strides=*/{400},
                                    /*box_dims=*/{64, 64},
                                    /*element_strides=*/{1, 1},
                                    /*element_byte_width=*/4),
              absl_testing::IsOk());
  constexpr uint64_t kValid32BSwizzleBoxDim = 32;
  EXPECT_THAT(
      TmaDescriptor::Create(/*global_dims=*/{500, 360},
                            /*global_strides=*/{1600},
                            /*box_dims=*/{kValid32BSwizzleBoxDim, 128},
                            /*element_strides=*/{1, 1},
                            /*element_byte_width=*/1,
                            /*interleave=*/TmaDescriptor::TmaInterleave::kNone,
                            /*swizzle=*/TmaDescriptor::TmaSwizzle::k32B),
      absl_testing::IsOk());
  constexpr uint64_t kValid64BSwizzleBoxDim = 64;
  EXPECT_THAT(
      TmaDescriptor::Create(/*global_dims=*/{500, 360},
                            /*global_strides=*/{1600},
                            /*box_dims=*/{kValid64BSwizzleBoxDim, 128},
                            /*element_strides=*/{1, 1},
                            /*element_byte_width=*/1,
                            /*interleave=*/TmaDescriptor::TmaInterleave::kNone,
                            /*swizzle=*/TmaDescriptor::TmaSwizzle::k64B),
      absl_testing::IsOk());
  constexpr uint64_t kValid128BSwizzleBoxDim = 128;
  EXPECT_THAT(
      TmaDescriptor::Create(/*global_dims=*/{500, 360},
                            /*global_strides=*/{1600},
                            /*box_dims=*/{kValid128BSwizzleBoxDim, 128},
                            /*element_strides=*/{1, 1},
                            /*element_byte_width=*/1,
                            /*interleave=*/TmaDescriptor::TmaInterleave::kNone,
                            /*swizzle=*/TmaDescriptor::TmaSwizzle::k128B),
      absl_testing::IsOk());
}

TEST(TmaMetadataTest, CreateInvalidTensorRankFailsGracefully) {
  EXPECT_THAT(TmaDescriptor::Create(/*global_dims=*/{}, /*global_strides=*/{},
                                    /*box_dims=*/{}, /*element_strides=*/{},
                                    /*element_byte_width=*/2),
              absl_testing::StatusIs(StatusCode::kInvalidArgument,
                                     HasSubstr("unsupported rank for TMA")));
  EXPECT_THAT(
      TmaDescriptor::Create(
          /*global_dims=*/{128, 128, 128, 128, 128, 128},
          /*global_strides=*/{1000, 1000, 1000, 1000, 1000},
          /*box_dims=*/{16, 16, 16, 16, 16, 16},
          /*element_strides=*/{1, 1, 1, 1, 1, 1}, /*element_byte_width=*/2),
      absl_testing::StatusIs(StatusCode::kInvalidArgument,
                             HasSubstr("unsupported rank for TMA")));
  EXPECT_THAT(
      TmaDescriptor::Create(/*global_dims=*/{500, 360},
                            /*global_strides=*/{1600},
                            /*box_dims=*/{128, 128},
                            /*element_strides=*/{1, 1},
                            /*element_byte_width=*/1,
                            /*interleave=*/TmaDescriptor::TmaInterleave::k16B),
      absl_testing::StatusIs(
          StatusCode::kFailedPrecondition,
          HasSubstr("If TmaInterleave is not kNone, then tensor rank must "
                    "additionally be >= 3")));
}

TEST(TmaMetadataTest, CreateMismatchedTensorRanksFailsGracefully) {
  auto kExpectedError = absl_testing::StatusIs(
      StatusCode::kFailedPrecondition, HasSubstr("must have the same rank"));
  EXPECT_THAT(TmaDescriptor::Create(/*global_dims=*/{128, 128, 128},
                                    /*global_strides=*/{500},
                                    /*box_dims=*/{16, 16},
                                    /*element_strides=*/{1, 1},
                                    /*element_byte_width=*/2),
              kExpectedError);
  EXPECT_THAT(
      TmaDescriptor::Create(/*global_dims=*/{128, 128},
                            /*global_strides=*/{},
                            /*box_dims=*/{16},
                            /*element_strides=*/{1}, /*element_byte_width=*/2),
      kExpectedError);
  EXPECT_THAT(
      TmaDescriptor::Create(/*global_dims=*/{128},
                            /*global_strides=*/{},
                            /*box_dims=*/{16, 64},
                            /*element_strides=*/{1}, /*element_byte_width=*/2),
      kExpectedError);
  EXPECT_THAT(TmaDescriptor::Create(/*global_dims=*/{128},
                                    /*global_strides=*/{},
                                    /*box_dims=*/{16},
                                    /*element_strides=*/{1, 1},
                                    /*element_byte_width=*/2),
              kExpectedError);
  EXPECT_THAT(
      TmaDescriptor::Create(/*global_dims=*/{128, 128},
                            /*global_strides=*/{500, 500},
                            /*box_dims=*/{16, 16},
                            /*element_strides=*/{1, 1},
                            /*element_byte_width=*/2),
      absl_testing::StatusIs(StatusCode::kFailedPrecondition,
                             HasSubstr("global_strides must have a rank of")));
}

TEST(TmaMetadataTest, CreateInvalidElementByteWidthFailsGracefully) {
  auto kExpectedError = absl_testing::StatusIs(
      StatusCode::kInvalidArgument, HasSubstr("unsupported element size"));
  EXPECT_THAT(TmaDescriptor::Create(/*global_dims=*/{500, 360},
                                    /*global_strides=*/{1000},
                                    /*box_dims=*/{128, 128},
                                    /*element_strides=*/{1, 1},
                                    /*element_byte_width=*/3),
              kExpectedError);
  EXPECT_THAT(TmaDescriptor::Create(/*global_dims=*/{500, 360},
                                    /*global_strides=*/{1000},
                                    /*box_dims=*/{128, 128},
                                    /*element_strides=*/{1, 1},
                                    /*element_byte_width=*/0),
              kExpectedError);
  EXPECT_THAT(TmaDescriptor::Create(/*global_dims=*/{500, 360},
                                    /*global_strides=*/{1000},
                                    /*box_dims=*/{128, 128},
                                    /*element_strides=*/{1, 1},
                                    /*element_byte_width=*/5),
              kExpectedError);
}

TEST(TmaMetadataTest, CreateInvalidGlobalDimsFailsGracefully) {
  auto kExpectedError =
      absl_testing::StatusIs(StatusCode::kInvalidArgument,
                             AllOf(HasSubstr("global_dims"),
                                   HasSubstr("must be non-zero and <= 2^32")));
  constexpr uint64_t kZeroGlobalDim = 0;
  EXPECT_THAT(TmaDescriptor::Create(/*global_dims=*/{kZeroGlobalDim, 360},
                                    /*global_strides=*/{1600},
                                    /*box_dims=*/{128, 128},
                                    /*element_strides=*/{1, 1},
                                    /*element_byte_width=*/1),
              kExpectedError);
  const uint64_t kOverMaxGlobalDim = static_cast<uint64_t>(pow(2, 33));
  EXPECT_THAT(TmaDescriptor::Create(
                  /*global_dims=*/{500, kOverMaxGlobalDim},
                  /*global_strides=*/{1600},
                  /*box_dims=*/{128, 128},
                  /*element_strides=*/{1, 1},
                  /*element_byte_width=*/1),
              kExpectedError);
}

TEST(TmaMetadataTest, CreateInvalidGlobalStridesFailsGracefully) {
  constexpr uint64_t kNotDivisibleBy16 = 1000;
  EXPECT_THAT(
      TmaDescriptor::Create(/*global_dims=*/{500, 360},
                            /*global_strides=*/{kNotDivisibleBy16},
                            /*box_dims=*/{128, 128},
                            /*element_strides=*/{1, 1},
                            /*element_byte_width=*/1),
      absl_testing::StatusIs(StatusCode::kInvalidArgument,
                             AllOf(HasSubstr("global_strides"),
                                   HasSubstr("must be a multiple of 16"))));
  const uint64_t kOverMaxGlobalStride = static_cast<uint64_t>(pow(2, 40));
  EXPECT_THAT(TmaDescriptor::Create(/*global_dims=*/{500, 360},
                                    /*global_strides=*/{kOverMaxGlobalStride},
                                    /*box_dims=*/{128, 128},
                                    /*element_strides=*/{1, 1},
                                    /*element_byte_width=*/1),
              absl_testing::StatusIs(
                  StatusCode::kInvalidArgument,
                  AllOf(HasSubstr("global_strides"), HasSubstr("< 2^40"))));
  constexpr uint64_t kNotDivisibleBy32 = 2000;
  EXPECT_THAT(
      TmaDescriptor::Create(/*global_dims=*/{500, 360, 200},
                            /*global_strides=*/
                            {kNotDivisibleBy32, kNotDivisibleBy32 * 500},
                            /*box_dims=*/{128, 128, 128},
                            /*element_strides=*/{1, 1, 1},
                            /*element_byte_width=*/1,
                            TmaDescriptor::TmaInterleave::k32B),
      absl_testing::StatusIs(
          StatusCode::kFailedPrecondition,
          AllOf(HasSubstr("global_strides"),
                HasSubstr("must be a multiple of 32 when interleave is 32B"))));
  constexpr uint64_t kNotDivisibleByStride0 = 6080;
  EXPECT_THAT(
      TmaDescriptor::Create(/*global_dims=*/{500, 360, 200},
                            /*global_strides=*/{1600, kNotDivisibleByStride0},
                            /*box_dims=*/{128, 128, 128},
                            /*element_strides=*/{1, 1, 1},
                            /*element_byte_width=*/1),
      absl_testing::StatusIs(
          StatusCode::kFailedPrecondition,
          AllOf(HasSubstr("global_stride"),
                HasSubstr("must be a multiple of the previous stride"))));
  constexpr uint64_t kSmallerThanGlobalDims = 160;
  EXPECT_THAT(
      TmaDescriptor::Create(/*global_dims=*/{500, 360},
                            /*global_strides=*/{kSmallerThanGlobalDims},
                            /*box_dims=*/{128, 128},
                            /*element_strides=*/{1, 1},
                            /*element_byte_width=*/1),
      absl_testing::StatusIs(StatusCode::kFailedPrecondition,
                             AllOf(HasSubstr("global_stride"),
                                   HasSubstr("must be >= global_dim"))));
}

TEST(TmaMetadataTest, CreateInvalidBoxDimsFailsGracefully) {
  constexpr uint64_t kZeroBoxDim = 0;
  EXPECT_THAT(
      TmaDescriptor::Create(/*global_dims=*/{500, 360},
                            /*global_strides=*/{1600},
                            /*box_dims=*/{kZeroBoxDim, 128},
                            /*element_strides=*/{1, 1},
                            /*element_byte_width=*/1),
      absl_testing::StatusIs(StatusCode::kInvalidArgument,
                             AllOf(HasSubstr("box_dims"),
                                   HasSubstr("must be non-zero and <= 256"))));
  const uint64_t kOverMaxBoxDim = 257;
  EXPECT_THAT(
      TmaDescriptor::Create(
          /*global_dims=*/{500, 360},
          /*global_strides=*/{1600},
          /*box_dims=*/{128, kOverMaxBoxDim},
          /*element_strides=*/{1, 1},
          /*element_byte_width=*/1),
      absl_testing::StatusIs(StatusCode::kInvalidArgument,
                             AllOf(HasSubstr("box_dims"),
                                   HasSubstr("must be non-zero and <= 256"))));
  const uint64_t kNotDivisibleBy16 = 17;
  EXPECT_THAT(TmaDescriptor::Create(
                  /*global_dims=*/{500, 360},
                  /*global_strides=*/{1600},
                  /*box_dims=*/{kNotDivisibleBy16, 128},
                  /*element_strides=*/{1, 1},
                  /*element_byte_width=*/1),
              absl_testing::StatusIs(
                  StatusCode::kFailedPrecondition,
                  AllOf(HasSubstr("when interleave is kNone, box_dims[0]"),
                        HasSubstr("must be a multiple of 16 bytes"))));
}

TEST(TmaMetadataTest, CreateInvalidElementStridesFailsGracefully) {
  auto kOutOfValidRangeError =
      absl_testing::StatusIs(StatusCode::kInvalidArgument,
                             AllOf(HasSubstr("element_strides"),
                                   HasSubstr("must be non-zero and <= 8")));
  auto kNotContiguousError = absl_testing::StatusIs(
      StatusCode::kInvalidArgument,
      HasSubstr("element_strides[0] must be 1 for TMA."));
  constexpr uint64_t kZeroElementStride = 0;
  EXPECT_THAT(TmaDescriptor::Create(/*global_dims=*/{500, 360},
                                    /*global_strides=*/{1600},
                                    /*box_dims=*/{128, 128},
                                    /*element_strides=*/{1, kZeroElementStride},
                                    /*element_byte_width=*/1),
              kOutOfValidRangeError);
  constexpr uint64_t kOverMaxElementStride = 9;
  EXPECT_THAT(
      TmaDescriptor::Create(/*global_dims=*/{500, 360},
                            /*global_strides=*/{1600},
                            /*box_dims=*/{128, 128},
                            /*element_strides=*/{1, kOverMaxElementStride},
                            /*element_byte_width=*/1),
      kOutOfValidRangeError);
  EXPECT_THAT(TmaDescriptor::Create(/*global_dims=*/{500, 360},
                                    /*global_strides=*/{1600},
                                    /*box_dims=*/{128, 128},
                                    /*element_strides=*/{2, 1},
                                    /*element_byte_width=*/1),
              kNotContiguousError);
}

TEST(TmaMetadataTest, CreateInvalidInterleaveSwizzleComboFailsGracefully) {
  constexpr uint64_t kGreaterThan32BSwizzle = 64;
  EXPECT_THAT(
      TmaDescriptor::Create(/*global_dims=*/{500, 360},
                            /*global_strides=*/{1600},
                            /*box_dims=*/{kGreaterThan32BSwizzle, 128},
                            /*element_strides=*/{1, 1},
                            /*element_byte_width=*/1,
                            /*interleave=*/TmaDescriptor::TmaInterleave::kNone,
                            /*swizzle=*/TmaDescriptor::TmaSwizzle::k32B),
      absl_testing::StatusIs(
          StatusCode::kFailedPrecondition,
          HasSubstr("when interleave is kNone and swizzle is k32B, "
                    "box_dims[0] * element_byte_width must be <= "
                    "32.")));
  constexpr uint64_t kGreaterThan64BSwizzle = 128;
  EXPECT_THAT(
      TmaDescriptor::Create(/*global_dims=*/{500, 360},
                            /*global_strides=*/{1600},
                            /*box_dims=*/{kGreaterThan64BSwizzle, 128},
                            /*element_strides=*/{1, 1},
                            /*element_byte_width=*/1,
                            /*interleave=*/TmaDescriptor::TmaInterleave::kNone,
                            /*swizzle=*/TmaDescriptor::TmaSwizzle::k64B),
      absl_testing::StatusIs(
          StatusCode::kFailedPrecondition,
          HasSubstr("when interleave is kNone and swizzle is k64B, "
                    "box_dims[0] * element_byte_width must be <= "
                    "64.")));
  constexpr uint64_t kGreaterThan128BSwizzle = 144;
  EXPECT_THAT(
      TmaDescriptor::Create(/*global_dims=*/{500, 360},
                            /*global_strides=*/{1600},
                            /*box_dims=*/{kGreaterThan128BSwizzle, 128},
                            /*element_strides=*/{1, 1},
                            /*element_byte_width=*/1,
                            /*interleave=*/TmaDescriptor::TmaInterleave::kNone,
                            /*swizzle=*/TmaDescriptor::TmaSwizzle::k128B),
      absl_testing::StatusIs(
          StatusCode::kFailedPrecondition,
          HasSubstr("when interleave is kNone and swizzle is k128B, "
                    "box_dims[0] * element_byte_width must be <= "
                    "128.")));
  const TmaDescriptor::TmaSwizzle kNot32BSwizzle =
      TmaDescriptor::TmaSwizzle::k128B;
  EXPECT_THAT(TmaDescriptor::Create(
                  /*global_dims=*/{500, 360, 200},
                  /*global_strides=*/{32 * 500, 32 * 500 * 360},
                  /*box_dims=*/{128, 128, 128},
                  /*element_strides=*/{1, 1, 1},
                  /*element_byte_width=*/1,
                  /*interleave=*/TmaDescriptor::TmaInterleave::k32B,
                  /*swizzle=*/kNot32BSwizzle),
              absl_testing::StatusIs(
                  StatusCode::kFailedPrecondition,
                  HasSubstr("when interleave is k32B, swizzle must be k32B.")));
}

TEST(TmaMetadataTest, ProtoSerializationDeserialization) {
  TmaDescriptorProto descriptor_proto;
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      R"pb(
        element_size: 4
        global_dims: 100
        global_dims: 100
        global_dims: 100
        global_strides: 400
        global_strides: 40000
        box_dims: 16
        box_dims: 16
        box_dims: 16
        element_strides: 1
        element_strides: 1
        element_strides: 1
        interleave: INTERLEAVE_BYTES16
        swizzle: SWIZZLE_BYTES32
        l2_promotion: L2_PROMOTION_BYTES64
        float_oob_fill: FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
      )pb",
      &descriptor_proto));

  TmaMetadataProto metadata_proto;
  metadata_proto.mutable_arg_index_to_tma_info()->emplace(
      /*arg_index=*/0, std::move(descriptor_proto));

  TF_ASSERT_OK_AND_ASSIGN(TmaMetadata metadata,
                          TmaMetadata::FromProto(metadata_proto));
  EXPECT_THAT(metadata.ToProto(), EqualsProto(metadata_proto));
}

}  // namespace
}  // namespace stream_executor::gpu
