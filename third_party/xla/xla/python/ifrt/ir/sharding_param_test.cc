/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/sharding_param.h"

#include <cstdint>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/tsl/platform/errors.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(ShardingParamTest, AxisSizesProductOverflowIsRejected) {
  // Two axis sizes whose product exceeds std::numeric_limits<int>::max()
  // (2 * 1,500,000,000 > INT_MAX), individually representable as plain ints.
  ShardingParam::MinorToMajor minor_to_major;
  minor_to_major.permutation = {0, 1};
  minor_to_major.axis_sizes = {2, 1500000000};

  ShardingParam sharding(/*dim_shards=*/{1, 1}, minor_to_major);

  EXPECT_THAT(sharding.verify(),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("`axis_sizes` product overflows")));
}

TEST(ShardingParamTest, NonPositiveAxisSizesAreRejected) {
  ShardingParam::MinorToMajor zero_axis;
  zero_axis.permutation = {0};
  zero_axis.axis_sizes = {0};
  ShardingParam zero_sharding(/*dim_shards=*/{1}, zero_axis);
  EXPECT_THAT(zero_sharding.verify(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`axis_sizes` must be positive")));

  ShardingParam::MinorToMajor negative_axes;
  negative_axes.permutation = {0, 1};
  negative_axes.axis_sizes = {-1, -2};
  ShardingParam negative_sharding(/*dim_shards=*/{1, 1}, negative_axes);
  EXPECT_THAT(negative_sharding.verify(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`axis_sizes` must be positive")));
}

TEST(ShardingParamTest, DimShardsProductOverflowIsRejected) {
  ShardingParam::MinorToMajor minor_to_major;
  minor_to_major.permutation = {0};
  minor_to_major.axis_sizes = {4};

  ShardingParam sharding(/*dim_shards=*/{2, 1500000000}, minor_to_major);

  EXPECT_THAT(sharding.verify(),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("`dim_shards` product overflows")));
}

TEST(ShardingParamTest, DimShardsExceedingInt32MaxIsRejected) {
  ShardingParam::MinorToMajor minor_to_major;
  minor_to_major.permutation = {0};
  minor_to_major.axis_sizes = {4};

  // `(1LL << 32) + 1` would wrap to `1` if narrowed to 32-bit `int`.
  ShardingParam sharding(/*dim_shards=*/{(1LL << 32) + 1}, minor_to_major);

  EXPECT_THAT(sharding.verify(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`dim_shards` product overflows")));
}

TEST(ShardingParamTest, GlobalShapeFromLocalShapeOverflowIsRejected) {
  ShardingParam::MinorToMajor minor_to_major;
  minor_to_major.permutation = {0};
  minor_to_major.axis_sizes = {2};
  ShardingParam sharding(/*dim_shards=*/{2}, minor_to_major);

  EXPECT_THAT(
      sharding.GlobalShapeFromLocalShape({std::numeric_limits<int64_t>::max()}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("overflows 64-bit signed integer")));
}

TEST(ShardingParamTest, ShapeConversionRankMismatchIsRejected) {
  ShardingParam::MinorToMajor minor_to_major;
  minor_to_major.permutation = {0, 1};
  minor_to_major.axis_sizes = {2, 2};
  ShardingParam sharding(/*dim_shards=*/{2, 2}, minor_to_major);

  EXPECT_THAT(sharding.LocalShapeFromGlobalShape({4}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Rank of global tensor differs from rank of "
                                 "`dim_shards`")));
  EXPECT_THAT(sharding.GlobalShapeFromLocalShape({4}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Rank of local tensor differs from rank of "
                                 "`dim_shards`")));
}

TEST(ShardingParamTest,
     LocalShapeFromGlobalShapeInvalidAndIndivisibleShardsAreRejected) {
  ShardingParam::MinorToMajor minor_to_major;
  minor_to_major.permutation = {0, 1};
  minor_to_major.axis_sizes = {2, 2};

  ShardingParam zero_dim_shard(/*dim_shards=*/{0, 2}, minor_to_major);
  EXPECT_THAT(zero_dim_shard.LocalShapeFromGlobalShape({4, 4}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`dim_shards` must be positive, got 0")));

  ShardingParam negative_dim_shard(/*dim_shards=*/{-1, 2}, minor_to_major);
  EXPECT_THAT(negative_dim_shard.LocalShapeFromGlobalShape({4, 4}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`dim_shards` must be positive, got -1")));

  ShardingParam valid_sharding(/*dim_shards=*/{2, 2}, minor_to_major);
  EXPECT_THAT(valid_sharding.LocalShapeFromGlobalShape({-4, 4}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Global shape dimensions must be "
                                 "non-negative, got -4 in dimension 0")));

  EXPECT_THAT(
      valid_sharding.LocalShapeFromGlobalShape({5, 4}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Global shape is not divisible by the number of "
                         "shards")));
}

TEST(ShardingParamTest,
     GlobalShapeFromLocalShapeInvalidShardsAndNegativeDimsAreRejected) {
  ShardingParam::MinorToMajor minor_to_major;
  minor_to_major.permutation = {0, 1};
  minor_to_major.axis_sizes = {2, 2};

  ShardingParam zero_dim_shard(/*dim_shards=*/{0, 2}, minor_to_major);
  EXPECT_THAT(zero_dim_shard.GlobalShapeFromLocalShape({4, 4}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`dim_shards` must be positive, got 0")));

  ShardingParam negative_dim_shard(/*dim_shards=*/{-2, 2}, minor_to_major);
  EXPECT_THAT(negative_dim_shard.GlobalShapeFromLocalShape({4, 4}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("`dim_shards` must be positive, got -2")));

  ShardingParam valid_sharding(/*dim_shards=*/{2, 2}, minor_to_major);
  EXPECT_THAT(valid_sharding.GlobalShapeFromLocalShape({4, -3}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Local shape dimensions must be "
                                 "non-negative, got -3 in dimension 1")));
}

TEST(ShardingParamTest, NonPositiveDimShardIsRejected) {
  ShardingParam::MinorToMajor minor_to_major;
  minor_to_major.permutation = {0};
  minor_to_major.axis_sizes = {4};

  ShardingParam sharding(/*dim_shards=*/{0}, minor_to_major);

  EXPECT_THAT(sharding.verify(),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("`dim_shards` must be positive")));
}

TEST(ShardingParamTest, ValidShardingIsAccepted) {
  ShardingParam::MinorToMajor minor_to_major;
  minor_to_major.permutation = {0, 1};
  minor_to_major.axis_sizes = {2, 4};

  ShardingParam sharding(/*dim_shards=*/{2, 4}, minor_to_major);

  EXPECT_OK(sharding.verify());
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
