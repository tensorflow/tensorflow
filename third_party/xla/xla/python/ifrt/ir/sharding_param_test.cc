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

#include <limits>
#include <vector>

#include <gtest/gtest.h>
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

TEST(ShardingParamTest, DimShardsProductOverflowIsRejected) {
  ShardingParam::MinorToMajor minor_to_major;
  minor_to_major.permutation = {0};
  minor_to_major.axis_sizes = {4};

  ShardingParam sharding(/*dim_shards=*/{2, 1500000000}, minor_to_major);

  EXPECT_THAT(sharding.verify(),
              StatusIs(tsl::error::INVALID_ARGUMENT,
                       HasSubstr("`dim_shards` product overflows")));
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
