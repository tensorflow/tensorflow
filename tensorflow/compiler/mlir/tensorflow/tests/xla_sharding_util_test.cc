/* Copyright 2023 Google Inc. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/utils/xla_sharding_util.h"

#include <string>

#include <gtest/gtest.h>
#include "llvm/ADT/StringRef.h"
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "xla/xla_data.pb.h"

inline constexpr llvm::StringRef kXlaShardingAttrName = "_XlaSharding";

namespace tensorflow {
namespace {

TEST(DecodeShardingAttributeTest, CheckInvalidString) {
  xla::OpSharding sharding;
  EXPECT_TRUE(DecodeShardingAttribute("", sharding).succeeded());
  EXPECT_TRUE(DecodeShardingAttribute("manual", sharding).failed());
}

TEST(DecodeShardingAttributeTest, CheckManualShardString) {
  xla::OpSharding sharding;
  EXPECT_TRUE(DecodeShardingAttribute("{manual}", sharding).succeeded());
  EXPECT_TRUE(sharding.type() == sharding.MANUAL);
  EXPECT_EQ(0, sharding.tile_assignment_devices_size());
}

TEST(DecodeShardingAttributeTest, CheckMaximalShardString) {
  xla::OpSharding sharding;
  EXPECT_TRUE(
      DecodeShardingAttribute("{maximal device=0}", sharding).succeeded());
  EXPECT_TRUE(sharding.type() == sharding.MAXIMAL);
  EXPECT_EQ(1, sharding.tile_assignment_devices_size());
}
}  // namespace

}  // namespace tensorflow
