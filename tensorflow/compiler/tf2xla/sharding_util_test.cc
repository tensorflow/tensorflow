/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/tf2xla/sharding_util.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(CoreUtilTest, ParseShardingFromDevice) {
  Graph graph(OpRegistry::Global());

  auto core_from_sharding =
      [](tensorflow::gtl::optional<xla::OpSharding> sharding) -> int64 {
    if (sharding.has_value() &&
        sharding.value().type() ==
            xla::OpSharding::Type::OpSharding_Type_MAXIMAL) {
      return sharding.value().tile_assignment_devices(0);
    } else {
      return -1;
    }
  };

  auto parse_status = ParseShardingFromDevice("", 1);
  TF_EXPECT_OK(parse_status.status());
  EXPECT_EQ(-1, core_from_sharding(parse_status.ValueOrDie()));
  parse_status = ParseShardingFromDevice("", 100);
  TF_EXPECT_OK(parse_status.status());
  EXPECT_EQ(-1, core_from_sharding(parse_status.ValueOrDie()));

  parse_status = ParseShardingFromDevice("/device:A_REPLICATED_CORE:-1", 100);
  EXPECT_FALSE(parse_status.ok());

  parse_status = ParseShardingFromDevice("/device:A_REPLICATED_CORE:55", 100);
  TF_EXPECT_OK(parse_status.status());
  EXPECT_EQ(55, core_from_sharding(parse_status.ValueOrDie()));

  parse_status = ParseShardingFromDevice("/device:A_REPLICATED_CORE:100", 100);
  EXPECT_FALSE(parse_status.ok());

  parse_status = ParseShardingFromDevice("/cpu:0", 100);
  TF_EXPECT_OK(parse_status.status());
  EXPECT_EQ(-1, core_from_sharding(parse_status.ValueOrDie()));
}

}  // namespace tensorflow
