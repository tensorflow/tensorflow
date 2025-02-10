/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/hlo_module_config.h"

#include <string>

#include "xla/tests/test_utils.h"
#include "xla/xla.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

TEST(HloModuleConfigTest, ShardableValueUpdatePairProtoRoundTrip) {
  const std::string text_proto = R"(
  shardable_value_update_pairs {
    input_parameter_number: 2
    parameter_shape_index: 0
    parameter_shape_index: 1
    output_shape_index: 1
    output_shape_index: 0
  }
  shardable_value_update_pairs {
    input_parameter_number: 1
    parameter_shape_index: 2
    output_shape_index: 3
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto input_proto,
                          ParseTextProto<HloModuleConfigProto>(text_proto));
  HloModuleConfig config;
  HloModuleConfig::AssignStructShardableValueUpdatePairs(
      config, input_proto.shardable_value_update_pairs());
  EXPECT_EQ(config.shardable_value_update_pairs().size(), 2);

  HloModuleConfigProto output_proto;
  HloModuleConfig::AssignProtoShardableValueUpdatePairs(
      output_proto.mutable_shardable_value_update_pairs(),
      config.shardable_value_update_pairs());
  EXPECT_EQ(input_proto.SerializeAsString(), output_proto.SerializeAsString());
}

}  // namespace
}  // namespace xla
