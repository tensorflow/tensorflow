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

#include "xla/xla.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::EqualsProto;

template <typename MessageType>
StatusOr<MessageType> ParseTextProto(const std::string& text_proto) {
  tsl::protobuf::TextFormat::Parser parser;
  MessageType parsed_proto;
  tsl::protobuf::io::ArrayInputStream input_stream(text_proto.data(),
                                                   text_proto.size());
  if (!parser.Parse(&input_stream, &parsed_proto)) {
    return tsl::errors::InvalidArgument("Could not parse text proto: ",
                                        text_proto);
  }
  return parsed_proto;
}

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
  EXPECT_THAT(input_proto, EqualsProto(output_proto));
}

}  // namespace
}  // namespace xla
