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

#include "xla/service/sharding_config.h"

#include <functional>

#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {

ShardingConfigProto ShardingConfig::ToProto(const ShardingConfig& config) {
  ShardingConfigProto sharding_config_proto;
  std::function<NodeShardingConfigProto(const NodeShardingConfig&)> convert;
  convert =
      [&convert](
          const NodeShardingConfig& node_config) -> NodeShardingConfigProto {
    NodeShardingConfigProto node_config_proto;
    if (node_config.sharding.has_value()) {
      *node_config_proto.mutable_sharding() = node_config.sharding->ToProto();
    }
    for (const NodeShardingConfig& node : node_config.nodes) {
      *node_config_proto.add_nodes() = convert(node);
    }
    return node_config_proto;
  };
  for (const NodeShardingConfig& node_config : config.nodes) {
    *sharding_config_proto.add_nodes() = convert(node_config);
  }
  return sharding_config_proto;
}

ShardingConfig ShardingConfig::FromProto(const ShardingConfigProto& proto) {
  ShardingConfig config;
  std::function<NodeShardingConfig(const NodeShardingConfigProto&)> convert;
  convert = [&convert](const NodeShardingConfigProto& node_config_proto)
      -> NodeShardingConfig {
    NodeShardingConfig node_config;
    if (node_config_proto.has_sharding()) {
      auto hlo_sharding = HloSharding::FromProto(node_config_proto.sharding());
      if (hlo_sharding.ok()) {
        node_config.sharding = *hlo_sharding;
      }
    }
    for (const NodeShardingConfigProto& node : node_config_proto.nodes()) {
      node_config.nodes.push_back(convert(node));
    }
    return node_config;
  };
  for (const NodeShardingConfigProto& node_config_proto : proto.nodes()) {
    config.nodes.push_back(convert(node_config_proto));
  }
  return config;
}

}  // namespace xla
