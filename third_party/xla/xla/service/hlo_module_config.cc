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

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xla/service/computation_layout.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape_layout.h"
#include "xla/xla.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {

using absl::StrAppend;

HloModuleConfig::HloModuleConfig(const ProgramShape& program_shape,
                                 bool ignore_layouts)
    : entry_computation_layout_(
          ComputationLayout(program_shape, ignore_layouts)) {}

HloModuleConfig::HloModuleConfig(ComputationLayout entry_computation_layout)
    : entry_computation_layout_(std::move(entry_computation_layout)) {}

void HloModuleConfig::SetDefaultComputationLayout(
    const ProgramShape& program_shape) {
  entry_computation_layout_ = ComputationLayout(program_shape);
}

void HloModuleConfig::SetComputationLayoutIfExists(
    const ProgramShape& program_shape) {
  entry_computation_layout_ = ComputationLayout(program_shape,
                                                /*ignore_layouts=*/false);
}

std::string HloModuleConfig::compilation_cache_key() const {
  std::string key = absl::StrCat("profiling=", hlo_profiling_enabled());
  StrAppend(&key, "::(");
  std::vector<std::string> params;
  if (entry_computation_layout_.has_value()) {
    for (const ShapeLayout& param_layout :
         entry_computation_layout_->parameter_layouts()) {
      params.push_back(param_layout.shape().DebugString());
    }
    StrAppend(&key, absl::StrJoin(params, ", "), ") => ",
              entry_computation_layout_->result_shape().SerializeAsString());
  }
  if (seed() != 0) {
    // TODO(b/32083678): force recompilation to reset global state.
    static std::atomic<int> counter{0};
    StrAppend(&key, "forcing recompile ", counter++);
  }
  if (replica_count() != 1) {
    StrAppend(&key, "::replica_count=", replica_count());
  }
  StrAppend(&key, debug_options_.DebugString());
  if (intra_op_parallelism_threads() > 0) {
    StrAppend(&key, "::intra_op_parallelism_threads=",
              intra_op_parallelism_threads());
  }
  if (!device_type().empty()) {
    StrAppend(&key, device_type());
  }
  StrAppend(&key, "::alias_passthrough_params=", alias_passthrough_params_);
  StrAppend(&key, "::allow_spmd_sharding_propagation_to_parameters={",
            absl::StrJoin(allow_spmd_sharding_propagation_to_parameters_, ","),
            "}");
  StrAppend(&key, "::allow_spmd_sharding_propagation_to_output={",
            absl::StrJoin(allow_spmd_sharding_propagation_to_output_, ","),
            "}");
  if (!fdo_profile().empty()) {
    StrAppend(&key, "::fdo_profile=", absl::BytesToHexString(fdo_profile()));
  }
  if (device_memory_size() != 0) {
    StrAppend(&key, "::device_memory_size=", device_memory_size());
  }
  return key;
}

/*static*/ void HloModuleConfig::AssignProtoShardableValueUpdatePairs(
    tsl::protobuf::RepeatedPtrField<ShardableValueUpdatePairProto>*
        proto_update_pairs,
    const std::vector<HloModuleConfig::ShardableValueUpdatePair>&
        update_pairs) {
  using ProtoShard = std::decay_t<decltype(proto_update_pairs->at(0))>;
  proto_update_pairs->Reserve(update_pairs.size());

  for (const auto& pair : update_pairs) {
    ProtoShard shard;
    shard.set_input_parameter_number(pair.input_parameter_number);
    for (int64_t val : pair.parameter_shape_index) {
      shard.add_parameter_shape_index(val);
    }
    for (int64_t val : pair.output_shape_index) {
      shard.add_output_shape_index(val);
    }
    proto_update_pairs->Add(std::move(shard));
  }
}

static HloModuleConfigProto::BoolList BoolVectorToProto(
    const std::vector<bool>& vals) {
  HloModuleConfigProto::BoolList list;
  for (int i = 0; i < vals.size(); ++i) {
    list.add_vals(vals[i]);
  }
  return list;
}

static void AssignProtoFusionConfig(
    HloModuleConfigProto& proto,
    const std::vector<std::vector<bool>>& fusion_config) {
  auto* proto_config = proto.mutable_fusion_config();
  proto_config->Reserve(fusion_config.size());
  for (const auto& vals : fusion_config) {
    proto_config->Add(BoolVectorToProto(vals));
  }
}

static void AssignProtoDotConfig(
    HloModuleConfigProto& proto,
    const absl::flat_hash_map<std::string, std::vector<int64_t>>& dot_config) {
  std::map<std::string, std::vector<int64_t>> sorted_dot_config;
  sorted_dot_config.insert(dot_config.begin(), dot_config.end());
  for (const auto& [key, list_vector] : sorted_dot_config) {
    HloModuleConfigProto::Int64List list;
    for (int64_t val : list_vector) {
      list.add_vals(val);
    }
    proto.mutable_dot_config()->try_emplace(key, std::move(list));
  }
}

static void AssignProtoLayoutConfig(
    HloModuleConfigProto& proto,
    const std::vector<std::vector<std::vector<int64_t>>>& layout_config) {
  auto* proto_layout_config = proto.mutable_layout_config();
  proto_layout_config->Reserve(layout_config.size());
  for (const auto& config_row : layout_config) {
    HloModuleConfigProto::Int64ListList proto_list_list;
    proto_list_list.mutable_lists()->Reserve(config_row.size());
    for (const auto& cell : config_row) {
      HloModuleConfigProto::Int64List list;
      for (int64_t val : cell) {
        list.add_vals(val);
      }
      *proto_list_list.add_lists() = std::move(list);
    }
    proto_layout_config->Add(std::move(proto_list_list));
  }
}

static void AssignProtoPhaseOrderingConfig(
    HloModuleConfigProto& proto,
    const std::vector<std::vector<bool>>& phase_config) {
  auto* proto_config = proto.mutable_phase_ordering_config();
  proto_config->Reserve(phase_config.size());
  for (const auto& vals : phase_config) {
    proto_config->Add(BoolVectorToProto(vals));
  }
}

/*static*/ void HloModuleConfig::AssignStructShardableValueUpdatePairs(
    HloModuleConfig& config,
    const tsl::protobuf::RepeatedPtrField<ShardableValueUpdatePairProto>&
        pairs) {
  std::vector<HloModuleConfig::ShardableValueUpdatePair> cfg_pairs;
  cfg_pairs.reserve(pairs.size());
  for (const auto& proto_pair : pairs) {
    HloModuleConfig::ShardableValueUpdatePair pair;
    pair.input_parameter_number = proto_pair.input_parameter_number();
    const auto param_idx = proto_pair.parameter_shape_index();
    pair.parameter_shape_index.assign(param_idx.begin(), param_idx.end());
    const auto output_idx = proto_pair.output_shape_index();
    pair.output_shape_index.assign(output_idx.begin(), output_idx.end());
    cfg_pairs.push_back(pair);
  }
  config.set_shardable_value_update_pairs(std::move(cfg_pairs));
}

static void AssignStructFusionConfig(HloModuleConfig& config,
                                     const HloModuleConfigProto& proto) {
  std::vector<std::vector<bool>> module_config;
  auto& proto_config = proto.fusion_config();
  module_config.reserve(proto_config.size());
  for (auto& list : proto_config) {
    std::vector<bool> temp;
    for (bool val : list.vals()) {
      temp.push_back(val);
    }
    module_config.push_back(std::move(temp));
  }
  *config.mutable_fusion_config() = std::move(module_config);
}

static void AssignStructDotConfig(HloModuleConfig& config,
                                  const HloModuleConfigProto& proto) {
  auto& proto_config = proto.dot_config();
  for (auto& [key, int_list] : proto_config) {
    std::vector<int64_t> value{int_list.vals().begin(), int_list.vals().end()};
    config.mutable_dot_config()->insert(std::pair{key, value});
  }
}

static void AssignStructLayoutConfig(HloModuleConfig& config,
                                     const HloModuleConfigProto& proto) {
  std::vector<std::vector<std::vector<int64_t>>> module_config;
  auto proto_config = proto.layout_config();
  module_config.reserve(proto_config.size());
  for (const auto& proto_row_wrapper : proto_config) {
    const auto& proto_row = proto_row_wrapper.lists();
    std::vector<std::vector<int64_t>> module_row;
    module_row.reserve(proto_row.size());
    for (const auto& proto_cell : proto_row) {
      const auto& cell = proto_cell.vals();
      module_row.push_back(std::vector<int64_t>(cell.begin(), cell.end()));
    }
    module_config.push_back(std::move(module_row));
  }
  *config.mutable_layout_config() = std::move(module_config);
}

static void AssignStructPhaseOrderingConfig(HloModuleConfig& config,
                                            const HloModuleConfigProto& proto) {
  std::vector<std::vector<bool>> module_config;
  auto& proto_config = proto.phase_ordering_config();
  module_config.reserve(proto_config.size());
  for (auto& list : proto_config) {
    std::vector<bool> temp;
    for (bool val : list.vals()) {
      temp.push_back(val);
    }
    module_config.push_back(std::move(temp));
  }
  *config.mutable_phase_ordering_config() = std::move(module_config);
}

absl::StatusOr<HloModuleConfigProto> HloModuleConfig::ToProto() const {
  HloModuleConfigProto proto;
  if (has_entry_computation_layout()) {
    *proto.mutable_entry_computation_layout() =
        entry_computation_layout().ComputeProgramShape().ToProto();
  }
  proto.set_seed(seed_);
  proto.set_launch_id(launch_id_);
  proto.set_replica_count(replica_count_);
  proto.set_num_partitions(num_partitions_);
  for (bool requirement : param_requires_broadcast_via_collectives_) {
    proto.add_param_requires_broadcast_via_collectives(requirement);
  }
  proto.set_use_spmd_partitioning(use_spmd_partitioning_);
  proto.set_use_auto_spmd_partitioning(use_auto_spmd_partitioning_);
  for (int64_t partitioning_shape : auto_spmd_partitioning_mesh_shape_) {
    proto.add_auto_spmd_partitioning_mesh_shape(partitioning_shape);
  }
  for (int64_t partitioning_id : auto_spmd_partitioning_mesh_ids_) {
    proto.add_auto_spmd_partitioning_mesh_ids(partitioning_id);
  }
  proto.set_deduplicate_hlo(deduplicate_hlo_);
  proto.set_intra_op_parallelism_threads(intra_op_parallelism_threads_);
  proto.set_device_type(device_type_);
  *proto.mutable_debug_options() = debug_options_;

  if (has_static_device_assignment()) {
    auto proto_assignment = proto.mutable_static_device_assignment();
    TF_RETURN_IF_ERROR(static_device_assignment_->Serialize(proto_assignment));
  }
  AssignProtoShardableValueUpdatePairs(
      proto.mutable_shardable_value_update_pairs(),
      shardable_value_update_pairs_);
  proto.set_alias_passthrough_params(alias_passthrough_params_);
  proto.set_content_aware_computation_sorting(
      content_aware_computation_sorting_);
  proto.set_fusion_config_collection(
      static_cast<HloModuleConfigProto::FusionConfigCollection>(
          fusion_config_collection_));
  AssignProtoFusionConfig(proto, fusion_config_);
  AssignProtoDotConfig(proto, dot_config_);
  AssignProtoLayoutConfig(proto, layout_config_);
  for (uint64_t cfg : memory_space_assignment_config_) {
    proto.add_memory_space_assignment_config(cfg);
  }
  AssignProtoPhaseOrderingConfig(proto, phase_ordering_config_);
  proto.set_phase_index(phase_index_);

  for (bool value : allow_spmd_sharding_propagation_to_parameters_) {
    proto.add_allow_spmd_sharding_propagation_to_parameters(value);
  }
  for (bool value : allow_spmd_sharding_propagation_to_output_) {
    proto.add_allow_spmd_sharding_propagation_to_output(value);
  }

  auto proto_analysis_map = proto.mutable_analysis_allowance_map();
  for (const auto& [key, value] : analysis_allowance_map_) {
    proto_analysis_map->insert({std::string(key), value});
  }
  proto.set_matrix_unit_operand_precision(matrix_unit_operand_precision_);
  proto.set_allow_separate_sharding_programs(allow_separate_sharding_programs_);
  proto.set_fdo_profile(fdo_profile_);
  proto.set_device_memory_size(device_memory_size_);
  return proto;
}

absl::StatusOr<std::unique_ptr<HloModuleConfig>>
HloModuleConfig::CreateFromProto(const HloModuleConfigProto& proto) {
  auto config = std::make_unique<HloModuleConfig>();

  if (proto.has_entry_computation_layout()) {
    auto comp_layout = ProgramShape{proto.entry_computation_layout()};
    config->SetComputationLayoutIfExists(comp_layout);
  } else {
    config->clear_entry_computation_layout();
  }
  config->seed_ = proto.seed();
  config->launch_id_ = proto.launch_id();
  config->replica_count_ = proto.replica_count();
  config->num_partitions_ = proto.num_partitions();
  config->param_requires_broadcast_via_collectives_.assign(
      proto.param_requires_broadcast_via_collectives().begin(),
      proto.param_requires_broadcast_via_collectives().end());
  config->use_spmd_partitioning_ = proto.use_spmd_partitioning();
  config->use_auto_spmd_partitioning_ = proto.use_auto_spmd_partitioning();
  config->auto_spmd_partitioning_mesh_shape_.assign(
      proto.auto_spmd_partitioning_mesh_shape().begin(),
      proto.auto_spmd_partitioning_mesh_shape().end());
  config->auto_spmd_partitioning_mesh_ids_.assign(
      proto.auto_spmd_partitioning_mesh_ids().begin(),
      proto.auto_spmd_partitioning_mesh_ids().end());
  config->deduplicate_hlo_ = proto.deduplicate_hlo();
  config->intra_op_parallelism_threads_ = proto.intra_op_parallelism_threads();
  config->device_type_ = proto.device_type();
  if (proto.has_debug_options()) {
    config->debug_options_ = proto.debug_options();
  }
  if (proto.has_static_device_assignment()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<DeviceAssignment> device_assignment,
        DeviceAssignment::Deserialize(proto.static_device_assignment()));
    config->static_device_assignment_ = std::move(*device_assignment);
  }
  AssignStructShardableValueUpdatePairs(*config,
                                        proto.shardable_value_update_pairs());
  config->alias_passthrough_params_ = proto.alias_passthrough_params();
  config->content_aware_computation_sorting_ =
      proto.content_aware_computation_sorting();
  config->fusion_config_collection_ =
      static_cast<FusionConfigCollection>(proto.fusion_config_collection());
  AssignStructFusionConfig(*config, proto);
  AssignStructDotConfig(*config, proto);
  AssignStructLayoutConfig(*config, proto);
  config->memory_space_assignment_config_.assign(
      proto.memory_space_assignment_config().begin(),
      proto.memory_space_assignment_config().end());
  AssignStructPhaseOrderingConfig(*config, proto);
  config->phase_index_ = proto.phase_index();
  config->allow_spmd_sharding_propagation_to_parameters_.assign(
      proto.allow_spmd_sharding_propagation_to_parameters().begin(),
      proto.allow_spmd_sharding_propagation_to_parameters().end());
  config->allow_spmd_sharding_propagation_to_output_.assign(
      proto.allow_spmd_sharding_propagation_to_output().begin(),
      proto.allow_spmd_sharding_propagation_to_output().end());
  config->analysis_allowance_map_.insert(proto.analysis_allowance_map().begin(),
                                         proto.analysis_allowance_map().end());
  config->matrix_unit_operand_precision_ =
      proto.matrix_unit_operand_precision();
  config->allow_separate_sharding_programs_ =
      proto.allow_separate_sharding_programs();
  config->fdo_profile_ = proto.fdo_profile();
  config->device_memory_size_ = proto.device_memory_size();
  return std::move(config);
}

}  // namespace xla
