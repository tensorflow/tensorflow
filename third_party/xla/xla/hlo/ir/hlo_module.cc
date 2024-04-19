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

#include "xla/hlo/ir/hlo_module.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_frontend_attributes.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/map_util.h"
#include "xla/printer.h"
#include "xla/service/compilation_environments.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/mapped_ptr_container_sorter.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/gtl/map_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {

HloModule::HloModule(const std::string& name, HloModuleConfig config)
    : HloModule(name, std::move(config),
                std::make_unique<CompilationEnvironments>()) {}

HloModule::HloModule(const std::string& name, HloModuleConfig config,
                     std::unique_ptr<CompilationEnvironments> comp_envs)
    : HloModule(name, std::make_unique<HloModuleConfig>(std::move(config)),
                std::move(comp_envs)) {}

HloModule::HloModule(const std::string& name,
                     std::variant<std::unique_ptr<HloModuleConfig>,
                                  std::shared_ptr<const HloModuleConfig>>
                         config,
                     std::unique_ptr<CompilationEnvironments> comp_envs)
    : name_(NameUniquer::GetSanitizedName(name)),
      config_(std::move(config)),
      unique_id_(next_unique_module_id_++),
      metadata_(tsl::Env::Default()),
      autofdo_fingerprint_(""),
      comp_envs_(std::move(comp_envs)) {
  metadata_.set_canonical_module_id(unique_id_);
}

Status HloModule::set_schedule(HloSchedule schedule) {
  TF_RET_CHECK(schedule.module() == this);
  TF_RETURN_IF_ERROR(schedule.Verify());
  schedule_ = std::move(schedule);
  return OkStatus();
}

void HloModule::ReplaceEntryComputation(HloComputation* entry_computation) {
  entry_computation_ = entry_computation;
  config_.get_mutable().SetDefaultComputationLayout(
      entry_computation_->ComputeProgramShape());
  input_output_alias_config_ = HloInputOutputAliasConfig(
      entry_computation_->root_instruction()->shape());
  buffer_donor_config_ = HloBufferDonorConfig();
}

HloModule::StackFrame HloModule::get_stack_frame(int id) const {
  HloModule::StackFrame stack_frame;
  if (!stack_frame_index_.has_value() || id < 1 ||
      id > stack_frame_index_->stack_frames().size()) {
    return stack_frame;
  }

  auto& frame = stack_frame_index_->stack_frames(id - 1);
  auto& file_location =
      stack_frame_index_->file_locations(frame.file_location_id() - 1);

  stack_frame.file_name =
      stack_frame_index_->file_names(file_location.file_name_id() - 1);
  stack_frame.function_name =
      stack_frame_index_->function_names(file_location.function_name_id() - 1);
  stack_frame.line = file_location.line();
  stack_frame.column = file_location.column();
  stack_frame.parent_frame_id = frame.parent_frame_id();

  return stack_frame;
}

HloComputation* HloModule::AddComputationInternal(
    std::unique_ptr<HloComputation> computation, bool is_entry,
    bool uniquify_identifiers, bool preserve_entry_layouts) {
  if (is_entry) {
    CHECK_EQ(nullptr, entry_computation_);
    entry_computation_ = computation.get();

    if (preserve_entry_layouts) {
      config_.get_mutable().SetComputationLayoutIfExists(
          entry_computation_->ComputeProgramShape());
    } else if (!config_.get().has_entry_computation_layout()) {
      // If the module configuration has no entry layout computation set, create
      // a default one based on the program shape.
      config_.get_mutable().SetDefaultComputationLayout(
          entry_computation_->ComputeProgramShape());
    }
    input_output_alias_config_ = HloInputOutputAliasConfig(
        entry_computation_->root_instruction()->shape());
    buffer_donor_config_ = HloBufferDonorConfig();
  }

  if (uniquify_identifiers) {
    computation->UniquifyName(&computation_name_uniquer_);
    for (auto* instruction : computation->instructions()) {
      instruction->UniquifyName(&instruction_name_uniquer_);
    }

    // Pick unique IDs for each instruction.
    for (auto* instruction : computation->instructions()) {
      instruction->SetUniqueId(NewUniqueInstructionId());
    }
    // Set unique id to this computation.
    CHECK_NE(computation->root_instruction()->unique_id(), -1)
        << "Root has no valid id: " << computation->ToString();
    computation->SetUniqueId(computation->root_instruction()->unique_id());
  } else {
    // Don't uniquify the names of the computation or instruction, but we must
    // run the names through the uniquifiers to prevent future name collisions
    // for computations and instructions created later. Also, set the
    // next_unique_id_ to the one greater than the max unique id of any
    // instruction (or the computation) to avoid ID collisions.
    computation_name_uniquer_.GetUniqueName(computation->name());
    for (auto* instruction : computation->instructions()) {
      instruction_name_uniquer_.GetUniqueName(instruction->name());
      next_unique_id_ = std::max(next_unique_id_, instruction->unique_id() + 1);
    }
    if (next_unique_id_ < computation->unique_id() + 1) {
      next_unique_id_ = computation->unique_id() + 1;
    }
  }

  computation->set_parent(this);
  computations_.push_back(std::move(computation));
  return computations_.back().get();
}

HloComputation* HloModule::AddEntryComputation(
    std::unique_ptr<HloComputation> computation) {
  return AddComputationInternal(std::move(computation), /*is_entry=*/true,
                                /*uniquify_identifiers=*/true,
                                /*preserve_entry_layouts=*/false);
}

HloComputation* HloModule::AddEntryComputationWithLayouts(
    std::unique_ptr<HloComputation> computation) {
  return AddComputationInternal(std::move(computation), /*is_entry=*/true,
                                /*uniquify_identifiers=*/true,
                                /*preserve_entry_layouts=*/true);
}

Status HloModule::RemoveEmbeddedComputation(HloComputation* to_remove) {
  if (has_schedule()) {
    schedule_->remove_computation(to_remove);
  }

  auto it = absl::c_find_if(
      computations_, [&to_remove](const std::unique_ptr<HloComputation>& comp) {
        return comp.get() == to_remove;
      });
  TF_RET_CHECK(it != computations_.end());
  TF_RET_CHECK(it->get() == to_remove);
  computations_.erase(it);
  return OkStatus();
}

HloComputation* HloModule::AddEmbeddedComputation(
    std::unique_ptr<HloComputation> computation) {
  return AddComputationInternal(std::move(computation), /*is_entry=*/false,
                                /*uniquify_identifiers=*/true,
                                /*preserve_entry_layouts=*/false);
}

void HloModule::MarkFusionDuplications(
    const absl::flat_hash_map<HloComputation*, HloComputation*>& replacements) {
  for (std::unique_ptr<HloComputation>& computation : computations_) {
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kFusion) {
        auto rep =
            replacements.find(instruction->fused_instructions_computation());
        if (rep != replacements.end()) {
          xla::HloComputation* new_comp = rep->second;
          if (new_comp->IsFusionComputation()) {
            auto dedup_name = new_comp->FusionInstruction()->name();
            new_comp->FusionInstruction()->set_metadata_deduplicated_name(
                std::string(dedup_name));
            instruction->set_metadata_deduplicated_name(
                std::string(dedup_name));
          }
        }
      }
    }
  }
}

void HloModule::MoveComputationsFrom(HloModule* module,
                                     bool make_names_unique) {
  for (size_t i = 0; i < module->computation_count(); ++i) {
    for (auto* instruction : module->computations_[i]->instructions()) {
      instruction->ClearUniqueIdInternal();
    }
    module->computations_[i]->ClearUniqueIdInternal();
    auto computation_raw_ptr = module->computations_[i].get();
    if (computation_raw_ptr->IsEntryComputation()) {
      this->entry_computation_ = nullptr;
    }
    this->AddComputationInternal(
        std::move(module->computations_[i]),
        /*is_entry=*/computation_raw_ptr->IsEntryComputation(),
        /*uniquify_identifiers=*/false,
        /*preserve_entry_layouts=*/false);
    if (make_names_unique) {
      computation_raw_ptr->UniquifyName(&computation_name_uniquer_);
      for (auto* instruction : computation_raw_ptr->instructions()) {
        instruction->UniquifyName(&instruction_name_uniquer_);
      }
    }
    // Pick unique IDs for each instruction.
    for (auto* instruction : computation_raw_ptr->instructions()) {
      instruction->SetUniqueId(NewUniqueInstructionId());
    }
    // Set unique id to this computation_raw_ptr.
    CHECK_NE(computation_raw_ptr->root_instruction()->unique_id(), -1)
        << "Root has no valid id: " << computation_raw_ptr->ToString();
    computation_raw_ptr->SetUniqueId(
        computation_raw_ptr->root_instruction()->unique_id());
  }
  // Since the computations no longer belong to the old module, clear the list.
  module->computations_.clear();
}

void HloModule::ReplaceComputations(
    const absl::flat_hash_map<HloComputation*, HloComputation*>& replacements) {
  // Replace all uses of non-canonical computations with their
  // representatives.
  std::vector<std::unique_ptr<HloComputation>> new_computations;
  new_computations.reserve(computations_.size());

  for (std::unique_ptr<HloComputation>& computation : computations_) {
    for (auto* instruction : computation->instructions()) {
      if (instruction->has_to_apply()) {
        HloComputation* new_arg = tsl::gtl::FindWithDefault(
            replacements, instruction->to_apply(), nullptr);
        if (new_arg != nullptr) {
          instruction->set_to_apply(new_arg);
        }
        continue;
      }
      switch (instruction->opcode()) {
        case HloOpcode::kWhile: {
          HloComputation* new_condition = tsl::gtl::FindWithDefault(
              replacements, instruction->while_condition(), nullptr);
          if (new_condition != nullptr) {
            instruction->set_while_condition(new_condition);
          }
          HloComputation* new_body = tsl::gtl::FindWithDefault(
              replacements, instruction->while_body(), nullptr);
          if (new_body != nullptr) {
            instruction->set_while_body(new_body);
          }
          break;
        }
        case HloOpcode::kConditional: {
          for (int b = 0; b < instruction->branch_count(); ++b) {
            HloComputation* new_computation = tsl::gtl::FindWithDefault(
                replacements, instruction->branch_computation(b), nullptr);
            if (new_computation != nullptr) {
              instruction->set_branch_computation(b, new_computation);
            }
          }
          break;
        }
        case HloOpcode::kSelectAndScatter: {
          HloComputation* new_select = tsl::gtl::FindWithDefault(
              replacements, instruction->select(), nullptr);
          if (new_select != nullptr) {
            instruction->set_select(new_select);
          }
          HloComputation* new_scatter = tsl::gtl::FindWithDefault(
              replacements, instruction->scatter(), nullptr);
          if (new_scatter != nullptr) {
            instruction->set_scatter(new_scatter);
          }
          break;
        }
        default:
          break;
      }
    }

    if (replacements.find(computation.get()) == replacements.end()) {
      new_computations.push_back(std::move(computation));
    }
  }

  // Replace entry_computation if necessary.
  entry_computation_ = tsl::gtl::FindWithDefault(
      replacements, entry_computation_, entry_computation_);

  computations_ = std::move(new_computations);
}

void HloModule::Print(Printer* printer, const HloPrintOptions& options) const {
  printer->Append("HloModule ");
  if (options.print_ids()) {
    // When print_ids() is false, exclude module's name because it includes and
    // leads to non-deterministic fingerprint.
    printer->Append(name());
  }
  if (has_schedule()) {
    TF_CHECK_OK(schedule().Verify());
    printer->Append(", is_scheduled=true");
  }
  std::string serialized_aliasing = input_output_alias_config().ToShortString();
  if (!serialized_aliasing.empty()) {
    printer->Append(", input_output_alias={ ");
    printer->Append(std::move(serialized_aliasing));
    printer->Append(" }");
  }
  std::string serialized_buffer_donor = buffer_donor_config().ToShortString();
  if (!serialized_buffer_donor.empty()) {
    printer->Append(", buffer_donor={ ");
    printer->Append(std::move(serialized_buffer_donor));
    printer->Append(" }");
  }
  const auto& config = config_.get();
  if (config.alias_passthrough_params()) {
    printer->Append(", alias_passthrough_params=true");
  }
  if (config.has_entry_computation_layout()) {
    printer->Append(", entry_computation_layout={");
    entry_computation_layout().Print(printer);
    printer->Append("}");
  }
  if (config.allow_spmd_sharding_propagation_to_parameters().size() != 1 ||
      config.allow_spmd_sharding_propagation_to_parameters().back()) {
    printer->Append(", allow_spmd_sharding_propagation_to_parameters={");
    AppendJoin(printer, config.allow_spmd_sharding_propagation_to_parameters(),
               ",", [](Printer* printer, bool i) {
                 printer->Append(i ? "true" : "false");
               });
    printer->Append("}");
  }
  if (config.allow_spmd_sharding_propagation_to_output().size() != 1 ||
      config.allow_spmd_sharding_propagation_to_output().back()) {
    printer->Append(", allow_spmd_sharding_propagation_to_output={");
    AppendJoin(printer, config.allow_spmd_sharding_propagation_to_output(), ",",
               [](Printer* printer, bool i) {
                 printer->Append(i ? "true" : "false");
               });
    printer->Append("}");
  }
  if (config.replica_count() != 1) {
    printer->Append(", replica_count=");
    printer->Append(config.replica_count());
  }
  if (config.num_partitions() != 1) {
    printer->Append(", num_partitions=");
    printer->Append(config.num_partitions());
  }
  if (!frontend_attributes_.map().empty()) {
    AppendCat(printer, ", frontend_attributes=",
              FrontendAttributesToString(frontend_attributes_));
  }
  printer->Append("\n\n");
  const auto& computations = options.canonicalize_computations()
                                 ? MakeComputationSorted()
                                 : MakeComputationPostOrder();
  for (const HloComputation* computation : computations) {
    // Don't print async computations when the sytax sugar is enabled since that
    // is redundant information.
    if (options.syntax_sugar_async_ops() && computation->IsAsyncComputation() &&
        computation->CanExpandIntoSingleInstruction()) {
      continue;
    }
    if (computation == entry_computation()) {
      printer->Append("ENTRY ");
    }
    if (has_schedule() && schedule().is_computation_scheduled(computation)) {
      computation->Print(printer, options,
                         schedule().sequence(computation).instructions());
    } else {
      computation->Print(printer, options);
    }
    printer->Append("\n\n");
  }
}

std::string HloModule::ToString(const HloPrintOptions& options) const {
  StringPrinter printer;
  Print(&printer, options);
  return std::move(printer).ToString();
}

absl::Cord HloModule::ToCord(const HloPrintOptions& options) const {
  CordPrinter printer;
  Print(&printer, options);
  return std::move(printer).ToCord();
}

HloModuleProto HloModule::ToProto() const {
  HloModuleProto proto;
  proto.set_id(unique_id_);
  proto.set_name(name_);
  if (entry_computation_) {
    *proto.mutable_entry_computation_name() =
        std::string(entry_computation_->name());
    proto.set_entry_computation_id(entry_computation_->unique_id());
    *proto.mutable_host_program_shape() =
        entry_computation_layout().ComputeProgramShape().ToProto();
  }
  for (const HloComputation* computation : MakeComputationPostOrder()) {
    HloComputationProto computation_proto = computation->ToProto();
    proto.add_computations()->Swap(&computation_proto);
  }
  if (has_schedule()) {
    *proto.mutable_schedule() = schedule().ToProto().value();
  }
  *proto.mutable_input_output_alias() = input_output_alias_config().ToProto();
  *proto.mutable_buffer_donor() = buffer_donor_config().ToProto();
  for (const auto& [parameter, indices, alt_memory_offset] :
       CrossProgramPrefetches()) {
    auto* prefetch = proto.mutable_cross_program_prefetches()->Add();
    prefetch->set_parameter(parameter);
    for (auto index : indices) {
      prefetch->add_index(index);
    }
    if (alt_memory_offset) {
      prefetch->set_offset(*alt_memory_offset);
    }
  }
  proto.set_is_dynamic(is_dynamic_);
  if (has_spmd_output_sharding()) {
    *proto.mutable_spmd_output_sharding() = spmd_output_sharding().ToProto();
  }

  *proto.mutable_frontend_attributes() = frontend_attributes_;

  if (has_spmd_parameters_shardings()) {
    for (const auto& parameter_sharding : spmd_parameters_shardings()) {
      *proto.add_spmd_parameters_shardings() = parameter_sharding.ToProto();
    }
  }

  proto.set_use_auto_spmd_partitioning(use_auto_spmd_partitioning_);

  for (const HloModuleProto::ProfileInfo& profile_info : profile_info_list_) {
    HloModuleProto::ProfileInfo& profile_info_proto =
        *proto.mutable_profile_info()->Add();
    profile_info_proto.set_profile_type(profile_info.profile_type());
    profile_info_proto.set_relative_speedup(profile_info.relative_speedup());
    profile_info_proto.set_profile_source(profile_info.profile_source());
    profile_info_proto.set_compilation_event(profile_info.compilation_event());
    profile_info_proto.set_fingerprint(profile_info.fingerprint());
  }
  if (config_.get().has_static_device_assignment()) {
    DeviceAssignmentProto device_assignment;
    TF_CHECK_OK(
        config_.get().static_device_assignment().Serialize(&device_assignment));
    (*proto.mutable_device_assignment()) = device_assignment;
  }

  if (stack_frame_index_.has_value()) {
    (*proto.mutable_stack_frame_index()) = *stack_frame_index_;
  }
  return proto;
}

absl::StatusOr<HloModuleProtoWithConfig> HloModule::ToProtoWithConfig() const {
  HloModuleProtoWithConfig result;
  TF_ASSIGN_OR_RETURN(*result.mutable_config(), config_.get().ToProto());
  *result.mutable_hlo_module() = ToProto();
  return result;
}

Status HloModule::CheckUniqueNamesAndIdsForComputationsAndInstructions() const {
  absl::flat_hash_set<absl::string_view> computation_names;
  absl::flat_hash_set<int> computation_ids;
  absl::flat_hash_set<absl::string_view> instruction_names;
  absl::flat_hash_set<int> instruction_ids;

  for (const HloComputation* computation : computations()) {
    TF_RET_CHECK(!ContainsKey(computation_names, computation->name()))
        << "Computation name is not unique: " << computation->name();
    computation_names.insert(computation->name());

    TF_RET_CHECK(!ContainsKey(computation_ids, computation->unique_id()))
        << "Computation id is not unique: " << computation->unique_id();
    computation_ids.insert(computation->unique_id());

    for (const HloInstruction* instruction : computation->instructions()) {
      TF_RET_CHECK(!ContainsKey(instruction_names, instruction->name()))
          << "Instruction name is not unique: " << instruction->name();
      instruction_names.insert(instruction->name());

      TF_RET_CHECK(!ContainsKey(instruction_ids, instruction->unique_id()))
          << "Instruction id is not unique: " << instruction->unique_id();
      instruction_ids.insert(instruction->unique_id());
    }
  }
  return OkStatus();
}

/* static */
absl::StatusOr<std::unique_ptr<HloModule>> HloModule::CreateFromProto(
    const HloModuleProto& proto, const HloModuleConfig& module_config,
    bool prohibit_empty_literal) {
  VLOG(2) << "CreateFromProto()";
  XLA_VLOG_LINES(3, proto.DebugString());

  // The ProgramShape in the passed in module config must match the shapes of
  // the entry parameters and root.
  TF_RET_CHECK(proto.has_host_program_shape())
      << "No program shape found in the proto";
  ProgramShape expected_program_shape(proto.host_program_shape());
  TF_RET_CHECK(expected_program_shape.parameters_size() ==
               module_config.entry_computation_layout().parameter_count());
  for (int i = 0; i < expected_program_shape.parameters_size(); ++i) {
    const Shape& parameter_shape =
        module_config.entry_computation_layout().parameter_layout(i).shape();
    TF_RET_CHECK(ShapeUtil::Compatible(expected_program_shape.parameters(i),
                                       parameter_shape))
        << "HloModuleConfig has different shape for parameter " << i
        << " than the HLO module. Expected: "
        << ShapeUtil::HumanStringWithLayout(
               expected_program_shape.parameters(i))
        << ", actual: " << ShapeUtil::HumanStringWithLayout(parameter_shape);
  }
  const Shape& result_shape =
      module_config.entry_computation_layout().result_layout().shape();
  TF_RET_CHECK(
      ShapeUtil::Compatible(expected_program_shape.result(), result_shape))
      << "HloModuleConfig has different result shape than the HLO module. "
         "Expected: "
      << ShapeUtil::HumanStringWithLayout(expected_program_shape.result())
      << ", actual: " << ShapeUtil::HumanStringWithLayout(result_shape);

  absl::flat_hash_map<int64_t, HloComputation*> computation_map;
  absl::flat_hash_map<HloComputation*, int64_t> to_proto_id;
  std::vector<std::unique_ptr<HloComputation>> computations;
  HloComputation* entry = nullptr;
  for (const HloComputationProto& computation_proto : proto.computations()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloComputation> computation,
        HloComputation::CreateFromProto(computation_proto, computation_map,
                                        prohibit_empty_literal));
    CHECK_NE(computation.get(), nullptr);
    int64_t computation_id = computation_proto.id();
    TF_RET_CHECK(computation_id != -1);
    TF_RET_CHECK(!ContainsKey(computation_map, computation_id));
    computation_map[computation_id] = computation.get();
    to_proto_id[computation.get()] = computation_id;
    if (computation_id == proto.entry_computation_id()) {
      entry = computation.get();
    }
    computations.push_back(std::move(computation));
  }
  TF_RET_CHECK(entry != nullptr);

  auto module = std::make_unique<HloModule>(proto.name(), module_config);

  // Sort the computations in the proto id's order.
  absl::c_sort(computations, [&](const std::unique_ptr<HloComputation>& a,
                                 const std::unique_ptr<HloComputation>& b) {
    return to_proto_id[a.get()] < to_proto_id[b.get()];
  });

  // Add sorted computations to the module.
  for (auto& computation : computations) {
    bool is_entry = computation.get() == entry;
    // Don't uniquify names because we want names to be stable across
    // serialization and deserialization.
    module->AddComputationInternal(std::move(computation), is_entry,
                                   /*uniquify_identifiers=*/false,
                                   /*preserve_entry_layouts=*/false);
  }
  TF_RET_CHECK(module->entry_computation_ != nullptr);
  TF_ASSIGN_OR_RETURN(
      module->input_output_alias_config_,
      HloInputOutputAliasConfig::CreateFromProto(
          entry->ComputeProgramShape().result(), proto.input_output_alias()));
  TF_ASSIGN_OR_RETURN(
      module->buffer_donor_config_,
      HloBufferDonorConfig::CreateFromProto(proto.buffer_donor()));

  TF_RETURN_IF_ERROR(
      module->CheckUniqueNamesAndIdsForComputationsAndInstructions());

  if (proto.has_schedule()) {
    TF_ASSIGN_OR_RETURN(
        HloSchedule schedule,
        HloSchedule::CreateFromProto(module.get(), proto.schedule()));
    TF_RETURN_IF_ERROR(module->set_schedule(std::move(schedule)));
  }

  for (const auto& prefetch : proto.cross_program_prefetches()) {
    module->AddCrossProgramPrefetch(
        prefetch.parameter(),
        ShapeIndex(prefetch.index().begin(), prefetch.index().end()),
        prefetch.offset());
  }

  module->set_is_dynamic(proto.is_dynamic());

  if (proto.has_frontend_attributes()) {
    module->set_frontend_attributes(proto.frontend_attributes());
  }

  if (proto.has_spmd_output_sharding()) {
    TF_ASSIGN_OR_RETURN(HloSharding hlo_sharding,
                        HloSharding::FromProto(proto.spmd_output_sharding()));
    module->set_spmd_output_sharding(hlo_sharding);
  }

  std::vector<HloSharding> param_shardings;
  for (const auto& sharding_proto : proto.spmd_parameters_shardings()) {
    TF_ASSIGN_OR_RETURN(HloSharding sharding,
                        HloSharding::FromProto(sharding_proto));
    param_shardings.push_back(sharding);
  }
  if (!param_shardings.empty()) {
    module->set_spmd_parameters_shardings(param_shardings);
  }

  module->set_use_auto_spmd_partitioning(proto.use_auto_spmd_partitioning());

  for (const auto& profile_info : proto.profile_info()) {
    module->add_profile_info(profile_info);
  }
  if (proto.has_device_assignment()) {
    if (!module->config_.get().has_static_device_assignment()) {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<DeviceAssignment> device_assignment,
          DeviceAssignment::Deserialize(proto.device_assignment()));
      module->config_.get_mutable().set_static_device_assignment(
          *device_assignment);
    }
  }

  if (proto.has_stack_frame_index()) {
    if (!module->stack_frame_index_.has_value()) {
      module->stack_frame_index_ = std::move(proto.stack_frame_index());
    }
  }
  return std::move(module);
}

/* static */
absl::StatusOr<HloModuleConfig> HloModule::CreateModuleConfigFromShape(
    const ProgramShape& program_shape, const DebugOptions& debug_options,
    const ExecutionOptions* execution_options) {
  HloModuleConfig module_config(ProgramShape{program_shape});
  module_config.set_debug_options(debug_options);
  if (execution_options) {
    if (execution_options->num_replicas() > 0) {
      module_config.set_replica_count(execution_options->num_replicas());
    }
    if (execution_options->num_partitions() > 0) {
      module_config.set_num_partitions(execution_options->num_partitions());
    }
    module_config.set_use_spmd_partitioning(
        execution_options->use_spmd_partitioning());
    module_config.set_use_auto_spmd_partitioning(
        execution_options->use_auto_spmd_partitioning());
    module_config.set_auto_spmd_partitioning_mesh_shape(std::vector<int64_t>(
        execution_options->auto_spmd_partitioning_mesh_shape().begin(),
        execution_options->auto_spmd_partitioning_mesh_shape().end()));
    module_config.set_auto_spmd_partitioning_mesh_ids(std::vector<int64_t>(
        execution_options->auto_spmd_partitioning_mesh_ids().begin(),
        execution_options->auto_spmd_partitioning_mesh_ids().end()));
    module_config.set_deduplicate_hlo(execution_options->deduplicate_hlo());
    if (!execution_options->allow_spmd_sharding_propagation_to_parameters()
             .empty()) {
      module_config.set_allow_spmd_sharding_propagation_to_parameters(
          execution_options->allow_spmd_sharding_propagation_to_parameters());
    }
    if (!execution_options->allow_spmd_sharding_propagation_to_output()
             .empty()) {
      module_config.set_allow_spmd_sharding_propagation_to_output(
          execution_options->allow_spmd_sharding_propagation_to_output());
    }
    if (execution_options->has_device_assignment()) {
      TF_ASSIGN_OR_RETURN(std::unique_ptr<DeviceAssignment> device_assignment,
                          DeviceAssignment::Deserialize(
                              execution_options->device_assignment()));
      module_config.set_static_device_assignment(*device_assignment);
      if (execution_options->num_replicas() > 0) {
        CHECK_EQ(module_config.static_device_assignment().replica_count(),
                 module_config.replica_count());
      }
      if (execution_options->num_partitions() > 0) {
        CHECK_EQ(module_config.static_device_assignment().computation_count(),
                 module_config.num_partitions());
      }
    }
    module_config.set_param_requires_broadcast_via_collectives(std::vector<
                                                               bool>(
        execution_options->param_requires_broadcast_via_collectives().begin(),
        execution_options->param_requires_broadcast_via_collectives().end()));
    module_config.set_allow_separate_sharding_programs(
        execution_options->allow_separate_sharding_programs());
    HloModuleConfig::AssignStructShardableValueUpdatePairs(
        module_config, execution_options->shardable_value_update_pairs());
  }

  // The module config is constructed with default layouts regardless of what is
  // passed in via the ProgramShape. Set the layouts to the appropriate values.
  ComputationLayout* entry_layout =
      module_config.mutable_entry_computation_layout();
  for (int64_t i = 0; i < entry_layout->parameter_count(); ++i) {
    TF_RETURN_IF_ERROR(
        entry_layout->mutable_parameter_layout(i)->CopyLayoutFromShape(
            program_shape.parameters(i)));
  }
  TF_RETURN_IF_ERROR(entry_layout->mutable_result_layout()->CopyLayoutFromShape(
      program_shape.result()));
  return module_config;
}

/* static */
absl::StatusOr<HloModuleConfig> HloModule::CreateModuleConfigFromProto(
    const HloModuleProto& module, const DebugOptions& debug_options,
    const ExecutionOptions* execution_options) {
  if (!module.has_host_program_shape()) {
    return tsl::errors::FailedPrecondition(
        "No program shape found in the proto");
  }
  ProgramShape program_shape(module.host_program_shape());
  TF_ASSIGN_OR_RETURN(HloModuleConfig config,
                      CreateModuleConfigFromShape(program_shape, debug_options,
                                                  execution_options));
  if (!config.has_static_device_assignment()) {
    if (module.has_device_assignment()) {
      // Get the proto from the exeuction options rather than the module proto.
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<DeviceAssignment> device_assignment,
          DeviceAssignment::Deserialize(module.device_assignment()));
      config.set_static_device_assignment(*device_assignment);
    }
  }
  return config;
}

absl::StatusOr<std::unique_ptr<HloModule>> HloModule::CreateFromProtoWithConfig(
    const HloModuleProtoWithConfig& proto, bool prohibit_empty_literal) {
  const auto& hlo_module_proto = proto.hlo_module();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModuleConfig> config_ptr,
                      HloModuleConfig::CreateFromProto(proto.config()));
  return HloModule::CreateFromProto(hlo_module_proto, *config_ptr,
                                    prohibit_empty_literal);
}

namespace {
// Returns whether `hlo` is used outside the given subcomputation.
// `instructions_in_subcomputation` is the instruction set of the given
// subcomputation.
bool IsUsedOutsideSubcomputation(const HloInstruction& hlo,
                                 const absl::flat_hash_set<HloInstruction*>&
                                     instructions_in_subcomputation) {
  return absl::c_any_of(hlo.users(), [&](HloInstruction* user) {
    return !instructions_in_subcomputation.contains(user);
  });
}
}  // anonymous namespace

HloInstruction* HloModule::OutlineExpressionFromComputation(
    absl::Span<HloInstruction* const> instructions_to_outline,
    const std::string& outlined_computation_name, HloComputation* computation) {
  auto builder = HloComputation::Builder(outlined_computation_name);

  // A map from original instructions to their counterparts in the new outlined
  // function.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> outlined_instructions;
  // A set that contains all instructions to be outlined.
  absl::flat_hash_set<HloInstruction*> instruction_set_to_outline(
      instructions_to_outline.begin(), instructions_to_outline.end());
  std::vector<HloInstruction*> arguments;
  std::vector<HloInstruction*> outputs;
  int64_t parameter_count = 0;
  for (HloInstruction* instruction_to_outline : instructions_to_outline) {
    // Clone the original instruction.
    HloInstruction* outlined_instruction =
        builder.AddInstruction(instruction_to_outline->Clone());

    // Replace its operands to their counterparts in the new function.
    for (int64_t operand_num = 0;
         operand_num < outlined_instruction->operand_count(); ++operand_num) {
      HloInstruction* old_operand =
          outlined_instruction->mutable_operand(operand_num);

      HloInstruction** operand_slot = &(outlined_instructions[old_operand]);
      if (*operand_slot == nullptr) {
        // Because instructions_to_outline is in topological order, if
        // old_operand is not in outlined_instructions, old_operand must be an
        // input of the outlined subcomputation and thus should be represented
        // as a parameter in the new function.
        arguments.push_back(old_operand);
        *operand_slot = builder.AddInstruction(HloInstruction::CreateParameter(
            parameter_count, old_operand->shape(), "p"));
        ++parameter_count;
      }
      TF_CHECK_OK(
          outlined_instruction->ReplaceOperandWith(operand_num, *operand_slot));
    }

    // Insert the new instruction into the outlined_instructions map.
    InsertOrDie(&outlined_instructions, instruction_to_outline,
                outlined_instruction);

    // Mark instruction_to_outline an output if it is used outside the
    // subcomputation or is the output of the original computation (i.e. used
    // externally).
    if (instruction_to_outline->user_count() == 0 ||
        IsUsedOutsideSubcomputation(*instruction_to_outline,
                                    instruction_set_to_outline)) {
      outputs.push_back(instruction_to_outline);
    }
  }

  if (outputs.size() != 1) {
    std::string error_message =
        "The subcomputation to outline has multiple outputs:\n";
    for (HloInstruction* output : outputs) {
      absl::StrAppend(&error_message, output->ToString(), "\n");
    }
    LOG(FATAL) << error_message;
  }
  HloInstruction* output = outputs[0];

  // Creates a call to the nested computation.
  HloComputation* nested_computation = AddEmbeddedComputation(
      builder.Build(FindOrDie(outlined_instructions, output)));
  HloInstruction* call = computation->AddInstruction(HloInstruction::CreateCall(
      output->shape(), arguments, nested_computation));

  VLOG(2) << "Outlining the following instructions";
  for (auto* instruction_to_outline : instructions_to_outline) {
    VLOG(2) << "  " << instruction_to_outline->ToString();
  }
  VLOG(2) << "as a call " << call->ToString();
  VLOG(2) << "to " << nested_computation->ToString();

  TF_CHECK_OK(output->ReplaceAllUsesWith(call));
  for (auto i = instructions_to_outline.rbegin();
       i != instructions_to_outline.rend(); ++i) {
    TF_CHECK_OK(computation->RemoveInstruction(*i));
  }

  return call;
}

int64_t HloModule::instruction_count() const {
  int64_t n = 0;
  for (const auto& computation : computations_) {
    n += computation->instruction_count();
  }
  return n;
}

std::vector<HloComputation*> HloModule::MakeComputationPostOrder(
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    const absl::flat_hash_set<HloComputation*>& allow_list) const {
  std::vector<HloComputation*> filtered_post_order(allow_list.size());
  auto post_order = this->MakeComputationPostOrder(execution_threads);

  int filtered_idx = 0;
  for (auto& computation : post_order) {
    if (allow_list.contains(computation)) {
      filtered_post_order[filtered_idx] = computation;
      filtered_idx += 1;
    }
  }

  return filtered_post_order;
}

std::vector<HloComputation*> HloModule::MakeComputationPostOrder(
    const absl::flat_hash_set<absl::string_view>& execution_threads) const {
  if (computations_.empty()) {
    return {};
  }
  // First determine all root computations by building a set of nonroot
  // computations (computations which are called by an instruction in the
  // module).
  absl::flat_hash_set<HloComputation*> nonroot_computations;
  nonroot_computations.reserve(computations_.size() - 1);
  for (auto& computation : computations_) {
    for (const HloInstructionInfo& inst :
         computation->instructions_with_info()) {
      if (HloInstruction::MightHaveCalledComputations(inst.opcode())) {
        for (HloComputation* called_computation : inst->called_computations()) {
          nonroot_computations.insert(called_computation);
        }
      }
    }
  }

  // Keep track of computations which have already been added to the post
  // order. This prevents duplication as an embedded computation may be called
  // from two different root computations.
  absl::flat_hash_set<HloComputation*> added_computations;
  std::vector<HloComputation*> post_order;
  added_computations.reserve(computations_.size());
  post_order.reserve(computations_.size());
  for (auto& computation : computations_) {
    if (nonroot_computations.contains(computation.get())) {
      continue;
    }
    for (HloComputation* embedded_computation :
         computation->MakeEmbeddedComputationsList()) {
      if (!added_computations.contains(embedded_computation)) {
        post_order.push_back(embedded_computation);
        added_computations.insert(embedded_computation);
      }
    }
    // Root computations should only be encountered once.
    CHECK(!added_computations.contains(computation.get()));
    post_order.push_back(computation.get());
    added_computations.insert(computation.get());
  }
  if (post_order.size() != computations_.size()) {
    for (HloComputation* computation : post_order) {
      LOG(ERROR) << "Post Order: " << computation->name() << " ("
                 << computation->parent()->name() << ")";
    }
    for (auto& computation : computations_) {
      LOG(ERROR) << "Computations: " << computation->name() << " ("
                 << computation->parent()->name() << ")";
    }
    LOG(FATAL) << "Mismatch computation count: post_order=" << post_order.size()
               << " computation_count=" << computations_.size();
  }
  if (execution_threads.empty()) {
    return post_order;
  }
  std::vector<HloComputation*> post_order_with_execution_threads;
  absl::c_copy_if(
      post_order, std::back_inserter(post_order_with_execution_threads),
      [&execution_threads](HloComputation* computation) {
        return execution_threads.find(computation->execution_thread()) !=
               execution_threads.end();
      });
  return post_order_with_execution_threads;
}

namespace {

class FingerprintMap {
 public:
  void Reserve(int capacity) { fingerprint_map_.reserve(capacity); }

  uint64_t GetFingerprint(const HloComputation* computation) {
    auto result = fingerprint_map_.try_emplace(computation, 0);
    if (result.second) {
      result.first->second =
          tsl::Fingerprint64(computation->ToString(print_options_));
    }
    return result.first->second;
  }

 private:
  HloPrintOptions print_options_ = HloPrintOptions::ModuleFingerprint();
  absl::flat_hash_map<const HloComputation*, uint64_t> fingerprint_map_;
};

void SortComputationsByContent(std::vector<HloComputation*>* computations) {
  FingerprintMap fingerprint_map;
  fingerprint_map.Reserve(computations->size());
  auto cmp = [&fingerprint_map](const HloComputation* a,
                                const HloComputation* b) {
    if (a->instruction_count() != b->instruction_count()) {
      return a->instruction_count() < b->instruction_count();
    }
    // Avoid computing fingerprints of (potentially) giant computation strings
    // just to compare when a == b
    if (a == b) return false;

    return fingerprint_map.GetFingerprint(a) <
           fingerprint_map.GetFingerprint(b);
  };
  absl::c_sort(*computations, cmp);
}

}  // anonymous namespace

std::vector<HloComputation*> HloModule::MakeComputationSorted(
    const absl::flat_hash_set<absl::string_view>& execution_threads) const {
  std::vector<HloComputation*> result =
      MakeComputationPostOrder(execution_threads);
  if (config().content_aware_computation_sorting()) {
    SortComputationsByContent(&result);
  }
  return result;
}

std::vector<HloComputation*> HloModule::MakeNonfusionComputations(
    const absl::flat_hash_set<absl::string_view>& execution_threads) const {
  std::vector<HloComputation*> result =
      MakeComputationPostOrder(execution_threads);
  result.erase(std::remove_if(
                   result.begin(), result.end(),
                   [](HloComputation* c) { return c->IsFusionComputation(); }),
               result.end());
  return result;
}

std::vector<HloComputation*> HloModule::MakeNonfusionComputationsSorted(
    const absl::flat_hash_set<absl::string_view>& execution_threads) const {
  auto result = MakeNonfusionComputations(execution_threads);
  if (config().content_aware_computation_sorting()) {
    SortComputationsByContent(&result);
  }
  return result;
}

std::unique_ptr<HloModule> HloModule::Clone(const std::string& suffix) const {
  return Clone(config_.FreezeAndShare(), suffix);
}

std::unique_ptr<HloModule> HloModule::Clone(const HloModuleConfig& config,
                                            const std::string& suffix) const {
  return Clone(std::make_shared<const HloModuleConfig>(config), suffix);
}

std::unique_ptr<HloModule> HloModule::Clone(
    std::shared_ptr<const HloModuleConfig> config,
    const std::string& suffix) const {
  VLOG(1) << "Cloning module :" << name_ << " --> " << suffix << "\n";
  auto module = std::make_unique<HloModule>(
      absl::StrCat(name_, suffix.empty() ? "" : "-", suffix), std::move(config),
      std::make_unique<CompilationEnvironments>(*comp_envs_));

  HloCloneContext context(module.get(), suffix);
  if (entry_computation_) {
    auto cloned_computation = entry_computation_->Clone(suffix, &context);
    module->AddEntryComputation(std::move(cloned_computation));
  }
  module->input_output_alias_config() = input_output_alias_config();
  module->buffer_donor_config() = buffer_donor_config();
  module->set_is_dynamic(is_dynamic());
  module->set_frontend_attributes(frontend_attributes());
  if (has_schedule() && schedule().Verify().ok()) {
    HloSchedule clone_schedule(module.get());
    for (HloComputation* computation : computations()) {
      if (schedule().is_computation_scheduled(computation)) {
        HloComputation* new_computation = context.FindComputation(computation);
        // The module being cloned may have computations that are dead, i.e.,
        // unreachable from the entry computation. In that case, new_computation
        // is nullptr.
        if (new_computation != nullptr) {
          HloInstructionSequence& clone_sequence =
              clone_schedule.GetOrCreateSequence(new_computation);
          for (const HloInstruction* instruction :
               schedule().sequence(computation).instructions()) {
            clone_sequence.push_back(context.GetInstruction(instruction));
          }
        }
      }
    }
    TF_CHECK_OK(module->set_schedule(std::move(clone_schedule)));
  }
  for (const auto& [parameter, indices, offset] : CrossProgramPrefetches()) {
    module->AddCrossProgramPrefetch(parameter, indices, offset);
  }

  // To make clone behavior match uncloned behavior, we reorder
  // module->computations_ to match the order in computations_.
  using ComputationSorter = MappedPtrContainerSorter<HloComputation>;
  auto computation_map_fn = [&context](const HloComputation* c) {
    return context.FindComputation(c);
  };
  auto status = ComputationSorter::Sort(
      computation_map_fn, ComputationSorter::IndexAfterMappedElementsFn(),
      computations_, module->computations_);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to sort module computations for " << name() << "; "
               << status;
  }

  return module;
}

Status HloModule::RemoveUnusedComputations() {
  std::string suffix = "tmp";
  auto module = std::make_unique<HloModule>(
      absl::StrCat(name_, "-", suffix), config(),
      std::make_unique<CompilationEnvironments>(*comp_envs_));
  HloCloneContext context(module.get(), suffix);
  entry_computation_->Clone(suffix, &context);
  std::vector<HloComputation*> to_remove;
  for (auto computation : computations()) {
    auto found_computation = context.FindComputation(computation);
    if (found_computation == nullptr) {
      to_remove.push_back(computation);
    }
  }
  for (auto computation : to_remove) {
    TF_RETURN_IF_ERROR(RemoveEmbeddedComputation(computation));
  }
  return OkStatus();
}

HloComputation* HloModule::DeepCloneComputation(HloComputation* computation,
                                                HloCloneContext* context) {
  HloComputation* new_computation;
  if (context != nullptr) {
    if ((new_computation = context->FindComputation(computation)) != nullptr) {
      return new_computation;
    }
    new_computation =
        AddEmbeddedComputation(computation->Clone(context->suffix(), context));
  } else {
    new_computation = AddEmbeddedComputation(computation->Clone(""));
  }
  return new_computation;
}

uint64_t HloModule::RandomNew64() const {
  absl::MutexLock l(&rng_mutex_);
  return rng_();
}

HloComputation* HloModule::GetComputationWithName(absl::string_view name) {
  auto computations_in_module = computations();
  auto it = absl::c_find_if(
      computations_in_module,
      [&](HloComputation* computation) { return computation->name() == name; });
  return it == computations_in_module.end() ? nullptr : *it;
}

std::string HloModule::GetFingerprint128(const HloPrintOptions& options) const {
  const tsl::Fprint128 fingerprint = tsl::Fingerprint128(ToString(options));
  absl::string_view fp_bytes(reinterpret_cast<const char*>(&fingerprint),
                             sizeof(tsl::Fprint128));
  return absl::BytesToHexString(fp_bytes);
}

/* static */ std::atomic<int> HloModule::next_unique_module_id_(0);

}  // namespace xla
