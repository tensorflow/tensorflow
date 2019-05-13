/* Copyright 2018 Graphcore Ltd

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_hash.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"

#include <map>

namespace xla {
namespace poplarplugin {

using tensorflow::Hash64Combine;

uint64 HloHash::GetHash() {
  if (!performed_hash_) HashModule();
  return hash_;
}

std::string HloHash::GetProtoStr() {
  if (!performed_hash_) HashModule();
  return proto_str_;
}

void HloHash::HashModule() {
  HloModuleProto proto = module_->ToProto();
  SanitizeHloModuleProto(&proto, module_);

  tensorflow::SerializeToStringDeterministic(proto, &proto_str_);

  hash_ = std::hash<string>()(proto_str_);
  hash_ = Hash64Combine(hash_, module_->config().seed());
  hash_ = Hash64Combine(hash_, module_->config().argument_count());
  hash_ = Hash64Combine(hash_, module_->config().resource_input_count());
  std::string s_inputs = absl::StrJoin(module_->config().input_mapping(), ",");
  hash_ = Hash64Combine(hash_, std::hash<string>()(s_inputs));
  std::string s_resource_update =
      absl::StrJoin(module_->config().resource_update_to_input_index(), ",");
  hash_ = Hash64Combine(hash_, std::hash<string>()(s_resource_update));
  performed_hash_ = true;
}

void HloHash::SanitizeHloModuleProto(HloModuleProto* proto,
                                     const HloModule* module) {
  // Always force the HloModule id to be 0 and set the name to "hlo_module"
  proto->set_id(0);
  proto->set_name("hlo_module");

  std::map<uint64, uint64> computation_id_map;

  // TODO Sort the computations into reliable post order

  // Serialize the computations
  uint64 serial_id = 0;
  for (const HloComputation* computation : module->MakeComputationPostOrder()) {
    HloComputationProto* computation_proto =
        proto->mutable_computations(serial_id);
    CHECK_EQ(computation_proto->name(), computation->name());
    auto old_id =
        SanitizeHloComputationProto(computation_proto, computation, serial_id);
    computation_id_map[old_id] = serial_id;
    if (computation->name() == module->entry_computation()->name()) {
      *proto->mutable_host_program_shape() = computation_proto->program_shape();
    }
    serial_id++;
  }

  // Patch up computation IDs with the renumbered ones
  for (uint64 comp_id = 0; comp_id < serial_id; comp_id++) {
    HloComputationProto* computation_proto =
        proto->mutable_computations(comp_id);
    for (uint64 inst_id = 0; inst_id < computation_proto->instructions().size();
         inst_id++) {
      HloInstructionProto* instruction_proto =
          computation_proto->mutable_instructions(inst_id);
      PatchComputationReferences(instruction_proto, computation_id_map);
    }
  }

  // Serialize entry_computation_name
  std::string entry_computation_name =
      std::to_string(proto->entry_computation_id());
  proto->set_entry_computation_name(entry_computation_name);
}

uint64 HloHash::SanitizeHloComputationProto(HloComputationProto* proto,
                                            const HloComputation* computation,
                                            uint64 new_id) {
  uint64 old_id = proto->id();

  // Replace computation name with id
  std::string name = std::to_string(new_id);
  proto->set_name(name);
  proto->set_id(new_id);

  std::map<uint64, uint64> instruction_id_map;

  // Serialize the instructions
  uint64 serial_id = 0;
  for (const HloInstruction* instruction :
       computation->MakeInstructionPostOrder()) {
    HloInstructionProto* instruction_proto =
        proto->mutable_instructions(serial_id);
    CHECK_EQ(instruction_proto->name(), instruction->name());
    auto old_id = SanitizeHloInstructionProto(instruction_proto, serial_id);
    instruction_id_map[old_id] = serial_id;
    serial_id++;
  }

  // Patch up instruction IDs with the renumbered ones
  for (uint64 id = 0; id < serial_id; id++) {
    HloInstructionProto* instruction_proto = proto->mutable_instructions(id);
    PatchInstructionReferences(instruction_proto, instruction_id_map);
  }

  // Serialize the shape
  SanitizeComputeProgramShape(proto->mutable_program_shape(), computation);

  return old_id;
}

uint64 HloHash::SanitizeHloInstructionProto(HloInstructionProto* proto,
                                            uint64 new_id) {
  uint64 old_id = proto->id();
  // Clear metadata - assuming metadata is irrelevant
  OpMetadata* metadata = proto->mutable_metadata();
  metadata->set_op_type("");
  metadata->set_op_name("");
  metadata->set_source_file("");
  metadata->set_source_line(0);
  // Replace instruction name with id
  std::string name = std::to_string(new_id);
  proto->set_name(name);
  proto->set_id(new_id);
  return old_id;
}

void HloHash::SanitizeComputeProgramShape(ProgramShapeProto* program_shape,
                                          const HloComputation* computation) {
  // Replace parameter names with unique ids
  for (auto* param_instruction : computation->parameter_instructions()) {
    uint64 serial_id = param_instruction->parameter_number();
    std::string name = std::to_string(serial_id);
    program_shape->set_parameter_names(serial_id, name);
  }
}

void HloHash::PatchInstructionReferences(
    HloInstructionProto* proto, const std::map<uint64, uint64>& id_map) {
  auto* operands = proto->mutable_operand_ids();
  for (auto it = operands->begin(); it != operands->end(); it++) {
    *it = id_map.at(*it);
  }
  auto* control_deps = proto->mutable_control_predecessor_ids();
  for (auto it = control_deps->begin(); it != control_deps->end(); it++) {
    *it = id_map.at(*it);
  }
}

void HloHash::PatchComputationReferences(
    HloInstructionProto* proto, const std::map<uint64, uint64>& id_map) {
  auto* ids = proto->mutable_called_computation_ids();
  for (auto it = ids->begin(); it != ids->end(); it++) {
    *it = id_map.at(*it);
  }
}

}  // namespace poplarplugin
}  // namespace xla
