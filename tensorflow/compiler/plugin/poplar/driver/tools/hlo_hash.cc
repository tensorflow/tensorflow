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
#include "tensorflow/compiler/plugin/poplar/driver/tools/hash.h"

#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/proto_serialization.h"

#include <map>
#include <queue>

namespace xla {
namespace poplarplugin {

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
  hash_ = hash_util::hash(
      proto_str_, module_->config().argument_count(),
      module_->config().resource_input_count(),
      absl::StrJoin(module_->config().input_mapping(), ","),
      absl::StrJoin(module_->config().resource_update_to_input_index(), ","));
  performed_hash_ = true;
}

void HloHash::SanitizeHloModuleProto(HloModuleProto* proto,
                                     const HloModule* module) {
  // Always force the HloModule id to be 0 and set the name to "hlo_module"
  proto->set_id(0);
  proto->set_name("hlo_module");

  std::map<uint64, uint64> computation_id_map;

  // Generate a reliable post order for the computations
  std::vector<HloComputation*> computation_list;
  std::set<HloComputation*> computation_set;

  std::queue<HloComputation*> comp_queue;
  comp_queue.push(module->entry_computation());
  while (!comp_queue.empty()) {
    HloComputation* comp = comp_queue.front();

    computation_list.push_back(comp);
    computation_set.insert(comp);
    comp_queue.pop();

    for (auto* inst : comp->MakeInstructionPostOrder()) {
      for (auto* called_comp : inst->called_computations()) {
        if (computation_set.count(called_comp) == 0) {
          comp_queue.push(called_comp);
        }
      }
    }
  }

  // Reorganise the computations in the proto into the reliable order, any
  // orphan computations will go to the end of the list
  for (uint64 serial_id = 0; serial_id < computation_list.size(); serial_id++) {
    auto comp_id = computation_list[serial_id]->unique_id();
    if (proto->computations(serial_id).id() != comp_id) {
      for (uint64 i = serial_id + 1; i < computation_list.size(); i++) {
        if (proto->computations(i).id() == comp_id) {
          auto c = proto->mutable_computations(i);
          proto->mutable_computations(serial_id)->Swap(c);
          break;
        }
      }
    }
  }

  // Serialize the computations
  for (auto comp_id = 0; comp_id < proto->computations().size(); comp_id++) {
    HloComputationProto* computation_proto =
        proto->mutable_computations(comp_id);
    auto old_id = SanitizeHloComputationProto(computation_proto, comp_id);
    computation_id_map[old_id] = comp_id;
  }

  // Patch up computation IDs with the renumbered ones
  for (auto comp_id = 0; comp_id < proto->computations().size(); comp_id++) {
    HloComputationProto* computation_proto =
        proto->mutable_computations(comp_id);
    for (uint64 inst_id = 0; inst_id < computation_proto->instructions().size();
         inst_id++) {
      HloInstructionProto* instruction_proto =
          computation_proto->mutable_instructions(inst_id);
      PatchComputationReferences(instruction_proto, computation_id_map);
    }
  }

  // Patch up entry computation id and name
  proto->set_entry_computation_id(
      computation_id_map[proto->entry_computation_id()]);
  std::string entry_computation_name =
      std::to_string(proto->entry_computation_id());
  proto->set_entry_computation_name(entry_computation_name);
}

uint64 HloHash::SanitizeHloComputationProto(HloComputationProto* proto,
                                            uint64 new_id) {
  uint64 old_id = proto->id();

  // Replace computation name with id
  std::string name = std::to_string(new_id);
  proto->set_name(name);
  proto->set_id(new_id);

  std::map<uint64, uint64> instruction_id_map;

  // Serialize the instructions
  for (uint64 inst_id = 0; inst_id < proto->instructions().size(); inst_id++) {
    HloInstructionProto* instruction_proto =
        proto->mutable_instructions(inst_id);
    auto old_id = SanitizeHloInstructionProto(instruction_proto, inst_id);
    instruction_id_map[old_id] = inst_id;
  }

  // Patch up instruction IDs with the renumbered ones
  for (uint64 id = 0; id < proto->instructions().size(); id++) {
    HloInstructionProto* instruction_proto = proto->mutable_instructions(id);
    PatchInstructionReferences(instruction_proto, instruction_id_map);
  }

  // Patch root instruction ID
  proto->set_root_id(instruction_id_map[proto->root_id()]);

  // Serialize the shape
  SanitizeComputeProgramShape(proto->mutable_program_shape());

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

void HloHash::SanitizeComputeProgramShape(ProgramShapeProto* program_shape) {
  // Replace parameter names with unique ids
  for (auto id = 0; id < program_shape->parameter_names().size(); id++) {
    program_shape->set_parameter_names(id, std::to_string(id));
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
