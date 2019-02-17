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
  SerializeHloModuleProto(&proto, module_);

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

void HloHash::SerializeHloModuleProto(HloModuleProto* proto,
                                      const HloModule* module) {
  // Always force the HloModule id to be 0 and set the name to "hlo_module"
  proto->set_id(0);
  proto->set_name("hlo_module");

  // Serialize the computations
  uint64 serial_id = 0;
  for (const HloComputation* computation : module->MakeComputationPostOrder()) {
    HloComputationProto* computation_proto =
        proto->mutable_computations(serial_id);
    CHECK_EQ(computation_proto->name(), computation->name());
    SerializeHloComputationProto(computation_proto, computation);
    if (computation->name() == module->entry_computation()->name()) {
      *proto->mutable_host_program_shape() = computation_proto->program_shape();
    }
    serial_id++;
  }

  // Serialize entry_computation_name
  std::string entry_computation_name =
      std::to_string(proto->entry_computation_id());
  proto->set_entry_computation_name(entry_computation_name);
}

void HloHash::SerializeHloComputationProto(HloComputationProto* proto,
                                           const HloComputation* computation) {
  // Replace computation name with id
  std::string name = std::to_string(proto->id());
  proto->set_name(name);

  // Serialize the instructions
  uint64 serial_id = 0;
  for (const HloInstruction* instruction :
       computation->MakeInstructionPostOrder()) {
    HloInstructionProto* instruction_proto =
        proto->mutable_instructions(serial_id);
    CHECK_EQ(instruction_proto->name(), instruction->name());
    SerializeHloInstructionProto(instruction_proto);
    serial_id++;
  }

  // Serialize the shape
  SerializeComputeProgramShape(proto->mutable_program_shape(), computation);
}

void HloHash::SerializeHloInstructionProto(HloInstructionProto* proto) {
  // Clear metadata - assuming metadata is irrelevant
  OpMetadata* metadata = proto->mutable_metadata();
  metadata->set_op_type("");
  metadata->set_op_name("");
  metadata->set_source_file("");
  metadata->set_source_line(0);
  // Replace instruction name with id
  std::string name = std::to_string(proto->id());
  proto->set_name(name);
}

void HloHash::SerializeComputeProgramShape(ProgramShapeProto* program_shape,
                                           const HloComputation* computation) {
  // Replace parameter names with unique ids
  uint64 serial_id = 0;
  for (auto* param_instruction : computation->parameter_instructions()) {
    CHECK_EQ(program_shape->parameter_names(serial_id),
             param_instruction->name());
    std::string name = std::to_string(param_instruction->unique_id());
    program_shape->set_parameter_names(serial_id, name);
    serial_id++;
  }
}

}  // namespace poplarplugin
}  // namespace xla
