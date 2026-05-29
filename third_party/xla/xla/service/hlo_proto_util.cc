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

#include "xla/service/hlo_proto_util.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/util.h"

namespace xla {

HloProto MakeHloProto(const HloModule& module,
                      const BufferAssignment& assignment) {
  HloProto proto;
  MakeHloProto(module, assignment, &proto);
  return proto;
}

void MakeHloProto(const HloModule& module, const BufferAssignment& assignment,
                  HloProto* proto) {
  MakeHloProto(module, proto);
  assignment.ToProto(proto->mutable_buffer_assignment());
}

HloProto MakeHloProto(const HloModule& module) {
  HloProto proto;
  MakeHloProto(module, &proto);
  return proto;
}

void MakeHloProto(const HloModule& module, HloProto* proto) {
  module.ToProto(proto->mutable_hlo_module());
}

absl::StatusOr<std::vector<const ShapeProto*>> EntryComputationParameterShapes(
    const HloProto& hlo_proto) {
  if (!hlo_proto.has_hlo_module()) {
    return NotFound("HloProto missing HloModuleProto.");
  }
  if (!hlo_proto.hlo_module().has_host_program_shape()) {
    return NotFound("HloProto missing program shape.");
  }

  std::vector<const ShapeProto*> parameter_shapes;
  const auto& program_shape = hlo_proto.hlo_module().host_program_shape();
  for (const ShapeProto& shape : program_shape.parameters()) {
    parameter_shapes.push_back(&shape);
  }
  return parameter_shapes;
}

absl::StatusOr<const ShapeProto*> EntryComputationOutputShape(
    const HloProto& hlo_proto) {
  if (!hlo_proto.has_hlo_module()) {
    return NotFound("HloProto missing HloModuleProto.");
  }
  if (!hlo_proto.hlo_module().has_host_program_shape()) {
    return NotFound("HloProto missing program shape.");
  }
  if (!hlo_proto.hlo_module().host_program_shape().has_result()) {
    return NotFound("HloProto missing result in its program shape");
  }

  return &hlo_proto.hlo_module().host_program_shape().result();
}

absl::StatusOr<std::string> GetBackendConfigString(
    const HloInstructionProto& instruction, const HloModuleProto* module) {
  const tsl::protobuf::RepeatedPtrField<std::string>* payloads =
      module ? &module->payloads() : nullptr;

  if (instruction.has_backend_config_payload()) {
    const Payload& payload = instruction.backend_config_payload();
    if (payload.has_id()) {
      if (module == nullptr) {
        return absl::InvalidArgumentError(
            "Module must be provided for external payload lookup.");
      }
      if (payload.id() < 0 || payload.id() >= payloads->size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Payload requested ID %d but payloads array has size %d",
            payload.id(), payloads ? payloads->size() : 0));
      }
      return payloads->at(payload.id());
    }
    return payload.value();
  }
  return instruction.backend_config();
}

HloInstructionProto ToProtoWithInlinedPayloads(HloInstructionProto proto,
                                               const HloModuleProto* module) {
  if (proto.has_backend_config_payload()) {
    const Payload& payload = proto.backend_config_payload();
    if (payload.has_id()) {
      if (module != nullptr) {
        const tsl::protobuf::RepeatedPtrField<std::string>& payloads =
            module->payloads();
        if (payload.id() >= 0 && payload.id() < payloads.size()) {
          const std::string& payload_string = payloads.at(payload.id());
          proto.mutable_backend_config_payload()->set_value(payload_string);
        }
      }
    }
  }
  return proto;
}

}  // namespace xla
