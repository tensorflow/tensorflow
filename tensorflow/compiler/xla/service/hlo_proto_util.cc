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

#include "tensorflow/compiler/xla/service/hlo_proto_util.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"

#include <string>

#include "tensorflow/compiler/xla/util.h"

namespace xla {

HloProto MakeHloProto(const HloModule& module,
                      const BufferAssignment& assignment) {
  BufferAssignmentProto proto_assignment = assignment.ToProto();
  HloProto proto = MakeHloProto(module);
  proto.mutable_buffer_assignment()->Swap(&proto_assignment);
  return proto;
}

HloProto MakeHloProto(const HloModule& module) {
  HloModuleProto proto_module = module.ToProto();
  HloProto proto;
  proto.mutable_hlo_module()->Swap(&proto_module);
  return proto;
}

StatusOr<std::unique_ptr<HloModule>> CreateModuleFromProto(
    const HloModuleProto& proto, const HloModuleConfig& module_config) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      HloModule::CreateFromProto(proto, module_config));
  TF_RETURN_IF_ERROR(
      HloVerifier(/*layout_sensitive=*/false, /*allow_mixed_precision=*/false)
          .Run(module.get())
          .status());
  return std::move(module);
}

StatusOr<std::vector<const Shape*>> EntryComputationParameterShapes(
    const HloProto& hlo_proto) {
  if (!hlo_proto.has_hlo_module()) {
    return NotFound("HloProto missing HloModuleProto.");
  }
  if (!hlo_proto.hlo_module().has_host_program_shape()) {
    return NotFound("HloProto missing program shape.");
  }

  std::vector<const Shape*> parameter_shapes;
  const auto& program_shape = hlo_proto.hlo_module().host_program_shape();
  for (const Shape& shape : program_shape.parameters()) {
    parameter_shapes.push_back(&shape);
  }
  return parameter_shapes;
}

StatusOr<const Shape*> EntryComputationOutputShape(const HloProto& hlo_proto) {
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

}  // namespace xla
