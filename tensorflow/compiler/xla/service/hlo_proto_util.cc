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

#include <string>

#include "tensorflow/compiler/xla/util.h"

namespace xla {

namespace {

// Returns the entry computation of the HLO module in the given HloProto.
StatusOr<const HloComputationProto*> GetEntryComputation(
    const HloProto& hlo_proto) {
  if (!hlo_proto.has_hlo_module()) {
    return NotFound("HloProto missing HloModuleProto.");
  }

  if (hlo_proto.hlo_module().entry_computation_name().empty()) {
    return NotFound("HloProto has empty entry computation name.");
  }

  const string& entry_computation_name =
      hlo_proto.hlo_module().entry_computation_name();
  const HloComputationProto* entry_computation = nullptr;
  for (const HloComputationProto& computation :
       hlo_proto.hlo_module().computations()) {
    if (computation.name() == entry_computation_name) {
      if (entry_computation == nullptr) {
        entry_computation = &computation;
      } else {
        return InvalidArgument(
            "HloProto has multiple computations with entry computation named "
            "%s.",
            entry_computation_name.c_str());
      }
    }
  }
  if (entry_computation == nullptr) {
    return InvalidArgument("HloProto has no entry computation named %s.",
                           entry_computation_name.c_str());
  }
  return entry_computation;
}

// Returns the root instruction of the given computation proto.
StatusOr<const HloInstructionProto*> GetRootInstruction(
    const HloComputationProto& computation) {
  if (computation.root_name().empty()) {
    return InvalidArgument("Missing root instruction name.");
  }

  const HloInstructionProto* root = nullptr;
  for (const HloInstructionProto& instruction : computation.instructions()) {
    if (instruction.name() == computation.root_name()) {
      if (root == nullptr) {
        root = &instruction;
      } else {
        return InvalidArgument(
            "Computation has multiple instructions named %s.",
            computation.root_name().c_str());
      }
    }
  }
  if (root == nullptr) {
    return InvalidArgument("Computation has no instruction named %s.",
                           computation.root_name().c_str());
  }
  return root;
}

// Returns the parameters of the given computation. Parameter numbers are
// checked for validity and contiguousness.
StatusOr<std::vector<const HloInstructionProto*>> GetParameters(
    const HloComputationProto& computation) {
  std::vector<const HloInstructionProto*> parameters;
  for (const HloInstructionProto& instruction : computation.instructions()) {
    if (instruction.opcode() == HloOpcodeString(HloOpcode::kParameter)) {
      parameters.push_back(&instruction);
    }
  }

  // Verify the uniqueness and validity of the parameter numbers.
  tensorflow::gtl::FlatSet<int64> parameter_numbers;
  for (const HloInstructionProto* parameter : parameters) {
    if (parameter->parameter_number() < 0 ||
        parameter->parameter_number() >= parameters.size()) {
      return InvalidArgument(
          "Parameter instruction %s has invalid parameter number %lld.",
          parameter->name().c_str(), parameter->parameter_number());
    }
    if (parameter_numbers.count(parameter->parameter_number()) != 0) {
      return InvalidArgument(
          "Multiple parameter instructions have parameter number %lld.",
          parameter->parameter_number());
    }
    parameter_numbers.insert(parameter->parameter_number());
  }

  std::sort(parameters.begin(), parameters.end(),
            [](const HloInstructionProto* a, const HloInstructionProto* b) {
              return a->parameter_number() < b->parameter_number();
            });

  return parameters;
}

}  // namespace

HloProto MakeHloProto(const HloModule& module,
                      const BufferAssignment& assignment) {
  HloOrderingProto proto_ordering =
      assignment.liveness().hlo_ordering().ToProto();
  BufferAssignmentProto proto_assignment = assignment.ToProto();
  HloProto proto = MakeHloProto(module);
  proto.mutable_hlo_ordering()->Swap(&proto_ordering);
  proto.mutable_buffer_assignment()->Swap(&proto_assignment);
  return proto;
}

HloProto MakeHloProto(const HloModule& module) {
  HloModuleProto proto_module = module.ToProto();
  HloProto proto;
  proto.mutable_hlo_module()->Swap(&proto_module);
  return proto;
}

StatusOr<std::vector<const Shape*>> EntryComputationParameterShapes(
    const HloProto& hlo_proto) {
  TF_ASSIGN_OR_RETURN(const HloComputationProto* entry_computation,
                      GetEntryComputation(hlo_proto));
  TF_ASSIGN_OR_RETURN(std::vector<const HloInstructionProto*> parameters,
                      GetParameters(*entry_computation));
  std::vector<const Shape*> parameter_shapes;
  for (const HloInstructionProto* parameter : parameters) {
    if (!parameter->has_shape()) {
      return InvalidArgument("Parameter instruction %s is missing shape.",
                             parameter->name().c_str());
    }
    parameter_shapes.push_back(&parameter->shape());
  }
  return parameter_shapes;
}

StatusOr<const Shape*> EntryComputationOutputShape(const HloProto& hlo_proto) {
  TF_ASSIGN_OR_RETURN(const HloComputationProto* entry_computation,
                      GetEntryComputation(hlo_proto));

  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      GetRootInstruction(*entry_computation));
  if (!root->has_shape()) {
    return InvalidArgument("Instruction %s is missing shape.",
                           root->name().c_str());
  }

  return &root->shape();
}

}  // namespace xla
