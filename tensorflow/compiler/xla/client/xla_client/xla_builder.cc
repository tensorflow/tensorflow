/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"

#include <string>
#include <utility>

#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"

namespace xla {

using tensorflow::strings::StrCat;

namespace {

int64 GetUniqueId() {
  static tensorflow::mutex mu(tensorflow::LINKER_INITIALIZED);
  static int64 built_counter = 0;
  tensorflow::mutex_lock loc(mu);
  const int64 id = built_counter++;
  return id;
}

// Returns true if an instruction with the given opcode can be the root of the
// computation.
bool CanBeRoot(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kSend:
    case HloOpcode::kOutfeed:
    case HloOpcode::kTrace:
      return false;
    default:
      return true;
  }
}

void SetOpcode(HloInstructionProto* instr, HloOpcode opcode) {
  instr->set_opcode(HloOpcodeString(opcode));
}

}  // namespace

StatusOr<std::unique_ptr<Shape>> XlaBuilder::GetShape(const XlaOp& op) const {
  TF_ASSIGN_OR_RETURN(auto instr, LookUpInstruction(op));
  return MakeUnique<Shape>(instr->shape());
}

StatusOr<Shape> XlaOp::GetShape() const {
  TF_RET_CHECK(builder_ != nullptr);
  TF_ASSIGN_OR_RETURN(auto shape, builder_->GetShape(*this));
  return *shape;
}

XlaBuilder::XlaBuilder(const string& computation_name)
    : name_(computation_name) {}

XlaBuilder::~XlaBuilder() {}

void XlaBuilder::NoteError(const Status& error) {
  CHECK(!error.ok());
  if (die_immediately_on_error_) {
    LOG(FATAL) << "error building computation: " << error;
  }

  if (first_error_.ok()) {
    first_error_ = error;
    first_error_backtrace_.CreateCurrent(/*skip_count=*/1);
  }
}

StatusOr<XlaComputation> XlaBuilder::Build() {
  if (!first_error_.ok()) {
    string backtrace;
    first_error_backtrace_.Dump(tensorflow::DebugWriteToString, &backtrace);
    return AppendStatus(first_error_, backtrace);
  }

  HloComputationProto entry;
  ProgramShape* program_shape = entry.mutable_program_shape();

  entry.set_name(name_);

  // Not all instructions can be roots. Walk backwards from the last added
  // instruction until a valid root is found.
  for (int64 i = instructions_.size() - 1; i >= 0; i--) {
    TF_ASSIGN_OR_RETURN(HloOpcode opcode,
                        StringToHloOpcode(instructions_[i].opcode()));
    if (CanBeRoot(opcode)) {
      entry.set_root_name(instructions_[i].name());
      *program_shape->mutable_result() = instructions_[i].shape();
      break;
    }
  }
  if (entry.root_name().empty()) {
    return FailedPrecondition("no root instruction was found");
  }

  // Check that the parameter numbers are continuous from 0, and add parameter
  // shapes and names to the program shape.
  const int64 param_count = parameter_numbers_.size();
  for (int64 i = 0; i < param_count; i++) {
    program_shape->add_parameters();
    program_shape->add_parameter_names();
  }
  for (const HloInstructionProto& instr : instructions_) {
    // Parameter number uniqueness is guaranteed in XlaBuilder::Parameter(). So
    // to verify continuity, we just need to verify that every parameter is in
    // the right range.
    if (instr.opcode() == HloOpcodeString(HloOpcode::kParameter)) {
      const int64 index = instr.parameter_number();
      TF_RET_CHECK(index >= 0 && index < param_count)
          << "invalid parameter number: " << index;
      *program_shape->mutable_parameters(index) = instr.shape();
      *program_shape->mutable_parameter_names(index) = instr.name();
    }
  }

  for (auto& instruction : instructions_) {
    entry.add_instructions()->Swap(&instruction);
  }

  const int64 id = GetUniqueId();
  entry.set_id(id);
  XlaComputation computation(id);
  HloModuleProto* module = computation.mutable_proto();
  module->set_name(entry.name());
  module->set_entry_computation_name(entry.name());
  *module->mutable_program_shape() = entry.program_shape();
  for (auto& e : embedded_) {
    module->add_computations()->Swap(&e.second);
  }
  module->add_computations()->Swap(&entry);

  return std::move(computation);
}

XlaOp XlaBuilder::Add(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  auto op = [&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    SetOpcode(&instr, HloOpcode::kAdd);
    TF_ASSIGN_OR_RETURN(const auto* lhs_instr, LookUpInstruction(lhs));
    TF_ASSIGN_OR_RETURN(const auto* rhs_instr, LookUpInstruction(rhs));
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferBinaryOpShape(
                            HloOpcode::kAdd, lhs_instr->shape(),
                            rhs_instr->shape(), broadcast_dimensions));
    instr.add_operand_names(lhs_instr->name());
    instr.add_operand_names(rhs_instr->name());
    return AddInstruction(std::move(instr));
  };
  return NoteErrorOrReturn(op());
}

XlaOp XlaBuilder::ConstantLiteral(const Literal& literal) {
  HloInstructionProto instr;
  SetOpcode(&instr, HloOpcode::kConstant);
  *instr.mutable_shape() = literal.shape();
  *instr.mutable_literal() = literal.ToProto();
  return AddInstruction(std::move(instr));
}

XlaOp XlaBuilder::Call(const XlaComputation& computation,
                       tensorflow::gtl::ArraySlice<XlaOp> operands) {
  auto op = [&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    SetOpcode(&instr, HloOpcode::kCall);
    std::vector<const Shape*> operand_shapes;
    for (const auto& operand : operands) {
      TF_ASSIGN_OR_RETURN(const auto* input, LookUpInstruction(operand));
      operand_shapes.push_back(&input->shape());
    }
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferCallShape(
                            operand_shapes,
                            /*to_apply=*/computation.GetProgramShape()));

    // Add input operands.
    for (const auto& operand : operands) {
      TF_ASSIGN_OR_RETURN(auto operand_instr, LookUpInstruction(operand));
      instr.add_operand_names(operand_instr->name());
    }

    // Add called computation.
    *instr.add_called_computation_names() = computation.proto().name();
    for (const HloComputationProto& e : computation.proto().computations()) {
      embedded_.insert({e.id(), e});
    }

    return AddInstruction(std::move(instr));
  };
  return NoteErrorOrReturn(op());
}

XlaOp XlaBuilder::Parameter(int64 parameter_number, const Shape& shape,
                            const string& name) {
  auto op = [&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    SetOpcode(&instr, HloOpcode::kParameter);
    if (parameter_numbers_.find(parameter_number) != parameter_numbers_.end()) {
      return InvalidArgument("parameter %lld already registered",
                             parameter_number);
    }
    parameter_numbers_.insert(parameter_number);
    instr.set_parameter_number(parameter_number);
    instr.set_name(name);
    *instr.mutable_shape() = shape;
    return AddInstruction(std::move(instr));
  };
  return NoteErrorOrReturn(op());
}

XlaOp XlaBuilder::AddInstruction(HloInstructionProto&& instr) {
  const int64 handle = instructions_.size();
  if (instr.name().empty()) {
    instr.set_name(StrCat(instr.opcode(), ".", handle));
  } else {
    // Append the handle to make sure the name is unique.
    instr.set_name(StrCat(instr.name(), ".", handle));
  }
  instructions_.push_back(instr);

  XlaOp op(handle, this);
  return op;
}

StatusOr<const HloInstructionProto*> XlaBuilder::LookUpInstruction(
    const XlaOp& op) const {
  TF_RET_CHECK(op.builder_ == this);
  if (op.handle() >= instructions_.size() || op.handle() < 0) {
    return InvalidArgument("no XlaOp value %lld", op.handle());
  }
  return &instructions_[op.handle()];
}

}  // namespace xla
