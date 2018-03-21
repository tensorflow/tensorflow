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

}  // namespace

StatusOr<Shape> XlaBuilder::GetShape(const XlaOp& op) const {
  TF_ASSIGN_OR_RETURN(auto instr, LookUpInstruction(op));
  return instr->shape();
}

StatusOr<Shape> XlaOp::GetShape() const {
  TF_RET_CHECK(builder_ != nullptr);
  return builder_->GetShape(*this);
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
  entry.set_root_id(-1);
  for (int64 i = instructions_.size() - 1; i >= 0; i--) {
    TF_ASSIGN_OR_RETURN(HloOpcode opcode,
                        StringToHloOpcode(instructions_[i].opcode()));
    if (CanBeRoot(opcode)) {
      entry.set_root_id(instructions_[i].id());
      *program_shape->mutable_result() = instructions_[i].shape();
      break;
    }
  }
  if (entry.root_id() == -1) {
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
  module->set_id(entry.id());
  module->set_entry_computation_name(entry.name());
  module->set_entry_computation_id(entry.id());
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
    TF_ASSIGN_OR_RETURN(const Shape& lhs_shape, lhs.GetShape());
    TF_ASSIGN_OR_RETURN(const Shape& rhs_shape, rhs.GetShape());
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferBinaryOpShape(HloOpcode::kAdd, lhs_shape,
                                           rhs_shape, broadcast_dimensions));
    return AddInstruction(std::move(instr), HloOpcode::kAdd, {lhs, rhs});
  };
  return NoteErrorOrReturn(op());
}

XlaOp XlaBuilder::ConstantLiteral(const Literal& literal) {
  HloInstructionProto instr;
  *instr.mutable_shape() = literal.shape();
  *instr.mutable_literal() = literal.ToProto();
  return AddInstruction(std::move(instr), HloOpcode::kConstant);
}

XlaOp XlaBuilder::Call(const XlaComputation& computation,
                       tensorflow::gtl::ArraySlice<XlaOp> operands) {
  auto op = [&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    std::vector<const Shape*> operand_shape_ptrs;
    std::vector<Shape> operand_shapes;
    for (const auto& operand : operands) {
      TF_ASSIGN_OR_RETURN(const Shape& shape, operand.GetShape());
      operand_shapes.push_back(shape);
    }
    c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferCallShape(
                            operand_shape_ptrs,
                            /*to_apply=*/computation.GetProgramShape()));

    // Add called computation.
    instr.add_called_computation_ids(
        computation.proto().entry_computation_id());
    for (const HloComputationProto& e : computation.proto().computations()) {
      embedded_.insert({e.id(), e});
    }

    return AddInstruction(std::move(instr), HloOpcode::kCall, operands);
  };
  return NoteErrorOrReturn(op());
}

XlaOp XlaBuilder::Parameter(int64 parameter_number, const Shape& shape,
                            const string& name) {
  auto op = [&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    if (parameter_numbers_.find(parameter_number) != parameter_numbers_.end()) {
      return InvalidArgument("parameter %lld already registered",
                             parameter_number);
    }
    parameter_numbers_.insert(parameter_number);
    instr.set_parameter_number(parameter_number);
    instr.set_name(name);
    *instr.mutable_shape() = shape;
    return AddInstruction(std::move(instr), HloOpcode::kParameter);
  };
  return NoteErrorOrReturn(op());
}

XlaOp XlaBuilder::AddInstruction(HloInstructionProto&& instr, HloOpcode opcode,
                                 tensorflow::gtl::ArraySlice<XlaOp> operands) {
  const int64 handle = instructions_.size();
  instr.set_id(handle);
  instr.set_opcode(HloOpcodeString(opcode));
  if (instr.name().empty()) {
    instr.set_name(StrCat(instr.opcode(), ".", handle));
  } else {
    // Append the handle to make sure the name is unique.
    instr.set_name(StrCat(instr.name(), ".", handle));
  }
  for (const auto& operand : operands) {
    instr.add_operand_ids(operand.handle());
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
