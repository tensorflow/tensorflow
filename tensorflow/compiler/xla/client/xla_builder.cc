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

#include "tensorflow/compiler/xla/client/xla_builder.h"

#include <functional>
#include <numeric>
#include <queue>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/client/sharding_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/platform/mutex.h"

namespace xla {

using absl::StrCat;

namespace {

int64 GetUniqueId() {
  static tensorflow::mutex mu(tensorflow::LINKER_INITIALIZED);
  static int64 built_counter = 0;
  tensorflow::mutex_lock loc(mu);
  const int64 id = built_counter++;
  return id;
}

}  // namespace

XlaOp operator-(const XlaOp& x) { return Neg(x); }
XlaOp operator+(const XlaOp& x, const XlaOp& y) { return Add(x, y); }
XlaOp operator-(const XlaOp& x, const XlaOp& y) { return Sub(x, y); }
XlaOp operator*(const XlaOp& x, const XlaOp& y) { return Mul(x, y); }
XlaOp operator/(const XlaOp& x, const XlaOp& y) { return Div(x, y); }
XlaOp operator%(const XlaOp& x, const XlaOp& y) { return Rem(x, y); }

XlaOp operator~(const XlaOp& x) { return Not(x); }
XlaOp operator&(const XlaOp& x, const XlaOp& y) { return And(x, y); }
XlaOp operator|(const XlaOp& x, const XlaOp& y) { return Or(x, y); }
XlaOp operator^(const XlaOp& x, const XlaOp& y) { return Xor(x, y); }
XlaOp operator<<(const XlaOp& x, const XlaOp& y) { return ShiftLeft(x, y); }

XlaOp operator>>(const XlaOp& x, const XlaOp& y) {
  XlaBuilder* builder = x.builder();
  return builder->ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(xla::Shape shape, builder->GetShape(x));
    if (!ShapeUtil::ElementIsIntegral(shape)) {
      return InvalidArgument(
          "Argument to >> operator does not have an integral type (%s).",
          ShapeUtil::HumanString(shape));
    }
    if (ShapeUtil::ElementIsSigned(shape)) {
      return ShiftRightArithmetic(x, y);
    } else {
      return ShiftRightLogical(x, y);
    }
  });
}

StatusOr<Shape> XlaBuilder::GetShape(const XlaOp& op) const {
  TF_RETURN_IF_ERROR(first_error_);

  TF_ASSIGN_OR_RETURN(auto instr, LookUpInstruction(op));
  return instr->shape();
}

StatusOr<std::vector<Shape>> XlaBuilder::GetOperandShapes(
    absl::Span<const XlaOp> operands) const {
  std::vector<Shape> operand_shapes;
  for (const XlaOp& operand : operands) {
    TF_ASSIGN_OR_RETURN(const Shape& shape, GetShape(operand));
    operand_shapes.push_back(shape);
  }
  return operand_shapes;
}

XlaBuilder::XlaBuilder(const string& computation_name)
    : name_(computation_name) {}

XlaBuilder::~XlaBuilder() {}

XlaOp XlaBuilder::ReportError(const Status& error) {
  CHECK(!error.ok());
  if (die_immediately_on_error_) {
    LOG(FATAL) << "error building computation: " << error;
  }

  if (first_error_.ok()) {
    first_error_ = error;
    first_error_backtrace_.CreateCurrent(/*skip_count=*/1);
  }
  return XlaOp(this);
}

XlaOp XlaBuilder::ReportErrorOrReturn(const StatusOr<XlaOp>& op) {
  if (!first_error_.ok()) {
    return XlaOp(this);
  }
  if (!op.ok()) {
    return ReportError(op.status());
  }
  return op.ValueOrDie();
}

XlaOp XlaBuilder::ReportErrorOrReturn(
    const std::function<StatusOr<XlaOp>()>& op_creator) {
  return ReportErrorOrReturn(op_creator());
}

StatusOr<ProgramShape> XlaBuilder::GetProgramShape(int64 root_id) const {
  TF_RETURN_IF_ERROR(first_error_);
  TF_RET_CHECK((root_id >= 0) && (root_id < instructions_.size()));

  ProgramShape program_shape;

  *program_shape.mutable_result() = instructions_[root_id].shape();

  // Check that the parameter numbers are continuous from 0, and add parameter
  // shapes and names to the program shape.
  const int64 param_count = parameter_numbers_.size();
  for (int64 i = 0; i < param_count; i++) {
    program_shape.add_parameters();
    program_shape.add_parameter_names();
  }
  for (const HloInstructionProto& instr : instructions_) {
    // Parameter number uniqueness is guaranteed in XlaBuilder::Parameter(). So
    // to verify continuity, we just need to verify that every parameter is in
    // the right range.
    if (instr.opcode() == HloOpcodeString(HloOpcode::kParameter)) {
      const int64 index = instr.parameter_number();
      TF_RET_CHECK(index >= 0 && index < param_count)
          << "invalid parameter number: " << index;
      *program_shape.mutable_parameters(index) = instr.shape();
      *program_shape.mutable_parameter_names(index) = instr.name();
    }
  }
  return program_shape;
}

StatusOr<ProgramShape> XlaBuilder::GetProgramShape() const {
  TF_RET_CHECK(!instructions_.empty());
  return GetProgramShape(instructions_.back().id());
}

StatusOr<ProgramShape> XlaBuilder::GetProgramShape(XlaOp root) const {
  if (root.builder_ != this) {
    return InvalidArgument("Given root operation is not in this computation.");
  }
  return GetProgramShape(root.handle());
}

void XlaBuilder::IsConstantVisitor(const int64 op_handle,
                                   std::set<int64>* visited,
                                   bool* is_constant) const {
  if (visited->count(op_handle) != 0 || !*is_constant) {
    return;
  }

  CHECK(op_handle < instructions_.size() && op_handle >= 0);

  const HloInstructionProto& instr = instructions_[op_handle];
  const HloOpcode opcode = StringToHloOpcode(instr.opcode()).ValueOrDie();
  switch (opcode) {
    default:
      for (const int64 operand_id : instr.operand_ids()) {
        IsConstantVisitor(operand_id, visited, is_constant);
      }
      // TODO(b/32495713): We aren't checking the called computations.
      break;

    // Non functional ops.
    case HloOpcode::kRng:
    case HloOpcode::kCrossReplicaSum:
      // TODO(b/33009255): Implmement constant folding for cross replica sum.
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kCall:
      // TODO(b/32495713): We aren't checking the to_apply computation itself,
      // so we conservatively say that computations containing the Call op
      // cannot be constant.  We cannot set is_functional=false in other similar
      // cases since we're already relying on IsConstant to return true.
    case HloOpcode::kCustomCall:
    case HloOpcode::kWhile:
      // TODO(b/32495713): We aren't checking the condition and body
      // computations themselves.
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
    case HloOpcode::kParameter:
      *is_constant = false;
      break;
  }
  if (!*is_constant) {
    VLOG(1) << "Non-constant: " << instr.name();
  }
  visited->insert(op_handle);
}

XlaComputation XlaBuilder::BuildAndNoteError() {
  DCHECK(parent_builder_ != nullptr);
  auto build_status = Build();
  if (!build_status.ok()) {
    parent_builder_->ReportError(
        AddStatus(build_status.status(), absl::StrCat("error from: ", name_)));
    return {};
  }
  return build_status.ConsumeValueOrDie();
}

StatusOr<XlaComputation> XlaBuilder::Build() {
  if (!first_error_.ok()) {
    string backtrace;
    first_error_backtrace_.Dump(tensorflow::DebugWriteToString, &backtrace);
    return AppendStatus(first_error_, backtrace);
  }
  return Build(instructions_.back().id());
}

StatusOr<XlaComputation> XlaBuilder::Build(XlaOp root) {
  if (root.builder_ != this) {
    return InvalidArgument("Given root operation is not in this computation.");
  }
  return Build(root.handle());
}

StatusOr<XlaComputation> XlaBuilder::Build(int64 root_id) {
  if (!first_error_.ok()) {
    string backtrace;
    first_error_backtrace_.Dump(tensorflow::DebugWriteToString, &backtrace);
    return AppendStatus(first_error_, backtrace);
  }

  HloComputationProto entry;
  entry.set_id(GetUniqueId());  // Give the computation a global unique id.
  entry.set_name(StrCat(name_, entry.id()));  // Ensure that the name is unique.

  TF_ASSIGN_OR_RETURN(*entry.mutable_program_shape(), GetProgramShape(root_id));
  entry.set_root_id(root_id);

  for (auto& instruction : instructions_) {
    // Ensures that the instruction names are unique among the whole graph.
    const string& new_name =
        StrCat(instruction.name(), ".", entry.id(), ".", instruction.id());
    instruction.set_name(new_name);
    entry.add_instructions()->Swap(&instruction);
  }

  XlaComputation computation(entry.id());
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

  // Clear data held by this builder.
  this->instructions_.clear();
  this->embedded_.clear();
  this->parameter_numbers_.clear();

  return std::move(computation);
}

StatusOr<XlaOp> XlaBuilder::InDimBroadcast(
    const Shape& shape, const XlaOp& operand,
    absl::Span<const int64> broadcast_dimensions) {
  TF_RETURN_IF_ERROR(first_error_);

  HloInstructionProto instr;
  *instr.mutable_shape() = shape;
  for (int64 dim : broadcast_dimensions) {
    instr.add_dimensions(dim);
  }
  return AddInstruction(std::move(instr), HloOpcode::kBroadcast, {operand});
}

StatusOr<XlaOp> XlaBuilder::AddBroadcastSequence(const Shape& output_shape,
                                                 const XlaOp& operand) {
  TF_RETURN_IF_ERROR(first_error_);

  TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));

  CHECK(ShapeUtil::IsScalar(operand_shape) ||
        ShapeUtil::Rank(operand_shape) == ShapeUtil::Rank(output_shape));
  Shape broadcast_shape =
      ShapeUtil::ChangeElementType(output_shape, operand_shape.element_type());

  // Do explicit broadcast for scalar.
  if (ShapeUtil::IsScalar(operand_shape)) {
    return InDimBroadcast(broadcast_shape, operand, {});
  }

  // Do explicit broadcast for degenerate broadcast.
  std::vector<int64> broadcast_dimensions;
  std::vector<int64> reshaped_dimensions;
  for (int i = 0; i < ShapeUtil::Rank(operand_shape); i++) {
    if (operand_shape.dimensions(i) == output_shape.dimensions(i)) {
      broadcast_dimensions.push_back(i);
      reshaped_dimensions.push_back(operand_shape.dimensions(i));
    } else {
      TF_RET_CHECK(operand_shape.dimensions(i) == 1)
          << "An explicit broadcast sequence requires the broadcasted "
             "dimensions to be trivial; operand shape: "
          << operand_shape << "; output_shape: " << output_shape;
    }
  }
  // Eliminate the size one dimensions.
  TF_ASSIGN_OR_RETURN(XlaOp reshaped_operand,
                      Reshape(ShapeUtil::MakeShape(operand_shape.element_type(),
                                                   reshaped_dimensions),
                              operand));
  // Broadcast 'reshape' up to the larger size.
  return InDimBroadcast(broadcast_shape, reshaped_operand,
                        broadcast_dimensions);
}

XlaOp XlaBuilder::UnaryOp(HloOpcode unop, const XlaOp& operand) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferUnaryOpShape(unop, operand_shape));
    return AddInstruction(std::move(instr), unop, {operand});
  });
}

XlaOp XlaBuilder::BinaryOp(HloOpcode binop, const XlaOp& lhs, const XlaOp& rhs,
                           absl::Span<const int64> broadcast_dimensions) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& lhs_shape, GetShape(lhs));
    TF_ASSIGN_OR_RETURN(const Shape& rhs_shape, GetShape(rhs));
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferBinaryOpShape(
                            binop, lhs_shape, rhs_shape, broadcast_dimensions));

    const int64 lhs_rank = ShapeUtil::Rank(lhs_shape);
    const int64 rhs_rank = ShapeUtil::Rank(rhs_shape);

    XlaOp updated_lhs = lhs;
    XlaOp updated_rhs = rhs;

    if (!broadcast_dimensions.empty() && lhs_rank != rhs_rank) {
      const bool should_broadcast_lhs = lhs_rank < rhs_rank;
      XlaOp from = should_broadcast_lhs ? lhs : rhs;
      const Shape& from_shape = should_broadcast_lhs ? lhs_shape : rhs_shape;

      std::vector<int64> to_size;
      for (int64 size : instr.shape().dimensions()) {
        to_size.push_back(size);
      }
      for (int64 from_dim = 0; from_dim < ShapeUtil::Rank(from_shape);
           from_dim++) {
        int64 to_dim = broadcast_dimensions[from_dim];
        to_size[to_dim] = from_shape.dimensions(from_dim);
      }

      const Shape& broadcasted_shape =
          ShapeUtil::MakeShape(from_shape.element_type(), to_size);
      TF_ASSIGN_OR_RETURN(
          XlaOp broadcasted_operand,
          InDimBroadcast(broadcasted_shape, from, broadcast_dimensions));

      updated_lhs = should_broadcast_lhs ? broadcasted_operand : lhs;
      updated_rhs = !should_broadcast_lhs ? broadcasted_operand : rhs;
    }

    TF_ASSIGN_OR_RETURN(Shape updated_lhs_shape, GetShape(updated_lhs));
    if (!ShapeUtil::SameDimensions(instr.shape(), updated_lhs_shape)) {
      TF_ASSIGN_OR_RETURN(updated_lhs,
                          AddBroadcastSequence(instr.shape(), updated_lhs));
    }
    TF_ASSIGN_OR_RETURN(Shape updated_rhs_shape, GetShape(updated_rhs));
    if (!ShapeUtil::SameDimensions(instr.shape(), updated_rhs_shape)) {
      TF_ASSIGN_OR_RETURN(updated_rhs,
                          AddBroadcastSequence(instr.shape(), updated_rhs));
    }

    return AddInstruction(std::move(instr), binop, {updated_lhs, updated_rhs});
  });
}

XlaOp XlaBuilder::TernaryOp(HloOpcode triop, const XlaOp& lhs, const XlaOp& rhs,
                            const XlaOp& ehs) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& lhs_shape, GetShape(lhs));
    TF_ASSIGN_OR_RETURN(const Shape& rhs_shape, GetShape(rhs));
    TF_ASSIGN_OR_RETURN(const Shape& ehs_shape, GetShape(ehs));
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferTernaryOpShape(
                            triop, lhs_shape, rhs_shape, ehs_shape));
    XlaOp updated_lhs = lhs;
    XlaOp updated_rhs = rhs;
    XlaOp updated_ehs = ehs;
    if (!ShapeUtil::IsTuple(instr.shape())) {
      if (!ShapeUtil::IsTuple(lhs_shape) &&
          !ShapeUtil::SameDimensions(instr.shape(), lhs_shape)) {
        // lhs is being implicitly broadcasted. Change to explicit.
        TF_ASSIGN_OR_RETURN(updated_lhs,
                            AddBroadcastSequence(instr.shape(), lhs));
      }
      if (!ShapeUtil::IsTuple(rhs_shape) &&
          !ShapeUtil::SameDimensions(instr.shape(), rhs_shape)) {
        // rhs is being implicitly broadcasted. Change to explicit.
        TF_ASSIGN_OR_RETURN(updated_rhs,
                            AddBroadcastSequence(instr.shape(), rhs));
      }
      if (!ShapeUtil::IsTuple(ehs_shape) &&
          !ShapeUtil::SameDimensions(instr.shape(), ehs_shape)) {
        // ehs is being implicitly broadcasted. Change to explicit.
        TF_ASSIGN_OR_RETURN(updated_ehs,
                            AddBroadcastSequence(instr.shape(), ehs));
      }
    }
    return AddInstruction(std::move(instr), triop,
                          {updated_lhs, updated_rhs, updated_ehs});
  });
}

XlaOp XlaBuilder::Add(const XlaOp& lhs, const XlaOp& rhs,
                      absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kAdd, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Mul(const XlaOp& lhs, const XlaOp& rhs,
                      absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kMultiply, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::ConstantLiteral(const LiteralSlice& literal) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = literal.shape();
    *instr.mutable_literal() = literal.ToProto();
    return AddInstruction(std::move(instr), HloOpcode::kConstant);
  });
}

XlaOp XlaBuilder::Iota(const Shape& shape, int64 iota_dimension) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = shape;
    instr.add_dimensions(iota_dimension);
    return AddInstruction(std::move(instr), HloOpcode::kIota);
  });
}

XlaOp XlaBuilder::Iota(PrimitiveType type, int64 size) {
  return Iota(ShapeUtil::MakeShape(type, {size}), /*iota_dimension=*/0);
}

XlaOp XlaBuilder::Call(const XlaComputation& computation,
                       absl::Span<const XlaOp> operands) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes, GetOperandShapes(operands));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(const ProgramShape& called_program_shape,
                        computation.GetProgramShape());
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferCallShape(operand_shape_ptrs,
                                       /*to_apply=*/called_program_shape));

    AddCalledComputation(computation, &instr);

    return AddInstruction(std::move(instr), HloOpcode::kCall, operands);
  });
}

XlaOp XlaBuilder::Parameter(int64 parameter_number, const Shape& shape,
                            const string& name) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    if (!parameter_numbers_.insert(parameter_number).second) {
      return InvalidArgument("parameter %d already registered",
                             parameter_number);
    }
    instr.set_parameter_number(parameter_number);
    instr.set_name(name);
    *instr.mutable_shape() = shape;
    return AddInstruction(std::move(instr), HloOpcode::kParameter);
  });
}

XlaOp XlaBuilder::Broadcast(const XlaOp& operand,
                            absl::Span<const int64> broadcast_sizes) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(
        const Shape& shape,
        ShapeInference::InferBroadcastShape(operand_shape, broadcast_sizes));

    // The client-level broadcast op just appends dimensions on the left (adds
    // lowest numbered dimensions). The HLO broadcast instruction is more
    // flexible and can add new dimensions anywhere. The instruction's
    // dimensions field maps operand dimensions to dimensions in the broadcast
    // output, so to append dimensions on the left the instruction's dimensions
    // should just be the n highest dimension numbers of the output shape where
    // n is the number of input dimensions.
    const int64 operand_rank = ShapeUtil::Rank(operand_shape);
    std::vector<int64> dimensions(operand_rank);
    for (int i = 0; i < operand_rank; ++i) {
      dimensions[i] = i + ShapeUtil::Rank(shape) - operand_rank;
    }
    return InDimBroadcast(shape, operand, dimensions);
  });
}

XlaOp XlaBuilder::BroadcastInDim(
    const XlaOp& operand, const Shape& shape,
    const absl::Span<const int64> broadcast_dimensions) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    return InDimBroadcast(shape, operand, broadcast_dimensions);
  });
}

StatusOr<XlaOp> XlaBuilder::Reshape(const Shape& shape, const XlaOp& operand) {
  TF_RETURN_IF_ERROR(first_error_);

  HloInstructionProto instr;
  *instr.mutable_shape() = shape;
  return AddInstruction(std::move(instr), HloOpcode::kReshape, {operand});
}

XlaOp XlaBuilder::Slice(const XlaOp& operand,
                        absl::Span<const int64> start_indices,
                        absl::Span<const int64> limit_indices,
                        absl::Span<const int64> strides) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferSliceShape(operand_shape, start_indices,
                                        limit_indices, strides));
    for (int i = 0; i < start_indices.size(); i++) {
      auto* slice_config = instr.add_slice_dimensions();
      slice_config->set_start(start_indices[i]);
      slice_config->set_limit(limit_indices[i]);
      slice_config->set_stride(strides[i]);
    }

    return AddInstruction(std::move(instr), HloOpcode::kSlice, {operand});
  });
}

XlaOp XlaBuilder::SliceInDim(const XlaOp& operand, int64 start_index,
                             int64 limit_index, int64 stride, int64 dimno) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape& shape, GetShape(operand));
    std::vector<int64> starts(ShapeUtil::Rank(shape), 0);
    std::vector<int64> limits(shape.dimensions().begin(),
                              shape.dimensions().end());
    std::vector<int64> strides(ShapeUtil::Rank(shape), 1);
    starts[dimno] = start_index;
    limits[dimno] = limit_index;
    strides[dimno] = stride;
    return Slice(operand, starts, limits, strides);
  });
}

XlaOp XlaBuilder::DynamicSlice(const XlaOp& operand, const XlaOp& start_indices,
                               absl::Span<const int64> slice_sizes) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(const Shape& start_indices_shape,
                        GetShape(start_indices));
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferDynamicSliceShape(
                            operand_shape, start_indices_shape, slice_sizes));

    for (int64 size : slice_sizes) {
      instr.add_dynamic_slice_sizes(size);
    }

    return AddInstruction(std::move(instr), HloOpcode::kDynamicSlice,
                          {operand, start_indices});
  });
}

XlaOp XlaBuilder::DynamicUpdateSlice(const XlaOp& operand, const XlaOp& update,
                                     const XlaOp& start_indices) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(const Shape& update_shape, GetShape(update));
    TF_ASSIGN_OR_RETURN(const Shape& start_indices_shape,
                        GetShape(start_indices));
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferDynamicUpdateSliceShape(
                            operand_shape, update_shape, start_indices_shape));

    return AddInstruction(std::move(instr), HloOpcode::kDynamicUpdateSlice,
                          {operand, update, start_indices});
  });
}

XlaOp XlaBuilder::ConcatInDim(absl::Span<const XlaOp> operands,
                              int64 dimension) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes, GetOperandShapes(operands));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferConcatOpShape(operand_shape_ptrs, dimension));

    instr.add_dimensions(dimension);

    return AddInstruction(std::move(instr), HloOpcode::kConcatenate, operands);
  });
}

XlaOp XlaBuilder::Pad(const XlaOp& operand, const XlaOp& padding_value,
                      const PaddingConfig& padding_config) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(const Shape& padding_value_shape,
                        GetShape(padding_value));
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferPadShape(operand_shape, padding_value_shape,
                                      padding_config));

    *instr.mutable_padding_config() = padding_config;

    return AddInstruction(std::move(instr), HloOpcode::kPad,
                          {operand, padding_value});
  });
}

XlaOp XlaBuilder::Reshape(const XlaOp& operand,
                          absl::Span<const int64> dimensions,
                          absl::Span<const int64> new_sizes) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(const Shape& shape,
                        ShapeInference::InferReshapeShape(
                            operand_shape, dimensions, new_sizes));
    XlaOp transposed = IsIdentityPermutation(dimensions)
                           ? operand
                           : Transpose(operand, dimensions);
    return Reshape(shape, transposed);
  });
}

XlaOp XlaBuilder::Reshape(const XlaOp& operand,
                          absl::Span<const int64> new_sizes) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, GetShape(operand));
    std::vector<int64> dimensions(shape.dimensions_size());
    std::iota(dimensions.begin(), dimensions.end(), 0);
    return Reshape(operand, dimensions, new_sizes);
  });
}

XlaOp XlaBuilder::Collapse(const XlaOp& operand,
                           absl::Span<const int64> dimensions) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (dimensions.size() <= 1) {
      // Not collapsing anything, trivially we can return the operand versus
      // enqueueing a trivial reshape.
      return operand;
    }

    // Out-of-order collapse is not supported.
    // Checks that the collapsed dimensions are in order and consecutive.
    for (absl::Span<const int64>::size_type i = 1; i < dimensions.size(); ++i) {
      if (dimensions[i] - 1 != dimensions[i - 1]) {
        return InvalidArgument(
            "Collapsed dimensions are not in consecutive order.");
      }
    }

    // Create a new sizes vector from the old shape, replacing the collapsed
    // dimensions by the product of their sizes.
    TF_ASSIGN_OR_RETURN(const Shape& original_shape, GetShape(operand));

    VLOG(3) << "original shape: " << ShapeUtil::HumanString(original_shape);
    VLOG(3) << "dims to collapse: " << absl::StrJoin(dimensions, ",");

    std::vector<int64> new_sizes;
    for (int i = 0; i < ShapeUtil::Rank(original_shape); ++i) {
      if (i <= dimensions.front() || i > dimensions.back()) {
        new_sizes.push_back(original_shape.dimensions(i));
      } else {
        new_sizes.back() *= original_shape.dimensions(i);
      }
    }

    VLOG(3) << "new sizes: [" << absl::StrJoin(new_sizes, ",") << "]";

    return Reshape(operand, new_sizes);
  });
}

void XlaBuilder::Trace(const string& tag, const XlaOp& operand) {
  ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = ShapeUtil::MakeNil();
    *instr.mutable_literal() = LiteralUtil::CreateR1U8(tag)->ToProto();
    return AddInstruction(std::move(instr), HloOpcode::kTrace, {operand});
  });
}

XlaOp XlaBuilder::Select(const XlaOp& pred, const XlaOp& on_true,
                         const XlaOp& on_false) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape& true_shape, GetShape(on_true));
    TF_ASSIGN_OR_RETURN(const Shape& false_shape, GetShape(on_false));
    TF_RET_CHECK(ShapeUtil::IsTuple(true_shape) ==
                 ShapeUtil::IsTuple(false_shape));
    HloOpcode opcode = ShapeUtil::IsTuple(true_shape) ? HloOpcode::kTupleSelect
                                                      : HloOpcode::kSelect;
    return TernaryOp(opcode, pred, on_true, on_false);
  });
}

XlaOp XlaBuilder::Tuple(absl::Span<const XlaOp> elements) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes, GetOperandShapes(elements));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferVariadicOpShape(
                            HloOpcode::kTuple, operand_shape_ptrs));
    return AddInstruction(std::move(instr), HloOpcode::kTuple, elements);
  });
}

XlaOp XlaBuilder::GetTupleElement(const XlaOp& tuple_data, int64 index) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& tuple_shape, GetShape(tuple_data));
    if (!ShapeUtil::IsTuple(tuple_shape)) {
      return InvalidArgument(
          "Operand to GetTupleElement() is not a tuple; got %s",
          ShapeUtil::HumanString(tuple_shape));
    }
    *instr.mutable_shape() =
        ShapeUtil::GetTupleElementShape(tuple_shape, index);

    instr.set_tuple_index(index);

    return AddInstruction(std::move(instr), HloOpcode::kGetTupleElement,
                          {tuple_data});
  });
}

XlaOp XlaBuilder::Eq(const XlaOp& lhs, const XlaOp& rhs,
                     absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kEq, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Ne(const XlaOp& lhs, const XlaOp& rhs,
                     absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kNe, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Ge(const XlaOp& lhs, const XlaOp& rhs,
                     absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kGe, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Gt(const XlaOp& lhs, const XlaOp& rhs,
                     absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kGt, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Le(const XlaOp& lhs, const XlaOp& rhs,
                     absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kLe, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Lt(const XlaOp& lhs, const XlaOp& rhs,
                     absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kLt, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Dot(const XlaOp& lhs, const XlaOp& rhs,
                      const PrecisionConfigProto* precision_config_proto) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape& lhs_shape, GetShape(lhs));

    DotDimensionNumbers dimension_numbers;
    dimension_numbers.add_lhs_contracting_dimensions(
        lhs_shape.dimensions_size() == 1 ? 0 : 1);
    dimension_numbers.add_rhs_contracting_dimensions(0);
    return DotGeneral(lhs, rhs, dimension_numbers, precision_config_proto);
  });
}

XlaOp XlaBuilder::DotGeneral(
    const XlaOp& lhs, const XlaOp& rhs,
    const DotDimensionNumbers& dimension_numbers,
    const PrecisionConfigProto* precision_config_proto) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& lhs_shape, GetShape(lhs));
    TF_ASSIGN_OR_RETURN(const Shape& rhs_shape, GetShape(rhs));
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferDotOpShape(lhs_shape, rhs_shape,
                                                        dimension_numbers));
    *instr.mutable_dot_dimension_numbers() = dimension_numbers;
    if (precision_config_proto != nullptr) {
      *instr.mutable_precision_config() = *precision_config_proto;
    }
    return AddInstruction(std::move(instr), HloOpcode::kDot, {lhs, rhs});
  });
}

Status XlaBuilder::VerifyConvolution(
    const Shape& lhs_shape, const Shape& rhs_shape,
    const ConvolutionDimensionNumbers& dimension_numbers) const {
  if (ShapeUtil::Rank(lhs_shape) != ShapeUtil::Rank(rhs_shape)) {
    return InvalidArgument(
        "Convolution arguments must have same number of "
        "dimensions. Got: %s and %s",
        ShapeUtil::HumanString(lhs_shape), ShapeUtil::HumanString(rhs_shape));
  }
  int num_dims = ShapeUtil::Rank(lhs_shape);
  if (num_dims < 2) {
    return InvalidArgument(
        "Convolution expects argument arrays with >= 3 dimensions. "
        "Got: %s and %s",
        ShapeUtil::HumanString(lhs_shape), ShapeUtil::HumanString(rhs_shape));
  }
  int num_spatial_dims = num_dims - 2;

  const auto check_spatial_dimensions =
      [&](const char* const field_name,
          const tensorflow::protobuf::RepeatedField<tensorflow::protobuf_int64>&
              numbers) {
        if (numbers.size() != num_spatial_dims) {
          return InvalidArgument("Expected %d elements for %s, but got %d.",
                                 num_spatial_dims, field_name, numbers.size());
        }
        for (int i = 0; i < numbers.size(); ++i) {
          if (numbers.Get(i) < 0 || numbers.Get(i) >= num_dims) {
            return InvalidArgument("Convolution %s[%d] is out of bounds: %d",
                                   field_name, i, numbers.Get(i));
          }
        }
        return Status::OK();
      };
  TF_RETURN_IF_ERROR(
      check_spatial_dimensions("input_spatial_dimensions",
                               dimension_numbers.input_spatial_dimensions()));
  TF_RETURN_IF_ERROR(
      check_spatial_dimensions("kernel_spatial_dimensions",
                               dimension_numbers.kernel_spatial_dimensions()));
  return check_spatial_dimensions(
      "output_spatial_dimensions",
      dimension_numbers.output_spatial_dimensions());
}

XlaOp XlaBuilder::Conv(const XlaOp& lhs, const XlaOp& rhs,
                       absl::Span<const int64> window_strides, Padding padding,
                       int64 feature_group_count,
                       const PrecisionConfigProto* precision_config_proto) {
  return ConvWithGeneralDimensions(
      lhs, rhs, window_strides, padding,
      CreateDefaultConvDimensionNumbers(window_strides.size()),
      feature_group_count, precision_config_proto);
}

XlaOp XlaBuilder::ConvWithGeneralPadding(
    const XlaOp& lhs, const XlaOp& rhs, absl::Span<const int64> window_strides,
    absl::Span<const std::pair<int64, int64>> padding,
    int64 feature_group_count,
    const PrecisionConfigProto* precision_config_proto) {
  return ConvGeneral(lhs, rhs, window_strides, padding,
                     CreateDefaultConvDimensionNumbers(window_strides.size()),
                     feature_group_count, precision_config_proto);
}

XlaOp XlaBuilder::ConvWithGeneralDimensions(
    const XlaOp& lhs, const XlaOp& rhs, absl::Span<const int64> window_strides,
    Padding padding, const ConvolutionDimensionNumbers& dimension_numbers,
    int64 feature_group_count,
    const PrecisionConfigProto* precision_config_proto) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape& lhs_shape, GetShape(lhs));
    TF_ASSIGN_OR_RETURN(const Shape& rhs_shape, GetShape(rhs));

    TF_RETURN_IF_ERROR(
        VerifyConvolution(lhs_shape, rhs_shape, dimension_numbers));

    std::vector<int64> base_area_dimensions(
        dimension_numbers.input_spatial_dimensions_size());
    for (std::vector<int64>::size_type i = 0; i < base_area_dimensions.size();
         ++i) {
      base_area_dimensions[i] =
          lhs_shape.dimensions(dimension_numbers.input_spatial_dimensions(i));
    }

    std::vector<int64> window_dimensions(
        dimension_numbers.kernel_spatial_dimensions_size());
    for (std::vector<int64>::size_type i = 0; i < window_dimensions.size();
         ++i) {
      window_dimensions[i] =
          rhs_shape.dimensions(dimension_numbers.kernel_spatial_dimensions(i));
    }

    return ConvGeneral(lhs, rhs, window_strides,
                       MakePadding(base_area_dimensions, window_dimensions,
                                   window_strides, padding),
                       dimension_numbers, feature_group_count,
                       precision_config_proto);
  });
}

XlaOp XlaBuilder::ConvGeneral(
    const XlaOp& lhs, const XlaOp& rhs, absl::Span<const int64> window_strides,
    absl::Span<const std::pair<int64, int64>> padding,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64 feature_group_count,
    const PrecisionConfigProto* precision_config_proto) {
  return ConvGeneralDilated(lhs, rhs, window_strides, padding, {}, {},
                            dimension_numbers, feature_group_count,
                            precision_config_proto);
}

XlaOp XlaBuilder::ConvGeneralDilated(
    const XlaOp& lhs, const XlaOp& rhs, absl::Span<const int64> window_strides,
    absl::Span<const std::pair<int64, int64>> padding,
    absl::Span<const int64> lhs_dilation, absl::Span<const int64> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers,
    int64 feature_group_count,
    const PrecisionConfigProto* precision_config_proto) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& lhs_shape, GetShape(lhs));
    TF_ASSIGN_OR_RETURN(const Shape& rhs_shape, GetShape(rhs));
    TF_RETURN_IF_ERROR(
        VerifyConvolution(lhs_shape, rhs_shape, dimension_numbers));

    std::vector<int64> window_dimensions(
        dimension_numbers.kernel_spatial_dimensions_size());
    for (std::vector<int64>::size_type i = 0; i < window_dimensions.size();
         ++i) {
      window_dimensions[i] =
          rhs_shape.dimensions(dimension_numbers.kernel_spatial_dimensions(i));
    }
    TF_ASSIGN_OR_RETURN(*instr.mutable_window(),
                        MakeWindow(window_dimensions, window_strides, padding,
                                   lhs_dilation, rhs_dilation));

    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferConvolveShape(
                            lhs_shape, rhs_shape, instr.window(),
                            dimension_numbers, feature_group_count));

    *instr.mutable_convolution_dimension_numbers() = dimension_numbers;
    instr.set_feature_group_count(feature_group_count);

    if (precision_config_proto != nullptr) {
      *instr.mutable_precision_config() = *precision_config_proto;
    }

    return AddInstruction(std::move(instr), HloOpcode::kConvolution,
                          {lhs, rhs});
  });
}

StatusOr<Window> XlaBuilder::MakeWindow(
    absl::Span<const int64> window_dimensions,
    absl::Span<const int64> window_strides,
    absl::Span<const std::pair<int64, int64>> padding,
    absl::Span<const int64> lhs_dilation,
    absl::Span<const int64> rhs_dilation) const {
  const auto verify_size = [&](const size_t x, const char* x_name) {
    if (x == 0 || x == window_dimensions.size()) {
      return Status::OK();
    } else {
      return InvalidArgument(
          "%s", absl::StrCat(
                    "Window has different number of window dimensions than of ",
                    x_name,
                    "\nNumber of window dimensions: ", window_dimensions.size(),
                    "\nNumber of ", x_name, ": ", x, "\n"));
    }
  };
  TF_RETURN_IF_ERROR(verify_size(window_strides.size(), "window strides"));
  TF_RETURN_IF_ERROR(verify_size(padding.size(), "padding entries"));
  TF_RETURN_IF_ERROR(verify_size(lhs_dilation.size(), "lhs dilation factors"));
  TF_RETURN_IF_ERROR(verify_size(rhs_dilation.size(), "rhs dilation factors"));

  Window window;
  for (size_t i = 0; i < window_dimensions.size(); i++) {
    auto dim = window.add_dimensions();
    dim->set_size(window_dimensions[i]);
    if (!window_strides.empty()) {
      dim->set_stride(window_strides[i]);
    } else {
      dim->set_stride(1);
    }
    if (!padding.empty()) {
      dim->set_padding_low(padding[i].first);
      dim->set_padding_high(padding[i].second);
    } else {
      dim->set_padding_low(0);
      dim->set_padding_high(0);
    }
    if (!lhs_dilation.empty()) {
      dim->set_base_dilation(lhs_dilation[i]);
    } else {
      dim->set_base_dilation(1);
    }
    if (!rhs_dilation.empty()) {
      dim->set_window_dilation(rhs_dilation[i]);
    } else {
      dim->set_window_dilation(1);
    }
    dim->set_window_reversal(false);
  }
  return window;
}

XlaOp XlaBuilder::Fft(const XlaOp& operand, const FftType fft_type,
                      const absl::Span<const int64> fft_length) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferFftShape(operand_shape, fft_type, fft_length));

    instr.set_fft_type(fft_type);
    for (int64 i : fft_length) {
      instr.add_fft_length(i);
    }

    return AddInstruction(std::move(instr), HloOpcode::kFft, {operand});
  });
}

XlaOp XlaBuilder::Infeed(const Shape& shape, const string& config) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    if (!LayoutUtil::HasLayout(shape)) {
      return InvalidArgument("Given shape to Infeed must have a layout");
    }
    const Shape infeed_instruction_shape =
        ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeTokenShape()});
    *instr.mutable_shape() = infeed_instruction_shape;
    instr.set_infeed_config(config);

    if (ShapeUtil::IsArray(shape) && sharding() &&
        sharding()->type() == OpSharding::Type::OpSharding_Type_OTHER) {
      // TODO(b/110793772): Support tiled array-shaped infeeds.
      return InvalidArgument(
          "Tiled sharding is not yet supported for array-shaped infeeds");
    }

    if (sharding() &&
        sharding()->type() == OpSharding::Type::OpSharding_Type_REPLICATED) {
      return InvalidArgument(
          "Replicated sharding is not yet supported for infeeds");
    }

    // Infeed takes a single token operand. Generate the token to pass to the
    // infeed.
    XlaOp token;
    auto make_token = [&]() {
      HloInstructionProto token_instr;
      *token_instr.mutable_shape() = ShapeUtil::MakeTokenShape();
      return AddInstruction(std::move(token_instr), HloOpcode::kAfterAll, {});
    };
    if (sharding()) {
      // Arbitrarily assign token to device 0.
      OpSharding sharding = sharding_builder::AssignDevice(0);
      XlaScopedShardingAssignment scoped_sharding(this, sharding);
      TF_ASSIGN_OR_RETURN(token, make_token());
    } else {
      TF_ASSIGN_OR_RETURN(token, make_token());
    }

    // The sharding is set by the client according to the data tuple shape.
    // However, the shape of the infeed instruction is a tuple containing the
    // data and a token. For tuple sharding type, the sharding must be changed
    // to accommodate the token.
    XlaOp infeed;
    if (sharding() &&
        sharding()->type() == OpSharding::Type::OpSharding_Type_TUPLE) {
      // TODO(b/80000000): Remove this when clients have been updated to handle
      // tokens.
      OpSharding infeed_instruction_sharding = *sharding();
      // Arbitrarily assign the token to device 0.
      *infeed_instruction_sharding.add_tuple_shardings() =
          sharding_builder::AssignDevice(0);
      XlaScopedShardingAssignment scoped_sharding(this,
                                                  infeed_instruction_sharding);
      TF_ASSIGN_OR_RETURN(infeed, AddInstruction(std::move(instr),
                                                 HloOpcode::kInfeed, {token}));
    } else {
      TF_ASSIGN_OR_RETURN(infeed, AddInstruction(std::move(instr),
                                                 HloOpcode::kInfeed, {token}));
    }

    // The infeed instruction produces a tuple of the infed data and a token
    // type. Return XLA op containing the data.
    // TODO(b/80000000): Remove this when clients have been updated to handle
    // tokens.
    HloInstructionProto infeed_data;
    *infeed_data.mutable_shape() = shape;
    infeed_data.set_tuple_index(0);
    return AddInstruction(std::move(infeed_data), HloOpcode::kGetTupleElement,
                          {infeed});
  });
}

XlaOp XlaBuilder::InfeedWithToken(const XlaOp& token, const Shape& shape,
                                  const string& config) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    if (!LayoutUtil::HasLayout(shape)) {
      return InvalidArgument("Given shape to Infeed must have a layout");
    }
    const Shape infeed_instruction_shape =
        ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeTokenShape()});
    *instr.mutable_shape() = infeed_instruction_shape;
    instr.set_infeed_config(config);

    if (ShapeUtil::IsArray(shape) && sharding() &&
        sharding()->type() == OpSharding::Type::OpSharding_Type_OTHER) {
      // TODO(b/110793772): Support tiled array-shaped infeeds.
      return InvalidArgument(
          "Tiled sharding is not yet supported for array-shaped infeeds");
    }

    if (sharding() &&
        sharding()->type() == OpSharding::Type::OpSharding_Type_REPLICATED) {
      return InvalidArgument(
          "Replicated sharding is not yet supported for infeeds");
    }

    return AddInstruction(std::move(instr), HloOpcode::kInfeed, {token});
  });
}

void XlaBuilder::Outfeed(const XlaOp& operand, const Shape& shape_with_layout,
                         const string& outfeed_config) {
  ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    *instr.mutable_shape() = ShapeUtil::MakeTokenShape();

    // Check and set outfeed shape.
    if (!LayoutUtil::HasLayout(shape_with_layout)) {
      return InvalidArgument("Given shape to Outfeed must have a layout");
    }
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    if (!ShapeUtil::Compatible(operand_shape, shape_with_layout)) {
      return InvalidArgument(
          "Outfeed shape %s must be compatible with operand shape %s",
          ShapeUtil::HumanStringWithLayout(shape_with_layout),
          ShapeUtil::HumanStringWithLayout(operand_shape));
    }
    *instr.mutable_outfeed_shape() = shape_with_layout;

    instr.set_outfeed_config(outfeed_config);

    // Outfeed takes a token as its second operand. Generate the token to pass
    // to the outfeed.
    HloInstructionProto token_instr;
    *token_instr.mutable_shape() = ShapeUtil::MakeTokenShape();
    TF_ASSIGN_OR_RETURN(XlaOp token, AddInstruction(std::move(token_instr),
                                                    HloOpcode::kAfterAll, {}));

    TF_RETURN_IF_ERROR(
        AddInstruction(std::move(instr), HloOpcode::kOutfeed, {operand, token})
            .status());

    // The outfeed instruction produces a token. However, existing users expect
    // a nil shape (empty tuple). This should only be relevant if the outfeed is
    // the root of a computation.
    // TODO(b/80000000): Remove this when clients have been updated to handle
    // tokens.
    HloInstructionProto tuple_instr;
    *tuple_instr.mutable_shape() = ShapeUtil::MakeNil();

    // The dummy tuple should have no sharding.
    {
      XlaScopedShardingAssignment scoped_sharding(this, OpSharding());
      TF_ASSIGN_OR_RETURN(
          XlaOp empty_tuple,
          AddInstruction(std::move(tuple_instr), HloOpcode::kTuple, {}));
      return empty_tuple;
    }
  });
}

XlaOp XlaBuilder::OutfeedWithToken(const XlaOp& operand, const XlaOp& token,
                                   const Shape& shape_with_layout,
                                   const string& outfeed_config) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    *instr.mutable_shape() = ShapeUtil::MakeTokenShape();

    // Check and set outfeed shape.
    if (!LayoutUtil::HasLayout(shape_with_layout)) {
      return InvalidArgument("Given shape to Outfeed must have a layout");
    }
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    if (!ShapeUtil::Compatible(operand_shape, shape_with_layout)) {
      return InvalidArgument(
          "Outfeed shape %s must be compatible with operand shape %s",
          ShapeUtil::HumanStringWithLayout(shape_with_layout),
          ShapeUtil::HumanStringWithLayout(operand_shape));
    }
    *instr.mutable_outfeed_shape() = shape_with_layout;

    instr.set_outfeed_config(outfeed_config);

    return AddInstruction(std::move(instr), HloOpcode::kOutfeed,
                          {operand, token});
  });
}

XlaOp XlaBuilder::CreateToken() {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = ShapeUtil::MakeTokenShape();
    return AddInstruction(std::move(instr), HloOpcode::kAfterAll);
  });
}

XlaOp XlaBuilder::AfterAll(absl::Span<const XlaOp> tokens) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (tokens.empty()) {
      return InvalidArgument("AfterAll requires at least one operand");
    }
    HloInstructionProto instr;
    *instr.mutable_shape() = ShapeUtil::MakeTokenShape();
    return AddInstruction(std::move(instr), HloOpcode::kAfterAll, tokens);
  });
}

XlaOp XlaBuilder::CustomCall(const string& call_target_name,
                             absl::Span<const XlaOp> operands,
                             const Shape& shape) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    if (absl::StartsWith(call_target_name, "$")) {
      return InvalidArgument(
          "Invalid custom_call_target \"%s\": Call targets that start with '$' "
          "are reserved for internal use.",
          call_target_name);
    }
    *instr.mutable_shape() = shape;
    instr.set_custom_call_target(call_target_name);
    return AddInstruction(std::move(instr), HloOpcode::kCustomCall, operands);
  });
}

XlaOp XlaBuilder::Complex(const XlaOp& real, const XlaOp& imag,
                          absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kComplex, real, imag, broadcast_dimensions);
}

XlaOp XlaBuilder::Conj(const XlaOp& operand) {
  return Complex(Real(operand), Neg(Imag(operand)));
}

XlaOp XlaBuilder::Sub(const XlaOp& lhs, const XlaOp& rhs,
                      absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kSubtract, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Div(const XlaOp& lhs, const XlaOp& rhs,
                      absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kDivide, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Rem(const XlaOp& lhs, const XlaOp& rhs,
                      absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kRemainder, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Max(const XlaOp& lhs, const XlaOp& rhs,
                      absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kMaximum, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Min(const XlaOp& lhs, const XlaOp& rhs,
                      absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kMinimum, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::And(const XlaOp& lhs, const XlaOp& rhs,
                      absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kAnd, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Or(const XlaOp& lhs, const XlaOp& rhs,
                     absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kOr, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Xor(const XlaOp& lhs, const XlaOp& rhs,
                      absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kXor, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Not(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kNot, operand);
}

XlaOp XlaBuilder::ShiftLeft(const XlaOp& lhs, const XlaOp& rhs,
                            absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kShiftLeft, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::ShiftRightArithmetic(
    const XlaOp& lhs, const XlaOp& rhs,
    absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kShiftRightArithmetic, lhs, rhs,
                  broadcast_dimensions);
}

XlaOp XlaBuilder::ShiftRightLogical(
    const XlaOp& lhs, const XlaOp& rhs,
    absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kShiftRightLogical, lhs, rhs,
                  broadcast_dimensions);
}

XlaOp XlaBuilder::Abs(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kAbs, operand);
}

XlaOp XlaBuilder::Atan2(const XlaOp& y, const XlaOp& x,
                        absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kAtan2, y, x, broadcast_dimensions);
}

XlaOp XlaBuilder::Exp(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kExp, operand);
}

XlaOp XlaBuilder::Expm1(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kExpm1, operand);
}

XlaOp XlaBuilder::Floor(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kFloor, operand);
}

XlaOp XlaBuilder::Ceil(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kCeil, operand);
}

XlaOp XlaBuilder::Round(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kRoundNearestAfz, operand);
}

XlaOp XlaBuilder::Log(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kLog, operand);
}

XlaOp XlaBuilder::Log1p(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kLog1p, operand);
}

XlaOp XlaBuilder::Sign(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kSign, operand);
}

XlaOp XlaBuilder::Clz(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kClz, operand);
}

XlaOp XlaBuilder::Cos(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kCos, operand);
}

XlaOp XlaBuilder::Sin(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kSin, operand);
}

XlaOp XlaBuilder::Tanh(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kTanh, operand);
}

XlaOp XlaBuilder::Real(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kReal, operand);
}

XlaOp XlaBuilder::Imag(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kImag, operand);
}

XlaOp XlaBuilder::IsFinite(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kIsFinite, operand);
}

XlaOp XlaBuilder::Transpose(const XlaOp& operand,
                            absl::Span<const int64> permutation) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferTransposeShape(operand_shape, permutation));
    for (int64 dim : permutation) {
      instr.add_dimensions(dim);
    }
    return AddInstruction(std::move(instr), HloOpcode::kTranspose, {operand});
  });
}

XlaOp XlaBuilder::Rev(const XlaOp& operand,
                      absl::Span<const int64> dimensions) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferReverseShape(operand_shape, dimensions));
    for (int64 dim : dimensions) {
      instr.add_dimensions(dim);
    }
    return AddInstruction(std::move(instr), HloOpcode::kReverse, {operand});
  });
}

XlaOp XlaBuilder::Sort(XlaOp keys, absl::optional<XlaOp> values,
                       int64 dimension) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const Shape& keys_shape, GetShape(keys));
    operand_shape_ptrs.push_back(&keys_shape);
    Shape values_shape;
    if (values.has_value()) {
      TF_ASSIGN_OR_RETURN(values_shape, GetShape(*values));
      operand_shape_ptrs.push_back(&values_shape);
    }
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferVariadicOpShape(
                            HloOpcode::kSort, operand_shape_ptrs));
    if (dimension == -1) {
      TF_ASSIGN_OR_RETURN(const Shape& keys_shape, GetShape(keys));
      dimension = ShapeUtil::Rank(keys_shape) - 1;
    }
    instr.add_dimensions(dimension);
    return values.has_value()
               ? AddInstruction(std::move(instr), HloOpcode::kSort,
                                {keys, *values})
               : AddInstruction(std::move(instr), HloOpcode::kSort, {keys});
  });
}

XlaOp XlaBuilder::Pow(const XlaOp& lhs, const XlaOp& rhs,
                      absl::Span<const int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kPower, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::ConvertElementType(const XlaOp& operand,
                                     PrimitiveType new_element_type) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferConvertShape(operand_shape, new_element_type));
    return AddInstruction(std::move(instr), HloOpcode::kConvert, {operand});
  });
}

XlaOp XlaBuilder::BitcastConvertType(const XlaOp& operand,
                                     PrimitiveType new_element_type) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferConvertShape(operand_shape, new_element_type));
    return AddInstruction(std::move(instr), HloOpcode::kBitcastConvert,
                          {operand});
  });
}

XlaOp XlaBuilder::Neg(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kNegate, operand);
}

XlaOp XlaBuilder::Clamp(const XlaOp& min, const XlaOp& operand,
                        const XlaOp& max) {
  return TernaryOp(HloOpcode::kClamp, min, operand, max);
}

XlaOp XlaBuilder::Map(absl::Span<const XlaOp> operands,
                      const XlaComputation& computation,
                      absl::Span<const int64> dimensions,
                      absl::Span<const XlaOp> static_operands) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (!static_operands.empty()) {
      return Unimplemented("static_operands is not supported in Map");
    }

    HloInstructionProto instr;
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes, GetOperandShapes(operands));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(const ProgramShape& called_program_shape,
                        computation.GetProgramShape());
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferMapShape(operand_shape_ptrs, called_program_shape,
                                      dimensions));

    const Shape& output_shape = instr.shape();
    const int64 output_rank = ShapeUtil::Rank(output_shape);
    AddCalledComputation(computation, &instr);
    std::vector<XlaOp> new_operands(operands.begin(), operands.end());
    for (XlaOp& new_operand : new_operands) {
      TF_ASSIGN_OR_RETURN(Shape shape, GetShape(new_operand));
      const int64 rank = ShapeUtil::Rank(shape);
      if (rank != output_rank) {
        TF_ASSIGN_OR_RETURN(new_operand,
                            InDimBroadcast(output_shape, new_operand, {}));
        TF_ASSIGN_OR_RETURN(shape, GetShape(new_operand));
      }
      if (!ShapeUtil::SameDimensions(output_shape, shape)) {
        TF_ASSIGN_OR_RETURN(new_operand,
                            AddBroadcastSequence(output_shape, new_operand));
      }
    }

    return AddInstruction(std::move(instr), HloOpcode::kMap, new_operands);
  });
}

XlaOp XlaBuilder::RngOp(RandomDistribution distribution,
                        absl::Span<const XlaOp> parameters,
                        const Shape& shape) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    // Check the number of parameters per RNG distribution.
    switch (distribution) {
      case RandomDistribution::RNG_NORMAL:
      case RandomDistribution::RNG_UNIFORM:
        if (parameters.size() != 2) {
          return InvalidArgument(
              "RNG distribution (%s) expects 2 parameters, but got %ld",
              RandomDistribution_Name(distribution), parameters.size());
        }
        break;
      default:
        LOG(FATAL) << "unhandled distribution " << distribution;
    }

    TF_RETURN_IF_ERROR(ShapeUtil::ValidateShapeWithOptionalLayout(shape));
    *instr.mutable_shape() = shape;

    instr.set_distribution(distribution);

    return AddInstruction(std::move(instr), HloOpcode::kRng, parameters);
  });
}

XlaOp XlaBuilder::RngNormal(const XlaOp& mu, const XlaOp& sigma,
                            const Shape& shape) {
  return RngOp(RandomDistribution::RNG_NORMAL, {mu, sigma}, shape);
}

XlaOp XlaBuilder::RngUniform(const XlaOp& a, const XlaOp& b,
                             const Shape& shape) {
  return RngOp(RandomDistribution::RNG_UNIFORM, {a, b}, shape);
}

XlaOp XlaBuilder::While(const XlaComputation& condition,
                        const XlaComputation& body, const XlaOp& init) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    // Infer shape.
    TF_ASSIGN_OR_RETURN(const auto& body_program_shape, body.GetProgramShape());
    TF_ASSIGN_OR_RETURN(const auto& condition_program_shape,
                        condition.GetProgramShape());
    TF_ASSIGN_OR_RETURN(const Shape& init_shape, GetShape(init));
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferWhileShape(condition_program_shape,
                                        body_program_shape, init_shape));
    // Body comes before condition computation in the vector.
    AddCalledComputation(body, &instr);
    AddCalledComputation(condition, &instr);
    return AddInstruction(std::move(instr), HloOpcode::kWhile, {init});
  });
}

XlaOp XlaBuilder::Gather(const XlaOp& input, const XlaOp& start_indices,
                         const GatherDimensionNumbers& dimension_numbers,
                         absl::Span<const int64> slice_sizes) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape& input_shape, GetShape(input));
    TF_ASSIGN_OR_RETURN(const Shape& start_indices_shape,
                        GetShape(start_indices));
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferGatherShape(input_shape, start_indices_shape,
                                         dimension_numbers, slice_sizes));

    *instr.mutable_gather_dimension_numbers() = dimension_numbers;
    for (int64 bound : slice_sizes) {
      instr.add_gather_slice_sizes(bound);
    }

    return AddInstruction(std::move(instr), HloOpcode::kGather,
                          {input, start_indices});
  });
}

XlaOp XlaBuilder::Scatter(const XlaOp& input, const XlaOp& scatter_indices,
                          const XlaOp& updates,
                          const XlaComputation& update_computation,
                          const ScatterDimensionNumbers& dimension_numbers) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape& input_shape, GetShape(input));
    TF_ASSIGN_OR_RETURN(const Shape& scatter_indices_shape,
                        GetShape(scatter_indices));
    TF_ASSIGN_OR_RETURN(const Shape& updates_shape, GetShape(updates));
    TF_ASSIGN_OR_RETURN(const ProgramShape& to_apply_shape,
                        update_computation.GetProgramShape());
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferScatterShape(
                            input_shape, scatter_indices_shape, updates_shape,
                            to_apply_shape, dimension_numbers));

    *instr.mutable_scatter_dimension_numbers() = dimension_numbers;

    AddCalledComputation(update_computation, &instr);
    return AddInstruction(std::move(instr), HloOpcode::kScatter,
                          {input, scatter_indices, updates});
  });
}

XlaOp XlaBuilder::Conditional(const XlaOp& predicate, const XlaOp& true_operand,
                              const XlaComputation& true_computation,
                              const XlaOp& false_operand,
                              const XlaComputation& false_computation) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape& predicate_shape, GetShape(predicate));
    TF_ASSIGN_OR_RETURN(const Shape& true_operand_shape,
                        GetShape(true_operand));
    TF_ASSIGN_OR_RETURN(const ProgramShape& true_computation_shape,
                        true_computation.GetProgramShape());
    TF_ASSIGN_OR_RETURN(const Shape& false_operand_shape,
                        GetShape(false_operand));
    TF_ASSIGN_OR_RETURN(const ProgramShape& false_computation_shape,
                        false_computation.GetProgramShape());
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferConditionalShape(
            predicate_shape, true_operand_shape, false_operand_shape,
            true_computation_shape, false_computation_shape));

    // The index of true_computation must be 0 and that of false computation
    // must be 1.
    AddCalledComputation(true_computation, &instr);
    AddCalledComputation(false_computation, &instr);

    return AddInstruction(std::move(instr), HloOpcode::kConditional,
                          {predicate, true_operand, false_operand});
  });
}

XlaOp XlaBuilder::Reduce(const XlaOp& operand, const XlaOp& init_value,
                         const XlaComputation& computation,
                         absl::Span<const int64> dimensions_to_reduce) {
  return Reduce(absl::Span<const XlaOp>({operand}),
                absl::Span<const XlaOp>({init_value}), computation,
                dimensions_to_reduce);
}

XlaOp XlaBuilder::Reduce(absl::Span<const XlaOp> operands,
                         absl::Span<const XlaOp> init_values,
                         const XlaComputation& computation,
                         absl::Span<const int64> dimensions_to_reduce) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const ProgramShape& called_program_shape,
                        computation.GetProgramShape());

    std::vector<XlaOp> all_operands;
    all_operands.insert(all_operands.end(), operands.begin(), operands.end());
    all_operands.insert(all_operands.end(), init_values.begin(),
                        init_values.end());

    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes,
                        GetOperandShapes(all_operands));
    absl::c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                      [](const Shape& shape) { return &shape; });

    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferReduceShape(
            operand_shape_ptrs, dimensions_to_reduce, called_program_shape));

    for (int64 dim : dimensions_to_reduce) {
      instr.add_dimensions(dim);
    }

    AddCalledComputation(computation, &instr);

    return AddInstruction(std::move(instr), HloOpcode::kReduce, all_operands);
  });
}

XlaOp XlaBuilder::ReduceAll(const XlaOp& operand, const XlaOp& init_value,
                            const XlaComputation& computation) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    std::vector<int64> all_dimnos(ShapeUtil::Rank(operand_shape));
    std::iota(all_dimnos.begin(), all_dimnos.end(), 0);
    return Reduce(operand, init_value, computation, all_dimnos);
  });
}

XlaOp XlaBuilder::ReduceWindow(const XlaOp& operand, const XlaOp& init_value,
                               const XlaComputation& computation,
                               absl::Span<const int64> window_dimensions,
                               absl::Span<const int64> window_strides,
                               Padding padding) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_RETURN_IF_ERROR(
        ValidatePaddingValues(AsInt64Slice(operand_shape.dimensions()),
                              window_dimensions, window_strides));

    std::vector<std::pair<int64, int64>> padding_values =
        MakePadding(AsInt64Slice(operand_shape.dimensions()), window_dimensions,
                    window_strides, padding);
    return ReduceWindowWithGeneralPadding(operand, init_value, computation,
                                          window_dimensions, window_strides,
                                          padding_values);
  });
}

XlaOp XlaBuilder::ReduceWindowWithGeneralPadding(
    const XlaOp& operand, const XlaOp& init_value,
    const XlaComputation& computation,
    absl::Span<const int64> window_dimensions,
    absl::Span<const int64> window_strides,
    absl::Span<const std::pair<int64, int64>> padding) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(const Shape& init_shape, GetShape(init_value));
    TF_ASSIGN_OR_RETURN(const ProgramShape& to_apply_shape,
                        computation.GetProgramShape());
    TF_ASSIGN_OR_RETURN(*instr.mutable_window(),
                        MakeWindow(window_dimensions, window_strides, padding,
                                   /*lhs_dilation=*/{}, /*rhs_dilation=*/{}));
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferReduceWindowShape(operand_shape, init_shape,
                                               instr.window(), to_apply_shape));

    AddCalledComputation(computation, &instr);
    return AddInstruction(std::move(instr), HloOpcode::kReduceWindow,
                          {operand, init_value});
  });
}

XlaOp XlaBuilder::BatchNormTraining(const XlaOp& operand, const XlaOp& scale,
                                    const XlaOp& offset, float epsilon,
                                    int64 feature_index) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(const Shape& scale_shape, GetShape(scale));
    TF_ASSIGN_OR_RETURN(const Shape& offset_shape, GetShape(offset));
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferBatchNormTrainingShape(
            operand_shape, scale_shape, offset_shape, feature_index));

    instr.set_epsilon(epsilon);
    instr.set_feature_index(feature_index);

    return AddInstruction(std::move(instr), HloOpcode::kBatchNormTraining,
                          {operand, scale, offset});
  });
}

XlaOp XlaBuilder::BatchNormInference(const XlaOp& operand, const XlaOp& scale,
                                     const XlaOp& offset, const XlaOp& mean,
                                     const XlaOp& variance, float epsilon,
                                     int64 feature_index) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(const Shape& scale_shape, GetShape(scale));
    TF_ASSIGN_OR_RETURN(const Shape& offset_shape, GetShape(offset));
    TF_ASSIGN_OR_RETURN(const Shape& mean_shape, GetShape(mean));
    TF_ASSIGN_OR_RETURN(const Shape& variance_shape, GetShape(variance));
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferBatchNormInferenceShape(
                            operand_shape, scale_shape, offset_shape,
                            mean_shape, variance_shape, feature_index));

    instr.set_epsilon(epsilon);
    instr.set_feature_index(feature_index);

    return AddInstruction(std::move(instr), HloOpcode::kBatchNormInference,
                          {operand, scale, offset, mean, variance});
  });
}

XlaOp XlaBuilder::BatchNormGrad(const XlaOp& operand, const XlaOp& scale,
                                const XlaOp& batch_mean, const XlaOp& batch_var,
                                const XlaOp& grad_output, float epsilon,
                                int64 feature_index) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(const Shape& scale_shape, GetShape(scale));
    TF_ASSIGN_OR_RETURN(const Shape& batch_mean_shape, GetShape(batch_mean));
    TF_ASSIGN_OR_RETURN(const Shape& batch_var_shape, GetShape(batch_var));
    TF_ASSIGN_OR_RETURN(const Shape& grad_output_shape, GetShape(grad_output));
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferBatchNormGradShape(
                            operand_shape, scale_shape, batch_mean_shape,
                            batch_var_shape, grad_output_shape, feature_index));

    instr.set_epsilon(epsilon);
    instr.set_feature_index(feature_index);

    return AddInstruction(std::move(instr), HloOpcode::kBatchNormGrad,
                          {operand, scale, batch_mean, batch_var, grad_output});
  });
}

XlaOp XlaBuilder::CrossReplicaSum(
    const XlaOp& operand, absl::Span<const ReplicaGroup> replica_groups) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape& shape, GetShape(operand));
    const Shape& scalar_shape = ShapeUtil::MakeShape(shape.element_type(), {});
    auto b = CreateSubBuilder("sum");
    b->Add(b->Parameter(/*parameter_number=*/0, scalar_shape, "x"),
           b->Parameter(/*parameter_number=*/1, scalar_shape, "y"));
    TF_ASSIGN_OR_RETURN(auto computation, b->Build());
    return CrossReplicaSum(operand, computation, replica_groups,
                           /*channel_id=*/absl::nullopt);
  });
}

XlaOp XlaBuilder::CrossReplicaSum(
    const XlaOp& operand, const XlaComputation& computation,
    absl::Span<const ReplicaGroup> replica_groups,
    const absl::optional<ChannelHandle>& channel_id) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferCrossReplicaSumShape({&operand_shape}));

    for (const ReplicaGroup& group : replica_groups) {
      *instr.add_replica_groups() = group;
    }

    if (channel_id.has_value()) {
      instr.set_all_reduce_id(channel_id->handle());
    }

    AddCalledComputation(computation, &instr);

    return AddInstruction(std::move(instr), HloOpcode::kCrossReplicaSum,
                          {operand});
  });
}

XlaOp XlaBuilder::AllToAll(const XlaOp& operand, int64 split_dimension,
                           int64 concat_dimension, int64 split_count,
                           const std::vector<ReplicaGroup>& replica_groups) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));

    // The HloInstruction for Alltoall currently only handles the data
    // communication: it accepts N already split parts and scatters them to N
    // cores, and each core gathers the N received parts into a tuple as the
    // output. So here we explicitly split the operand before the hlo alltoall,
    // and concat the tuple elements.
    //
    // First, run shape inference to make sure the shapes are valid.
    TF_RETURN_IF_ERROR(
        ShapeInference::InferAllToAllShape(operand_shape, split_dimension,
                                           concat_dimension, split_count)
            .status());

    // Split into N parts.
    std::vector<XlaOp> slices;
    slices.reserve(split_count);
    const int64 block_size =
        operand_shape.dimensions(split_dimension) / split_count;
    for (int i = 0; i < split_count; i++) {
      slices.push_back(SliceInDim(operand, /*start_index=*/i * block_size,
                                  /*limit_index=*/(i + 1) * block_size,
                                  /*stride=*/1, /*dimno=*/split_dimension));
    }

    // Handle data communication.
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(auto slice_shapes, this->GetOperandShapes(slices));
    std::vector<const Shape*> slice_shape_ptrs;
    absl::c_transform(slice_shapes, std::back_inserter(slice_shape_ptrs),
                      [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferAllToAllTupleShape(slice_shape_ptrs));
    for (const ReplicaGroup& group : replica_groups) {
      *instr.add_replica_groups() = group;
    }
    TF_ASSIGN_OR_RETURN(
        XlaOp alltoall,
        AddInstruction(std::move(instr), HloOpcode::kAllToAll, slices));

    // Concat the N received parts.
    std::vector<XlaOp> received;
    received.reserve(split_count);
    for (int i = 0; i < split_count; i++) {
      received.push_back(this->GetTupleElement(alltoall, i));
    }
    return this->ConcatInDim(received, concat_dimension);
  });
}

XlaOp XlaBuilder::CollectivePermute(
    const XlaOp& operand,
    const std::vector<std::pair<int64, int64>>& source_target_pairs) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferCollectivePermuteShape(operand_shape));

    for (const auto& pair : source_target_pairs) {
      auto* proto_pair = instr.add_source_target_pairs();
      proto_pair->set_source(pair.first);
      proto_pair->set_target(pair.second);
    }

    return AddInstruction(std::move(instr), HloOpcode::kCollectivePermute,
                          {operand});
  });
}

XlaOp XlaBuilder::SelectAndScatter(const XlaOp& operand,
                                   const XlaComputation& select,
                                   absl::Span<const int64> window_dimensions,
                                   absl::Span<const int64> window_strides,
                                   Padding padding, const XlaOp& source,
                                   const XlaOp& init_value,
                                   const XlaComputation& scatter) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    return SelectAndScatterWithGeneralPadding(
        operand, select, window_dimensions, window_strides,
        MakePadding(AsInt64Slice(operand_shape.dimensions()), window_dimensions,
                    window_strides, padding),
        source, init_value, scatter);
  });
}

XlaOp XlaBuilder::SelectAndScatterWithGeneralPadding(
    const XlaOp& operand, const XlaComputation& select,
    absl::Span<const int64> window_dimensions,
    absl::Span<const int64> window_strides,
    absl::Span<const std::pair<int64, int64>> padding, const XlaOp& source,
    const XlaOp& init_value, const XlaComputation& scatter) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(const Shape& source_shape, GetShape(source));
    TF_ASSIGN_OR_RETURN(const Shape& init_shape, GetShape(init_value));
    TF_ASSIGN_OR_RETURN(const ProgramShape& select_shape,
                        select.GetProgramShape());
    TF_ASSIGN_OR_RETURN(const ProgramShape& scatter_shape,
                        scatter.GetProgramShape());
    TF_ASSIGN_OR_RETURN(*instr.mutable_window(),
                        MakeWindow(window_dimensions, window_strides, padding,
                                   /*lhs_dilation=*/{}, /*rhs_dilation=*/{}));
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferSelectAndScatterShape(
                            operand_shape, select_shape, instr.window(),
                            source_shape, init_shape, scatter_shape));

    AddCalledComputation(select, &instr);
    AddCalledComputation(scatter, &instr);

    return AddInstruction(std::move(instr), HloOpcode::kSelectAndScatter,
                          {operand, source, init_value});
  });
}

XlaOp XlaBuilder::ReducePrecision(const XlaOp& operand, const int exponent_bits,
                                  const int mantissa_bits) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferReducePrecisionShape(
                            operand_shape, exponent_bits, mantissa_bits));
    instr.set_exponent_bits(exponent_bits);
    instr.set_mantissa_bits(mantissa_bits);
    return AddInstruction(std::move(instr), HloOpcode::kReducePrecision,
                          {operand});
  });
}

void XlaBuilder::Send(const XlaOp& operand, const ChannelHandle& handle) {
  ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    // Send HLO takes two operands: a data operand and a token. Generate the
    // token to pass into the send.
    // TODO(b/80000000): Remove this when clients have been updated to handle
    // tokens.
    HloInstructionProto token_instr;
    *token_instr.mutable_shape() = ShapeUtil::MakeTokenShape();
    TF_ASSIGN_OR_RETURN(XlaOp token, AddInstruction(std::move(token_instr),
                                                    HloOpcode::kAfterAll, {}));

    return SendWithToken(operand, token, handle);
  });
}

XlaOp XlaBuilder::SendWithToken(const XlaOp& operand, const XlaOp& token,
                                const ChannelHandle& handle) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (handle.type() != ChannelHandle::DEVICE_TO_DEVICE) {
      return InvalidArgument("Send must use a device-to-device channel");
    }

    // Send instruction produces a tuple of {aliased operand, U32 context,
    // token}.
    HloInstructionProto send_instr;
    TF_ASSIGN_OR_RETURN(const Shape& shape, GetShape(operand));
    *send_instr.mutable_shape() = ShapeUtil::MakeTupleShape(
        {shape, ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeTokenShape()});
    send_instr.set_channel_id(handle.handle());
    TF_ASSIGN_OR_RETURN(XlaOp send,
                        AddInstruction(std::move(send_instr), HloOpcode::kSend,
                                       {operand, token}));

    HloInstructionProto send_done_instr;
    *send_done_instr.mutable_shape() = ShapeUtil::MakeTokenShape();
    send_done_instr.set_channel_id(handle.handle());
    return AddInstruction(std::move(send_done_instr), HloOpcode::kSendDone,
                          {send});
  });
}

XlaOp XlaBuilder::Recv(const Shape& shape, const ChannelHandle& handle) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    // Recv HLO takes a single token operand. Generate the token to pass into
    // the Recv and RecvDone instructions.
    // TODO(b/80000000): Remove this when clients have been updated to handle
    // tokens.
    HloInstructionProto token_instr;
    *token_instr.mutable_shape() = ShapeUtil::MakeTokenShape();
    TF_ASSIGN_OR_RETURN(XlaOp token, AddInstruction(std::move(token_instr),
                                                    HloOpcode::kAfterAll, {}));

    XlaOp recv = RecvWithToken(token, shape, handle);

    // The RecvDone instruction produces a tuple of the data and a token
    // type. Return XLA op containing the data.
    // TODO(b/80000000): Remove this when clients have been updated to handle
    // tokens.
    HloInstructionProto recv_data;
    *recv_data.mutable_shape() = shape;
    recv_data.set_tuple_index(0);
    return AddInstruction(std::move(recv_data), HloOpcode::kGetTupleElement,
                          {recv});
  });
}

XlaOp XlaBuilder::RecvWithToken(const XlaOp& token, const Shape& shape,
                                const ChannelHandle& handle) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (handle.type() != ChannelHandle::DEVICE_TO_DEVICE) {
      return InvalidArgument("Recv must use a device-to-device channel");
    }

    // Recv instruction produces a tuple of {receive buffer, U32 context,
    // token}.
    HloInstructionProto recv_instr;
    *recv_instr.mutable_shape() = ShapeUtil::MakeTupleShape(
        {shape, ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeTokenShape()});
    recv_instr.set_channel_id(handle.handle());
    TF_ASSIGN_OR_RETURN(XlaOp recv, AddInstruction(std::move(recv_instr),
                                                   HloOpcode::kRecv, {token}));

    HloInstructionProto recv_done_instr;
    *recv_done_instr.mutable_shape() =
        ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeTokenShape()});
    recv_done_instr.set_channel_id(handle.handle());
    return AddInstruction(std::move(recv_done_instr), HloOpcode::kRecvDone,
                          {recv});
  });
}

XlaOp XlaBuilder::SendToHost(const XlaOp& operand, const XlaOp& token,
                             const Shape& shape_with_layout,
                             const ChannelHandle& handle) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (!LayoutUtil::HasLayout(shape_with_layout)) {
      return InvalidArgument("Shape passed to SendToHost must have a layout");
    }
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    if (!ShapeUtil::Compatible(operand_shape, shape_with_layout)) {
      return InvalidArgument(
          "SendToHost shape %s must be compatible with operand shape %s",
          ShapeUtil::HumanStringWithLayout(shape_with_layout),
          ShapeUtil::HumanStringWithLayout(operand_shape));
    }
    // TODO(b/111544877): Support tuple shapes.
    if (!ShapeUtil::IsArray(operand_shape)) {
      return InvalidArgument("SendToHost only supports array shapes, shape: %s",
                             ShapeUtil::HumanString(operand_shape));
    }

    if (handle.type() != ChannelHandle::DEVICE_TO_HOST) {
      return InvalidArgument("SendToHost must use a device-to-host channel");
    }

    // Send instruction produces a tuple of {aliased operand, U32 context,
    // token}.
    HloInstructionProto send_instr;
    *send_instr.mutable_shape() = ShapeUtil::MakeTupleShape(
        {shape_with_layout, ShapeUtil::MakeShape(U32, {}),
         ShapeUtil::MakeTokenShape()});
    send_instr.set_channel_id(handle.handle());
    send_instr.set_is_host_transfer(true);
    TF_ASSIGN_OR_RETURN(XlaOp send,
                        AddInstruction(std::move(send_instr), HloOpcode::kSend,
                                       {operand, token}));

    HloInstructionProto send_done_instr;
    *send_done_instr.mutable_shape() = ShapeUtil::MakeTokenShape();
    send_done_instr.set_channel_id(handle.handle());
    send_done_instr.set_is_host_transfer(true);
    return AddInstruction(std::move(send_done_instr), HloOpcode::kSendDone,
                          {send});
  });
}

XlaOp XlaBuilder::RecvFromHost(const XlaOp& token, const Shape& shape,
                               const ChannelHandle& handle) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (!LayoutUtil::HasLayout(shape)) {
      return InvalidArgument("Shape passed to RecvFromHost must have a layout");
    }

    // TODO(b/111544877): Support tuple shapes.
    if (!ShapeUtil::IsArray(shape)) {
      return InvalidArgument(
          "RecvFromHost only supports array shapes, shape: %s",
          ShapeUtil::HumanString(shape));
    }

    if (handle.type() != ChannelHandle::HOST_TO_DEVICE) {
      return InvalidArgument("RecvFromHost must use a host-to-device channel");
    }

    // Recv instruction produces a tuple of {receive buffer, U32 context,
    // token}.
    HloInstructionProto recv_instr;
    *recv_instr.mutable_shape() = ShapeUtil::MakeTupleShape(
        {shape, ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeTokenShape()});
    recv_instr.set_channel_id(handle.handle());
    recv_instr.set_is_host_transfer(true);
    TF_ASSIGN_OR_RETURN(XlaOp recv, AddInstruction(std::move(recv_instr),
                                                   HloOpcode::kRecv, {token}));

    HloInstructionProto recv_done_instr;
    *recv_done_instr.mutable_shape() =
        ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeTokenShape()});
    recv_done_instr.set_channel_id(handle.handle());
    recv_done_instr.set_is_host_transfer(true);
    return AddInstruction(std::move(recv_done_instr), HloOpcode::kRecvDone,
                          {recv});
  });
}

StatusOr<bool> XlaBuilder::IsConstant(const XlaOp& operand) const {
  TF_RETURN_IF_ERROR(first_error_);

  // Verify that the handle is valid.
  TF_RETURN_IF_ERROR(LookUpInstruction(operand).status());

  bool is_constant = true;
  std::set<int64> visited;
  IsConstantVisitor(operand.handle(), &visited, &is_constant);
  return is_constant;
}

StatusOr<XlaComputation> XlaBuilder::BuildConstantSubGraph(
    const XlaOp& root_op) const {
  TF_ASSIGN_OR_RETURN(bool is_constant, IsConstant(root_op));
  if (!is_constant) {
    auto op_status = LookUpInstruction(root_op);
    string op_string =
        op_status.ok() ? op_status.ValueOrDie()->name() : "<unknown operation>";
    return InvalidArgument(
        "Operand to BuildConstantSubGraph depends on a parameter.\n\n"
        "  op requested for constant subgraph: %s\n\n"
        "This is an internal error that typically happens when the XLA user "
        "(e.g. TensorFlow) is attempting to determine a value that must be a "
        "compile-time constant (e.g. an array dimension) but it is not capable "
        "of being evaluated at XLA compile time.\n\n"
        "Please file a usability bug with the framework being used (e.g. "
        "TensorFlow).",
        op_string);
  }

  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      LookUpInstruction(root_op));

  HloComputationProto entry;
  entry.set_id(GetUniqueId());  // Give the computation a global unique id.
  entry.set_name(StrCat(name_, entry.id(), "_compute_constant"));
  entry.set_root_id(root->id());
  ProgramShape* program_shape = entry.mutable_program_shape();
  *program_shape->mutable_result() = root->shape();

  // We use std::set to keep the instruction ids in ascending order (which is
  // also a valid denpendency order). The related ops will be added to the
  // subgraph in the same order.
  std::set<int64> related_ops;
  tensorflow::gtl::FlatSet<int64> related_calls;  // Related computations.
  std::queue<int64> worklist;
  worklist.push(root->id());
  related_ops.insert(root->id());
  while (!worklist.empty()) {
    int64 node = worklist.front();
    worklist.pop();
    for (int64 id : instructions_[node].operand_ids()) {
      if (related_ops.insert(id).second) {
        worklist.push(id);
      }
    }
    for (int64 called_id : instructions_[node].called_computation_ids()) {
      related_calls.insert(called_id);
    }
  }

  // Add related ops to the computation.
  for (int64 id : related_ops) {
    auto* instr = entry.add_instructions();
    *instr = instructions_[id];
    // Ensures that the instruction names are unique among the graph.
    const string& new_name =
        StrCat(instr->name(), ".", entry.id(), ".", instr->id());
    instr->set_name(new_name);
  }

  XlaComputation computation(entry.id());
  HloModuleProto* module = computation.mutable_proto();
  module->set_name(entry.name());
  module->set_id(entry.id());
  module->set_entry_computation_name(entry.name());
  module->set_entry_computation_id(entry.id());
  *module->mutable_program_shape() = *program_shape;
  for (auto& e : embedded_) {
    if (related_calls.find(e.second.id()) != related_calls.end()) {
      *module->add_computations() = e.second;
    }
  }
  *module->add_computations() = std::move(entry);

  return std::move(computation);
}

std::unique_ptr<XlaBuilder> XlaBuilder::CreateSubBuilder(
    const string& computation_name) {
  auto sub_builder = absl::make_unique<XlaBuilder>(computation_name);
  sub_builder->parent_builder_ = this;
  sub_builder->die_immediately_on_error_ = this->die_immediately_on_error_;
  return sub_builder;
}

/* static */ ConvolutionDimensionNumbers
XlaBuilder::CreateDefaultConvDimensionNumbers(int num_spatial_dims) {
  ConvolutionDimensionNumbers dimension_numbers;
  dimension_numbers.set_input_batch_dimension(kConvBatchDimension);
  dimension_numbers.set_input_feature_dimension(kConvFeatureDimension);
  dimension_numbers.set_output_batch_dimension(kConvBatchDimension);
  dimension_numbers.set_output_feature_dimension(kConvFeatureDimension);
  dimension_numbers.set_kernel_output_feature_dimension(
      kConvKernelOutputDimension);
  dimension_numbers.set_kernel_input_feature_dimension(
      kConvKernelInputDimension);
  for (int i = 0; i < num_spatial_dims; ++i) {
    dimension_numbers.add_input_spatial_dimensions(i + 2);
    dimension_numbers.add_kernel_spatial_dimensions(i + 2);
    dimension_numbers.add_output_spatial_dimensions(i + 2);
  }
  return dimension_numbers;
}

/* static */ Status XlaBuilder::Validate(
    const ConvolutionDimensionNumbers& dnum) {
  if (dnum.input_spatial_dimensions_size() < 2) {
    return FailedPrecondition("input spacial dimension < 2: %d",
                              dnum.input_spatial_dimensions_size());
  }
  if (dnum.kernel_spatial_dimensions_size() < 2) {
    return FailedPrecondition("kernel spacial dimension < 2: %d",
                              dnum.kernel_spatial_dimensions_size());
  }
  if (dnum.output_spatial_dimensions_size() < 2) {
    return FailedPrecondition("output spacial dimension < 2: %d",
                              dnum.output_spatial_dimensions_size());
  }

  if (std::set<int64>(
          {dnum.input_batch_dimension(), dnum.input_feature_dimension(),
           dnum.input_spatial_dimensions(0), dnum.input_spatial_dimensions(1)})
          .size() != 4) {
    return FailedPrecondition(
        "dimension numbers for the input are not unique: (%d, %d, %d, "
        "%d)",
        dnum.input_batch_dimension(), dnum.input_feature_dimension(),
        dnum.input_spatial_dimensions(0), dnum.input_spatial_dimensions(1));
  }
  if (std::set<int64>({dnum.kernel_output_feature_dimension(),
                       dnum.kernel_input_feature_dimension(),
                       dnum.kernel_spatial_dimensions(0),
                       dnum.kernel_spatial_dimensions(1)})
          .size() != 4) {
    return FailedPrecondition(
        "dimension numbers for the weight are not unique: (%d, %d, %d, "
        "%d)",
        dnum.kernel_output_feature_dimension(),
        dnum.kernel_input_feature_dimension(),
        dnum.kernel_spatial_dimensions(0), dnum.kernel_spatial_dimensions(1));
  }
  if (std::set<int64>({dnum.output_batch_dimension(),
                       dnum.output_feature_dimension(),
                       dnum.output_spatial_dimensions(0),
                       dnum.output_spatial_dimensions(1)})
          .size() != 4) {
    return FailedPrecondition(
        "dimension numbers for the output are not unique: (%d, %d, %d, "
        "%d)",
        dnum.output_batch_dimension(), dnum.output_feature_dimension(),
        dnum.output_spatial_dimensions(0), dnum.output_spatial_dimensions(1));
  }
  return Status::OK();
}

StatusOr<XlaOp> XlaBuilder::AddInstruction(HloInstructionProto&& instr,
                                           HloOpcode opcode,
                                           absl::Span<const XlaOp> operands) {
  TF_RETURN_IF_ERROR(first_error_);

  const int64 handle = instructions_.size();
  instr.set_id(handle);
  instr.set_opcode(HloOpcodeString(opcode));
  if (instr.name().empty()) {
    instr.set_name(StrCat(instr.opcode()));
  }
  for (const auto& operand : operands) {
    if (operand.builder_ == nullptr) {
      return InvalidArgument("invalid XlaOp with handle %d", operand.handle());
    }
    if (operand.builder_ != this) {
      return InvalidArgument("Do not add XlaOp from builder %s to builder %s",
                             operand.builder_->name(), this->name());
    }
    instr.add_operand_ids(operand.handle());
  }

  *instr.mutable_metadata() = metadata_;
  if (sharding_) {
    *instr.mutable_sharding() = *sharding_;
  }

  instructions_.push_back(instr);

  XlaOp op(handle, this);
  return op;
}

void XlaBuilder::AddCalledComputation(const XlaComputation& computation,
                                      HloInstructionProto* instr) {
  instr->add_called_computation_ids(computation.proto().entry_computation_id());
  for (const HloComputationProto& e : computation.proto().computations()) {
    embedded_.insert({e.id(), e});
  }
}

StatusOr<const HloInstructionProto*> XlaBuilder::LookUpInstruction(
    const XlaOp& op) const {
  TF_RETURN_IF_ERROR(first_error_);

  if (op.builder_ == nullptr) {
    return InvalidArgument(
        "invalid XlaOp with handle %d; the builder of this op is freed",
        op.handle());
  }
  if (op.builder_ != this) {
    return InvalidArgument(
        "XlaOp with handle %d is built by builder '%s', but is trying to use "
        "it in builder '%s'",
        op.handle(), op.builder_->name(), this->name());
  }

  if (op.handle() >= instructions_.size() || op.handle() < 0) {
    return InvalidArgument("no XlaOp value %d", op.handle());
  }
  return &instructions_[op.handle()];
}

// Enqueues a "retrieve parameter value" instruction for a parameter that was
// passed to the computation.
XlaOp Parameter(XlaBuilder* builder, int64 parameter_number, const Shape& shape,
                const string& name) {
  return builder->Parameter(parameter_number, shape, name);
}

// Enqueues a constant with the value of the given literal onto the
// computation.
XlaOp ConstantLiteral(XlaBuilder* builder, const LiteralSlice& literal) {
  return builder->ConstantLiteral(literal);
}

XlaOp Broadcast(const XlaOp& operand, absl::Span<const int64> broadcast_sizes) {
  return operand.builder()->Broadcast(operand, broadcast_sizes);
}

XlaOp BroadcastInDim(const XlaOp& operand, const Shape& shape,
                     const absl::Span<const int64> broadcast_dimensions) {
  return operand.builder()->BroadcastInDim(operand, shape,
                                           broadcast_dimensions);
}

XlaOp Pad(const XlaOp& operand, const XlaOp& padding_value,
          const PaddingConfig& padding_config) {
  return operand.builder()->Pad(operand, padding_value, padding_config);
}

XlaOp Reshape(const XlaOp& operand, absl::Span<const int64> dimensions,
              absl::Span<const int64> new_sizes) {
  return operand.builder()->Reshape(operand, dimensions, new_sizes);
}

XlaOp Reshape(const XlaOp& operand, absl::Span<const int64> new_sizes) {
  return operand.builder()->Reshape(operand, new_sizes);
}

XlaOp Collapse(const XlaOp& operand, absl::Span<const int64> dimensions) {
  return operand.builder()->Collapse(operand, dimensions);
}

XlaOp Slice(const XlaOp& operand, absl::Span<const int64> start_indices,
            absl::Span<const int64> limit_indices,
            absl::Span<const int64> strides) {
  return operand.builder()->Slice(operand, start_indices, limit_indices,
                                  strides);
}

XlaOp SliceInDim(const XlaOp& operand, int64 start_index, int64 limit_index,
                 int64 stride, int64 dimno) {
  return operand.builder()->SliceInDim(operand, start_index, limit_index,
                                       stride, dimno);
}

XlaOp DynamicSlice(const XlaOp& operand, const XlaOp& start_indices,
                   absl::Span<const int64> slice_sizes) {
  return operand.builder()->DynamicSlice(operand, start_indices, slice_sizes);
}

XlaOp DynamicUpdateSlice(const XlaOp& operand, const XlaOp& update,
                         const XlaOp& start_indices) {
  return operand.builder()->DynamicUpdateSlice(operand, update, start_indices);
}

XlaOp ConcatInDim(XlaBuilder* builder, absl::Span<const XlaOp> operands,
                  int64 dimension) {
  return builder->ConcatInDim(operands, dimension);
}

void Trace(const string& tag, const XlaOp& operand) {
  return operand.builder()->Trace(tag, operand);
}

XlaOp Select(const XlaOp& pred, const XlaOp& on_true, const XlaOp& on_false) {
  return pred.builder()->Select(pred, on_true, on_false);
}

XlaOp Tuple(XlaBuilder* builder, absl::Span<const XlaOp> elements) {
  return builder->Tuple(elements);
}

XlaOp GetTupleElement(const XlaOp& tuple_data, int64 index) {
  return tuple_data.builder()->GetTupleElement(tuple_data, index);
}

XlaOp Eq(const XlaOp& lhs, const XlaOp& rhs,
         absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->Eq(lhs, rhs, broadcast_dimensions);
}

XlaOp Ne(const XlaOp& lhs, const XlaOp& rhs,
         absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->Ne(lhs, rhs, broadcast_dimensions);
}

XlaOp Ge(const XlaOp& lhs, const XlaOp& rhs,
         absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->Ge(lhs, rhs, broadcast_dimensions);
}

XlaOp Gt(const XlaOp& lhs, const XlaOp& rhs,
         absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->Gt(lhs, rhs, broadcast_dimensions);
}

XlaOp Lt(const XlaOp& lhs, const XlaOp& rhs,
         absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->Lt(lhs, rhs, broadcast_dimensions);
}

XlaOp Le(const XlaOp& lhs, const XlaOp& rhs,
         absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->Le(lhs, rhs, broadcast_dimensions);
}

XlaOp Dot(const XlaOp& lhs, const XlaOp& rhs,
          const PrecisionConfigProto* precision_config_proto) {
  return lhs.builder()->Dot(lhs, rhs, precision_config_proto);
}

XlaOp DotGeneral(const XlaOp& lhs, const XlaOp& rhs,
                 const DotDimensionNumbers& dimension_numbers,
                 const PrecisionConfigProto* precision_config_proto) {
  return lhs.builder()->DotGeneral(lhs, rhs, dimension_numbers,
                                   precision_config_proto);
}

XlaOp Conv(const XlaOp& lhs, const XlaOp& rhs,
           absl::Span<const int64> window_strides, Padding padding,
           int64 feature_group_count,
           const PrecisionConfigProto* precision_config_proto) {
  return lhs.builder()->Conv(lhs, rhs, window_strides, padding,
                             feature_group_count, precision_config_proto);
}

XlaOp ConvWithGeneralPadding(
    const XlaOp& lhs, const XlaOp& rhs, absl::Span<const int64> window_strides,
    absl::Span<const std::pair<int64, int64>> padding,
    int64 feature_group_count,
    const PrecisionConfigProto* precision_config_proto) {
  return lhs.builder()->ConvWithGeneralPadding(lhs, rhs, window_strides,
                                               padding, feature_group_count,
                                               precision_config_proto);
}

XlaOp ConvWithGeneralDimensions(
    const XlaOp& lhs, const XlaOp& rhs, absl::Span<const int64> window_strides,
    Padding padding, const ConvolutionDimensionNumbers& dimension_numbers,
    int64 feature_group_count,
    const PrecisionConfigProto* precision_config_proto) {
  return lhs.builder()->ConvWithGeneralDimensions(
      lhs, rhs, window_strides, padding, dimension_numbers, feature_group_count,
      precision_config_proto);
}

XlaOp ConvGeneral(const XlaOp& lhs, const XlaOp& rhs,
                  absl::Span<const int64> window_strides,
                  absl::Span<const std::pair<int64, int64>> padding,
                  const ConvolutionDimensionNumbers& dimension_numbers,
                  int64 feature_group_count,
                  const PrecisionConfigProto* precision_config_proto) {
  return lhs.builder()->ConvGeneral(lhs, rhs, window_strides, padding,
                                    dimension_numbers, feature_group_count,
                                    precision_config_proto);
}

XlaOp ConvGeneralDilated(const XlaOp& lhs, const XlaOp& rhs,
                         absl::Span<const int64> window_strides,
                         absl::Span<const std::pair<int64, int64>> padding,
                         absl::Span<const int64> lhs_dilation,
                         absl::Span<const int64> rhs_dilation,
                         const ConvolutionDimensionNumbers& dimension_numbers,
                         int64 feature_group_count,
                         const PrecisionConfigProto* precision_config_proto) {
  return lhs.builder()->ConvGeneralDilated(
      lhs, rhs, window_strides, padding, lhs_dilation, rhs_dilation,
      dimension_numbers, feature_group_count, precision_config_proto);
}

XlaOp Fft(const XlaOp& operand, FftType fft_type,
          absl::Span<const int64> fft_length) {
  return operand.builder()->Fft(operand, fft_type, fft_length);
}

XlaOp Infeed(XlaBuilder* builder, const Shape& shape, const string& config) {
  return builder->Infeed(shape, config);
}

void Outfeed(const XlaOp& operand, const Shape& shape_with_layout,
             const string& outfeed_config) {
  return operand.builder()->Outfeed(operand, shape_with_layout, outfeed_config);
}

XlaOp Call(XlaBuilder* builder, const XlaComputation& computation,
           absl::Span<const XlaOp> operands) {
  return builder->Call(computation, operands);
}

XlaOp CustomCall(XlaBuilder* builder, const string& call_target_name,
                 absl::Span<const XlaOp> operands, const Shape& shape) {
  return builder->CustomCall(call_target_name, operands, shape);
}

XlaOp Complex(const XlaOp& real, const XlaOp& imag,
              absl::Span<const int64> broadcast_dimensions) {
  return real.builder()->Complex(real, imag, broadcast_dimensions);
}

XlaOp Conj(const XlaOp& operand) { return operand.builder()->Conj(operand); }

XlaOp Add(const XlaOp& lhs, const XlaOp& rhs,
          absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->Add(lhs, rhs, broadcast_dimensions);
}

XlaOp Sub(const XlaOp& lhs, const XlaOp& rhs,
          absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->Sub(lhs, rhs, broadcast_dimensions);
}

XlaOp Mul(const XlaOp& lhs, const XlaOp& rhs,
          absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->Mul(lhs, rhs, broadcast_dimensions);
}

XlaOp Div(const XlaOp& lhs, const XlaOp& rhs,
          absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->Div(lhs, rhs, broadcast_dimensions);
}

XlaOp Rem(const XlaOp& lhs, const XlaOp& rhs,
          absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->Rem(lhs, rhs, broadcast_dimensions);
}

XlaOp Max(const XlaOp& lhs, const XlaOp& rhs,
          absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->Max(lhs, rhs, broadcast_dimensions);
}

XlaOp Min(const XlaOp& lhs, const XlaOp& rhs,
          absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->Min(lhs, rhs, broadcast_dimensions);
}

XlaOp And(const XlaOp& lhs, const XlaOp& rhs,
          absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->And(lhs, rhs, broadcast_dimensions);
}

XlaOp Or(const XlaOp& lhs, const XlaOp& rhs,
         absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->Or(lhs, rhs, broadcast_dimensions);
}

XlaOp Xor(const XlaOp& lhs, const XlaOp& rhs,
          absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->Xor(lhs, rhs, broadcast_dimensions);
}

XlaOp Not(const XlaOp& operand) { return operand.builder()->Not(operand); }

XlaOp ShiftLeft(const XlaOp& lhs, const XlaOp& rhs,
                absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->ShiftLeft(lhs, rhs, broadcast_dimensions);
}

XlaOp ShiftRightArithmetic(const XlaOp& lhs, const XlaOp& rhs,
                           absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->ShiftRightArithmetic(lhs, rhs, broadcast_dimensions);
}

XlaOp ShiftRightLogical(const XlaOp& lhs, const XlaOp& rhs,
                        absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->ShiftRightLogical(lhs, rhs, broadcast_dimensions);
}

XlaOp Reduce(const XlaOp& operand, const XlaOp& init_value,
             const XlaComputation& computation,
             absl::Span<const int64> dimensions_to_reduce) {
  return operand.builder()->Reduce(operand, init_value, computation,
                                   dimensions_to_reduce);
}

// Reduces several arrays simultaneously among the provided dimensions, given
// "computation" as a reduction operator.
XlaOp Reduce(XlaBuilder* builder, absl::Span<const XlaOp> operands,
             absl::Span<const XlaOp> init_values,
             const XlaComputation& computation,
             absl::Span<const int64> dimensions_to_reduce) {
  return builder->Reduce(operands, init_values, computation,
                         dimensions_to_reduce);
}

XlaOp ReduceAll(const XlaOp& operand, const XlaOp& init_value,
                const XlaComputation& computation) {
  return operand.builder()->ReduceAll(operand, init_value, computation);
}

XlaOp ReduceWindow(const XlaOp& operand, const XlaOp& init_value,
                   const XlaComputation& computation,
                   absl::Span<const int64> window_dimensions,
                   absl::Span<const int64> window_strides, Padding padding) {
  return operand.builder()->ReduceWindow(operand, init_value, computation,
                                         window_dimensions, window_strides,
                                         padding);
}

XlaOp ReduceWindowWithGeneralPadding(
    const XlaOp& operand, const XlaOp& init_value,
    const XlaComputation& computation,
    absl::Span<const int64> window_dimensions,
    absl::Span<const int64> window_strides,
    absl::Span<const std::pair<int64, int64>> padding) {
  return operand.builder()->ReduceWindowWithGeneralPadding(
      operand, init_value, computation, window_dimensions, window_strides,
      padding);
}

XlaOp CrossReplicaSum(const XlaOp& operand,
                      absl::Span<const ReplicaGroup> replica_groups) {
  return operand.builder()->CrossReplicaSum(operand, replica_groups);
}

XlaOp CrossReplicaSum(const XlaOp& operand, const XlaComputation& computation,
                      absl::Span<const ReplicaGroup> replica_groups,
                      const absl::optional<ChannelHandle>& channel_id) {
  return operand.builder()->CrossReplicaSum(operand, computation,
                                            replica_groups, channel_id);
}

XlaOp AllToAll(const XlaOp& operand, int64 split_dimension,
               int64 concat_dimension, int64 split_count,
               const std::vector<ReplicaGroup>& replica_groups) {
  return operand.builder()->AllToAll(operand, split_dimension, concat_dimension,
                                     split_count, replica_groups);
}

XlaOp CollectivePermute(
    const XlaOp& operand,
    const std::vector<std::pair<int64, int64>>& source_target_pairs) {
  return operand.builder()->CollectivePermute(operand, source_target_pairs);
}

XlaOp SelectAndScatter(const XlaOp& operand, const XlaComputation& select,
                       absl::Span<const int64> window_dimensions,
                       absl::Span<const int64> window_strides, Padding padding,
                       const XlaOp& source, const XlaOp& init_value,
                       const XlaComputation& scatter) {
  return operand.builder()->SelectAndScatter(operand, select, window_dimensions,
                                             window_strides, padding, source,
                                             init_value, scatter);
}

XlaOp SelectAndScatterWithGeneralPadding(
    const XlaOp& operand, const XlaComputation& select,
    absl::Span<const int64> window_dimensions,
    absl::Span<const int64> window_strides,
    absl::Span<const std::pair<int64, int64>> padding, const XlaOp& source,
    const XlaOp& init_value, const XlaComputation& scatter) {
  return operand.builder()->SelectAndScatterWithGeneralPadding(
      operand, select, window_dimensions, window_strides, padding, source,
      init_value, scatter);
}

XlaOp Abs(const XlaOp& operand) { return operand.builder()->Abs(operand); }

XlaOp Atan2(const XlaOp& y, const XlaOp& x,
            absl::Span<const int64> broadcast_dimensions) {
  return y.builder()->Atan2(y, x, broadcast_dimensions);
}

XlaOp Exp(const XlaOp& operand) { return operand.builder()->Exp(operand); }

XlaOp Expm1(const XlaOp& operand) { return operand.builder()->Expm1(operand); }

XlaOp Floor(const XlaOp& operand) { return operand.builder()->Floor(operand); }

XlaOp Ceil(const XlaOp& operand) { return operand.builder()->Ceil(operand); }

XlaOp Round(const XlaOp& operand) { return operand.builder()->Round(operand); }

XlaOp Log(const XlaOp& operand) { return operand.builder()->Log(operand); }

XlaOp Log1p(const XlaOp& operand) { return operand.builder()->Log1p(operand); }

XlaOp Sign(const XlaOp& operand) { return operand.builder()->Sign(operand); }

XlaOp Clz(const XlaOp& operand) { return operand.builder()->Clz(operand); }

XlaOp Cos(const XlaOp& operand) { return operand.builder()->Cos(operand); }

XlaOp Sin(const XlaOp& operand) { return operand.builder()->Sin(operand); }

XlaOp Tanh(const XlaOp& operand) { return operand.builder()->Tanh(operand); }

XlaOp Real(const XlaOp& operand) { return operand.builder()->Real(operand); }

XlaOp Imag(const XlaOp& operand) { return operand.builder()->Imag(operand); }

XlaOp Pow(const XlaOp& lhs, const XlaOp& rhs,
          absl::Span<const int64> broadcast_dimensions) {
  return lhs.builder()->Pow(lhs, rhs, broadcast_dimensions);
}

XlaOp IsFinite(const XlaOp& operand) {
  return operand.builder()->IsFinite(operand);
}

XlaOp ConvertElementType(const XlaOp& operand, PrimitiveType new_element_type) {
  return operand.builder()->ConvertElementType(operand, new_element_type);
}

XlaOp BitcastConvertType(const XlaOp& operand, PrimitiveType new_element_type) {
  return operand.builder()->BitcastConvertType(operand, new_element_type);
}

XlaOp Neg(const XlaOp& operand) { return operand.builder()->Neg(operand); }

XlaOp Transpose(const XlaOp& operand, absl::Span<const int64> permutation) {
  return operand.builder()->Transpose(operand, permutation);
}

XlaOp Rev(const XlaOp& operand, absl::Span<const int64> dimensions) {
  return operand.builder()->Rev(operand, dimensions);
}

XlaOp Sort(XlaOp keys, absl::optional<XlaOp> values, int64 dimension) {
  return keys.builder()->Sort(keys, std::move(values), dimension);
}

XlaOp Clamp(const XlaOp& min, const XlaOp& operand, const XlaOp& max) {
  return min.builder()->Clamp(min, operand, max);
}

XlaOp Map(XlaBuilder* builder, absl::Span<const XlaOp> operands,
          const XlaComputation& computation, absl::Span<const int64> dimensions,
          absl::Span<const XlaOp> static_operands) {
  return builder->Map(operands, computation, dimensions, static_operands);
}

XlaOp RngNormal(const XlaOp& mu, const XlaOp& sigma, const Shape& shape) {
  return mu.builder()->RngNormal(mu, sigma, shape);
}

XlaOp RngUniform(const XlaOp& a, const XlaOp& b, const Shape& shape) {
  return a.builder()->RngUniform(a, b, shape);
}

XlaOp While(const XlaComputation& condition, const XlaComputation& body,
            const XlaOp& init) {
  return init.builder()->While(condition, body, init);
}

XlaOp Conditional(const XlaOp& predicate, const XlaOp& true_operand,
                  const XlaComputation& true_computation,
                  const XlaOp& false_operand,
                  const XlaComputation& false_computation) {
  return predicate.builder()->Conditional(predicate, true_operand,
                                          true_computation, false_operand,
                                          false_computation);
}

XlaOp ReducePrecision(const XlaOp& operand, const int exponent_bits,
                      const int mantissa_bits) {
  return operand.builder()->ReducePrecision(operand, exponent_bits,
                                            mantissa_bits);
}

XlaOp Gather(const XlaOp& input, const XlaOp& start_indices,
             const GatherDimensionNumbers& dimension_numbers,
             absl::Span<const int64> slice_sizes) {
  return input.builder()->Gather(input, start_indices, dimension_numbers,
                                 slice_sizes);
}

XlaOp Scatter(const XlaOp& input, const XlaOp& scatter_indices,
              const XlaOp& updates, const XlaComputation& update_computation,
              const ScatterDimensionNumbers& dimension_numbers) {
  return input.builder()->Scatter(input, scatter_indices, updates,
                                  update_computation, dimension_numbers);
}

void Send(const XlaOp& operand, const ChannelHandle& handle) {
  return operand.builder()->Send(operand, handle);
}

XlaOp Recv(XlaBuilder* builder, const Shape& shape,
           const ChannelHandle& handle) {
  return builder->Recv(shape, handle);
}

XlaOp SendWithToken(const XlaOp& operand, const XlaOp& token,
                    const ChannelHandle& handle) {
  return operand.builder()->SendWithToken(operand, token, handle);
}

XlaOp RecvWithToken(const XlaOp& token, const Shape& shape,
                    const ChannelHandle& handle) {
  return token.builder()->RecvWithToken(token, shape, handle);
}

XlaOp SendToHost(const XlaOp& operand, const XlaOp& token,
                 const Shape& shape_with_layout, const ChannelHandle& handle) {
  return operand.builder()->SendToHost(operand, token, shape_with_layout,
                                       handle);
}

XlaOp RecvFromHost(const XlaOp& token, const Shape& shape,
                   const ChannelHandle& handle) {
  return token.builder()->RecvFromHost(token, shape, handle);
}

XlaOp InfeedWithToken(const XlaOp& token, const Shape& shape,
                      const string& config) {
  return token.builder()->InfeedWithToken(token, shape, config);
}

XlaOp OutfeedWithToken(const XlaOp& operand, const XlaOp& token,
                       const Shape& shape_with_layout,
                       const string& outfeed_config) {
  return operand.builder()->OutfeedWithToken(operand, token, shape_with_layout,
                                             outfeed_config);
}

XlaOp CreateToken(XlaBuilder* builder) { return builder->CreateToken(); }

XlaOp AfterAll(XlaBuilder* builder, absl::Span<const XlaOp> tokens) {
  return builder->AfterAll(tokens);
}

XlaOp BatchNormTraining(const XlaOp& operand, const XlaOp& scale,
                        const XlaOp& offset, float epsilon,
                        int64 feature_index) {
  return operand.builder()->BatchNormTraining(operand, scale, offset, epsilon,
                                              feature_index);
}

XlaOp BatchNormInference(const XlaOp& operand, const XlaOp& scale,
                         const XlaOp& offset, const XlaOp& mean,
                         const XlaOp& variance, float epsilon,
                         int64 feature_index) {
  return operand.builder()->BatchNormInference(
      operand, scale, offset, mean, variance, epsilon, feature_index);
}

XlaOp BatchNormGrad(const XlaOp& operand, const XlaOp& scale,
                    const XlaOp& batch_mean, const XlaOp& batch_var,
                    const XlaOp& grad_output, float epsilon,
                    int64 feature_index) {
  return operand.builder()->BatchNormGrad(operand, scale, batch_mean, batch_var,
                                          grad_output, epsilon, feature_index);
}

XlaOp Iota(XlaBuilder* builder, PrimitiveType type, int64 size) {
  return builder->Iota(type, size);
}

XlaOp Iota(XlaBuilder* builder, const Shape& shape, int64 iota_dimension) {
  return builder->Iota(shape, iota_dimension);
}

}  // namespace xla
