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

#include <functional>
#include <numeric>
#include <queue>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/client/sharding_builder.h"
#include "tensorflow/compiler/xla/execution_options_util.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/flatset.h"
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
    case HloOpcode::kSendDone:
    case HloOpcode::kOutfeed:
    case HloOpcode::kTrace:
      return false;
    default:
      return true;
  }
}

}  // namespace

StatusOr<Shape> XlaBuilder::GetShape(const XlaOp& op) const {
  TF_RETURN_IF_ERROR(first_error_);

  TF_ASSIGN_OR_RETURN(auto instr, LookUpInstruction(op));
  return instr->shape();
}

StatusOr<std::vector<Shape>> XlaBuilder::GetOperandShapes(
    tensorflow::gtl::ArraySlice<XlaOp> operands) const {
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

StatusOr<ProgramShape> XlaBuilder::GetProgramShape(int64* root_id) const {
  TF_RETURN_IF_ERROR(first_error_);

  TF_RET_CHECK(root_id != nullptr);

  ProgramShape program_shape;

  // Not all instructions can be roots. Walk backwards from the last added
  // instruction until a valid root is found.
  int64 index = instructions_.size() - 1;
  for (; index >= 0; index--) {
    TF_ASSIGN_OR_RETURN(HloOpcode opcode,
                        StringToHloOpcode(instructions_[index].opcode()));
    if (CanBeRoot(opcode)) {
      break;
    }
  }
  if (index < 0) {
    return FailedPrecondition("no root instruction was found");
  }
  *root_id = instructions_[index].id();
  *program_shape.mutable_result() = instructions_[index].shape();

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
  int64 root;
  return GetProgramShape(&root);
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
    case HloOpcode::kHostCompute:
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
        AddStatus(build_status.status(),
                  tensorflow::strings::StrCat("error from: ", name_)));
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

  HloComputationProto entry;
  entry.set_id(GetUniqueId());  // Give the computation a global unique id.
  entry.set_name(StrCat(name_, entry.id()));  // Ensure that the name is unique.

  {
    int64 root_id;
    TF_ASSIGN_OR_RETURN(*entry.mutable_program_shape(),
                        GetProgramShape(&root_id));
    entry.set_root_id(root_id);
  }

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
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
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

XlaOp XlaBuilder::BinaryOp(
    HloOpcode binop, const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
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
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kAdd, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Mul(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
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

XlaOp XlaBuilder::Call(const XlaComputation& computation,
                       tensorflow::gtl::ArraySlice<XlaOp> operands) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes, GetOperandShapes(operands));
    c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
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
      return InvalidArgument("parameter %lld already registered",
                             parameter_number);
    }
    instr.set_parameter_number(parameter_number);
    instr.set_name(name);
    *instr.mutable_shape() = shape;
    return AddInstruction(std::move(instr), HloOpcode::kParameter);
  });
}

XlaOp XlaBuilder::Broadcast(
    const XlaOp& operand, tensorflow::gtl::ArraySlice<int64> broadcast_sizes) {
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

StatusOr<XlaOp> XlaBuilder::Reshape(const Shape& shape, const XlaOp& operand) {
  TF_RETURN_IF_ERROR(first_error_);

  HloInstructionProto instr;
  *instr.mutable_shape() = shape;
  return AddInstruction(std::move(instr), HloOpcode::kReshape, {operand});
}

XlaOp XlaBuilder::Slice(const XlaOp& operand,
                        tensorflow::gtl::ArraySlice<int64> start_indices,
                        tensorflow::gtl::ArraySlice<int64> limit_indices,
                        tensorflow::gtl::ArraySlice<int64> strides) {
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
                               tensorflow::gtl::ArraySlice<int64> slice_sizes) {
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

XlaOp XlaBuilder::ConcatInDim(tensorflow::gtl::ArraySlice<XlaOp> operands,
                              int64 dimension) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes, GetOperandShapes(operands));
    c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
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
                          tensorflow::gtl::ArraySlice<int64> dimensions,
                          tensorflow::gtl::ArraySlice<int64> new_sizes) {
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
                          tensorflow::gtl::ArraySlice<int64> new_sizes) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, GetShape(operand));
    std::vector<int64> dimensions(shape.dimensions_size());
    std::iota(dimensions.begin(), dimensions.end(), 0);
    return Reshape(operand, dimensions, new_sizes);
  });
}

XlaOp XlaBuilder::Collapse(const XlaOp& operand,
                           tensorflow::gtl::ArraySlice<int64> dimensions) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (dimensions.size() <= 1) {
      // Not collapsing anything, trivially we can return the operand versus
      // enqueueing a trivial reshape.
      return operand;
    }

    // Out-of-order collapse is not supported.
    // Checks that the collapsed dimensions are in order and consecutive.
    for (tensorflow::gtl::ArraySlice<int64>::size_type i = 1;
         i < dimensions.size(); ++i) {
      if (dimensions[i] - 1 != dimensions[i - 1]) {
        return InvalidArgument(
            "Collapsed dimensions are not in consecutive order.");
      }
    }

    // Create a new sizes vector from the old shape, replacing the collapsed
    // dimensions by the product of their sizes.
    TF_ASSIGN_OR_RETURN(const Shape& original_shape, GetShape(operand));

    VLOG(3) << "original shape: " << ShapeUtil::HumanString(original_shape);
    VLOG(3) << "dims to collapse: "
            << tensorflow::str_util::Join(dimensions, ",");

    std::vector<int64> new_sizes;
    for (int i = 0; i < ShapeUtil::Rank(original_shape); ++i) {
      if (i <= dimensions.front() || i > dimensions.back()) {
        new_sizes.push_back(original_shape.dimensions(i));
      } else {
        new_sizes.back() *= original_shape.dimensions(i);
      }
    }

    VLOG(3) << "new sizes: [" << tensorflow::str_util::Join(new_sizes, ",")
            << "]";

    return Reshape(operand, new_sizes);
  });
}

void XlaBuilder::Trace(const string& tag, const XlaOp& operand) {
  ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = ShapeUtil::MakeNil();
    *instr.mutable_literal() = Literal::CreateR1U8(tag)->ToProto();
    return AddInstruction(std::move(instr), HloOpcode::kTrace, {operand});
  });
}

XlaOp XlaBuilder::Select(const XlaOp& pred, const XlaOp& on_true,
                         const XlaOp& on_false) {
  return TernaryOp(HloOpcode::kSelect, pred, on_true, on_false);
}

XlaOp XlaBuilder::Tuple(tensorflow::gtl::ArraySlice<XlaOp> elements) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes, GetOperandShapes(elements));
    c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
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
          ShapeUtil::HumanString(tuple_shape).c_str());
    }
    *instr.mutable_shape() =
        ShapeUtil::GetTupleElementShape(tuple_shape, index);

    instr.set_tuple_index(index);

    return AddInstruction(std::move(instr), HloOpcode::kGetTupleElement,
                          {tuple_data});
  });
}

XlaOp XlaBuilder::Eq(const XlaOp& lhs, const XlaOp& rhs,
                     tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kEq, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Ne(const XlaOp& lhs, const XlaOp& rhs,
                     tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kNe, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Ge(const XlaOp& lhs, const XlaOp& rhs,
                     tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kGe, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Gt(const XlaOp& lhs, const XlaOp& rhs,
                     tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kGt, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Le(const XlaOp& lhs, const XlaOp& rhs,
                     tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kLe, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Lt(const XlaOp& lhs, const XlaOp& rhs,
                     tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kLt, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Dot(const XlaOp& lhs, const XlaOp& rhs) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape& lhs_shape, GetShape(lhs));

    DotDimensionNumbers dimension_numbers;
    dimension_numbers.add_lhs_contracting_dimensions(
        lhs_shape.dimensions_size() == 1 ? 0 : 1);
    dimension_numbers.add_rhs_contracting_dimensions(0);
    return DotGeneral(lhs, rhs, dimension_numbers);
  });
}

XlaOp XlaBuilder::DotGeneral(const XlaOp& lhs, const XlaOp& rhs,
                             const DotDimensionNumbers& dimension_numbers) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& lhs_shape, GetShape(lhs));
    TF_ASSIGN_OR_RETURN(const Shape& rhs_shape, GetShape(rhs));
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferDotOpShape(lhs_shape, rhs_shape,
                                                        dimension_numbers));
    *instr.mutable_dot_dimension_numbers() = dimension_numbers;
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
        ShapeUtil::HumanString(lhs_shape).c_str(),
        ShapeUtil::HumanString(rhs_shape).c_str());
  }
  int num_dims = ShapeUtil::Rank(lhs_shape);
  if (num_dims < 2) {
    return InvalidArgument(
        "Convolution expects argument arrays with >= 3 dimensions. "
        "Got: %s and %s",
        ShapeUtil::HumanString(lhs_shape).c_str(),
        ShapeUtil::HumanString(rhs_shape).c_str());
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
            return InvalidArgument("Convolution %s[%d] is out of bounds: %lld",
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
                       tensorflow::gtl::ArraySlice<int64> window_strides,
                       Padding padding) {
  return ConvWithGeneralDimensions(
      lhs, rhs, window_strides, padding,
      CreateDefaultConvDimensionNumbers(window_strides.size()));
}

XlaOp XlaBuilder::ConvWithGeneralPadding(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding) {
  return ConvGeneral(lhs, rhs, window_strides, padding,
                     CreateDefaultConvDimensionNumbers(window_strides.size()));
}

XlaOp XlaBuilder::ConvWithGeneralDimensions(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding,
    const ConvolutionDimensionNumbers& dimension_numbers) {
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
                       dimension_numbers);
  });
}

XlaOp XlaBuilder::ConvGeneral(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
    const ConvolutionDimensionNumbers& dimension_numbers) {
  return ConvGeneralDilated(lhs, rhs, window_strides, padding, {}, {},
                            dimension_numbers);
}

XlaOp XlaBuilder::ConvGeneralDilated(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
    tensorflow::gtl::ArraySlice<int64> lhs_dilation,
    tensorflow::gtl::ArraySlice<int64> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers) {
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

    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferConvolveShape(lhs_shape, rhs_shape, instr.window(),
                                           dimension_numbers));

    *instr.mutable_convolution_dimension_numbers() = dimension_numbers;

    return AddInstruction(std::move(instr), HloOpcode::kConvolution,
                          {lhs, rhs});
  });
}

StatusOr<Window> XlaBuilder::MakeWindow(
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
    tensorflow::gtl::ArraySlice<int64> lhs_dilation,
    tensorflow::gtl::ArraySlice<int64> rhs_dilation) const {
  const auto verify_size = [&](const size_t x, const char* x_name) {
    if (x == 0 || x == window_dimensions.size()) {
      return Status::OK();
    } else {
      return InvalidArgument(
          "%s", tensorflow::strings::StrCat(
                    "Window has different number of window dimensions than of ",
                    x_name,
                    "\nNumber of window dimensions: ", window_dimensions.size(),
                    "\nNumber of ", x_name, ": ", x, "\n")
                    .c_str());
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
                      const tensorflow::gtl::ArraySlice<int64> fft_length) {
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
      TF_ASSIGN_OR_RETURN(infeed,
                          AddInstruction(std::move(instr), HloOpcode::kInfeed));
    } else {
      TF_ASSIGN_OR_RETURN(infeed,
                          AddInstruction(std::move(instr), HloOpcode::kInfeed));
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
          ShapeUtil::HumanStringWithLayout(shape_with_layout).c_str(),
          ShapeUtil::HumanStringWithLayout(operand_shape).c_str());
    }
    *instr.mutable_outfeed_shape() = shape_with_layout;

    instr.set_outfeed_config(outfeed_config);

    TF_RETURN_IF_ERROR(
        AddInstruction(std::move(instr), HloOpcode::kOutfeed, {operand})
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

XlaOp XlaBuilder::CustomCall(const string& call_target_name,
                             tensorflow::gtl::ArraySlice<XlaOp> operands,
                             const Shape& shape) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    if (tensorflow::str_util::StartsWith(call_target_name, "$")) {
      return InvalidArgument(
          "Invalid custom_call_target \"%s\": Call targets that start with '$' "
          "are reserved for internal use.",
          call_target_name.c_str());
    }
    *instr.mutable_shape() = shape;
    instr.set_custom_call_target(call_target_name);
    return AddInstruction(std::move(instr), HloOpcode::kCustomCall, operands);
  });
}

XlaOp XlaBuilder::HostCompute(tensorflow::gtl::ArraySlice<XlaOp> operands,
                              const string& channel_name,
                              int64 cost_estimate_ns, const Shape& shape) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = shape;
    instr.set_channel_name(channel_name);
    instr.set_cost_estimate_ns(cost_estimate_ns);
    return AddInstruction(std::move(instr), HloOpcode::kHostCompute, operands);
  });
}

XlaOp XlaBuilder::Complex(
    const XlaOp& real, const XlaOp& imag,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kComplex, real, imag, broadcast_dimensions);
}

XlaOp XlaBuilder::Conj(const XlaOp& operand) {
  return Complex(Real(operand), Neg(Imag(operand)));
}

XlaOp XlaBuilder::Sub(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kSubtract, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Div(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kDivide, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Rem(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kRemainder, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Max(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kMaximum, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Min(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kMinimum, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::And(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kAnd, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Or(const XlaOp& lhs, const XlaOp& rhs,
                     tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kOr, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Xor(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kXor, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::Not(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kNot, operand);
}

XlaOp XlaBuilder::ShiftLeft(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kShiftLeft, lhs, rhs, broadcast_dimensions);
}

XlaOp XlaBuilder::ShiftRightArithmetic(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kShiftRightArithmetic, lhs, rhs,
                  broadcast_dimensions);
}

XlaOp XlaBuilder::ShiftRightLogical(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kShiftRightLogical, lhs, rhs,
                  broadcast_dimensions);
}

XlaOp XlaBuilder::Abs(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kAbs, operand);
}

XlaOp XlaBuilder::Atan2(
    const XlaOp& y, const XlaOp& x,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
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
                            tensorflow::gtl::ArraySlice<int64> permutation) {
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
                      tensorflow::gtl::ArraySlice<int64> dimensions) {
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

XlaOp XlaBuilder::Sort(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kSort, operand);
}

XlaOp XlaBuilder::SqrtF32(const XlaOp& operand) {
  return BinaryOp(HloOpcode::kPower, operand, ConstantR0<float>(0.5),
                  /*broadcast_dimensions=*/{});
}

XlaOp XlaBuilder::Pow(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
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

XlaOp XlaBuilder::SquareF32(const XlaOp& operand) {
  return BinaryOp(HloOpcode::kPower, operand, ConstantR0<float>(2.0),
                  /*broadcast_dimensions=*/{});
}

XlaOp XlaBuilder::ReciprocalF32(const XlaOp& operand) {
  return BinaryOp(HloOpcode::kPower, operand, ConstantR0<float>(-1.0),
                  /*broadcast_dimensions=*/{});
}

XlaOp XlaBuilder::Neg(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kNegate, operand);
}

XlaOp XlaBuilder::Clamp(const XlaOp& min, const XlaOp& operand,
                        const XlaOp& max) {
  return TernaryOp(HloOpcode::kClamp, min, operand, max);
}

XlaOp XlaBuilder::Map(tensorflow::gtl::ArraySlice<XlaOp> operands,
                      const XlaComputation& computation,
                      tensorflow::gtl::ArraySlice<int64> dimensions,
                      tensorflow::gtl::ArraySlice<XlaOp> static_operands) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (!static_operands.empty()) {
      return Unimplemented("static_operands is not supported in Map");
    }

    HloInstructionProto instr;

    std::vector<const Shape*> operand_shape_ptrs;
    TF_ASSIGN_OR_RETURN(const auto& operand_shapes, GetOperandShapes(operands));
    c_transform(operand_shapes, std::back_inserter(operand_shape_ptrs),
                [](const Shape& shape) { return &shape; });
    TF_ASSIGN_OR_RETURN(const ProgramShape& called_program_shape,
                        computation.GetProgramShape());
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferMapShape(operand_shape_ptrs, called_program_shape,
                                      dimensions));

    AddCalledComputation(computation, &instr);

    return AddInstruction(std::move(instr), HloOpcode::kMap, operands);
  });
}

XlaOp XlaBuilder::RngOp(RandomDistribution distribution,
                        tensorflow::gtl::ArraySlice<XlaOp> parameters,
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
              RandomDistribution_Name(distribution).c_str(), parameters.size());
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

XlaOp XlaBuilder::Gather(const XlaOp& input, const XlaOp& gather_indices,
                         const GatherDimensionNumbers& dimension_numbers,
                         tensorflow::gtl::ArraySlice<int64> window_bounds) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape& input_shape, GetShape(input));
    TF_ASSIGN_OR_RETURN(const Shape& gather_indices_shape,
                        GetShape(gather_indices));
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferGatherShape(input_shape, gather_indices_shape,
                                         dimension_numbers, window_bounds));

    *instr.mutable_gather_dimension_numbers() = dimension_numbers;
    for (int64 bound : window_bounds) {
      instr.add_gather_window_bounds(bound);
    }

    return AddInstruction(std::move(instr), HloOpcode::kGather,
                          {input, gather_indices});
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

XlaOp XlaBuilder::Reduce(
    const XlaOp& operand, const XlaOp& init_value,
    const XlaComputation& computation,
    tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(const Shape& init_shape, GetShape(init_value));
    TF_ASSIGN_OR_RETURN(const ProgramShape& called_program_shape,
                        computation.GetProgramShape());
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferReduceShape(
                            operand_shape, init_shape, dimensions_to_reduce,
                            called_program_shape));

    for (int64 dim : dimensions_to_reduce) {
      instr.add_dimensions(dim);
    }

    AddCalledComputation(computation, &instr);

    return AddInstruction(std::move(instr), HloOpcode::kReduce,
                          {operand, init_value});
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

XlaOp XlaBuilder::ReduceWindow(
    const XlaOp& operand, const XlaOp& init_value,
    const XlaComputation& computation,
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding) {
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
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding) {
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
    const XlaOp& operand,
    tensorflow::gtl::ArraySlice<int64> replica_group_ids) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape& shape, GetShape(operand));
    const Shape& scalar_shape = ShapeUtil::MakeShape(shape.element_type(), {});
    auto b = CreateSubBuilder("sum");
    b->Add(b->Parameter(/*parameter_number=*/0, scalar_shape, "x"),
           b->Parameter(/*parameter_number=*/1, scalar_shape, "y"));
    TF_ASSIGN_OR_RETURN(auto computation, b->Build());
    return CrossReplicaSum(operand, computation, replica_group_ids,
                           /*channel_id=*/tensorflow::gtl::nullopt);
  });
}

XlaOp XlaBuilder::CrossReplicaSum(
    const XlaOp& operand, const XlaComputation& computation,
    tensorflow::gtl::ArraySlice<int64> replica_group_ids,
    const tensorflow::gtl::optional<ChannelHandle>& channel_id) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    if (channel_id.has_value()) {
      return Unimplemented("channel_id is not supported in AllReduce");
    }

    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, GetShape(operand));
    TF_ASSIGN_OR_RETURN(
        *instr.mutable_shape(),
        ShapeInference::InferCrossReplicaSumShape({&operand_shape}));
    for (int64 replica_group_id : replica_group_ids) {
      instr.add_replica_group_ids(replica_group_id);
    }

    AddCalledComputation(computation, &instr);

    return AddInstruction(std::move(instr), HloOpcode::kCrossReplicaSum,
                          {operand});
  });
}

XlaOp XlaBuilder::SelectAndScatter(
    const XlaOp& operand, const XlaComputation& select,
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding,
    const XlaOp& source, const XlaOp& init_value,
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
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
    const XlaOp& source, const XlaOp& init_value,
    const XlaComputation& scatter) {
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
    HloInstructionProto instr;

    // Send instruction produces a tuple of {aliased operand, U32 context}.
    TF_ASSIGN_OR_RETURN(const Shape& shape, GetShape(operand));
    *instr.mutable_shape() =
        ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeShape(U32, {})});
    instr.set_channel_id(handle.handle());
    TF_ASSIGN_OR_RETURN(
        XlaOp send,
        AddInstruction(std::move(instr), HloOpcode::kSend, {operand}));

    HloInstructionProto send_done_instr;
    *send_done_instr.mutable_shape() = ShapeUtil::MakeNil();
    send_done_instr.set_channel_id(handle.handle());
    return AddInstruction(std::move(send_done_instr), HloOpcode::kSendDone,
                          {send});
  });
}

XlaOp XlaBuilder::Recv(const Shape& shape, const ChannelHandle& handle) {
  return ReportErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;

    // Recv instruction produces a tuple of {receive buffer, U32 context}.
    *instr.mutable_shape() =
        ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeShape(U32, {})});
    instr.set_channel_id(handle.handle());
    TF_ASSIGN_OR_RETURN(XlaOp recv,
                        AddInstruction(std::move(instr), HloOpcode::kRecv, {}));

    HloInstructionProto recv_done_instr;
    *recv_done_instr.mutable_shape() = shape;
    recv_done_instr.set_channel_id(handle.handle());
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
        op_string.c_str());
  }

  TF_ASSIGN_OR_RETURN(const HloInstructionProto* root,
                      LookUpInstruction(root_op));
  TF_ASSIGN_OR_RETURN(HloOpcode opcode, StringToHloOpcode(root->opcode()));
  if (!CanBeRoot(opcode)) {
    return InvalidArgument("the operand with opcode %s cannot be root",
                           root->opcode().c_str());
  }

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
  auto sub_builder = MakeUnique<XlaBuilder>(computation_name);
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
        "dimension numbers for the input are not unique: (%lld, %lld, %lld, "
        "%lld)",
        dnum.input_batch_dimension(), dnum.input_feature_dimension(),
        dnum.input_spatial_dimensions(0), dnum.input_spatial_dimensions(1));
  }
  if (std::set<int64>({dnum.kernel_output_feature_dimension(),
                       dnum.kernel_input_feature_dimension(),
                       dnum.kernel_spatial_dimensions(0),
                       dnum.kernel_spatial_dimensions(1)})
          .size() != 4) {
    return FailedPrecondition(
        "dimension numbers for the weight are not unique: (%lld, %lld, %lld, "
        "%lld)",
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
        "dimension numbers for the output are not unique: (%lld, %lld, %lld, "
        "%lld)",
        dnum.output_batch_dimension(), dnum.output_feature_dimension(),
        dnum.output_spatial_dimensions(0), dnum.output_spatial_dimensions(1));
  }
  return Status::OK();
}

StatusOr<XlaOp> XlaBuilder::AddInstruction(
    HloInstructionProto&& instr, HloOpcode opcode,
    tensorflow::gtl::ArraySlice<XlaOp> operands) {
  TF_RETURN_IF_ERROR(first_error_);

  const int64 handle = instructions_.size();
  instr.set_id(handle);
  instr.set_opcode(HloOpcodeString(opcode));
  if (instr.name().empty()) {
    instr.set_name(StrCat(instr.opcode()));
  }
  for (const auto& operand : operands) {
    if (operand.builder_ == nullptr) {
      return InvalidArgument("invalid XlaOp with handle %lld",
                             operand.handle());
    }
    if (operand.builder_ != this) {
      return InvalidArgument("Do not add XlaOp from builder %s to builder %s",
                             operand.builder_->name().c_str(),
                             this->name().c_str());
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
        "invalid XlaOp with handle %lld; the builder of this op is freed",
        op.handle());
  }
  if (op.builder_ != this) {
    return InvalidArgument(
        "XlaOp with handle %lld is built by builder '%s', but is trying to use "
        "it in builder '%s'",
        op.handle(), op.builder_->name().c_str(), this->name().c_str());
  }

  if (op.handle() >= instructions_.size() || op.handle() < 0) {
    return InvalidArgument("no XlaOp value %lld", op.handle());
  }
  return &instructions_[op.handle()];
}

}  // namespace xla
