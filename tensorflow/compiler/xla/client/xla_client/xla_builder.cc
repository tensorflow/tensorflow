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
    case HloOpcode::kSendDone:
    case HloOpcode::kOutfeed:
    case HloOpcode::kTrace:
      return false;
    default:
      return true;
  }
}

StatusOr<std::vector<Shape>> GetOperandShapes(
    tensorflow::gtl::ArraySlice<XlaOp> operands) {
  std::vector<Shape> operand_shapes;
  for (const XlaOp& operand : operands) {
    TF_ASSIGN_OR_RETURN(const Shape& shape, operand.GetShape());
    operand_shapes.push_back(shape);
  }
  return operand_shapes;
}

}  // namespace

StatusOr<Shape> XlaBuilder::GetShape(const XlaOp& op) const {
  TF_RETURN_IF_ERROR(first_error_);

  TF_ASSIGN_OR_RETURN(auto instr, LookUpInstruction(op));
  return instr->shape();
}

StatusOr<Shape> XlaOp::GetShape() const {
  if (builder_ == nullptr) {
    return InvalidArgument(
        "cannot GetShape for an invalid XlaOp with handle %lld", handle());
  }
  return builder_->GetShape(*this);
}

XlaBuilder::XlaBuilder(const string& computation_name)
    : name_(computation_name), unique_id_(GetUniqueId()) {}

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

XlaOp XlaBuilder::NoteErrorOrReturn(
    const std::function<StatusOr<XlaOp>()>& op_creator) {
  if (!first_error_.ok()) {
    return {};
  }
  auto op = op_creator();
  if (!op.ok()) {
    NoteError(op.status());
    return {};
  }
  return op.ConsumeValueOrDie();
}

StatusOr<ProgramShape> XlaBuilder::GetProgramShape(int64* root_id) {
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

StatusOr<ProgramShape> XlaBuilder::GetProgramShape() {
  int64 root_id;
  return GetProgramShape(&root_id);
}

XlaComputation XlaBuilder::BuildAndNoteError() {
  DCHECK(parent_builder_ != nullptr);
  auto build_status = Build();
  if (!build_status.ok()) {
    parent_builder_->NoteError(
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

  {
    int64 root_id;
    ProgramShape program_shape;
    TF_ASSIGN_OR_RETURN(program_shape, GetProgramShape(&root_id));
    entry.mutable_program_shape()->Swap(&program_shape);
    entry.set_root_id(root_id);
  }

  for (auto& instruction : instructions_) {
    entry.add_instructions()->Swap(&instruction);
  }

  entry.set_id(unique_id_);
  entry.set_name(StrCat(name_, entry.id()));  // Ensure that the name is unique.
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

  TF_ASSIGN_OR_RETURN(const Shape& operand_shape, operand.GetShape());

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
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, operand.GetShape());
    TF_ASSIGN_OR_RETURN(*instr.mutable_shape(),
                        ShapeInference::InferUnaryOpShape(unop, operand_shape));
    return AddInstruction(std::move(instr), unop, {operand});
  });
}

XlaOp XlaBuilder::BinaryOp(
    HloOpcode binop, const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& lhs_shape, lhs.GetShape());
    TF_ASSIGN_OR_RETURN(const Shape& rhs_shape, rhs.GetShape());
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

    TF_ASSIGN_OR_RETURN(Shape updated_lhs_shape, updated_lhs.GetShape());
    if (!ShapeUtil::SameDimensions(instr.shape(), updated_lhs_shape)) {
      TF_ASSIGN_OR_RETURN(updated_lhs,
                          AddBroadcastSequence(instr.shape(), updated_lhs));
    }
    TF_ASSIGN_OR_RETURN(Shape updated_rhs_shape, updated_rhs.GetShape());
    if (!ShapeUtil::SameDimensions(instr.shape(), updated_rhs_shape)) {
      TF_ASSIGN_OR_RETURN(updated_rhs,
                          AddBroadcastSequence(instr.shape(), updated_rhs));
    }

    return AddInstruction(std::move(instr), binop, {updated_lhs, updated_rhs});
  });
}

XlaOp XlaBuilder::TernaryOp(HloOpcode triop, const XlaOp& lhs, const XlaOp& rhs,
                            const XlaOp& ehs) {
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& lhs_shape, lhs.GetShape());
    TF_ASSIGN_OR_RETURN(const Shape& rhs_shape, rhs.GetShape());
    TF_ASSIGN_OR_RETURN(const Shape& ehs_shape, ehs.GetShape());
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

XlaOp XlaBuilder::ConstantLiteral(const Literal& literal) {
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    *instr.mutable_shape() = literal.shape();
    *instr.mutable_literal() = literal.ToProto();
    return AddInstruction(std::move(instr), HloOpcode::kConstant);
  });
}

XlaOp XlaBuilder::Call(const XlaComputation& computation,
                       tensorflow::gtl::ArraySlice<XlaOp> operands) {
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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
  });
}

XlaOp XlaBuilder::Broadcast(
    const XlaOp& operand, tensorflow::gtl::ArraySlice<int64> broadcast_sizes) {
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, operand.GetShape());
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
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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
  return UnimplementedOp();
}

XlaOp XlaBuilder::DynamicSlice(const XlaOp& operand, const XlaOp& start_indices,
                               tensorflow::gtl::ArraySlice<int64> slice_sizes) {
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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
  return UnimplementedOp();
}

XlaOp XlaBuilder::Reshape(const XlaOp& operand,
                          tensorflow::gtl::ArraySlice<int64> dimensions,
                          tensorflow::gtl::ArraySlice<int64> new_sizes) {
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, operand.GetShape());
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
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, operand.GetShape());
    std::vector<int64> dimensions(shape.dimensions_size());
    std::iota(dimensions.begin(), dimensions.end(), 0);
    return Reshape(operand, dimensions, new_sizes);
  });
}

XlaOp XlaBuilder::Collapse(const XlaOp& operand,
                           tensorflow::gtl::ArraySlice<int64> dimensions) {
  return UnimplementedOp();
}

void XlaBuilder::Trace(const string& tag, const XlaOp& operand) {
  NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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

XlaOp XlaBuilder::Conv(const XlaOp& lhs, const XlaOp& rhs,
                       tensorflow::gtl::ArraySlice<int64> window_strides,
                       Padding padding) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::ConvWithGeneralPadding(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::ConvWithGeneralDimensions(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding,
    const ConvolutionDimensionNumbers& dimension_numbers) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::ConvGeneral(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
    const ConvolutionDimensionNumbers& dimension_numbers) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::ConvGeneralDilated(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
    tensorflow::gtl::ArraySlice<int64> lhs_dilation,
    tensorflow::gtl::ArraySlice<int64> rhs_dilation,
    const ConvolutionDimensionNumbers& dimension_numbers) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Fft(const XlaOp& operand, const FftType fft_type,
                      const tensorflow::gtl::ArraySlice<int64> fft_length) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Infeed(const Shape& shape, const string& config) {
  return UnimplementedOp();
}

void XlaBuilder::Outfeed(const XlaOp& operand, const Shape& shape_with_layout,
                         const string& outfeed_config) {
  UnimplementedOp();
}

XlaOp XlaBuilder::CustomCall(const string& call_target_name,
                             tensorflow::gtl::ArraySlice<XlaOp> operands,
                             const Shape& shape) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::HostCompute(tensorflow::gtl::ArraySlice<XlaOp> operands,
                              const string& channel_name,
                              int64 cost_estimate_ns, const Shape& shape) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Complex(
    const XlaOp& real, const XlaOp& imag,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return BinaryOp(HloOpcode::kComplex, real, imag, broadcast_dimensions);
}

XlaOp XlaBuilder::Conj(const XlaOp& operand) { return UnimplementedOp(); }

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

// TODO(b/65209188): Create a dedicated lowering for Xor.
XlaOp XlaBuilder::Xor(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return Or(And(Not(lhs), rhs, broadcast_dimensions),
            And(lhs, Not(rhs), broadcast_dimensions));
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

XlaOp XlaBuilder::Sign(const XlaOp& operand) {
  return UnaryOp(HloOpcode::kSign, operand);
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
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
    HloInstructionProto instr;
    TF_ASSIGN_OR_RETURN(const Shape& operand_shape, operand.GetShape());
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
  return UnimplementedOp();
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
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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
  return UnimplementedOp();
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
  return UnimplementedOp();
}

XlaOp XlaBuilder::RngOp(RandomDistribution distribution,
                        tensorflow::gtl::ArraySlice<XlaOp> parameters,
                        const Shape& shape) {
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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
  return UnimplementedOp();
}

XlaOp XlaBuilder::Conditional(const XlaOp& predicate, const XlaOp& true_operand,
                              const XlaComputation& true_computation,
                              const XlaOp& false_operand,
                              const XlaComputation& false_computation) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Reduce(
    const XlaOp& operand, const XlaOp& init_value,
    const XlaComputation& computation,
    tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce) {
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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
  return UnimplementedOp();
}

XlaOp XlaBuilder::ReduceWindow(
    const XlaOp& operand, const XlaOp& init_value,
    const XlaComputation& computation,
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::ReduceWindowWithGeneralPadding(
    const XlaOp& operand, const XlaOp& init_value,
    const XlaComputation& computation,
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::BatchNormTraining(const XlaOp& operand, const XlaOp& scale,
                                    const XlaOp& offset, float epsilon,
                                    int64 feature_index) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::BatchNormInference(const XlaOp& operand, const XlaOp& scale,
                                     const XlaOp& offset, const XlaOp& mean,
                                     const XlaOp& variance, float epsilon,
                                     int64 feature_index) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::BatchNormGrad(const XlaOp& operand, const XlaOp& scale,
                                const XlaOp& batch_mean, const XlaOp& batch_var,
                                const XlaOp& grad_output, float epsilon,
                                int64 feature_index) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::CrossReplicaSum(const XlaOp& operand) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::SelectAndScatter(
    const XlaOp& operand, const XlaComputation& select,
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides, Padding padding,
    const XlaOp& source, const XlaOp& init_value,
    const XlaComputation& scatter) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::SelectAndScatterWithGeneralPadding(
    const XlaOp& operand, const XlaComputation& select,
    tensorflow::gtl::ArraySlice<int64> window_dimensions,
    tensorflow::gtl::ArraySlice<int64> window_strides,
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding,
    const XlaOp& source, const XlaOp& init_value,
    const XlaComputation& scatter) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::ReducePrecision(const XlaOp& operand, const int exponent_bits,
                                  const int mantissa_bits) {
  return UnimplementedOp();
}

void XlaBuilder::Send(const XlaOp& operand, const ChannelHandle& handle) {
  NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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

StatusOr<bool> XlaBuilder::IsConstant(const XlaOp& operand,
                                      int64 num_parameters) {
  return Unimplemented("IsConstant is not implemented.");
}

StatusOr<std::unique_ptr<Literal>> XlaBuilder::ComputeConstant(
    const XlaOp& operand, const Layout* output_layout,
    tensorflow::gtl::ArraySlice<Literal> parameters) {
  return Unimplemented("ComputeConstant is not implemented");
}

std::unique_ptr<XlaBuilder> XlaBuilder::CreateSubBuilder(
    const string& computation_name) {
  auto sub_builder = MakeUnique<XlaBuilder>(computation_name);
  sub_builder->parent_builder_ = this;
  sub_builder->die_immediately_on_error_ = this->die_immediately_on_error_;
  return sub_builder;
}

Status XlaBuilder::SetReturnValue(const XlaOp& operand) {
  return Unimplemented("SetReturnValue is not implemented.");
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
    instr.set_name(StrCat(instr.opcode(), ".", unique_id_, ".", handle));
  } else {
    // Append the handle to make sure the name is unique.
    instr.set_name(StrCat(instr.name(), ".", unique_id_, ".", handle));
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

  if (op.builder_ != this) {
    return InvalidArgument("invalid XlaOp with handle %lld", op.handle());
  }

  TF_RET_CHECK(op.builder_ == this);
  if (op.handle() >= instructions_.size() || op.handle() < 0) {
    return InvalidArgument("no XlaOp value %lld", op.handle());
  }
  return &instructions_[op.handle()];
}

XlaOp XlaBuilder::UnimplementedOp() {
  NoteError(Unimplemented("Op not implemented"));
  return {};
}

}  // namespace xla
