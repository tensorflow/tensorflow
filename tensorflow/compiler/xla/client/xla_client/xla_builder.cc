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

StatusOr<ProgramShape> XlaBuilder::GetProgramShape(int64* root_id) {
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

StatusOr<XlaComputation> XlaBuilder::Build() {
  if (!first_error_.ok()) {
    string backtrace;
    first_error_backtrace_.Dump(tensorflow::DebugWriteToString, &backtrace);
    return AppendStatus(first_error_, backtrace);
  }

  HloComputationProto entry;
  entry.set_name(name_);

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

StatusOr<XlaOp> XlaBuilder::InDimBroadcast(
    const Shape& shape, const XlaOp& operand,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  HloInstructionProto instr;
  *instr.mutable_shape() = shape;
  for (int64 dim : broadcast_dimensions) {
    instr.add_dimensions(dim);
  }
  return AddInstruction(std::move(instr), HloOpcode::kBroadcast, {operand});
}

StatusOr<XlaOp> XlaBuilder::AddBroadcastSequence(const Shape& output_shape,
                                                 const XlaOp& operand) {
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
  }());
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
  }());
}

XlaOp XlaBuilder::Call(const XlaComputation& computation,
                       tensorflow::gtl::ArraySlice<XlaOp> operands) {
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
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
  }());
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
  }());
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
  }());
}

StatusOr<XlaOp> XlaBuilder::Reshape(const Shape& shape, const XlaOp& operand) {
  HloInstructionProto instr;
  *instr.mutable_shape() = shape;
  return AddInstruction(std::move(instr), HloOpcode::kReshape, {operand});
}

XlaOp XlaBuilder::Slice(const XlaOp& operand,
                        tensorflow::gtl::ArraySlice<int64> start_indices,
                        tensorflow::gtl::ArraySlice<int64> limit_indices,
                        tensorflow::gtl::ArraySlice<int64> strides) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::SliceInDim(const XlaOp& operand, int64 start_index,
                             int64 limit_index, int64 stride, int64 dimno) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::DynamicSlice(const XlaOp& operand, const XlaOp& start_indices,
                               tensorflow::gtl::ArraySlice<int64> slice_sizes) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::DynamicUpdateSlice(const XlaOp& operand, const XlaOp& update,
                                     const XlaOp& start_indices) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::ConcatInDim(tensorflow::gtl::ArraySlice<XlaOp> operands,
                              int64 dimension) {
  return UnimplementedOp();
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
  }());
}

XlaOp XlaBuilder::Reshape(const XlaOp& operand,
                          tensorflow::gtl::ArraySlice<int64> new_sizes) {
  return NoteErrorOrReturn([&]() -> StatusOr<XlaOp> {
    TF_ASSIGN_OR_RETURN(auto shape, operand.GetShape());
    std::vector<int64> dimensions(shape.dimensions_size());
    std::iota(dimensions.begin(), dimensions.end(), 0);
    return Reshape(operand, dimensions, new_sizes);
  }());
}

XlaOp XlaBuilder::Collapse(const XlaOp& operand,
                           tensorflow::gtl::ArraySlice<int64> dimensions) {
  return UnimplementedOp();
}

void XlaBuilder::Trace(const string& tag, const XlaOp& operand) {
  UnimplementedOp();
}

XlaOp XlaBuilder::Select(const XlaOp& pred, const XlaOp& on_true,
                         const XlaOp& on_false) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Tuple(tensorflow::gtl::ArraySlice<XlaOp> elements) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::GetTupleElement(const XlaOp& tuple_data, int64 index) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Eq(const XlaOp& lhs, const XlaOp& rhs,
                     tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Ne(const XlaOp& lhs, const XlaOp& rhs,
                     tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Ge(const XlaOp& lhs, const XlaOp& rhs,
                     tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Gt(const XlaOp& lhs, const XlaOp& rhs,
                     tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Le(const XlaOp& lhs, const XlaOp& rhs,
                     tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Lt(const XlaOp& lhs, const XlaOp& rhs,
                     tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Dot(const XlaOp& lhs, const XlaOp& rhs) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::DotGeneral(const XlaOp& lhs, const XlaOp& rhs,
                             const DotDimensionNumbers& dimension_numbers) {
  return UnimplementedOp();
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
  return UnimplementedOp();
}

XlaOp XlaBuilder::Conj(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::Sub(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Div(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Rem(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Max(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Min(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::And(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Or(const XlaOp& lhs, const XlaOp& rhs,
                     tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Xor(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Not(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::ShiftLeft(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::ShiftRightArithmetic(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::ShiftRightLogical(
    const XlaOp& lhs, const XlaOp& rhs,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Abs(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::Atan2(
    const XlaOp& y, const XlaOp& x,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Exp(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::Floor(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::Ceil(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::Round(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::Log(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::Sign(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::Cos(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::Sin(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::Tanh(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::Real(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::Imag(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::IsFinite(const XlaOp& operand) { return UnimplementedOp(); }

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
  }());
}

XlaOp XlaBuilder::Rev(const XlaOp& operand,
                      tensorflow::gtl::ArraySlice<int64> dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Sort(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::SqrtF32(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::Pow(const XlaOp& lhs, const XlaOp& rhs,
                      tensorflow::gtl::ArraySlice<int64> broadcast_dimensions) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::ConvertElementType(const XlaOp& operand,
                                     PrimitiveType new_element_type) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::BitcastConvertType(const XlaOp& operand,
                                     PrimitiveType new_element_type) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::SquareF32(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::ReciprocalF32(const XlaOp& operand) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Neg(const XlaOp& operand) { return UnimplementedOp(); }

XlaOp XlaBuilder::Clamp(const XlaOp& min, const XlaOp& operand,
                        const XlaOp& max) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::Map(tensorflow::gtl::ArraySlice<XlaOp> operands,
                      const XlaComputation& computation,
                      tensorflow::gtl::ArraySlice<int64> dimensions,
                      tensorflow::gtl::ArraySlice<XlaOp> static_operands) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::RngNormal(const XlaOp& mu, const XlaOp& sigma,
                            const Shape& shape) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::RngUniform(const XlaOp& a, const XlaOp& b,
                             const Shape& shape) {
  return UnimplementedOp();
}

XlaOp XlaBuilder::While(const XlaComputation& condition,
                        const XlaComputation& body, const XlaOp& init) {
  return UnimplementedOp();
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
  return UnimplementedOp();
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
  UnimplementedOp();
}

XlaOp XlaBuilder::Recv(const Shape& shape, const ChannelHandle& handle) {
  return UnimplementedOp();
}

StatusOr<XlaOp> XlaBuilder::AddInstruction(
    HloInstructionProto&& instr, HloOpcode opcode,
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
    TF_RET_CHECK(operand.builder_ != nullptr);
    TF_RET_CHECK(operand.builder_ == this)
        << "Do not add XlaOp from builder " << operand.builder_->name()
        << " to builder " << this->name();
    instr.add_operand_ids(operand.handle());
    // TODO(b/74197823): Set metadata and sharding.
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

XlaOp XlaBuilder::UnimplementedOp() {
  NoteError(Unimplemented("Op not yet implemented"));
  return {};
}

}  // namespace xla
