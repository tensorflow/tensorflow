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

#include "tensorflow/compiler/xla/service/hlo_instructions.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"

namespace xla {

using ::tensorflow::str_util::Join;
using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;

HloBatchNormInstruction::HloBatchNormInstruction(
    HloOpcode opcode, const Shape& shape, HloInstruction* operand,
    HloInstruction* scale, float epsilon, int64 feature_index)
    : HloInstruction(opcode, shape),
      epsilon_(epsilon),
      feature_index_(feature_index) {
  AppendOperand(operand);
  AppendOperand(scale);
}

bool HloBatchNormInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other = static_cast<const HloBatchNormInstruction&>(other);
  return feature_index() == casted_other.feature_index() &&
         epsilon() == casted_other.epsilon();
}

HloInstructionProto HloBatchNormInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_epsilon(epsilon_);
  proto.set_feature_index(feature_index_);
  return proto;
}

std::vector<string> HloBatchNormInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {StrCat("epsilon=", epsilon()),
          StrCat("feature_index=", feature_index())};
}

HloBatchNormTrainingInstruction::HloBatchNormTrainingInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* scale,
    HloInstruction* offset, float epsilon, int64 feature_index)
    : HloBatchNormInstruction(HloOpcode::kBatchNormTraining, shape, operand,
                              scale, epsilon, feature_index) {
  AppendOperand(offset);
}

std::unique_ptr<HloInstruction>
HloBatchNormTrainingInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 3);
  return MakeUnique<HloBatchNormTrainingInstruction>(
      shape, new_operands[0], new_operands[1], new_operands[2], epsilon(),
      feature_index());
}

HloBatchNormInferenceInstruction::HloBatchNormInferenceInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* scale,
    HloInstruction* offset, HloInstruction* mean, HloInstruction* variance,
    float epsilon, int64 feature_index)
    : HloBatchNormInstruction(HloOpcode::kBatchNormInference, shape, operand,
                              scale, epsilon, feature_index) {
  AppendOperand(offset);
  AppendOperand(mean);
  AppendOperand(variance);
}

std::unique_ptr<HloInstruction>
HloBatchNormInferenceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 5);
  return MakeUnique<HloBatchNormInferenceInstruction>(
      shape, new_operands[0], new_operands[1], new_operands[2], new_operands[3],
      new_operands[4], epsilon(), feature_index());
}

HloBatchNormGradInstruction::HloBatchNormGradInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* scale,
    HloInstruction* mean, HloInstruction* variance, HloInstruction* grad_output,
    float epsilon, int64 feature_index)
    : HloBatchNormInstruction(HloOpcode::kBatchNormGrad, shape, operand, scale,
                              epsilon, feature_index) {
  AppendOperand(mean);
  AppendOperand(variance);
  AppendOperand(grad_output);
}

std::unique_ptr<HloInstruction>
HloBatchNormGradInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 5);
  return MakeUnique<HloBatchNormGradInstruction>(
      shape, new_operands[0], new_operands[1], new_operands[2], new_operands[3],
      new_operands[4], epsilon(), feature_index());
}

HloFftInstruction::HloFftInstruction(
    const Shape& shape, HloInstruction* operand, FftType fft_type,
    tensorflow::gtl::ArraySlice<int64> fft_length)
    : HloInstruction(HloOpcode::kFft, shape), fft_type_(fft_type) {
  fft_length_.assign(fft_length.begin(), fft_length.end());
  AppendOperand(operand);
}

HloInstructionProto HloFftInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_fft_type(fft_type_);
  for (int64 fft_len : fft_length_) {
    proto.add_fft_length(fft_len);
  }
  return proto;
}

std::vector<string> HloFftInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {StrCat("fft_type=", FftType_Name(fft_type())),
          StrCat("fft_length={", Join(fft_length(), ","), "}")};
}

bool HloFftInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other = static_cast<const HloFftInstruction&>(other);
  return fft_type() == casted_other.fft_type() &&
         fft_length() == casted_other.fft_length();
}

std::unique_ptr<HloInstruction> HloFftInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return MakeUnique<HloFftInstruction>(shape, new_operands[0], fft_type_,
                                       fft_length_);
}

HloSendRecvInstruction::HloSendRecvInstruction(HloOpcode opcode,
                                               const Shape& shape,
                                               int64 channel_id)
    : HloInstruction(opcode, shape), channel_id_(channel_id) {}

HloInstructionProto HloSendRecvInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_channel_id(channel_id_);
  return proto;
}

std::vector<string> HloSendRecvInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {StrCat("channel_id=", channel_id_)};
}

bool HloSendRecvInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  // Not yet supported.
  return false;
}

// Send instruction produces a tuple of {aliased operand, U32 context}.
HloSendInstruction::HloSendInstruction(HloInstruction* operand,
                                       int64 channel_id)
    : HloSendRecvInstruction(
          HloOpcode::kSend,
          ShapeUtil::MakeTupleShape(
              {CHECK_NOTNULL(operand)->shape(), ShapeUtil::MakeShape(U32, {})}),
          channel_id) {
  AppendOperand(operand);
}

std::unique_ptr<HloInstruction> HloSendInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return MakeUnique<HloSendInstruction>(new_operands[0], channel_id());
}

HloSendDoneInstruction::HloSendDoneInstruction(HloSendInstruction* operand)
    : HloSendRecvInstruction(HloOpcode::kSendDone, ShapeUtil::MakeNil(),
                             CHECK_NOTNULL(operand)->channel_id()) {
  AppendOperand(operand);
}

std::unique_ptr<HloInstruction>
HloSendDoneInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return MakeUnique<HloSendDoneInstruction>(
      Cast<HloSendInstruction>(new_operands[0]));
}

// Recv instruction produces a tuple of {receive buffer, U32 context}.
HloRecvInstruction::HloRecvInstruction(const Shape& shape, int64 channel_id)
    : HloSendRecvInstruction(
          HloOpcode::kRecv,
          ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeShape(U32, {})}),
          channel_id) {}

std::unique_ptr<HloInstruction> HloRecvInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 0);
  return MakeUnique<HloRecvInstruction>(
      ShapeUtil::GetTupleElementShape(shape, 0), channel_id());
}

HloRecvDoneInstruction::HloRecvDoneInstruction(HloRecvInstruction* operand)
    : HloSendRecvInstruction(
          HloOpcode::kRecvDone,
          ShapeUtil::GetTupleElementShape(operand->shape(), 0),
          CHECK_NOTNULL(operand)->channel_id()) {
  AppendOperand(operand);
}

std::unique_ptr<HloInstruction>
HloRecvDoneInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return MakeUnique<HloRecvDoneInstruction>(
      Cast<HloRecvInstruction>(new_operands[0]));
}

HloReverseInstruction::HloReverseInstruction(
    const Shape& shape, HloInstruction* operand,
    tensorflow::gtl::ArraySlice<int64> dimensions)
    : HloInstruction(HloOpcode::kReverse, shape),
      dimensions_(dimensions.begin(), dimensions.end()) {
  AppendOperand(operand);
}

HloInstructionProto HloReverseInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64 dimension : dimensions_) {
    proto.add_dimensions(dimension);
  }
  return proto;
}

std::vector<string> HloReverseInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {StrCat("dimensions={", Join(dimensions(), ","), "}")};
}

bool HloReverseInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other = static_cast<const HloReverseInstruction&>(other);
  return dimensions() == casted_other.dimensions();
}

std::unique_ptr<HloInstruction> HloReverseInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return MakeUnique<HloReverseInstruction>(shape, new_operands[0],
                                           dimensions());
}

HloConcatenateInstruction::HloConcatenateInstruction(
    const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    int64 dimension)
    : HloInstruction(HloOpcode::kConcatenate, shape), dimensions_({dimension}) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
}

HloInstructionProto HloConcatenateInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64 dimension : dimensions_) {
    proto.add_dimensions(dimension);
  }
  return proto;
}

std::vector<string> HloConcatenateInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {StrCat("dimensions={", Join(dimensions(), ","), "}")};
}

bool HloConcatenateInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloConcatenateInstruction&>(other);
  return dimensions() == casted_other.dimensions();
}

std::unique_ptr<HloInstruction>
HloConcatenateInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  return MakeUnique<HloConcatenateInstruction>(shape, new_operands,
                                               dimensions(0));
}

HloReduceInstruction::HloReduceInstruction(
    const Shape& shape, HloInstruction* arg, HloInstruction* init_value,
    tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce,
    HloComputation* reduce_computation)
    : HloInstruction(HloOpcode::kReduce, shape),
      dimensions_(dimensions_to_reduce.begin(), dimensions_to_reduce.end()) {
  AppendOperand(arg);
  AppendOperand(init_value);
  AppendComputation(reduce_computation);
}

HloInstructionProto HloReduceInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64 dimension : dimensions_) {
    proto.add_dimensions(dimension);
  }
  return proto;
}

std::vector<string> HloReduceInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {StrCat("dimensions={", Join(dimensions(), ","), "}")};
}

bool HloReduceInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other = static_cast<const HloReduceInstruction&>(other);
  // Reduction results are determined by the reduction dimension and the
  // reduction computation.
  return dimensions() == casted_other.dimensions() &&
         eq_computations(to_apply(), casted_other.to_apply());
}

std::unique_ptr<HloInstruction> HloReduceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return MakeUnique<HloReduceInstruction>(
      shape, new_operands[0], new_operands[1], dimensions(), to_apply());
}

HloTransposeInstruction::HloTransposeInstruction(
    const Shape& shape, HloInstruction* operand,
    tensorflow::gtl::ArraySlice<int64> dimensions)
    : HloInstruction(HloOpcode::kTranspose, shape),
      dimensions_(dimensions.begin(), dimensions.end()) {
  CHECK_EQ(shape.dimensions().size(), dimensions.size());
  CHECK_EQ(shape.dimensions().size(), operand->shape().dimensions().size());
  CHECK(std::equal(operand->shape().dimensions().begin(),
                   operand->shape().dimensions().end(),
                   Permute(dimensions, shape.dimensions()).begin()))
      << "shape: " << ShapeUtil::HumanString(shape)
      << ", operand->shape(): " << ShapeUtil::HumanString(shape)
      << ", dimensions: {" << Join(dimensions, ", ") << "}";
  AppendOperand(operand);
}

bool HloTransposeInstruction::IsRank2Transpose() const {
  return dimensions() == std::vector<int64>({1, 0}) &&
         shape().dimensions_size() == 2 &&
         std::equal(shape().dimensions().begin(), shape().dimensions().end(),
                    operand(0)->shape().dimensions().rbegin());
}

HloInstructionProto HloTransposeInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64 dimension : dimensions_) {
    proto.add_dimensions(dimension);
  }
  return proto;
}

std::vector<string> HloTransposeInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {StrCat("dimensions={", Join(dimensions(), ","), "}")};
}

bool HloTransposeInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other = static_cast<const HloTransposeInstruction&>(other);
  return dimensions() == casted_other.dimensions();
}

std::unique_ptr<HloInstruction>
HloTransposeInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return MakeUnique<HloTransposeInstruction>(shape, new_operands[0],
                                             dimensions());
}

HloBroadcastInstruction::HloBroadcastInstruction(
    const Shape& shape, HloInstruction* operand,
    tensorflow::gtl::ArraySlice<int64> broadcast_dimension)
    : HloInstruction(HloOpcode::kBroadcast, shape),
      dimensions_(broadcast_dimension.begin(), broadcast_dimension.end()) {
  AppendOperand(operand);
}

HloInstructionProto HloBroadcastInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64 dimension : dimensions_) {
    proto.add_dimensions(dimension);
  }
  return proto;
}

std::vector<string> HloBroadcastInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {StrCat("dimensions={", Join(dimensions(), ","), "}")};
}

bool HloBroadcastInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other = static_cast<const HloBroadcastInstruction&>(other);
  return dimensions() == casted_other.dimensions();
}

std::unique_ptr<HloInstruction>
HloBroadcastInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return MakeUnique<HloBroadcastInstruction>(shape, new_operands[0],
                                             dimensions());
}

HloMapInstruction::HloMapInstruction(
    const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    HloComputation* map_computation,
    tensorflow::gtl::ArraySlice<HloInstruction*> static_operands)
    : HloInstruction(HloOpcode::kMap, shape) {
  CHECK(static_operands.empty()) << "static_operands not yet supported";
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  AppendComputation(map_computation);
  // TODO(b/65689298) Remove code below once Map is generalized to accept
  // arbitrary map dimensions.
  dimensions_.resize(ShapeUtil::Rank(shape));
  std::iota(dimensions_.begin(), dimensions_.end(), 0);
}

HloInstructionProto HloMapInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64 dimension : dimensions_) {
    proto.add_dimensions(dimension);
  }
  return proto;
}

bool HloMapInstruction::IsElementwise() const {
  if (!dimensions().empty()) {
    // Check that the map is executed in elementwise compatible dimensions.
    if (dimensions().size() != shape().dimensions_size()) {
      return false;
    }
    for (int i = 0; i < dimensions().size(); ++i) {
      if (dimensions()[i] != i) {
        return false;
      }
    }
  }
  return true;
}

std::vector<string> HloMapInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {StrCat("dimensions={", Join(dimensions(), ","), "}")};
}

bool HloMapInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  return eq_computations(to_apply(), other.to_apply());
}

std::unique_ptr<HloInstruction> HloMapInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  return MakeUnique<HloMapInstruction>(shape, new_operands, to_apply());
}

HloSliceInstruction::HloSliceInstruction(
    const Shape& shape, HloInstruction* operand,
    tensorflow::gtl::ArraySlice<int64> start_indices,
    tensorflow::gtl::ArraySlice<int64> limit_indices,
    tensorflow::gtl::ArraySlice<int64> strides)
    : HloInstruction(HloOpcode::kSlice, shape),
      slice_starts_(start_indices.begin(), start_indices.end()),
      slice_limits_(limit_indices.begin(), limit_indices.end()),
      slice_strides_(strides.begin(), strides.end()) {
  AppendOperand(operand);
  // For backward compatibility with old serialized computations: if there are
  // no strides, assume all strides are 1.
  // TODO(b/63317920): remove this code.
  if (slice_strides_.empty()) {
    slice_strides_ = std::vector<int64>(start_indices.size(), 1LL);
  }
}

HloInstructionProto HloSliceInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int i = 0; i < slice_starts_.size(); ++i) {
    auto* slice_dimension = proto.add_slice_dimensions();
    slice_dimension->set_start(slice_starts_[i]);
    slice_dimension->set_limit(slice_limits_[i]);
    slice_dimension->set_stride(slice_strides_[i]);
  }
  return proto;
}

std::vector<string> HloSliceInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<string> bounds;
  bounds.reserve(slice_starts_.size());
  const bool omit_stride =
      std::all_of(slice_strides_.begin(), slice_strides_.end(),
                  [](int64 stride) { return stride == 1; });
  for (int i = 0; i < slice_starts_.size(); ++i) {
    string stride_str = omit_stride ? "" : StrCat(":", slice_strides_[i]);
    bounds.push_back(
        StrCat("[", slice_starts_[i], ":", slice_limits_[i], stride_str, "]"));
  }
  return {StrCat("slice={", Join(bounds, ", "), "}")};
}

bool HloSliceInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& other_slice = static_cast<const HloSliceInstruction&>(other);
  return slice_starts_ == other_slice.slice_starts_ &&
         slice_limits_ == other_slice.slice_limits_ &&
         slice_strides_ == other_slice.slice_strides_;
}

std::unique_ptr<HloInstruction> HloSliceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return MakeUnique<HloSliceInstruction>(shape, new_operands[0], slice_starts_,
                                         slice_limits_, slice_strides_);
}

HloConstantInstruction::HloConstantInstruction(std::unique_ptr<Literal> literal)
    : HloInstruction(HloOpcode::kConstant, CHECK_NOTNULL(literal)->shape()),
      literal_(std::move(literal)) {}

HloInstructionProto HloConstantInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_literal() = literal_->ToProto();
  return proto;
}

bool HloConstantInstruction::IsElementwise() const { return true; }

void HloConstantInstruction::RelayoutConstant(const Layout& new_layout,
                                              const ShapeIndex& shape_index) {
  Shape* mutable_array_subshape =
      ShapeUtil::GetMutableSubshape(mutable_shape(), shape_index);
  CHECK(ShapeUtil::IsArray(*mutable_array_subshape));

  // Normally array_subshape will always have a layout, but this invariant is
  // temporarily broken in LayoutAssignment::AssignLayouts.

  if (!mutable_array_subshape->has_layout() ||
      !LayoutUtil::Equal(mutable_array_subshape->layout(), new_layout)) {
    literal_ = literal_->Relayout(new_layout, shape_index);
    *mutable_array_subshape->mutable_layout() = new_layout;
  }
}

bool HloConstantInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& other_slice = static_cast<const HloSliceInstruction&>(other);
  return literal() == other_slice.literal();
}

std::unique_ptr<HloInstruction>
HloConstantInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  return MakeUnique<HloConstantInstruction>(literal_->CloneToUnique());
}

string HloConstantInstruction::OperandsToStringWithCanonicalNameMap(
    const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
  string operands;
  // For constants, show the actual value in place of an empty operand list.
  if ((!ShapeUtil::IsTuple(shape()) && ShapeUtil::ElementsIn(shape()) <= 10) ||
      options.print_large_constants()) {
    // Literal::ToString emits multidimensional arrays over multiple
    // lines. Compact this into one line by stripping out white space.
    string tmp = literal().ToString();
    std::replace(tmp.begin(), tmp.end(), '\n', ' ');
    std::vector<string> v = tensorflow::str_util::Split(tmp, ' ');
    bool first = true;
    // Concatenate elements in "v" with spaces separating them, but ignoring
    // empty entries.
    for (const auto& s : v) {
      if (s.empty()) {
        continue;
      }
      StrAppend(&operands, (first ? "" : " "), s);
      first = false;
    }
  } else {
    // Do not show large constants or tuples.
    operands = "{...}";
  }
  return operands;
}

HloTraceInstruction::HloTraceInstruction(const string& tag,
                                         HloInstruction* operand)
    : HloInstruction(HloOpcode::kTrace, ShapeUtil::MakeNil()),
      literal_(Literal::CreateR1U8(tag)) {
  AppendOperand(operand);
  operand->set_tracing(this);
}

HloInstructionProto HloTraceInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_literal() = literal_->ToProto();
  return proto;
}

bool HloTraceInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  return false;
}

std::unique_ptr<HloInstruction> HloTraceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  LOG(FATAL) << "Not yet implemented, clone: " << HloOpcodeString(opcode());
}
}  // namespace xla
