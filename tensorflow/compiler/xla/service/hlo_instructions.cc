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

#include <deque>

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace {

using ::tensorflow::str_util::CEscape;
using ::tensorflow::str_util::Join;
using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;

bool IsInstructionElementwiseOnOperand(const HloInstruction* instruction,
                                       const HloInstruction* operand) {
  std::vector<int64> operand_indices = instruction->OperandIndices(operand);
  return std::all_of(
      operand_indices.begin(), operand_indices.end(),
      [instruction](int64 operand_index) {
        return instruction->IsElementwiseOnOperand(operand_index);
      });
}
}  // namespace

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

HloAllReduceInstruction::HloAllReduceInstruction(
    const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    HloComputation* reduce_computation,
    tensorflow::gtl::ArraySlice<int64> replica_group_ids,
    tensorflow::StringPiece barrier,
    const tensorflow::gtl::optional<int64>& all_reduce_id)
    : HloInstruction(HloOpcode::kCrossReplicaSum, shape),
      replica_group_ids_(replica_group_ids.begin(), replica_group_ids.end()),
      cross_replica_sum_barrier_(barrier.begin(), barrier.end()),
      all_reduce_id_(all_reduce_id) {
  // TODO(b/79737069): Remove the CHECK when supported.
  CHECK(!all_reduce_id_);
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  AppendComputation(reduce_computation);
}

HloInstructionProto HloAllReduceInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64 i : replica_group_ids_) {
    proto.add_replica_group_ids(i);
  }
  // Proto3 is so sad.
  if (all_reduce_id_) {
    proto.set_all_reduce_id(*all_reduce_id_);
  }
  proto.set_cross_replica_sum_barrier(cross_replica_sum_barrier_);
  return proto;
}

std::vector<string> HloAllReduceInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& /*options*/) const {
  std::vector<string> result = {
      StrCat("replica_group_ids={", Join(replica_group_ids(), ","), "}")};
  if (!cross_replica_sum_barrier().empty()) {
    result.push_back(StrCat("barrier=\"", cross_replica_sum_barrier(), "\""));
  }
  if (all_reduce_id_) {
    result.push_back(StrCat("all_reduce_id=", *all_reduce_id_));
  }
  return result;
}

bool HloAllReduceInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other = static_cast<const HloAllReduceInstruction&>(other);
  return replica_group_ids() == casted_other.replica_group_ids() &&
         eq_computations(to_apply(), casted_other.to_apply()) &&
         cross_replica_sum_barrier() ==
             casted_other.cross_replica_sum_barrier() &&
         all_reduce_id() == casted_other.all_reduce_id();
}

std::unique_ptr<HloInstruction>
HloAllReduceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* /*context*/) const {
  return MakeUnique<HloAllReduceInstruction>(
      shape, new_operands, to_apply(), replica_group_ids(),
      cross_replica_sum_barrier(), all_reduce_id());
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

bool HloMapInstruction::IsElementwiseImpl(
    const tensorflow::gtl::optional<int64>& operand_idx) const {
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

HloConstantInstruction::HloConstantInstruction(const Shape& shape)
    : HloInstruction(HloOpcode::kConstant, shape) {}

HloInstructionProto HloConstantInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  if (literal_ != nullptr) {
    *proto.mutable_literal() = literal_->ToProto();
  }
  return proto;
}

bool HloConstantInstruction::IsElementwiseImpl(
    const tensorflow::gtl::optional<int64>& operand_idx) const {
  return true;
}

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
  if (literal_ != nullptr &&
      ((ShapeUtil::IsArray(shape()) && ShapeUtil::ElementsIn(shape()) <= 10) ||
       options.print_large_constants())) {
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

HloFusionInstruction::HloFusionInstruction(const Shape& shape,
                                           FusionKind fusion_kind,
                                           HloInstruction* fused_root)
    : HloInstruction(HloOpcode::kFusion, shape), fusion_kind_(fusion_kind) {
  CHECK(fused_root != nullptr);
  SetAndSanitizeName("fusion");
  set_parent(fused_root->parent());
  set_metadata(fused_root->metadata());
  CloneAndFuseInternal(fused_root);
}

HloFusionInstruction::HloFusionInstruction(
    const Shape& shape, FusionKind fusion_kind,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    HloComputation* fusion_computation)
    : HloInstruction(HloOpcode::kFusion, shape), fusion_kind_(fusion_kind) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  SetAndSanitizeName("fusion");
  AppendComputation(fusion_computation);
  fusion_computation->SetFusionInstruction(this);
}

string HloFusionInstruction::ToCategory() const {
  switch (fusion_kind()) {
    case FusionKind::kLoop:
      return "loop fusion";
    case FusionKind::kInput:
      return "input fusion";
    case FusionKind::kOutput:
      return "output fusion";
    case FusionKind::kCustom:
      return "custom fusion";
  }
}

HloInstructionProto HloFusionInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_fusion_kind(xla::ToString(fusion_kind()));
  proto.add_called_computation_ids(
      fused_instructions_computation()->unique_id());
  return proto;
}

bool HloFusionInstruction::IsElementwiseImpl(
    const tensorflow::gtl::optional<int64>& operand_idx) const {
  if (!operand_idx.has_value()) {
    for (auto* fused : fused_instructions()) {
      if (fused->opcode() != HloOpcode::kParameter && !fused->IsElementwise()) {
        return false;
      }
    }
    return true;
  }
  // A loop-fusion is elementwise on an operand if all operations (computed
  // using BFS) between the operand and the fused root are elementwise.
  std::deque<HloInstruction*> worklist;
  std::unordered_set<const HloInstruction*> visited;
  worklist.push_back(fused_parameter(operand_idx.value()));
  visited.insert(fused_parameter(operand_idx.value()));
  while (!worklist.empty()) {
    HloInstruction* operand = worklist.front();
    worklist.pop_front();
    for (HloInstruction* user : operand->users()) {
      CHECK_GE(user->unique_id(), 0);
      if (ContainsKey(visited, user)) {
        continue;
      }
      if (user->IsElementwise() ||
          IsInstructionElementwiseOnOperand(user, operand)) {
        worklist.push_back(user);
        visited.insert(user);
      } else {
        return false;
      }
    }
  }
  return true;
}

HloInstruction* HloFusionInstruction::AddFusionOperand(
    HloInstruction* new_operand) {
  CHECK_EQ(operand_count(),
           fused_instructions_computation()->parameter_instructions().size());
  const int64 param_no = operand_count();
  // Name the parameter after the instruction it represents in the outer
  // (non-fusion) computation.
  string param_name = StrCat(new_operand->name(), ".param_", param_no);
  HloInstruction* fused_parameter =
      fused_instructions_computation()->AddParameter(
          HloInstruction::CreateParameter(param_no, new_operand->shape(),
                                          param_name));
  AppendOperand(new_operand);
  return fused_parameter;
}

void HloFusionInstruction::MergeFusionInstruction(
    HloFusionInstruction* instruction_to_merge) {
  CHECK(std::find(operands().begin(), operands().end(), instruction_to_merge) !=
        operands().end());
  // Clone the instruction from which to merge fused instructions.
  std::unique_ptr<HloInstruction> cloned = instruction_to_merge->Clone();
  HloFusionInstruction* cloned_fusion =
      static_cast<HloFusionInstruction*>(cloned.get());
  // Replace uses of fused parameters with the corresponding operand of the
  // fusion.  Add all non-parameter fused instructions to
  // 'unfused_instructions' to be merged into 'this'.  This is done in reverse
  // post order.
  std::vector<HloInstruction*> unfused_instructions;
  auto fused_instructions = cloned_fusion->fused_instructions_computation()
                                ->MakeInstructionPostOrder();
  for (auto fused_it = fused_instructions.rbegin();
       fused_it != fused_instructions.rend(); ++fused_it) {
    auto fused_instruction = *fused_it;
    if (fused_instruction->opcode() == HloOpcode::kParameter) {
      TF_CHECK_OK(
          fused_instruction->ReplaceAllUsesWith(cloned_fusion->mutable_operand(
              fused_instruction->parameter_number())));
    } else {
      unfused_instructions.push_back(fused_instruction);
    }
  }
  CHECK(unfused_instructions.front() == cloned_fusion->fused_expression_root());
  // Replace instruction_to_merge use of 'this' with unfused_root.
  TF_CHECK_OK(
      instruction_to_merge->ReplaceUseWith(this, unfused_instructions.front()));
  // Fuse 'unfused_instructions' into 'this'.
  for (auto& instruction : unfused_instructions) {
    FuseInstruction(instruction);
  }
  CHECK_EQ(0, cloned_fusion->user_count());
  TF_CHECK_OK(parent()->parent()->RemoveEmbeddedComputation(
      cloned_fusion->fused_instructions_computation()));
}

void HloFusionInstruction::MergeFusionInstructionIntoMultiOutput(
    HloFusionInstruction* instruction_to_merge) {
  // Add all non-parameter fused instructions to 'unfused_instructions' to be
  // merged into 'this'. `old_to_new' maps the instructions in the fused node
  // to the disaseembled fusion instructions.
  // Note that we add the unfused instructions to this->parent_ computation.
  // This is necessary because the unique_id needs for an instruction and
  // it's only added when inserting to the computation.
  tensorflow::gtl::FlatMap<HloInstruction*, HloInstruction*> old_to_new;
  std::vector<HloInstruction*> unfused_instructions;
  auto computation_to_merge =
      instruction_to_merge->fused_instructions_computation();
  auto post_order = computation_to_merge->MakeInstructionPostOrder();
  for (auto rit = post_order.rbegin(); rit != post_order.rend(); ++rit) {
    auto fused_instruction = *rit;
    if (fused_instruction->opcode() == HloOpcode::kParameter) {
      InsertOrDie(&old_to_new, fused_instruction,
                  instruction_to_merge->mutable_operand(
                      fused_instruction->parameter_number()));
      continue;
    }

    // Here we clone the insertion and call FuseInstructionIntoMultiOutput()
    // which clones again. This can be improved.
    auto cloned_instruction =
        parent()->AddInstruction(fused_instruction->Clone());
    unfused_instructions.push_back(cloned_instruction);
    InsertOrDie(&old_to_new, fused_instruction, cloned_instruction);
  }
  for (auto unfused_instruction : unfused_instructions) {
    for (int64 index = 0; index < unfused_instruction->operand_count();
         index++) {
      auto new_operand =
          FindOrDie(old_to_new, unfused_instruction->mutable_operand(index));
      TF_CHECK_OK(unfused_instruction->ReplaceOperandWith(index, new_operand));
    }
  }

  HloInstruction* unfused_root = unfused_instructions.front();
  TF_CHECK_OK(instruction_to_merge->ReplaceAllUsesWith(unfused_root));

  TF_CHECK_OK(
      instruction_to_merge->parent()->RemoveInstruction(instruction_to_merge));
  if (GetModule()) {
    TF_CHECK_OK(GetModule()->RemoveEmbeddedComputation(computation_to_merge));
  }

  // Fuse the root instruction and generate multiple outputs.
  FuseInstructionIntoMultiOutput(unfused_root);
  TF_CHECK_OK(unfused_root->parent()->RemoveInstruction(unfused_root));
  // The rest instructions are of normal fusing.
  for (int64 i = 1; i < unfused_instructions.size(); i++) {
    auto instruction = unfused_instructions[i];
    FuseInstruction(instruction);
    TF_CHECK_OK(instruction->parent()->RemoveInstruction(instruction));
  }
}

HloComputation* HloFusionInstruction::fused_instructions_computation() const {
  CHECK(!called_computations().empty());
  auto* fused_instructions_computation = called_computations().front();
  CHECK(fused_instructions_computation->IsFusionComputation())
      << "Computation " << fused_instructions_computation->name()
      << " is not a fusion kind";
  return fused_instructions_computation;
}

HloInstruction* HloFusionInstruction::fused_expression_root() const {
  return fused_instructions_computation()->root_instruction();
}

HloInstruction* HloFusionInstruction::fused_parameter(
    int64 parameter_number) const {
  return fused_instructions_computation()->parameter_instruction(
      parameter_number);
}

const std::vector<HloInstruction*>& HloFusionInstruction::fused_parameters()
    const {
  return fused_instructions_computation()->parameter_instructions();
}

const tensorflow::gtl::iterator_range<UnwrappingIterator<
    std::list<std::unique_ptr<HloInstruction>>::const_iterator>>
HloFusionInstruction::fused_instructions() const {
  const HloComputation* subcomp = fused_instructions_computation();
  return subcomp->instructions();
}

const tensorflow::gtl::iterator_range<
    UnwrappingIterator<std::list<std::unique_ptr<HloInstruction>>::iterator>>
HloFusionInstruction::fused_instructions() {
  return fused_instructions_computation()->instructions();
}

int64 HloFusionInstruction::fused_instruction_count() const {
  return fused_instructions_computation()->instruction_count();
}

HloInstruction* HloFusionInstruction::FuseInstructionInternal(
    HloInstruction* instruction_to_fuse, bool add_output) {
  // When add_output is false, this fusion instruction must be a user of
  // instruction_to_fuse.
  if (!add_output) {
    CHECK(IsUserOf(instruction_to_fuse));
  }
  HloInstruction* fused_instruction =
      CloneAndFuseInternal(instruction_to_fuse, add_output);
  return fused_instruction;
}

HloInstruction* HloFusionInstruction::CloneAndFuseInternal(
    HloInstruction* instruction_to_fuse, bool add_output) {
  CHECK(instruction_to_fuse->IsFusable()) << instruction_to_fuse->ToString();
  VLOG(3) << "CloneAndFuseInternal:\n" << instruction_to_fuse->ToString();
  HloInstruction* clone = nullptr;
  if (called_computations().empty()) {
    // New fusion instruction. It should not be a multioutput instruction.
    CHECK(!add_output);
    auto builder = HloComputation::Builder("fused_computation", this);
    builder.AddInstruction(instruction_to_fuse->Clone(/*suffix=*/""));
    AppendComputation(
        CHECK_NOTNULL(GetModule())->AddEmbeddedComputation(builder.Build()));
    clone = fused_expression_root();
  } else {
    clone = fused_instructions_computation()->AddInstruction(
        instruction_to_fuse->Clone(/*suffix=*/""));
    // When add_output is false, instruction_to_fuse is necessarily an operand
    // of the fusion instruction. After fusion this will no longer be the
    // case. Remove the operand from the operand list and remove its
    // corresponding fused parameter instruction. Renumber parameters as
    // necessary to make parameter numbers consistent with their index in the
    // fused_parameter_ vector.
    bool in_operand_list = std::find(operands().begin(), operands().end(),
                                     instruction_to_fuse) != operands().end();
    CHECK(add_output || in_operand_list);
    const std::vector<HloInstruction*>& fused_parameters =
        fused_instructions_computation()->parameter_instructions();
    for (int64 operand_num = 0; operand_num < operand_count(); ++operand_num) {
      if (instruction_to_fuse == operand(operand_num)) {
        // replace the fused parameter instruction's uses with the clone.
        HloInstruction* fused_parameter = fused_parameters[operand_num];
        TF_CHECK_OK(fused_parameter->ReplaceAllUsesWith(clone));

        // Remove the corresponding fused parameter and operand from their
        // respective vectors.
        TF_CHECK_OK(
            fused_instructions_computation()->RemoveParameter(operand_num));
        RemoveOperandAt(operand_num);
        break;
      }
    }
    // We've cloned instruction_to_fuse into this fusion instruction, so this
    // fusion instruction is no longer a use of instruction_to_fuse.
    if (in_operand_list) {
      DetachFrom(instruction_to_fuse);
      // When the instruction_to_fuse does not have other users, we don't need
      // to generate a multioutput fusion instruction.
      if (instruction_to_fuse->user_count() == 0) {
        add_output = false;
      }
    }
  }

  // Reread the parameters in the computation.
  const std::vector<HloInstruction*>& fused_parameters =
      fused_instructions_computation()->parameter_instructions();

  // Add each operand of the clone as an operand of the fusion instruction. A
  // complication is that some clone operands may already be operands of the
  // fusion instruction.
  for (int64 operand_num = 0; operand_num < clone->operand_count();
       ++operand_num) {
    HloInstruction* operand = clone->mutable_operand(operand_num);

    // See if this operand is already an operand of the fusion node.
    CHECK_EQ(operands().size(), fused_parameters.size());
    HloInstruction* fused_param = nullptr;
    for (int64 i = 0; i < operands().size(); ++i) {
      if (this->operand(i) == operand) {
        fused_param = fused_parameters[i];
        break;
      }
    }

    if (fused_param == nullptr) {
      // Clone's operand was not already an operand of the fusion
      // instruction. Add it as an operand and add a corresponding fused
      // parameter instruction.
      fused_param = AddFusionOperand(operand);
    }
    TF_CHECK_OK(clone->ReplaceOperandWith(operand_num, fused_param));
  }

  if (add_output) {
    CHECK_GT(instruction_to_fuse->user_count(), 0);
    // If this is already a multioutput fusion instruction, expand the root
    // tuple by 1.
    HloInstruction* fused_root = fused_expression_root();
    HloInstruction::InstructionVector tuple_elements;
    bool newly_created_tuple_instr = false;
    if (fused_root->opcode() == HloOpcode::kTuple) {
      tuple_elements = fused_root->operands();
    } else {
      tuple_elements.push_back(fused_root);
      newly_created_tuple_instr = true;
    }
    if (clone->opcode() == HloOpcode::kTuple) {
      for (auto inst : clone->operands()) {
        tuple_elements.push_back(inst);
      }
    } else {
      tuple_elements.push_back(clone);
    }
    HloInstruction* new_root = fused_instructions_computation()->AddInstruction(
        HloInstruction::CreateTuple(tuple_elements));
    fused_instructions_computation()->set_root_instruction(new_root);
    *mutable_shape() = new_root->shape();
    if (fused_root->opcode() == HloOpcode::kTuple) {
      TF_CHECK_OK(
          fused_instructions_computation()->RemoveInstruction(fused_root));
    }

    // If this is a newly created multioutput instruction, we need to update
    // the use of the original fusion instruction.
    if (newly_created_tuple_instr) {
      HloInstruction* new_instr = parent()->AddInstruction(
          HloInstruction::CreateGetTupleElement(fused_root->shape(), this, 0));
      TF_CHECK_OK(ReplaceAllUsesWith(new_instr));
    }
    int64 index = tuple_elements.size();
    if (instruction_to_fuse->opcode() == HloOpcode::kTuple) {
      index -= instruction_to_fuse->operand_count();
      std::vector<HloInstruction*> to_be_removed;
      for (auto old_gte : instruction_to_fuse->users()) {
        CHECK_EQ(old_gte->opcode(), HloOpcode::kGetTupleElement);
        int64 old_tuple_index = old_gte->tuple_index();
        HloInstruction* new_gte =
            parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
                old_gte->shape(), this, index + old_tuple_index));
        TF_CHECK_OK(old_gte->ReplaceAllUsesWith(new_gte));
        to_be_removed.push_back(old_gte);
      }
      for (auto old_gte : to_be_removed) {
        TF_CHECK_OK(parent()->RemoveInstruction(old_gte));
      }
      TF_CHECK_OK(fused_instructions_computation()->RemoveInstruction(clone));
    } else {
      HloInstruction* new_gte =
          parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
              clone->shape(), this, index - 1));
      TF_CHECK_OK(instruction_to_fuse->ReplaceAllUsesWith(new_gte));
    }
  }

  VLOG(2) << "New clone:\n" << clone->ToString();
  return clone;
}

std::vector<string> HloFusionInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {StrCat("kind=", xla::ToString(fusion_kind()))};
}

bool HloFusionInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  return fusion_kind() == other.fusion_kind() &&
         eq_computations(fused_instructions_computation(),
                         other.fused_instructions_computation());
}

std::unique_ptr<HloInstruction> HloFusionInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  HloModule* module = context != nullptr ? context->module() : GetModule();
  HloComputation* new_fused_computation = nullptr;
  if (context != nullptr) {
    new_fused_computation =
        context->FindComputation(fused_instructions_computation());
  }
  if (new_fused_computation == nullptr) {
    new_fused_computation = module->AddEmbeddedComputation(
        fused_instructions_computation()->Clone("clone", context));
  }
  return MakeUnique<HloFusionInstruction>(shape, fusion_kind(), new_operands,
                                          new_fused_computation);
}

HloRngInstruction::HloRngInstruction(
    const Shape& shape, RandomDistribution distribution,
    tensorflow::gtl::ArraySlice<HloInstruction*> parameters)
    : HloInstruction(HloOpcode::kRng, shape), distribution_(distribution) {
  for (HloInstruction* param : parameters) {
    AppendOperand(param);
  }
}

HloInstructionProto HloRngInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_distribution(distribution_);
  return proto;
}

std::vector<string> HloRngInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {StrCat("distribution=", RandomDistributionToString(distribution_))};
}

bool HloRngInstruction::IsElementwiseImpl(
    const tensorflow::gtl::optional<int64>& operand_idx) const {
  return true;
}

bool HloRngInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  return false;
}

std::unique_ptr<HloInstruction> HloRngInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  return MakeUnique<HloRngInstruction>(shape, distribution_, new_operands);
}

HloParameterInstruction::HloParameterInstruction(int64 parameter_number,
                                                 const Shape& shape,
                                                 const string& name)
    : HloInstruction(HloOpcode::kParameter, shape),
      parameter_number_(parameter_number) {
  SetAndSanitizeName(name);
}

HloInstructionProto HloParameterInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_parameter_number(parameter_number_);
  return proto;
}

string HloParameterInstruction::OperandsToStringWithCanonicalNameMap(
    const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
  return StrCat(parameter_number_);
}

bool HloParameterInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other = static_cast<const HloParameterInstruction&>(other);
  return parameter_number() == casted_other.parameter_number();
}

std::unique_ptr<HloInstruction>
HloParameterInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  return MakeUnique<HloParameterInstruction>(parameter_number_, shape, name());
}

HloGetTupleElementInstruction::HloGetTupleElementInstruction(
    const Shape& shape, HloInstruction* operand, int64 index)
    : HloInstruction(HloOpcode::kGetTupleElement, shape), tuple_index_(index) {
  CHECK(ShapeUtil::IsTuple(operand->shape()));
  AppendOperand(operand);
}

HloInstructionProto HloGetTupleElementInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_tuple_index(tuple_index_);
  return proto;
}

std::vector<string> HloGetTupleElementInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {StrCat("index=", tuple_index())};
}

bool HloGetTupleElementInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloGetTupleElementInstruction&>(other);
  return tuple_index() == casted_other.tuple_index();
}

std::unique_ptr<HloInstruction>
HloGetTupleElementInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return MakeUnique<HloGetTupleElementInstruction>(shape, new_operands[0],
                                                   tuple_index());
}

HloReducePrecisionInstruction::HloReducePrecisionInstruction(
    const Shape& shape, HloInstruction* operand, const int exponent_bits,
    const int mantissa_bits)
    : HloInstruction(HloOpcode::kReducePrecision, shape),
      exponent_bits_(exponent_bits),
      mantissa_bits_(mantissa_bits) {
  AppendOperand(operand);
}

HloInstructionProto HloReducePrecisionInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_exponent_bits(exponent_bits_);
  proto.set_mantissa_bits(mantissa_bits_);
  return proto;
}

std::vector<string> HloReducePrecisionInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {StrCat("exponent_bits=", exponent_bits_),
          StrCat("mantissa_bits=", mantissa_bits_)};
}

bool HloReducePrecisionInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloReducePrecisionInstruction&>(other);
  // A reduce-precision operation is determined by the bit sizes.
  return exponent_bits() == casted_other.exponent_bits() &&
         mantissa_bits() == casted_other.mantissa_bits();
}

std::unique_ptr<HloInstruction>
HloReducePrecisionInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return MakeUnique<HloReducePrecisionInstruction>(
      shape, new_operands[0], exponent_bits(), mantissa_bits());
}

HloInfeedInstruction::HloInfeedInstruction(const Shape& shape,
                                           const string& config)
    : HloInstruction(HloOpcode::kInfeed, shape), infeed_config_(config) {}

HloInstructionProto HloInfeedInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_infeed_config(infeed_config_);
  return proto;
}

std::vector<string> HloInfeedInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  if (infeed_config_.empty()) {
    return {};
  }
  return {StrCat("infeed_config=\"", CEscape(infeed_config_), "\"")};
}

bool HloInfeedInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  // Not yet supported.
  return false;
}

std::unique_ptr<HloInstruction> HloInfeedInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 0);
  return MakeUnique<HloInfeedInstruction>(shape, infeed_config());
}

HloOutfeedInstruction::HloOutfeedInstruction(
    const Shape& shape, HloInstruction* operand,
    tensorflow::StringPiece outfeed_config)
    : HloInstruction(HloOpcode::kOutfeed, ShapeUtil::MakeNil()),
      outfeed_shape_(shape),
      outfeed_config_(outfeed_config.begin(), outfeed_config.end()) {
  CHECK(ShapeUtil::Compatible(operand->shape(), shape))
      << "Outfeed shape " << shape << " must be compatible with operand shape "
      << operand->shape();
  AppendOperand(operand);
}

HloInstructionProto HloOutfeedInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_outfeed_config(outfeed_config());
  *proto.mutable_outfeed_shape() = outfeed_shape();
  return proto;
}

std::vector<string> HloOutfeedInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  if (outfeed_config_.empty()) {
    return {};
  }
  return {StrCat("outfeed_config=\"", CEscape(outfeed_config_), "\"")};
}

bool HloOutfeedInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  // Not yet supported.
  return false;
}

std::unique_ptr<HloInstruction> HloOutfeedInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return MakeUnique<HloOutfeedInstruction>(outfeed_shape(), new_operands[0],
                                           outfeed_config());
}

HloConvolutionInstruction::HloConvolutionInstruction(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    const Window& window, const ConvolutionDimensionNumbers& dimension_numbers)
    : HloInstruction(HloOpcode::kConvolution, shape),
      window_(window),
      convolution_dimension_numbers_(dimension_numbers) {
  if (window_util::HasBaseDilation(window)) {
    SetAndSanitizeName(StrCat(name(), "-base-dilated"));
  }
  if (window_util::HasWindowDilation(window)) {
    SetAndSanitizeName(StrCat(name(), "-window-dilated"));
  }
  AppendOperand(lhs);
  AppendOperand(rhs);
}

string HloConvolutionInstruction::ToCategory() const {
  string category = "convolution";
  if (window_util::HasBaseDilation(window())) {
    category += " base-dilated";
  }
  if (window_util::HasWindowDilation(window())) {
    category += " window-dilated";
  }
  return category;
}

HloInstructionProto HloConvolutionInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_window() = window_;
  *proto.mutable_convolution_dimension_numbers() =
      convolution_dimension_numbers_;
  return proto;
}

std::vector<string> HloConvolutionInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<string> extra;
  if (window_.dimensions_size() != 0) {
    extra.push_back(StrCat("window={", window_util::ToString(window()), "}"));
  }
  extra.push_back(StrCat("dim_labels=", ConvolutionDimensionNumbersToString(
                                            convolution_dimension_numbers_)));
  return extra;
}

bool HloConvolutionInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloConvolutionInstruction&>(other);
  return protobuf_util::ProtobufEquals(window(), casted_other.window()) &&
         protobuf_util::ProtobufEquals(
             convolution_dimension_numbers(),
             casted_other.convolution_dimension_numbers());
}

std::unique_ptr<HloInstruction>
HloConvolutionInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return MakeUnique<HloConvolutionInstruction>(shape, new_operands[0],
                                               new_operands[1], window(),
                                               convolution_dimension_numbers_);
}

HloReduceWindowInstruction::HloReduceWindowInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* init_value,
    const Window& window, HloComputation* reduce_computation)
    : HloInstruction(HloOpcode::kReduceWindow, shape), window_(window) {
  AppendOperand(operand);
  AppendOperand(init_value);
  AppendComputation(reduce_computation);
}

HloInstructionProto HloReduceWindowInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_window() = window_;
  return proto;
}

std::vector<string> HloReduceWindowInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<string> extra;
  if (window_.dimensions_size() != 0) {
    extra.push_back(StrCat("window={", window_util::ToString(window()), "}"));
  }
  return extra;
}

bool HloReduceWindowInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloReduceWindowInstruction&>(other);
  return eq_computations(to_apply(), casted_other.to_apply()) &&
         protobuf_util::ProtobufEquals(window(), casted_other.window());
}

std::unique_ptr<HloInstruction>
HloReduceWindowInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return MakeUnique<HloReduceWindowInstruction>(
      shape, new_operands[0], new_operands[1], window(), to_apply());
}

HloSelectAndScatterInstruction::HloSelectAndScatterInstruction(
    const Shape& shape, HloInstruction* operand, HloComputation* select,
    const Window& window, HloInstruction* source, HloInstruction* init_value,
    HloComputation* scatter)
    : HloInstruction(HloOpcode::kSelectAndScatter, shape), window_(window) {
  AppendOperand(operand);
  AppendOperand(source);
  AppendOperand(init_value);
  // Select comes before scatter in the vector.
  AppendComputation(select);
  AppendComputation(scatter);
}

HloInstructionProto HloSelectAndScatterInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_window() = window_;
  return proto;
}

std::vector<string> HloSelectAndScatterInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<string> extra;
  if (window_.dimensions_size() != 0) {
    extra.push_back(StrCat("window={", window_util::ToString(window()), "}"));
  }
  return extra;
}

bool HloSelectAndScatterInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloSelectAndScatterInstruction&>(other);
  return eq_computations(select(), casted_other.select()) &&
         eq_computations(scatter(), casted_other.scatter()) &&
         protobuf_util::ProtobufEquals(window(), casted_other.window());
}

std::unique_ptr<HloInstruction>
HloSelectAndScatterInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 3);
  return MakeUnique<HloSelectAndScatterInstruction>(
      shape, new_operands[0], select(), window(), new_operands[1],
      new_operands[2], scatter());
}

HloCustomCallInstruction::HloCustomCallInstruction(
    const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    tensorflow::StringPiece custom_call_target)
    : HloInstruction(HloOpcode::kCustomCall, shape),
      custom_call_target_(custom_call_target.begin(),
                          custom_call_target.end()) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
}

HloInstructionProto HloCustomCallInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  if (window_ != nullptr) {
    *proto.mutable_window() = *window_;
  }
  if (convolution_dimension_numbers_ != nullptr) {
    *proto.mutable_convolution_dimension_numbers() =
        *convolution_dimension_numbers_;
  }
  proto.set_custom_call_target(custom_call_target_);
  return proto;
}

std::vector<string> HloCustomCallInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<string> extra;
  if (window_ != nullptr && window_->dimensions_size() != 0) {
    extra.push_back(StrCat("window={", window_util::ToString(*window_), "}"));
  }
  if (convolution_dimension_numbers_ != nullptr) {
    extra.push_back(StrCat(
        "dim_labels=",
        ConvolutionDimensionNumbersToString(*convolution_dimension_numbers_)));
  }
  // By contract, we print the custom call target even if
  // options.print_subcomputation_mode() == kOff, because the call target is not
  // an HloComputation.
  extra.push_back(
      StrCat("custom_call_target=\"", CEscape(custom_call_target_), "\""));
  return extra;
}

bool HloCustomCallInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloCustomCallInstruction&>(other);
  if ((window_ == nullptr) != (casted_other.window_ == nullptr) ||
      (window_ != nullptr &&
       !protobuf_util::ProtobufEquals(*window_, *casted_other.window_))) {
    return false;
  }
  if ((convolution_dimension_numbers_ == nullptr) !=
          (casted_other.convolution_dimension_numbers_ == nullptr) ||
      (convolution_dimension_numbers_ != nullptr &&
       !protobuf_util::ProtobufEquals(
           convolution_dimension_numbers(),
           casted_other.convolution_dimension_numbers()))) {
    return false;
  }
  return custom_call_target_ == casted_other.custom_call_target_;
}

std::unique_ptr<HloInstruction>
HloCustomCallInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  auto cloned = MakeUnique<HloCustomCallInstruction>(shape, new_operands,
                                                     custom_call_target());
  if (window_ != nullptr) {
    cloned->set_window(*window_);
  }
  if (convolution_dimension_numbers_ != nullptr) {
    cloned->set_convolution_dimension_numbers(*convolution_dimension_numbers_);
  }
  return std::move(cloned);
}

HloHostComputeInstruction::HloHostComputeInstruction(
    const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
    tensorflow::StringPiece channel_name, const int64 cost_estimate_ns)
    : HloInstruction(HloOpcode::kHostCompute, shape),
      channel_name_(channel_name.begin(), channel_name.end()),
      cost_estimate_ns_(cost_estimate_ns) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
}

HloInstructionProto HloHostComputeInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_channel_name(channel_name_);
  proto.set_cost_estimate_ns(cost_estimate_ns_);
  return proto;
}

bool HloHostComputeInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  // Not yet supported.
  return false;
}

std::unique_ptr<HloInstruction>
HloHostComputeInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  return MakeUnique<HloHostComputeInstruction>(
      shape, new_operands, channel_name_, cost_estimate_ns_);
}

HloPadInstruction::HloPadInstruction(const Shape& shape,
                                     HloInstruction* operand,
                                     HloInstruction* padding_value,
                                     const PaddingConfig& padding_config)
    : HloInstruction(HloOpcode::kPad, shape), padding_config_(padding_config) {
  AppendOperand(operand);
  AppendOperand(padding_value);
}

HloInstructionProto HloPadInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_padding_config() = padding_config_;
  return proto;
}

std::vector<string> HloPadInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {StrCat("padding=", xla::PaddingConfigToString(padding_config_))};
}

bool HloPadInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other = static_cast<const HloPadInstruction&>(other);
  return protobuf_util::ProtobufEquals(padding_config(),
                                       casted_other.padding_config());
}

std::unique_ptr<HloInstruction> HloPadInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return MakeUnique<HloPadInstruction>(shape, new_operands[0], new_operands[1],
                                       padding_config_);
}

HloDynamicSliceInstruction::HloDynamicSliceInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* start_indices,
    tensorflow::gtl::ArraySlice<int64> slice_sizes)
    : HloInstruction(HloOpcode::kDynamicSlice, shape),
      dynamic_slice_sizes_(slice_sizes.begin(), slice_sizes.end()) {
  AppendOperand(operand);
  AppendOperand(start_indices);
}

HloInstructionProto HloDynamicSliceInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64 slice_size : dynamic_slice_sizes_) {
    proto.add_dynamic_slice_sizes(slice_size);
  }
  return proto;
}

std::vector<string> HloDynamicSliceInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {
      StrCat("dynamic_slice_sizes={", Join(dynamic_slice_sizes(), ","), "}")};
}

bool HloDynamicSliceInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  return true;
}

std::unique_ptr<HloInstruction>
HloDynamicSliceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape,
    tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return MakeUnique<HloDynamicSliceInstruction>(
      shape, new_operands[0], new_operands[1], dynamic_slice_sizes_);
}
}  // namespace xla
