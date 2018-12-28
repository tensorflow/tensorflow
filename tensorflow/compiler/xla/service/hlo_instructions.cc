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

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_sharding_metadata.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace {

using absl::CEscape;
using absl::StrAppend;
using absl::StrCat;
using absl::StrJoin;

bool IsInstructionElementwiseOnOperand(const HloInstruction* instruction,
                                       const HloInstruction* operand) {
  std::vector<int64> operand_indices = instruction->OperandIndices(operand);
  return std::all_of(
      operand_indices.begin(), operand_indices.end(),
      [instruction](int64 operand_index) {
        return instruction->IsElementwiseOnOperand(operand_index);
      });
}

string PrecisionConfigToString(const PrecisionConfig& precision_config) {
  if (absl::c_all_of(precision_config.operand_precision(), [](int32 precision) {
        return static_cast<PrecisionConfig::Precision>(precision) ==
               PrecisionConfig::DEFAULT;
      })) {
    return "";
  }

  return StrCat(
      "operand_precision={",
      StrJoin(
          precision_config.operand_precision(), ",",
          [](string* out, int32 precision) {
            CHECK(PrecisionConfig::Precision_IsValid(precision)) << precision;
            StrAppend(out,
                      PrecisionToString(
                          static_cast<PrecisionConfig::Precision>(precision)));
          }),
      "}");
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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 3);
  return absl::make_unique<HloBatchNormTrainingInstruction>(
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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 5);
  return absl::make_unique<HloBatchNormInferenceInstruction>(
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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 5);
  return absl::make_unique<HloBatchNormGradInstruction>(
      shape, new_operands[0], new_operands[1], new_operands[2], new_operands[3],
      new_operands[4], epsilon(), feature_index());
}

HloFftInstruction::HloFftInstruction(const Shape& shape,
                                     HloInstruction* operand, FftType fft_type,
                                     absl::Span<const int64> fft_length)
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
          StrCat("fft_length={", StrJoin(fft_length(), ","), "}")};
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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloFftInstruction>(shape, new_operands[0], fft_type_,
                                              fft_length_);
}

HloSendRecvInstruction::HloSendRecvInstruction(HloOpcode opcode,
                                               const Shape& shape,
                                               int64 channel_id,
                                               bool is_host_transfer)
    : HloInstruction(opcode, shape),
      channel_id_(channel_id),
      is_host_transfer_(is_host_transfer) {}

HloInstructionProto HloSendRecvInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_channel_id(channel_id_);
  proto.set_is_host_transfer(is_host_transfer_);
  return proto;
}

std::vector<string> HloSendRecvInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<string> attrs;
  attrs.push_back(StrCat("channel_id=", channel_id_));
  if (is_host_transfer()) {
    attrs.push_back("is_host_transfer=true");
  }
  return attrs;
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
                                       HloInstruction* token, int64 channel_id,
                                       bool is_host_transfer)
    : HloSendRecvInstruction(
          HloOpcode::kSend,
          ShapeUtil::MakeTupleShape({CHECK_NOTNULL(operand)->shape(),
                                     ShapeUtil::MakeShape(U32, {}),
                                     ShapeUtil::MakeTokenShape()}),
          channel_id, is_host_transfer) {
  AppendOperand(operand);
  AppendOperand(token);
}

std::unique_ptr<HloInstruction> HloSendInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloSendInstruction>(
      new_operands[0], new_operands[1], channel_id(), is_host_transfer());
}

HloSendDoneInstruction::HloSendDoneInstruction(HloSendInstruction* operand,
                                               bool is_host_transfer)
    : HloSendRecvInstruction(HloOpcode::kSendDone, ShapeUtil::MakeTokenShape(),
                             CHECK_NOTNULL(operand)->channel_id(),
                             is_host_transfer) {
  AppendOperand(operand);
}

std::unique_ptr<HloInstruction>
HloSendDoneInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloSendDoneInstruction>(
      Cast<HloSendInstruction>(new_operands[0]), is_host_transfer());
}

// Recv instruction produces a tuple of {receive buffer, U32 context}.
HloRecvInstruction::HloRecvInstruction(const Shape& shape,
                                       HloInstruction* token, int64 channel_id,
                                       bool is_host_transfer)
    : HloSendRecvInstruction(
          HloOpcode::kRecv,
          ShapeUtil::MakeTupleShape({shape, ShapeUtil::MakeShape(U32, {}),
                                     ShapeUtil::MakeTokenShape()}),
          channel_id, is_host_transfer) {
  AppendOperand(token);
}

std::unique_ptr<HloInstruction> HloRecvInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloRecvInstruction>(
      ShapeUtil::GetTupleElementShape(shape, 0), new_operands[0], channel_id(),
      is_host_transfer());
}

HloRecvDoneInstruction::HloRecvDoneInstruction(HloRecvInstruction* operand,
                                               bool is_host_transfer)
    : HloSendRecvInstruction(
          HloOpcode::kRecvDone,
          ShapeUtil::MakeTupleShape(
              {ShapeUtil::GetTupleElementShape(operand->shape(), 0),
               ShapeUtil::MakeTokenShape()}),
          CHECK_NOTNULL(operand)->channel_id(), is_host_transfer) {
  AppendOperand(operand);
}

std::unique_ptr<HloInstruction>
HloRecvDoneInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloRecvDoneInstruction>(
      Cast<HloRecvInstruction>(new_operands[0]), is_host_transfer());
}

HloCollectiveInstruction::HloCollectiveInstruction(
    HloOpcode opcode, const Shape& shape,
    absl::Span<HloInstruction* const> operands,
    const std::vector<ReplicaGroup>& replica_groups)
    : HloInstruction(opcode, shape), replica_groups_(replica_groups) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
}

HloInstructionProto HloCollectiveInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_replica_groups() = {replica_groups_.begin(),
                                     replica_groups_.end()};
  return proto;
}

std::vector<string> HloCollectiveInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& /*options*/) const {
  std::vector<string> result;
  std::vector<string> replica_group_str;
  for (const ReplicaGroup& group : replica_groups()) {
    replica_group_str.push_back(
        StrCat("{", StrJoin(group.replica_ids(), ","), "}"));
  }
  result.push_back(
      StrCat("replica_groups={", StrJoin(replica_group_str, ","), "}"));
  return result;
}

bool HloCollectiveInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
    /*eq_computations*/) const {
  const auto& casted_other =
      static_cast<const HloCollectiveInstruction&>(other);
  return absl::c_equal(replica_groups(), casted_other.replica_groups(),
                       [](const ReplicaGroup& a, const ReplicaGroup& b) {
                         return absl::c_equal(a.replica_ids(), b.replica_ids());
                       });
}

HloAllReduceInstruction::HloAllReduceInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    HloComputation* reduce_computation,
    const std::vector<ReplicaGroup>& replica_groups, absl::string_view barrier,
    const absl::optional<int64>& all_reduce_id)
    : HloCollectiveInstruction(HloOpcode::kAllReduce, shape, operands,
                               replica_groups),
      all_reduce_barrier_(barrier),
      all_reduce_id_(all_reduce_id) {
  AppendComputation(reduce_computation);
}

void HloAllReduceInstruction::set_all_reduce_id(
    const absl::optional<int64>& all_reduce_id) {
  all_reduce_id_ = all_reduce_id;
}

HloInstructionProto HloAllReduceInstruction::ToProto() const {
  HloInstructionProto proto = HloCollectiveInstruction::ToProto();
  // Proto3 is so sad.
  if (all_reduce_id_) {
    proto.set_all_reduce_id(*all_reduce_id_);
  }
  proto.set_all_reduce_barrier(all_reduce_barrier_);
  return proto;
}

std::vector<string> HloAllReduceInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<string> result =
      HloCollectiveInstruction::ExtraAttributesToStringImpl(options);
  if (!all_reduce_barrier().empty()) {
    result.push_back(StrCat("barrier=\"", all_reduce_barrier(), "\""));
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
  return HloCollectiveInstruction::IdenticalSlowPath(other, eq_computations) &&
         eq_computations(to_apply(), casted_other.to_apply()) &&
         all_reduce_barrier() == casted_other.all_reduce_barrier() &&
         all_reduce_id() == casted_other.all_reduce_id();
}

std::unique_ptr<HloInstruction>
HloAllReduceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  return absl::make_unique<HloAllReduceInstruction>(
      shape, new_operands, to_apply(), replica_groups(), all_reduce_barrier(),
      all_reduce_id());
}

HloAllToAllInstruction::HloAllToAllInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    const std::vector<ReplicaGroup>& replica_groups)
    : HloCollectiveInstruction(HloOpcode::kAllToAll, shape, operands,
                               replica_groups) {}

std::unique_ptr<HloInstruction>
HloAllToAllInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  return absl::make_unique<HloAllToAllInstruction>(shape, new_operands,
                                                   replica_groups());
}

HloCollectivePermuteInstruction::HloCollectivePermuteInstruction(
    const Shape& shape, HloInstruction* operand,
    const std::vector<std::pair<int64, int64>>& source_target_pairs)
    : HloInstruction(HloOpcode::kCollectivePermute, shape),
      source_target_pairs_(source_target_pairs) {
  AppendOperand(operand);
}

HloInstructionProto HloCollectivePermuteInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (const auto& pair : source_target_pairs()) {
    auto* proto_pair = proto.add_source_target_pairs();
    proto_pair->set_source(pair.first);
    proto_pair->set_target(pair.second);
  }
  return proto;
}

std::vector<string>
HloCollectivePermuteInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& /*options*/) const {
  std::vector<string> result;
  std::vector<string> strs;
  for (const auto& pair : source_target_pairs()) {
    strs.push_back(StrCat("{", pair.first, ",", pair.second, "}"));
  }
  result.push_back(StrCat("source_target_pairs={", StrJoin(strs, ","), "}"));
  return result;
}

bool HloCollectivePermuteInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
    /*eq_computations*/) const {
  const auto& casted_other =
      static_cast<const HloCollectivePermuteInstruction&>(other);
  return absl::c_equal(source_target_pairs(),
                       casted_other.source_target_pairs(),
                       [](const std::pair<int64, int64>& a,
                          const std::pair<int64, int64>& b) { return a == b; });
}

std::unique_ptr<HloInstruction>
HloCollectivePermuteInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  return absl::make_unique<HloCollectivePermuteInstruction>(
      shape, new_operands[0], source_target_pairs());
}

HloReverseInstruction::HloReverseInstruction(const Shape& shape,
                                             HloInstruction* operand,
                                             absl::Span<const int64> dimensions)
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
  return {StrCat("dimensions={", StrJoin(dimensions(), ","), "}")};
}

bool HloReverseInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other = static_cast<const HloReverseInstruction&>(other);
  return dimensions() == casted_other.dimensions();
}

std::unique_ptr<HloInstruction> HloReverseInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloReverseInstruction>(shape, new_operands[0],
                                                  dimensions());
}

HloConcatenateInstruction::HloConcatenateInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
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
  return {StrCat("dimensions={", StrJoin(dimensions(), ","), "}")};
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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return absl::make_unique<HloConcatenateInstruction>(shape, new_operands,
                                                      dimensions(0));
}

HloReduceInstruction::HloReduceInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> args,
    absl::Span<const int64> dimensions_to_reduce,
    HloComputation* reduce_computation)
    : HloInstruction(HloOpcode::kReduce, shape),
      dimensions_(dimensions_to_reduce.begin(), dimensions_to_reduce.end()) {
  for (HloInstruction* arg : args) {
    AppendOperand(arg);
  }
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
  return {StrCat("dimensions={", StrJoin(dimensions(), ","), "}")};
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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size() % 2, 0);
  return absl::make_unique<HloReduceInstruction>(shape, new_operands,
                                                 dimensions(), to_apply());
}

HloSortInstruction::HloSortInstruction(const Shape& shape, int64 dimension,
                                       HloInstruction* keys,
                                       absl::Span<HloInstruction* const> values)
    : HloInstruction(HloOpcode::kSort, shape), dimensions_({dimension}) {
  AppendOperand(keys);
  for (auto* value : values) {
    AppendOperand(value);
  }
}

HloInstructionProto HloSortInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  for (int64 dimension : dimensions_) {
    proto.add_dimensions(dimension);
  }
  return proto;
}

std::vector<string> HloSortInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {StrCat("dimensions={", StrJoin(dimensions(), ","), "}")};
}

bool HloSortInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other = static_cast<const HloSortInstruction&>(other);
  return dimensions() == casted_other.dimensions();
}

std::unique_ptr<HloInstruction> HloSortInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  HloInstruction* keys = new_operands[0];
  return absl::make_unique<HloSortInstruction>(shape, dimensions(0), keys,
                                               new_operands.subspan(1));
}

HloTransposeInstruction::HloTransposeInstruction(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64> dimensions)
    : HloInstruction(HloOpcode::kTranspose, shape),
      dimensions_(dimensions.begin(), dimensions.end()) {
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
  return {StrCat("dimensions={", StrJoin(dimensions(), ","), "}")};
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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloTransposeInstruction>(shape, new_operands[0],
                                                    dimensions());
}

HloBroadcastInstruction::HloBroadcastInstruction(
    const Shape& shape, HloInstruction* operand,
    absl::Span<const int64> broadcast_dimension)
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
  return {StrCat("dimensions={", StrJoin(dimensions(), ","), "}")};
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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloBroadcastInstruction>(shape, new_operands[0],
                                                    dimensions());
}

HloMapInstruction::HloMapInstruction(const Shape& shape,
                                     absl::Span<HloInstruction* const> operands,
                                     HloComputation* map_computation)
    : HloInstruction(HloOpcode::kMap, shape) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
  AppendComputation(map_computation);
  // TODO(b/65689298) Remove code below once Map is generalized to accept
  // arbitrary map dimensions.
  dimensions_.resize(shape.rank());
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
    const absl::optional<int64>& operand_idx) const {
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
  return {StrCat("dimensions={", StrJoin(dimensions(), ","), "}")};
}

bool HloMapInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  return eq_computations(to_apply(), other.to_apply());
}

std::unique_ptr<HloInstruction> HloMapInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return absl::make_unique<HloMapInstruction>(shape, new_operands, to_apply());
}

HloSliceInstruction::HloSliceInstruction(const Shape& shape,
                                         HloInstruction* operand,
                                         absl::Span<const int64> start_indices,
                                         absl::Span<const int64> limit_indices,
                                         absl::Span<const int64> strides)
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
  return {StrCat("slice={", StrJoin(bounds, ", "), "}")};
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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloSliceInstruction>(
      shape, new_operands[0], slice_starts_, slice_limits_, slice_strides_);
}

HloConstantInstruction::HloConstantInstruction(Literal literal)
    : HloInstruction(HloOpcode::kConstant, literal.shape()),
      literal_(std::move(literal)) {}

HloConstantInstruction::HloConstantInstruction(const Shape& shape)
    : HloInstruction(HloOpcode::kConstant, shape) {}

HloInstructionProto HloConstantInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  if (literal_.has_value()) {
    *proto.mutable_literal() = literal_->ToProto();
  }
  return proto;
}

bool HloConstantInstruction::IsElementwiseImpl(
    const absl::optional<int64>& operand_idx) const {
  return true;
}

void HloConstantInstruction::RelayoutConstant(const Layout& new_layout,
                                              const ShapeIndex& shape_index) {
  Shape* mutable_array_subshape =
      ShapeUtil::GetMutableSubshape(mutable_shape(), shape_index);
  CHECK(mutable_array_subshape->IsArray());

  // Normally array_subshape will always have a layout, but this invariant is
  // temporarily broken in LayoutAssignment::AssignLayouts.

  if (!mutable_array_subshape->has_layout() ||
      !LayoutUtil::Equal(mutable_array_subshape->layout(), new_layout)) {
    *literal_ = literal_->Relayout(new_layout, shape_index);
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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK(literal_.has_value());
  return absl::make_unique<HloConstantInstruction>(literal_->Clone());
}

string HloConstantInstruction::OperandsToStringWithCanonicalNameMap(
    const HloPrintOptions& options,
    CanonicalNameMap* canonical_name_map) const {
  string operands;
  // For constants, show the actual value in place of an empty operand list.
  if (literal_.has_value() &&
      ((shape().IsArray() && ShapeUtil::ElementsIn(shape()) <= 10) ||
       options.print_large_constants())) {
    // Literal::ToString emits multidimensional arrays over multiple
    // lines. Compact this into one line by stripping out white space.
    string tmp = literal().ToStringWithoutShape();
    std::replace(tmp.begin(), tmp.end(), '\n', ' ');
    std::vector<string> v = absl::StrSplit(tmp, ' ');
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
      literal_(LiteralUtil::CreateR1U8(tag)) {
  AppendOperand(operand);
  operand->set_tracing(this);
}

HloInstructionProto HloTraceInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_literal() = literal_.ToProto();
  return proto;
}

bool HloTraceInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  return false;
}

std::unique_ptr<HloInstruction> HloTraceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
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
    absl::Span<HloInstruction* const> operands,
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
    const absl::optional<int64>& operand_idx) const {
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
  // string param_name = StrCat(new_operand->name(), ".param_", param_no);
  string param_name = StrCat("param_", param_no);
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
  absl::flat_hash_map<HloInstruction*, HloInstruction*> old_to_new;
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
  CHECK(instruction_to_fuse->IsFusible()) << instruction_to_fuse->ToString();
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
    // When add_output is false, instruction_to_fuse is necessarily an operand
    // of the fusion instruction. After fusion this will no longer be the
    // case. Remove the operand from the operand list and remove its
    // corresponding fused parameter instruction. Renumber parameters as
    // necessary to make parameter numbers consistent with their index in the
    // fused_parameter_ vector.
    bool in_operand_list = std::find(operands().begin(), operands().end(),
                                     instruction_to_fuse) != operands().end();
    CHECK(add_output || in_operand_list);
    if (instruction_to_fuse->opcode() == HloOpcode::kTuple) {
      // We assume all uses of a kTuple operation are GTE ops, not another
      // fusion node. In this case, we don't need to clone
      // 'instruction_to_fuse'.
      CHECK(!in_operand_list);
      clone = instruction_to_fuse;
    } else {
      clone = fused_instructions_computation()->AddInstruction(
          instruction_to_fuse->Clone(/*suffix=*/""));
    }
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
      CHECK_EQ(clone, instruction_to_fuse);
      index -= clone->operand_count();
      std::vector<HloInstruction*> to_be_removed;
      for (auto old_gte : clone->users()) {
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
    } else {
      HloInstruction* new_gte =
          parent()->AddInstruction(HloInstruction::CreateGetTupleElement(
              clone->shape(), this, index - 1));
      TF_CHECK_OK(instruction_to_fuse->ReplaceAllUsesWith(new_gte));
    }
  }

  if (clone != instruction_to_fuse) {
    VLOG(2) << "New clone:\n" << clone->ToString();
  }
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

static uint64 HashOperandRecursive(const HloInstruction* hlo) {
  return hlo->Hash(HashOperandRecursive);
}

uint64 HloFusionInstruction::InnerHash() const {
  // Use HashOperandRecursive to recursively compute hash on inner operands.
  return fused_instructions_computation()->root_instruction()->Hash(
      HashOperandRecursive);
}

std::unique_ptr<HloInstruction> HloFusionInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
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
  return absl::make_unique<HloFusionInstruction>(
      shape, fusion_kind(), new_operands, new_fused_computation);
}

Status HloFusionInstruction::DeduplicateFusionOperands() {
  absl::flat_hash_map<const HloInstruction*, int> operand_indices;
  std::vector<int> operands_to_remove;
  for (int i = 0; i < operand_count(); ++i) {
    auto emplace_result = operand_indices.emplace(operand(i), i);
    if (!emplace_result.second) {
      TF_RETURN_IF_ERROR(fused_parameter(i)->ReplaceAllUsesWith(
          fused_parameter(emplace_result.first->second)));
      operands_to_remove.push_back(i);
    }
  }
  if (operands_to_remove.empty()) {
    return Status::OK();
  }
  TF_RETURN_IF_ERROR(
      fused_instructions_computation()->RemoveUnusedParameters());
  RemoveOperandsAtAscendingIndices(operands_to_remove);
  return Status::OK();
}

HloRngInstruction::HloRngInstruction(
    const Shape& shape, RandomDistribution distribution,
    absl::Span<HloInstruction* const> parameters)
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
    const absl::optional<int64>& operand_idx) const {
  return true;
}

bool HloRngInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  return false;
}

std::unique_ptr<HloInstruction> HloRngInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return absl::make_unique<HloRngInstruction>(shape, distribution_,
                                              new_operands);
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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return absl::make_unique<HloParameterInstruction>(parameter_number_, shape,
                                                    name());
}

HloGetTupleElementInstruction::HloGetTupleElementInstruction(
    const Shape& shape, HloInstruction* operand, int64 index)
    : HloInstruction(HloOpcode::kGetTupleElement, shape), tuple_index_(index) {
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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloGetTupleElementInstruction>(
      shape, new_operands[0], tuple_index());
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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloReducePrecisionInstruction>(
      shape, new_operands[0], exponent_bits(), mantissa_bits());
}

HloInfeedInstruction::HloInfeedInstruction(const Shape& infeed_shape,
                                           HloInstruction* token_operand,
                                           const string& config)
    : HloInstruction(HloOpcode::kInfeed,
                     ShapeUtil::MakeTupleShape(
                         {infeed_shape, ShapeUtil::MakeTokenShape()})),
      infeed_config_(config) {
  AppendOperand(token_operand);
}

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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloInfeedInstruction>(
      infeed_shape(), new_operands[0], infeed_config());
}

HloOutfeedInstruction::HloOutfeedInstruction(const Shape& outfeed_shape,
                                             HloInstruction* operand,
                                             HloInstruction* token_operand,
                                             absl::string_view outfeed_config)
    : HloInstruction(HloOpcode::kOutfeed, ShapeUtil::MakeTokenShape()),
      outfeed_shape_(outfeed_shape),
      outfeed_config_(outfeed_config) {
  AppendOperand(operand);
  AppendOperand(token_operand);
}

HloInstructionProto HloOutfeedInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.set_outfeed_config(outfeed_config());
  *proto.mutable_outfeed_shape() = outfeed_shape().ToProto();
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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloOutfeedInstruction>(
      outfeed_shape(), new_operands[0], new_operands[1], outfeed_config());
}

HloConvolutionInstruction::HloConvolutionInstruction(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    int64 feature_group_count, int64 batch_group_count, const Window& window,
    const ConvolutionDimensionNumbers& dimension_numbers,
    const PrecisionConfig& precision_config)
    : HloInstruction(HloOpcode::kConvolution, shape),
      feature_group_count_(feature_group_count),
      batch_group_count_(batch_group_count),
      window_(window),
      convolution_dimension_numbers_(dimension_numbers),
      precision_config_(precision_config) {
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
  proto.set_feature_group_count(feature_group_count_);
  *proto.mutable_precision_config() = precision_config_;
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
  if (feature_group_count_ != 1) {
    extra.push_back(StrCat("feature_group_count=", feature_group_count_));
  }

  string precision_config_string = PrecisionConfigToString(precision_config_);
  if (!precision_config_string.empty()) {
    extra.push_back(precision_config_string);
  }

  return extra;
}

bool HloConvolutionInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other =
      static_cast<const HloConvolutionInstruction&>(other);
  if (feature_group_count_ != other.feature_group_count()) {
    return false;
  }
  return protobuf_util::ProtobufEquals(window(), casted_other.window()) &&
         protobuf_util::ProtobufEquals(
             convolution_dimension_numbers(),
             casted_other.convolution_dimension_numbers()) &&
         protobuf_util::ProtobufEquals(precision_config(),
                                       casted_other.precision_config());
}

std::unique_ptr<HloInstruction>
HloConvolutionInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloConvolutionInstruction>(
      shape, new_operands[0], new_operands[1], feature_group_count_,
      batch_group_count_, window(), convolution_dimension_numbers_,
      precision_config_);
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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloReduceWindowInstruction>(
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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 3);
  return absl::make_unique<HloSelectAndScatterInstruction>(
      shape, new_operands[0], select(), window(), new_operands[1],
      new_operands[2], scatter());
}

HloCustomCallInstruction::HloCustomCallInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::string_view custom_call_target, absl::string_view opaque)
    : HloInstruction(HloOpcode::kCustomCall, shape),
      custom_call_target_(custom_call_target.begin(), custom_call_target.end()),
      opaque_(opaque.begin(), opaque.end()),
      feature_group_count_(1),
      layout_constrained_(false) {
  for (auto operand : operands) {
    AppendOperand(operand);
  }
}

HloCustomCallInstruction::HloCustomCallInstruction(
    const Shape& shape, absl::Span<HloInstruction* const> operands,
    absl::string_view custom_call_target, absl::string_view opaque,
    absl::Span<const Shape> operand_shapes_with_layout)
    : HloInstruction(HloOpcode::kCustomCall, shape),
      custom_call_target_(custom_call_target.begin(), custom_call_target.end()),
      opaque_(opaque.begin(), opaque.end()),
      feature_group_count_(1),
      layout_constrained_(true),
      operand_shapes_with_layout_(operand_shapes_with_layout.begin(),
                                  operand_shapes_with_layout.end()) {
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
  proto.set_custom_call_opaque(opaque_);
  proto.set_feature_group_count(feature_group_count_);
  if (layout_constrained()) {
    proto.set_constrain_layout(true);
    for (const Shape& shape : operand_shapes_with_layout_) {
      *proto.add_operand_shapes_with_layout() = shape.ToProto();
    }
  }
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
  if (feature_group_count_ != 1) {
    extra.push_back(StrCat("feature_group_count=", feature_group_count_));
  }
  // By contract, we print the custom call target even if
  // options.print_subcomputation_mode() == kOff, because the call target is not
  // an HloComputation.
  extra.push_back(
      StrCat("custom_call_target=\"", CEscape(custom_call_target_), "\""));
  // If the opaque string becomes enormous we may want to reconsider printing
  // this inline and consider other options.
  if (!opaque_.empty()) {
    extra.push_back(StrCat("opaque=\"", CEscape(opaque_), "\""));
  }
  if (layout_constrained()) {
    std::vector<string> shape_strings;
    for (const Shape& shape : operand_shapes_with_layout_) {
      shape_strings.push_back(ShapeUtil::HumanStringWithLayout(shape));
    }
    extra.push_back(StrCat("operand_layout_constraints={",
                           StrJoin(shape_strings, ", "), "}"));
  }
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
  if (feature_group_count_ != casted_other.feature_group_count_) {
    return false;
  }
  return custom_call_target_ == casted_other.custom_call_target_ &&
         opaque_ == casted_other.opaque_;
}

std::unique_ptr<HloInstruction>
HloCustomCallInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  auto cloned = absl::make_unique<HloCustomCallInstruction>(
      shape, new_operands, custom_call_target(), opaque());
  if (window_ != nullptr) {
    cloned->set_window(*window_);
  }
  if (convolution_dimension_numbers_ != nullptr) {
    cloned->set_convolution_dimension_numbers(*convolution_dimension_numbers_);
  }
  cloned->set_feature_group_count(feature_group_count_);
  return std::move(cloned);
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
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloPadInstruction>(shape, new_operands[0],
                                              new_operands[1], padding_config_);
}

HloDynamicSliceInstruction::HloDynamicSliceInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* start_indices,
    absl::Span<const int64> slice_sizes)
    : HloDynamicIndexInstruction(HloOpcode::kDynamicSlice, shape),
      dynamic_slice_sizes_(slice_sizes.begin(), slice_sizes.end()) {
  AppendOperand(operand);
  AppendOperand(start_indices);
}

HloDynamicSliceInstruction::HloDynamicSliceInstruction(
    const Shape& shape, HloInstruction* operand,
    absl::Span<HloInstruction* const> start_indices,
    absl::Span<const int64> slice_sizes)
    : HloDynamicIndexInstruction(HloOpcode::kDynamicSlice, shape),
      dynamic_slice_sizes_(slice_sizes.begin(), slice_sizes.end()) {
  AppendOperand(operand);
  for (HloInstruction* index : start_indices) {
    AppendOperand(index);
  }
}

HloDynamicUpdateSliceInstruction::HloDynamicUpdateSliceInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* update,
    HloInstruction* start_indices)
    : HloDynamicIndexInstruction(HloOpcode::kDynamicUpdateSlice, shape) {
  AppendOperand(operand);
  AppendOperand(update);
  AppendOperand(start_indices);
}

HloDynamicUpdateSliceInstruction::HloDynamicUpdateSliceInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* update,
    absl::Span<HloInstruction* const> start_indices)
    : HloDynamicIndexInstruction(HloOpcode::kDynamicUpdateSlice, shape) {
  AppendOperand(operand);
  AppendOperand(update);
  for (HloInstruction* index : start_indices) {
    AppendOperand(index);
  }
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
  return {StrCat("dynamic_slice_sizes={", StrJoin(dynamic_slice_sizes(), ","),
                 "}")};
}

bool HloDynamicSliceInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  return true;
}

std::unique_ptr<HloInstruction>
HloDynamicSliceInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  if (new_operands.size() == 2 && new_operands[1]->shape().rank() == 1) {
    // TODO(b/118437727): Old form, remove this path.
    return absl::make_unique<HloDynamicSliceInstruction>(
        shape, new_operands[0], new_operands[1], dynamic_slice_sizes_);
  } else {
    return absl::make_unique<HloDynamicSliceInstruction>(
        shape, new_operands[0], new_operands.subspan(1), dynamic_slice_sizes_);
  }
}

HloGatherInstruction::HloGatherInstruction(
    const Shape& shape, HloInstruction* operand, HloInstruction* start_indices,
    const GatherDimensionNumbers& gather_dim_numbers,
    absl::Span<const int64> slice_sizes)
    : HloInstruction(HloOpcode::kGather, shape) {
  AppendOperand(operand);
  AppendOperand(start_indices);
  gather_dimension_numbers_ =
      absl::make_unique<GatherDimensionNumbers>(gather_dim_numbers);
  absl::c_copy(slice_sizes, std::back_inserter(gather_slice_sizes_));
}

string HloGatherInstruction::GatherDimensionNumbersToString() const {
  CHECK(gather_dimension_numbers_ != nullptr);
  string offset_dims =
      StrCat("offset_dims={",
             StrJoin(gather_dimension_numbers_->offset_dims(), ","), "}");
  string collapsed_slice_dims = StrCat(
      "collapsed_slice_dims={",
      StrJoin(gather_dimension_numbers_->collapsed_slice_dims(), ","), "}");
  string start_index_map =
      StrCat("start_index_map={",
             StrJoin(gather_dimension_numbers_->start_index_map(), ","), "}");
  string index_vector_dim = StrCat(
      "index_vector_dim=", gather_dimension_numbers_->index_vector_dim());

  return StrJoin<std::initializer_list<string>>(
      {offset_dims, collapsed_slice_dims, start_index_map, index_vector_dim},
      ", ");
}

/* static */ GatherDimensionNumbers HloGatherInstruction::MakeGatherDimNumbers(
    absl::Span<const int64> offset_dims,
    absl::Span<const int64> collapsed_slice_dims,
    absl::Span<const int64> start_index_map, int64 index_vector_dim) {
  GatherDimensionNumbers gather_dim_numbers;
  for (int64 output_window_dim : offset_dims) {
    gather_dim_numbers.add_offset_dims(output_window_dim);
  }
  for (int64 elided_window_dim : collapsed_slice_dims) {
    gather_dim_numbers.add_collapsed_slice_dims(elided_window_dim);
  }
  for (int64 gather_dim_to_input_dim : start_index_map) {
    gather_dim_numbers.add_start_index_map(gather_dim_to_input_dim);
  }

  gather_dim_numbers.set_index_vector_dim(index_vector_dim);
  return gather_dim_numbers;
}

HloInstructionProto HloGatherInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_gather_dimension_numbers() = gather_dimension_numbers();
  for (int64 bound : gather_slice_sizes()) {
    proto.add_gather_slice_sizes(bound);
  }
  return proto;
}

std::vector<string> HloGatherInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {GatherDimensionNumbersToString(),
          StrCat("slice_sizes={", StrJoin(gather_slice_sizes(), ","), "}")};
}

bool HloGatherInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other = static_cast<const HloGatherInstruction&>(other);
  return protobuf_util::ProtobufEquals(
             gather_dimension_numbers(),
             casted_other.gather_dimension_numbers()) &&
         gather_slice_sizes() == casted_other.gather_slice_sizes();
}

std::unique_ptr<HloInstruction> HloGatherInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloGatherInstruction>(
      shape, new_operands[0], new_operands[1], gather_dimension_numbers(),
      gather_slice_sizes());
}

HloScatterInstruction::HloScatterInstruction(
    const Shape& shape, HloInstruction* operand,
    HloInstruction* scatter_indices, HloInstruction* updates,
    HloComputation* update_computation,
    const ScatterDimensionNumbers& scatter_dim_numbers)
    : HloInstruction(HloOpcode::kScatter, shape) {
  AppendOperand(operand);
  AppendOperand(scatter_indices);
  AppendOperand(updates);
  AppendComputation(update_computation);
  scatter_dimension_numbers_ =
      absl::make_unique<ScatterDimensionNumbers>(scatter_dim_numbers);
}

string HloScatterInstruction::ScatterDimensionNumbersToString() const {
  string update_window_dims = StrCat(
      "update_window_dims={",
      StrJoin(scatter_dimension_numbers().update_window_dims(), ","), "}");
  string inserted_window_dims = StrCat(
      "inserted_window_dims={",
      StrJoin(scatter_dimension_numbers().inserted_window_dims(), ","), "}");
  string scatter_dims_to_operand_dims = StrCat(
      "scatter_dims_to_operand_dims={",
      StrJoin(scatter_dimension_numbers().scatter_dims_to_operand_dims(), ","),
      "}");
  string index_vector_dim = StrCat(
      "index_vector_dim=", scatter_dimension_numbers().index_vector_dim());

  return StrJoin<std::initializer_list<string>>(
      {update_window_dims, inserted_window_dims, scatter_dims_to_operand_dims,
       index_vector_dim},
      ", ");
}

/* static */ ScatterDimensionNumbers
HloScatterInstruction::MakeScatterDimNumbers(
    absl::Span<const int64> update_window_dims,
    absl::Span<const int64> inserted_window_dims,
    absl::Span<const int64> scatter_dims_to_operand_dims,
    int64 index_vector_dim) {
  ScatterDimensionNumbers scatter_dim_numbers;
  for (int64 update_window_dim : update_window_dims) {
    scatter_dim_numbers.add_update_window_dims(update_window_dim);
  }
  for (int64 inserted_window_dim : inserted_window_dims) {
    scatter_dim_numbers.add_inserted_window_dims(inserted_window_dim);
  }
  for (int64 scatter_dim_to_operand_dim : scatter_dims_to_operand_dims) {
    scatter_dim_numbers.add_scatter_dims_to_operand_dims(
        scatter_dim_to_operand_dim);
  }
  scatter_dim_numbers.set_index_vector_dim(index_vector_dim);
  return scatter_dim_numbers;
}

HloInstructionProto HloScatterInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_scatter_dimension_numbers() = scatter_dimension_numbers();
  return proto;
}

std::vector<string> HloScatterInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {ScatterDimensionNumbersToString()};
}

bool HloScatterInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other = static_cast<const HloScatterInstruction&>(other);
  return protobuf_util::ProtobufEquals(
             scatter_dimension_numbers(),
             casted_other.scatter_dimension_numbers()) &&
         eq_computations(to_apply(), casted_other.to_apply());
}

std::unique_ptr<HloInstruction> HloScatterInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 3);
  return absl::make_unique<HloScatterInstruction>(
      shape, new_operands[0], new_operands[1], new_operands[2], to_apply(),
      scatter_dimension_numbers());
}

HloIotaInstruction::HloIotaInstruction(const Shape& shape, int64 iota_dimension)
    : HloInstruction(HloOpcode::kIota, shape),
      iota_dimension_(iota_dimension) {}

HloInstructionProto HloIotaInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.add_dimensions(iota_dimension());
  return proto;
}

std::vector<string> HloIotaInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  return {StrCat("iota_dimension=", iota_dimension())};
}

bool HloIotaInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other = static_cast<const HloIotaInstruction&>(other);
  return iota_dimension() == casted_other.iota_dimension();
}

std::unique_ptr<HloInstruction> HloIotaInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  return absl::make_unique<HloIotaInstruction>(shape, iota_dimension());
}

HloDotInstruction::HloDotInstruction(
    const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
    const DotDimensionNumbers& dimension_numbers,
    const PrecisionConfig& precision_config)
    : HloInstruction(HloOpcode::kDot, shape),
      dot_dimension_numbers_(dimension_numbers),
      precision_config_(precision_config) {
  AppendOperand(lhs);
  AppendOperand(rhs);
}

HloInstructionProto HloDotInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  *proto.mutable_dot_dimension_numbers() = dot_dimension_numbers_;
  *proto.mutable_precision_config() = precision_config_;
  return proto;
}

std::vector<string> HloDotInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  std::vector<string> extra = {DotDimensionNumbersToString()};

  string precision_config_string = PrecisionConfigToString(precision_config_);
  if (!precision_config_string.empty()) {
    extra.push_back(precision_config_string);
  }
  return extra;
}

bool HloDotInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other = static_cast<const HloDotInstruction&>(other);
  return protobuf_util::ProtobufEquals(dot_dimension_numbers(),
                                       casted_other.dot_dimension_numbers()) &&
         protobuf_util::ProtobufEquals(precision_config(),
                                       casted_other.precision_config());
}

std::unique_ptr<HloInstruction> HloDotInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 2);
  return absl::make_unique<HloDotInstruction>(
      shape, new_operands[0], new_operands[1], dot_dimension_numbers_,
      precision_config_);
}

string HloDotInstruction::DotDimensionNumbersToString() const {
  std::vector<string> result;
  const DotDimensionNumbers& dnums = dot_dimension_numbers_;
  if (!dnums.lhs_batch_dimensions().empty()) {
    result.push_back(StrCat("lhs_batch_dims={",
                            StrJoin(dnums.lhs_batch_dimensions(), ","), "}"));
  }
  result.push_back(StrCat("lhs_contracting_dims={",
                          StrJoin(dnums.lhs_contracting_dimensions(), ","),
                          "}"));

  if (!dnums.rhs_batch_dimensions().empty()) {
    result.push_back(StrCat("rhs_batch_dims={",
                            StrJoin(dnums.rhs_batch_dimensions(), ","), "}"));
  }
  result.push_back(StrCat("rhs_contracting_dims={",
                          StrJoin(dnums.rhs_contracting_dimensions(), ","),
                          "}"));

  return StrJoin(result, ", ");
}

HloDomainInstruction::HloDomainInstruction(
    const Shape& shape, HloInstruction* operand,
    std::unique_ptr<DomainMetadata> operand_side_metadata,
    std::unique_ptr<DomainMetadata> user_side_metadata)
    : HloInstruction(HloOpcode::kDomain, shape),
      operand_side_metadata_(std::move(operand_side_metadata)),
      user_side_metadata_(std::move(user_side_metadata)) {
  AppendOperand(operand);
}

std::vector<string> HloDomainInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& options) const {
  if (operand_side_metadata_ != nullptr && user_side_metadata_ != nullptr) {
    return {StrCat("domain={kind=\"", operand_side_metadata_->Kind(),
                   "\", entry=", user_side_metadata_->ToString(),
                   ", exit=", operand_side_metadata_->ToString(), "}")};
  }
  return {};
}

bool HloDomainInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
        eq_computations) const {
  const auto& casted_other = static_cast<const HloDomainInstruction&>(other);
  return operand_side_metadata().Matches(
             casted_other.operand_side_metadata()) &&
         user_side_metadata().Matches(casted_other.user_side_metadata());
}

std::unique_ptr<HloInstruction> HloDomainInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* context) const {
  CHECK_EQ(new_operands.size(), 1);
  return absl::make_unique<HloDomainInstruction>(
      shape, new_operands[0], operand_side_metadata_->Clone(),
      user_side_metadata_->Clone());
}

HloInstructionProto HloDomainInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  auto operand_side_sharding =
      dynamic_cast<const ShardingMetadata*>(operand_side_metadata_.get());
  if (operand_side_sharding && operand_side_sharding->sharding() != nullptr) {
    *proto.mutable_domain_entry_sharding() =
        operand_side_sharding->sharding()->ToProto();
  }

  auto user_side_sharding =
      dynamic_cast<const ShardingMetadata*>(user_side_metadata_.get());
  if (user_side_sharding && user_side_sharding->sharding() != nullptr) {
    *proto.mutable_domain_exit_sharding() =
        user_side_sharding->sharding()->ToProto();
  }

  return proto;
}

HloGetDimensionSizeInstruction::HloGetDimensionSizeInstruction(
    const Shape& shape, HloInstruction* operand, int64 dimension)
    : HloInstruction(HloOpcode::kGetDimensionSize, shape),
      dimension_(dimension) {
  AppendOperand(operand);
}

HloInstructionProto HloGetDimensionSizeInstruction::ToProto() const {
  HloInstructionProto proto = HloInstruction::ToProto();
  proto.add_dimensions(dimension());
  return proto;
}

std::vector<string> HloGetDimensionSizeInstruction::ExtraAttributesToStringImpl(
    const HloPrintOptions& /*options*/) const {
  return {StrCat("dimensions={", dimension(), "}")};
}

bool HloGetDimensionSizeInstruction::IdenticalSlowPath(
    const HloInstruction& other,
    const std::function<bool(const HloComputation*, const HloComputation*)>&
    /*eq_computations*/) const {
  const auto& casted_other =
      static_cast<const HloGetDimensionSizeInstruction&>(other);
  return dimension() == casted_other.dimension();
}

std::unique_ptr<HloInstruction>
HloGetDimensionSizeInstruction::CloneWithNewOperandsImpl(
    const Shape& shape, absl::Span<HloInstruction* const> new_operands,
    HloCloneContext* /*context*/) const {
  if (new_operands.size() != 1) {
    LOG(FATAL) << "expects 1 operand";
  }
  return absl::make_unique<HloGetDimensionSizeInstruction>(
      shape, new_operands[0], dimension());
}

}  // namespace xla
