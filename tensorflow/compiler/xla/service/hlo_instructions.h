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

// All HloInstruction subclasses are put in this file.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTIONS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTIONS_H_

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"

namespace xla {

class HloBatchNormInstruction : public HloInstruction {
 public:
  // Returns feature_index field associated with the instruction. The index
  // represents the index of the feature dimension.
  int64 feature_index() const { return feature_index_; }

  // Returns a epsilon value associated with the instruction. The is a small
  // number added to the variance to avoid divide-by-zero error.
  float epsilon() const { return epsilon_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 protected:
  explicit HloBatchNormInstruction(HloOpcode opcode, const Shape& shape,
                                   HloInstruction* operand,
                                   HloInstruction* scale, float epsilon,
                                   int64 feature_index);

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // A small float number added to the variance to avoid divide-by-zero error.
  float epsilon_ = 0.0f;

  // An integer value representing the index of the feature dimension.
  int64 feature_index_ = -1;
};

class HloBatchNormTrainingInstruction : public HloBatchNormInstruction {
 public:
  explicit HloBatchNormTrainingInstruction(const Shape& shape,
                                           HloInstruction* operand,
                                           HloInstruction* scale,
                                           HloInstruction* offset,
                                           float epsilon, int64 feature_index);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloBatchNormInferenceInstruction : public HloBatchNormInstruction {
 public:
  explicit HloBatchNormInferenceInstruction(
      const Shape& shape, HloInstruction* operand, HloInstruction* scale,
      HloInstruction* offset, HloInstruction* mean, HloInstruction* variance,
      float epsilon, int64 feature_index);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloBatchNormGradInstruction : public HloBatchNormInstruction {
 public:
  explicit HloBatchNormGradInstruction(
      const Shape& shape, HloInstruction* operand, HloInstruction* scale,
      HloInstruction* mean, HloInstruction* variance,
      HloInstruction* grad_output, float epsilon, int64 feature_index);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloFftInstruction : public HloInstruction {
 public:
  explicit HloFftInstruction(const Shape& shape, HloInstruction* operand,
                             FftType fft_type,
                             absl::Span<const int64> fft_length);
  FftType fft_type() const { return fft_type_; }

  const std::vector<int64>& fft_length() const { return fft_length_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Describes FFT type for an FFT instruction.
  FftType fft_type_ = FftType::FFT;

  // Indicates the FFT length for an FFT instruction.
  std::vector<int64> fft_length_;
};

class HloSendRecvInstruction : public HloInstruction {
 public:
  // Returns the channel id associated with the instruction. The id is
  // shared between each Send/Recv pair and is globally unique to identify each
  // channel.
  int64 channel_id() const { return channel_id_; }

  // Returns whether this send/recv instruction sends data to/from the host.
  bool is_host_transfer() const { return is_host_transfer_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 protected:
  explicit HloSendRecvInstruction(HloOpcode opcode, const Shape& shape,
                                  int64 channel_id, bool is_host_transfer);

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Represents a unique identifier for each Send/Recv instruction pair.
  int64 channel_id_;

  // Whether this send/recv instruction sends data to/from the host.
  bool is_host_transfer_;
};

class HloSendInstruction : public HloSendRecvInstruction {
 public:
  explicit HloSendInstruction(HloInstruction* operand, HloInstruction* token,
                              int64 channel_id, bool is_host_transfer);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloSendDoneInstruction : public HloSendRecvInstruction {
 public:
  explicit HloSendDoneInstruction(HloSendInstruction* operand,
                                  bool is_host_transfer);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloRecvInstruction : public HloSendRecvInstruction {
 public:
  explicit HloRecvInstruction(const Shape& shape, HloInstruction* token,
                              int64 channel_id, bool is_host_transfer);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloRecvDoneInstruction : public HloSendRecvInstruction {
 public:
  explicit HloRecvDoneInstruction(HloRecvInstruction* operand,
                                  bool is_host_transfer);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloCollectiveInstruction : public HloInstruction {
 public:
  const std::vector<ReplicaGroup>& replica_groups() const {
    return replica_groups_;
  }

 protected:
  explicit HloCollectiveInstruction(
      HloOpcode opcode, const Shape& shape,
      absl::Span<HloInstruction* const> operands,
      const std::vector<ReplicaGroup>& replica_groups);

  HloInstructionProto ToProto() const override;

  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;

  std::vector<ReplicaGroup> replica_groups_;
};

class HloAllReduceInstruction : public HloCollectiveInstruction {
 public:
  explicit HloAllReduceInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* reduce_computation,
      const std::vector<ReplicaGroup>& replica_groups,
      absl::string_view barrier, const absl::optional<int64>& all_reduce_id);

  // Returns the barrier config used for the CrossReplicaSum implementation of
  // each backend.
  string cross_replica_sum_barrier() const {
    return cross_replica_sum_barrier_;
  }
  void set_cross_replica_sum_barrier(string barrier) {
    cross_replica_sum_barrier_ = barrier;
  }

  absl::optional<int64> all_reduce_id() const { return all_reduce_id_; }
  void set_all_reduce_id(const absl::optional<int64>& all_reduce_id);

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // The string representation of the barrier config used for CrossReplicaSum.
  string cross_replica_sum_barrier_;

  // For Allreduce nodes from different modules, if they have the same
  // all_reduce_id, they will be 'Allreduce'd. If empty, Allreduce will not be
  // applied cross modules.
  absl::optional<int64> all_reduce_id_;
};

class HloAllToAllInstruction : public HloCollectiveInstruction {
 public:
  explicit HloAllToAllInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      const std::vector<ReplicaGroup>& replica_groups);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloCollectivePermuteInstruction : public HloInstruction {
 public:
  explicit HloCollectivePermuteInstruction(
      const Shape& shape, HloInstruction* operand,
      const std::vector<std::pair<int64, int64>>& source_target_pairs);

  const std::vector<std::pair<int64, int64>>& source_target_pairs() const {
    return source_target_pairs_;
  }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  const std::vector<std::pair<int64, int64>> source_target_pairs_;
};

class HloReverseInstruction : public HloInstruction {
 public:
  explicit HloReverseInstruction(const Shape& shape, HloInstruction* operand,
                                 absl::Span<const int64> dimensions);
  // Returns the dimension sizes or numbers associated with this instruction.
  const std::vector<int64>& dimensions() const override { return dimensions_; }
  int64 dimensions(int64 index) const override { return dimensions()[index]; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::vector<int64> dimensions_;
};

class HloConcatenateInstruction : public HloInstruction {
 public:
  explicit HloConcatenateInstruction(const Shape& shape,
                                     absl::Span<HloInstruction* const> operands,
                                     int64 dimension);
  // Returns the dimension sizes or numbers associated with this instruction.
  const std::vector<int64>& dimensions() const override { return dimensions_; }
  int64 dimensions(int64 index) const override { return dimensions()[index]; }
  // Accessor for the dimension in which a concatenate HLO should occur.
  int64 concatenate_dimension() const { return dimensions(0); }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::vector<int64> dimensions_;
};

class HloReduceInstruction : public HloInstruction {
 public:
  explicit HloReduceInstruction(const Shape& shape,
                                absl::Span<HloInstruction* const> args,
                                absl::Span<const int64> dimensions_to_reduce,
                                HloComputation* reduce_computation);
  // Returns the dimension sizes or numbers associated with this instruction.
  const std::vector<int64>& dimensions() const override { return dimensions_; }
  int64 dimensions(int64 index) const override { return dimensions()[index]; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Returns the number of input arrays (and, consequentially, the number of
  // init values) this reduce has.
  int64 input_count() const { return operand_count() / 2; }

  // Returns the input tensors to be reduced.
  absl::Span<HloInstruction* const> inputs() const {
    return absl::MakeSpan(operands()).subspan(0, input_count());
  }

  // Returns the init values of the reduction.
  absl::Span<HloInstruction* const> init_values() const {
    return absl::MakeSpan(operands()).subspan(input_count(), operand_count());
  }

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::vector<int64> dimensions_;
};

class HloSortInstruction : public HloInstruction {
 public:
  explicit HloSortInstruction(const Shape& shape, int64 dimension,
                              HloInstruction* keys,
                              absl::Span<HloInstruction* const> values = {});
  // Returns the dimension sizes or numbers associated with this instruction.
  const std::vector<int64>& dimensions() const override { return dimensions_; }
  int64 dimensions(int64 index) const override { return dimensions()[index]; }
  // Returns the sort dimension for this instruction
  int64 sort_dimension() const { return dimensions(0); }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;
  // Returns the key operand to this instruction.
  const HloInstruction* keys() const { return operand(0); }
  HloInstruction* mutable_keys() { return mutable_operand(0); }
  // Returns the number of value operands.
  int64 values_count() const { return operand_count() - 1; }

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::vector<int64> dimensions_;
};

class HloTransposeInstruction : public HloInstruction {
 public:
  explicit HloTransposeInstruction(const Shape& shape, HloInstruction* operand,
                                   absl::Span<const int64> dimensions);
  // Returns the dimension sizes or numbers associated with this instruction.
  const std::vector<int64>& dimensions() const override { return dimensions_; }
  int64 dimensions(int64 index) const override { return dimensions()[index]; }
  // Returns whether this instruction does a rank-2 transposition.
  bool IsRank2Transpose() const;
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::vector<int64> dimensions_;
};

class HloBroadcastInstruction : public HloInstruction {
 public:
  explicit HloBroadcastInstruction(const Shape& shape, HloInstruction* operand,
                                   absl::Span<const int64> broadcast_dimension);
  // Returns the dimension sizes or numbers associated with this instruction.
  const std::vector<int64>& dimensions() const override { return dimensions_; }
  int64 dimensions(int64 index) const override { return dimensions()[index]; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::vector<int64> dimensions_;
};

class HloMapInstruction : public HloInstruction {
 public:
  explicit HloMapInstruction(const Shape& shape,
                             absl::Span<HloInstruction* const> operands,
                             HloComputation* map_computation);
  // Returns the dimension sizes or numbers associated with this instruction.
  const std::vector<int64>& dimensions() const override { return dimensions_; }
  int64 dimensions(int64 index) const override { return dimensions()[index]; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  bool IsElementwiseImpl(
      const absl::optional<int64>& operand_idx) const override;
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::vector<int64> dimensions_;
};

class HloSliceInstruction : public HloInstruction {
 public:
  explicit HloSliceInstruction(const Shape& shape, HloInstruction* operand,
                               absl::Span<const int64> start_indices,
                               absl::Span<const int64> limit_indices,
                               absl::Span<const int64> strides);

  HloInstructionProto ToProto() const override;

  // Returns the start index in the given dimension for a slice node.
  int64 slice_starts(int64 dimension) const { return slice_starts_[dimension]; }
  const std::vector<int64>& slice_starts() const { return slice_starts_; }

  // Returns the (exclusive) limit index in the given dimension for a slice
  // node.
  int64 slice_limits(int64 dimension) const { return slice_limits_[dimension]; }
  const std::vector<int64>& slice_limits() const { return slice_limits_; }

  // Returns the stride in the given dimension for a slice node.
  int64 slice_strides(int64 dimension) const {
    return slice_strides_[dimension];
  }
  const std::vector<int64>& slice_strides() const { return slice_strides_; }

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Describes the [begin, end) index range for a slice.
  std::vector<int64> slice_starts_;
  std::vector<int64> slice_limits_;
  std::vector<int64> slice_strides_;
};

class HloConstantInstruction : public HloInstruction {
 public:
  explicit HloConstantInstruction(Literal literal);
  // Used when the literal is too large and dropped.
  explicit HloConstantInstruction(const Shape& shape);
  // Returns the literal associated with this instruction.
  const Literal& literal() const { return *literal_; }
  // Returns whether there is literal associated with this instruction.
  bool HasLiteral() const { return literal_.has_value(); }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Change the layout for an Constant Hlo instruction to match new_layout.  For
  // tuple shaped constants shape_index is the path to the internal array
  // subshape whose layout needs to be changed.
  void RelayoutConstant(const Layout& new_layout,
                        const ShapeIndex& shape_index = {});

 private:
  bool IsElementwiseImpl(
      const absl::optional<int64>& operand_idx) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  string OperandsToStringWithCanonicalNameMap(
      const HloPrintOptions& options,
      CanonicalNameMap* canonical_name_map) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  absl::optional<Literal> literal_;
};

class HloTraceInstruction : public HloInstruction {
 public:
  explicit HloTraceInstruction(const string& tag, HloInstruction* operand);
  // Returns a tag to be used in tracing.
  string TracingTag() const { return literal_.GetR1U8AsString(); }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  Literal literal_;
};

class HloFusionInstruction : public HloInstruction {
 public:
  explicit HloFusionInstruction(const Shape& shape, FusionKind fusion_kind,
                                HloInstruction* fused_root);

  explicit HloFusionInstruction(const Shape& shape, FusionKind fusion_kind,
                                absl::Span<HloInstruction* const> operands,
                                HloComputation* fusion_computation);

  string ToCategory() const override;
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Adds a new operand the fusion instruction.
  HloInstruction* AddFusionOperand(HloInstruction* new_operand);

  // Merges the fused instructions from 'instruction_to_merge' into the
  // fused instruction set of 'this', updating operands as necessary.
  //
  // Predondition: 'instruction_to_merge' must be an operand of 'this'.
  void MergeFusionInstruction(HloFusionInstruction* instruction_to_merge);

  // Merges the fused instructions from instruction_to_merge into the fused
  // instruction set of 'this' and generates multioutput fusion instructions.
  // All the users of instruction_to_merge will be redirected to 'this'
  // instruction. instruction_to_merge will be removed from its parent
  // computation.
  void MergeFusionInstructionIntoMultiOutput(
      HloFusionInstruction* instruction_to_merge);

  // Fuses the given instruction in this fusion instruction. instruction_to_fuse
  // is cloned and the clone is placed in the fusion
  // instruction. instruction_to_fuse is unchanged. Instruction is cloned rather
  // than moved to cleanly handle the case where the instruction has a use
  // outside the fusion instruction. Moving such an instruction into a fusion
  // instruction would violate the single-result invariant of HLO instructions
  // and significantly complicate code generation.
  HloInstruction* FuseInstruction(HloInstruction* instruction_to_fuse) {
    return FuseInstructionInternal(instruction_to_fuse);
  }

  // Fuses the given instruction in this fusion instruction and generate
  // multioutput fusion instruction. A clone of the instruction_to_fuse will
  // be part of the output of fusion instructions. The users of
  // instruction_to_fuse will be redirected to this fusion instructions.
  // instruction_to_fuse will be removed from its parent computation.
  HloInstruction* FuseInstructionIntoMultiOutput(
      HloInstruction* instruction_to_fuse) {
    return FuseInstructionInternal(instruction_to_fuse, /* add_output */ true);
  }

  // Returns the computation for this fused instruction.
  HloComputation* fused_instructions_computation() const;

  // Returns the root instruction of the fused expression contained within this
  // fusion instruction.
  HloInstruction* fused_expression_root() const;

  // Returns the list of fused instructions inside this fusion instruction.  The
  // returned type is a range of HloInstruction*s.
  const tensorflow::gtl::iterator_range<UnwrappingIterator<
      std::list<std::unique_ptr<HloInstruction>>::const_iterator>>
  fused_instructions() const;

  const tensorflow::gtl::iterator_range<
      UnwrappingIterator<std::list<std::unique_ptr<HloInstruction>>::iterator>>
  fused_instructions();

  // Gets the number of instructions inside this fusion instruction.
  int64 fused_instruction_count() const;

  // Returns the fused parameter instruction in this fusion instruction
  // corresponding to the given parameter number.
  HloInstruction* fused_parameter(int64 parameter_number) const;

  // Returns the vector of fused parameters inside this fusion instruction.
  const std::vector<HloInstruction*>& fused_parameters() const;

  // Returns true if this instruction is a fusion instruction that generates
  // multiple outputs.
  const bool IsMultiOutputFusion() const {
    return fused_expression_root()->opcode() == HloOpcode::kTuple;
  }

  FusionKind fusion_kind() const { return fusion_kind_; }

  void set_fusion_kind(FusionKind kind) { fusion_kind_ = kind; }

  // If multiple operands are the same instruction, keeps only one of them.
  Status DeduplicateFusionOperands();

 private:
  // Fuses the given instruction into this fusion instruction. When add_output
  // is false (which is the default), instruction_to_fuse is cloned and the
  // clone is placed in the fusion instruction. instruction_to_fuse is
  // unchanged.
  //
  // When add_output is true, a clone of the instruction_to_fuse will be part
  // of the output of fusion instructions. The users of instruction_to_fuse
  // will be redirected to this fusion instructions. instruction_to_fuse will
  // be removed from its parent computation.
  HloInstruction* FuseInstructionInternal(HloInstruction* instruction_to_fuse,
                                          bool add_output = false);
  // Clones the given instruction_to_fuse and insert the clone into this fusion
  // instruction. If add_output is true, a clone of instruction_to_fuse will
  // be in the output of the this fusion instruction (part of the tuple of the
  // fusion root).
  HloInstruction* CloneAndFuseInternal(HloInstruction* instruction_to_fuse,
                                       bool add_output = false);

  bool IsElementwiseImpl(
      const absl::optional<int64>& operand_idx) const override;
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // The type of the fusion. Used by kFusion only.
  FusionKind fusion_kind_;
};

class HloRngInstruction : public HloInstruction {
 public:
  explicit HloRngInstruction(const Shape& shape,
                             RandomDistribution distribution,
                             absl::Span<HloInstruction* const> parameters);
  // Returns the random distribution for this rng node.
  RandomDistribution random_distribution() const { return distribution_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  bool IsElementwiseImpl(
      const absl::optional<int64>& operand_idx) const override;
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // The distribution requested for random number generation.
  RandomDistribution distribution_;
};

class HloParameterInstruction : public HloInstruction {
 public:
  explicit HloParameterInstruction(int64 parameter_number, const Shape& shape,
                                   const string& name);
  int64 parameter_number() const { return parameter_number_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  string OperandsToStringWithCanonicalNameMap(
      const HloPrintOptions& options,
      CanonicalNameMap* canonical_name_map) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64 parameter_number_ = 0;
};

class HloGetTupleElementInstruction : public HloInstruction {
 public:
  explicit HloGetTupleElementInstruction(const Shape& shape,
                                         HloInstruction* operand, int64 index);
  // Returns the tuple index associated with this instruction.
  int64 tuple_index() const { return tuple_index_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64 tuple_index_ = -1;
};

class HloReducePrecisionInstruction : public HloInstruction {
 public:
  explicit HloReducePrecisionInstruction(const Shape& shape,
                                         HloInstruction* operand,
                                         const int exponent_bits,
                                         const int mantissa_bits);
  // Returns the number of exponent bits for a reduce-precision node.
  int32 exponent_bits() const { return exponent_bits_; }
  // Returns the number of mantissa bits for a reduce-precision node.
  int32 mantissa_bits() const { return mantissa_bits_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // The bit sizes for a reduce-precision operation.
  int32 exponent_bits_ = 0;
  int32 mantissa_bits_ = 0;
};

class HloInfeedInstruction : public HloInstruction {
 public:
  explicit HloInfeedInstruction(const Shape& infeed_shape,
                                HloInstruction* token_operand,
                                const string& config);
  // Returns the infeed configuration string. The infeed configuration includes
  // any metadata needed for the backend compiler (e.g., infeed buffer address)
  // and is target-dependent.
  string infeed_config() const { return infeed_config_; }
  void set_infeed_config(const string& config) { infeed_config_ = config; }
  // Returns the shape of the data received by the infeed. This is not the same
  // as the shape of the infeed instruction which produces a tuple containing
  // the infeed data shape and a TOKEN.
  const Shape& infeed_shape() const {
    TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(shape()));
    return ShapeUtil::GetSubshape(shape(), {0});
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // The string representation of the infeed configuration.
  string infeed_config_;
};

class HloOutfeedInstruction : public HloInstruction {
 public:
  explicit HloOutfeedInstruction(const Shape& outfeed_shape,
                                 HloInstruction* operand,
                                 HloInstruction* token_operand,
                                 absl::string_view outfeed_config);
  // Returns the shape for the Outfeed instruction.
  const Shape& outfeed_shape() const {
    return outfeed_shape_;
  }
  // Returns the config for the Outfeed instruction.
  const string& outfeed_config() const { return outfeed_config_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Shape of outfeed request.
  Shape outfeed_shape_;
  // Outfeed configuration information, only present for kOutfeed.
  string outfeed_config_;
};

class HloConvolutionInstruction : public HloInstruction {
 public:
  explicit HloConvolutionInstruction(
      const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
      int64 feature_group_count, const Window& window,
      const ConvolutionDimensionNumbers& dimension_numbers,
      const PrecisionConfig& precision_config);
  const Window& window() const override { return window_; }
  void set_window(const Window& window) override { window_ = window; }
  const ConvolutionDimensionNumbers& convolution_dimension_numbers() const {
    return convolution_dimension_numbers_;
  }
  void set_convolution_dimension_numbers(
      const ConvolutionDimensionNumbers& dnums) {
    convolution_dimension_numbers_ = dnums;
  }
  // The number of feature groups. Must be a divisor of the input feature
  // dimension and output feature dimension.
  int64 feature_group_count() const { return feature_group_count_; }

  // Returns the information used to tell the implementation information about
  // what sort of precision is requested. The meaning of the field is backend
  // specific. At the moment, it is only supported for kConvolution and kDot.
  // Transformations on one kDot or kConvolution to another will preserve this
  // information. Transformations to other HLOs will not preserve this
  // information but it is presumed that the alternate lowering is strictly
  // superior.
  const PrecisionConfig& precision_config() const { return precision_config_; }
  PrecisionConfig* mutable_precision_config() { return &precision_config_; }

  string ToCategory() const override;
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  // The number of feature groups. Must be a divisor of the input feature
  // dimension and output feature dimension.
  int64 feature_group_count_;
  // Describes the window used for a convolution.
  Window window_;
  // Describes the dimension numbers used for a convolution.
  ConvolutionDimensionNumbers convolution_dimension_numbers_;
  // Information used to communicate to the implementation about the algorithm
  // used to produce results. See the documentation on precision_config().
  PrecisionConfig precision_config_;
};

class HloReduceWindowInstruction : public HloInstruction {
 public:
  explicit HloReduceWindowInstruction(const Shape& shape,
                                      HloInstruction* operand,
                                      HloInstruction* init_value,
                                      const Window& window,
                                      HloComputation* reduce_computation);
  const Window& window() const override { return window_; }
  void set_window(const Window& window) override { window_ = window; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  Window window_;
};

class HloSelectAndScatterInstruction : public HloInstruction {
 public:
  explicit HloSelectAndScatterInstruction(
      const Shape& shape, HloInstruction* operand, HloComputation* select,
      const Window& window, HloInstruction* source, HloInstruction* init_value,
      HloComputation* scatter);
  const Window& window() const override { return window_; }
  void set_window(const Window& window) override { window_ = window; }
  // Gets/sets the select or scatter HloComputation for SelectAndScatter. The
  // setters should only be called by HloModule or HloComputation methods.
  HloComputation* select() const {
    return called_computations()[kSelectComputationIndex];
  }

  HloComputation* scatter() const {
    return called_computations()[kScatterComputationIndex];
  }

  void set_select(HloComputation* computation) {
    // Don't allow changing the computation for fused instructions so we don't
    // have to recompute called_instructions for the entire fusion instruction.
    CHECK(!IsFused());
    set_called_computation(kSelectComputationIndex, computation);
  }

  void set_scatter(HloComputation* computation) {
    // Don't allow changing the computation for fused instructions so we don't
    // have to recompute called_instructions for the entire fusion instruction.
    CHECK(!IsFused());
    set_called_computation(kScatterComputationIndex, computation);
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  Window window_;
};

class HloCustomCallInstruction : public HloInstruction {
 public:
  HloCustomCallInstruction(const Shape& shape,
                           absl::Span<HloInstruction* const> operands,
                           absl::string_view custom_call_target,
                           absl::string_view opaque);

  // Constructor for a custom call with constrained layout. 'shape' and
  // 'operands_with_layout' must all have layouts.
  HloCustomCallInstruction(const Shape& shape,
                           absl::Span<HloInstruction* const> operands,
                           absl::string_view custom_call_target,
                           absl::string_view opaque,
                           absl::Span<const Shape> operand_shapes_with_layout);

  const Window& window() const override {
    CHECK(window_ != nullptr);
    return *window_;
  }

  void set_window(const Window& window) override {
    window_ = absl::make_unique<Window>(window);
  }

  const ConvolutionDimensionNumbers& convolution_dimension_numbers() const {
    CHECK(convolution_dimension_numbers_ != nullptr);
    return *convolution_dimension_numbers_;
  }

  void set_convolution_dimension_numbers(
      const ConvolutionDimensionNumbers& dnums) {
    convolution_dimension_numbers_ =
        absl::make_unique<ConvolutionDimensionNumbers>(dnums);
  }
  const string& opaque() const { return opaque_; }
  const string& custom_call_target() const { return custom_call_target_; }
  void set_feature_group_count(int64 feature_group_count) {
    feature_group_count_ = feature_group_count;
  }
  int64 feature_group_count() const { return feature_group_count_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Returns whether the result and operand layouts are constrained.
  bool layout_constrained() const { return layout_constrained_; }

  // Returns the shapes (with layout) of the operands. CHECKs if this custom
  // call does not have constrained layouts.
  const std::vector<Shape>& operand_shapes_with_layout() const {
    CHECK(layout_constrained());
    return operand_shapes_with_layout_;
  }

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  // Name of a global symbol to call.
  string custom_call_target_;
  // Opaque string interpreted by the backend.
  string opaque_;
  // Describes the window in a windowed operation such as convolution.
  std::unique_ptr<Window> window_;
  // Describes the dimension numbers used for a convolution.
  std::unique_ptr<ConvolutionDimensionNumbers> convolution_dimension_numbers_;
  // The number of feature groups. This is used for grouped convolutions.
  int64 feature_group_count_;
  // Whether the result and operand layouts are constrained.
  bool layout_constrained_;
  // For layout-constrained custom calls, this vector holds the shape with
  // layout for each operand.
  std::vector<Shape> operand_shapes_with_layout_;
};

class HloPadInstruction : public HloInstruction {
 public:
  explicit HloPadInstruction(const Shape& shape, HloInstruction* operand,
                             HloInstruction* padding_value,
                             const PaddingConfig& padding_config);
  // Returns the padding configuration for a pad node.
  const PaddingConfig& padding_config() const { return padding_config_; }
  // Returns the padding value.
  const HloInstruction* padding_value() const { return operand(1); }
  HloInstruction* mutable_padding_value() { return mutable_operand(1); }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // The padding configuration that describes the edge padding and interior
  // padding of this pad instruction.
  PaddingConfig padding_config_;
};

class HloDynamicSliceInstruction : public HloInstruction {
 public:
  explicit HloDynamicSliceInstruction(const Shape& shape,
                                      HloInstruction* operand,
                                      HloInstruction* start_indices,
                                      absl::Span<const int64> slice_sizes);
  // Old methods kept for smooth subclassing transition END.
  // Returns the size of the slice in the given dimension for a dynamic
  // slice node.
  int64 slice_sizes(int64 dimension) const {
    return dynamic_slice_sizes_[dimension];
  }
  const std::vector<int64>& dynamic_slice_sizes() const {
    return dynamic_slice_sizes_;
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Describes the [start, start + size) range size for a dynamic slice
  // ('start' is specified dynamically in the second operand of the operation).
  std::vector<int64> dynamic_slice_sizes_;
};

class HloGatherInstruction : public HloInstruction {
 public:
  explicit HloGatherInstruction(
      const Shape& shape, HloInstruction* operand,
      HloInstruction* start_indices,
      const GatherDimensionNumbers& gather_dim_numbers,
      absl::Span<const int64> slice_sizes);
  const GatherDimensionNumbers& gather_dimension_numbers() const {
    CHECK(gather_dimension_numbers_ != nullptr);
    return *gather_dimension_numbers_;
  }
  absl::Span<const int64> gather_slice_sizes() const {
    return gather_slice_sizes_;
  }
  // Returns the dump string of the gather dimension numbers.
  string GatherDimensionNumbersToString() const;
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Creates an instance of GatherDimensionNumbers.
  static GatherDimensionNumbers MakeGatherDimNumbers(
      absl::Span<const int64> offset_dims,
      absl::Span<const int64> collapsed_slice_dims,
      absl::Span<const int64> start_index_map, int64 index_vector_dim);

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::unique_ptr<GatherDimensionNumbers> gather_dimension_numbers_;
  std::vector<int64> gather_slice_sizes_;
};

class HloScatterInstruction : public HloInstruction {
 public:
  explicit HloScatterInstruction(
      const Shape& shape, HloInstruction* operand,
      HloInstruction* scatter_indices, HloInstruction* updates,
      HloComputation* update_computation,
      const ScatterDimensionNumbers& scatter_dim_numbers);
  const ScatterDimensionNumbers& scatter_dimension_numbers() const {
    CHECK(scatter_dimension_numbers_ != nullptr);
    return *scatter_dimension_numbers_;
  }
  // Returns the dump string of the scatter dimension numbers.
  string ScatterDimensionNumbersToString() const;
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Creates an instance of ScatterDimensionNumbers.
  static ScatterDimensionNumbers MakeScatterDimNumbers(
      absl::Span<const int64> update_window_dims,
      absl::Span<const int64> inserted_window_dims,
      absl::Span<const int64> scatter_dims_to_operand_dims,
      int64 index_vector_dim);

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::unique_ptr<ScatterDimensionNumbers> scatter_dimension_numbers_;
};

class HloIotaInstruction : public HloInstruction {
 public:
  explicit HloIotaInstruction(const Shape& shape, int64 iota_dimension);
  // Returns the dimension sizes or numbers associated with this instruction.
  int64 iota_dimension() const { return iota_dimension_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  const int64 iota_dimension_;
};

class HloDotInstruction : public HloInstruction {
 public:
  // Creates a dot op with operands 'lhs' and 'rhs' with contracting and batch
  // dimensions specified in 'dimension_numbers'.
  explicit HloDotInstruction(const Shape& shape, HloInstruction* lhs,
                             HloInstruction* rhs,
                             const DotDimensionNumbers& dimension_numbers,
                             const PrecisionConfig& precision_config);

  // Returns data on the dimension numbers used for a dot operation.
  const DotDimensionNumbers& dot_dimension_numbers() const {
    return dot_dimension_numbers_;
  }

  // Returns the information used to tell the implementation information about
  // what sort of precision is requested. The meaning of the field is backend
  // specific. At the moment, it is only supported for kConvolution and kDot.
  // Transformations on one kDot or kConvolution to another will preserve this
  // information. Transformations to other HLOs will not preserve this
  // information but it is presumed that the alternate lowering is strictly
  // superior.
  const PrecisionConfig& precision_config() const { return precision_config_; }
  PrecisionConfig* mutable_precision_config() { return &precision_config_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  // Returns the dump string of the dot dimension numbers.
  string DotDimensionNumbersToString() const;

  // Describes the dimension numbers used for a dot.
  DotDimensionNumbers dot_dimension_numbers_;

  // Information used to communicate to the implementation about the algorithm
  // used to produce results. See the documentation on precision_config().
  PrecisionConfig precision_config_;
};

class HloDomainInstruction : public HloInstruction {
 public:
  explicit HloDomainInstruction(
      const Shape& shape, HloInstruction* operand,
      std::unique_ptr<DomainMetadata> operand_side_metadata,
      std::unique_ptr<DomainMetadata> user_side_metadata);

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Retrieves the operand side metadata of a kDomain instruction.
  const DomainMetadata& operand_side_metadata() const {
    return *operand_side_metadata_;
  }
  // Retrieves the user side metadata of a kDomain instruction.
  const DomainMetadata& user_side_metadata() const {
    return *user_side_metadata_;
  }

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::unique_ptr<DomainMetadata> operand_side_metadata_;
  std::unique_ptr<DomainMetadata> user_side_metadata_;
};

class HloGetDimensionSizeInstruction : public HloInstruction {
 public:
  explicit HloGetDimensionSizeInstruction(const Shape& shape,
                                          HloInstruction* operand,
                                          int64 dimension);

  // Returns the dimension sizes or numbers associated with this instruction.
  int64 dimension() const { return dimension_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64 dimension_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTIONS_H_
