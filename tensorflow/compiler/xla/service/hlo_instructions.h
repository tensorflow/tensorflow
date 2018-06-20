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
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
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
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
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
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;
};

class HloFftInstruction : public HloInstruction {
 public:
  explicit HloFftInstruction(const Shape& shape, HloInstruction* operand,
                             FftType fft_type,
                             tensorflow::gtl::ArraySlice<int64> fft_length);
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
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
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

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 protected:
  explicit HloSendRecvInstruction(HloOpcode opcode, const Shape& shape,
                                  int64 channel_id);

 private:
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Represents a unique identifier for each Send/Recv instruction pair.
  int64 channel_id_;
};

class HloSendInstruction : public HloSendRecvInstruction {
 public:
  explicit HloSendInstruction(HloInstruction* operand, int64 channel_id);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;
};

class HloSendDoneInstruction : public HloSendRecvInstruction {
 public:
  explicit HloSendDoneInstruction(HloSendInstruction* operand);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;
};

class HloRecvInstruction : public HloSendRecvInstruction {
 public:
  explicit HloRecvInstruction(const Shape& shape, int64 channel_id);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;
};

class HloRecvDoneInstruction : public HloSendRecvInstruction {
 public:
  explicit HloRecvDoneInstruction(HloRecvInstruction* operand);

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;
};

class HloAllReduceInstruction : public HloInstruction {
 public:
  explicit HloAllReduceInstruction(
      const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
      HloComputation* reduce_computation,
      tensorflow::gtl::ArraySlice<int64> replica_group_ids,
      tensorflow::StringPiece barrier,
      const tensorflow::gtl::optional<int64>& all_reduce_id =
          tensorflow::gtl::nullopt);

  // Returns the group ids of each replica for CrossReplicaSum op.
  const std::vector<int64>& replica_group_ids() const {
    return replica_group_ids_;
  }

  // Returns the barrier config used for the CrossReplicaSum implementation of
  // each backend.
  string cross_replica_sum_barrier() const {
    return cross_replica_sum_barrier_;
  }
  void set_cross_replica_sum_barrier(string barrier) {
    cross_replica_sum_barrier_ = barrier;
  }

  tensorflow::gtl::optional<int64> all_reduce_id() const {
    return all_reduce_id_;
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
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;

  // The group id of each replica for CrossReplicaSum.
  std::vector<int64> replica_group_ids_;

  // The string representation of the barrier config used for CrossReplicaSum.
  string cross_replica_sum_barrier_;

  // For Allreduce nodes from different modules, if they have the same
  // all_reduce_id, they will be 'Allreduce'd. If empty, Allreduce will not be
  // applied cross modules.
  tensorflow::gtl::optional<int64> all_reduce_id_;
};

class HloReverseInstruction : public HloInstruction {
 public:
  explicit HloReverseInstruction(const Shape& shape, HloInstruction* operand,
                                 tensorflow::gtl::ArraySlice<int64> dimensions);
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
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;

  std::vector<int64> dimensions_;
};

class HloConcatenateInstruction : public HloInstruction {
 public:
  explicit HloConcatenateInstruction(
      const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
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
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;

  std::vector<int64> dimensions_;
};

class HloReduceInstruction : public HloInstruction {
 public:
  explicit HloReduceInstruction(
      const Shape& shape, HloInstruction* arg, HloInstruction* init_value,
      tensorflow::gtl::ArraySlice<int64> dimensions_to_reduce,
      HloComputation* reduce_computation);
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
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;

  std::vector<int64> dimensions_;
};

class HloTransposeInstruction : public HloInstruction {
 public:
  explicit HloTransposeInstruction(
      const Shape& shape, HloInstruction* operand,
      tensorflow::gtl::ArraySlice<int64> dimensions);
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
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;

  std::vector<int64> dimensions_;
};

class HloBroadcastInstruction : public HloInstruction {
 public:
  explicit HloBroadcastInstruction(
      const Shape& shape, HloInstruction* operand,
      tensorflow::gtl::ArraySlice<int64> broadcast_dimension);
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
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;

  std::vector<int64> dimensions_;
};

class HloMapInstruction : public HloInstruction {
 public:
  explicit HloMapInstruction(
      const Shape& shape, tensorflow::gtl::ArraySlice<HloInstruction*> operands,
      HloComputation* map_computation,
      tensorflow::gtl::ArraySlice<HloInstruction*> static_operands = {});
  // Returns the dimension sizes or numbers associated with this instruction.
  const std::vector<int64>& dimensions() const override { return dimensions_; }
  int64 dimensions(int64 index) const override { return dimensions()[index]; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  bool IsElementwiseImpl(
      const tensorflow::gtl::optional<int64>& operand_idx) const override;
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;

  std::vector<int64> dimensions_;
};

class HloSliceInstruction : public HloInstruction {
 public:
  explicit HloSliceInstruction(const Shape& shape, HloInstruction* operand,
                               tensorflow::gtl::ArraySlice<int64> start_indices,
                               tensorflow::gtl::ArraySlice<int64> limit_indices,
                               tensorflow::gtl::ArraySlice<int64> strides);

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

  // Returns the flag that describes whether a slice must be lowered into an
  // offset into the original operand.
  bool IsInPlaceSlice() const { return is_in_place_slice_; }

  // Sets and returns the flag that describes whether a slice must be lowered
  // into an offset into the original operand.
  bool SetIsInPlaceSlice(bool value) {
    is_in_place_slice_ = value;
    return value;
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
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;

  // Describes the [begin, end) index range for a slice.
  std::vector<int64> slice_starts_;
  std::vector<int64> slice_limits_;
  std::vector<int64> slice_strides_;

  // Describes whether the slice can be lowered to an offset into the operand.
  bool is_in_place_slice_ = false;
};

class HloConstantInstruction : public HloInstruction {
 public:
  explicit HloConstantInstruction(std::unique_ptr<Literal> literal);
  // Used when the literal is too large and dropped.
  explicit HloConstantInstruction(const Shape& shape);
  // Returns the literal associated with this instruction.
  const Literal& literal() const { return *literal_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Change the layout for an Constant Hlo instruction to match new_layout.  For
  // tuple shaped constants shape_index is the path to the internal array
  // subshape whose layout needs to be changed.
  void RelayoutConstant(const Layout& new_layout,
                        const ShapeIndex& shape_index = {});

 private:
  bool IsElementwiseImpl(
      const tensorflow::gtl::optional<int64>& operand_idx) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  string OperandsToStringWithCanonicalNameMap(
      const HloPrintOptions& options,
      CanonicalNameMap* canonical_name_map) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;
  // TODO(b/36360764): Remove unique_ptr wrapping.
  std::unique_ptr<Literal> literal_;
};

class HloTraceInstruction : public HloInstruction {
 public:
  explicit HloTraceInstruction(const string& tag, HloInstruction* operand);
  // Returns a tag to be used in tracing.
  string TracingTag() const { return literal_->GetR1U8AsString(); }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;
  // TODO(b/36360764): Remove unique_ptr wrapping.
  std::unique_ptr<Literal> literal_;
};

class HloFusionInstruction : public HloInstruction {
 public:
  explicit HloFusionInstruction(const Shape& shape, FusionKind fusion_kind,
                                HloInstruction* fused_root);

  explicit HloFusionInstruction(
      const Shape& shape, FusionKind fusion_kind,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands,
      HloComputation* fusion_computation);

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
      const tensorflow::gtl::optional<int64>& operand_idx) const override;
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;

  // The type of the fusion. Used by kFusion only.
  FusionKind fusion_kind_;
};

class HloRngInstruction : public HloInstruction {
 public:
  explicit HloRngInstruction(
      const Shape& shape, RandomDistribution distribution,
      tensorflow::gtl::ArraySlice<HloInstruction*> parameters);
  // Returns the random distribution for this rng node.
  RandomDistribution random_distribution() const { return distribution_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

 private:
  bool IsElementwiseImpl(
      const tensorflow::gtl::optional<int64>& operand_idx) const override;
  std::vector<string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      const std::function<bool(const HloComputation*, const HloComputation*)>&
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
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
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
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
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
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
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;

  // The bit sizes for a reduce-precision operation.
  int32 exponent_bits_ = 0;
  int32 mantissa_bits_ = 0;
};

class HloInfeedInstruction : public HloInstruction {
 public:
  explicit HloInfeedInstruction(const Shape& shape, const string& config);
  // Returns the infeed configuration string. The infeed configuration includes
  // any metadata needed for the backend compiler (e.g., infeed buffer address)
  // and is target-dependent.
  string infeed_config() const { return infeed_config_; }
  void set_infeed_config(const string& config) { infeed_config_ = config; }
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
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;

  // The string representation of the infeed configuration.
  string infeed_config_;
};

class HloOutfeedInstruction : public HloInstruction {
 public:
  explicit HloOutfeedInstruction(const Shape& shape, HloInstruction* operand,
                                 tensorflow::StringPiece outfeed_config);
  // Returns the shape for the Outfeed instruction.
  const Shape& outfeed_shape() const {
    TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(shape()));
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
      const Shape& shape,
      tensorflow::gtl::ArraySlice<HloInstruction*> new_operands,
      HloCloneContext* context) const override;

  // Shape of outfeed request.
  Shape outfeed_shape_;
  // Outfeed configuration information, only present for kOutfeed.
  string outfeed_config_;
};
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_INSTRUCTIONS_H_
