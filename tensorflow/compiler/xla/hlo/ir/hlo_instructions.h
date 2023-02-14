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

#ifndef TENSORFLOW_COMPILER_XLA_HLO_IR_HLO_INSTRUCTIONS_H_
#define TENSORFLOW_COMPILER_XLA_HLO_IR_HLO_INSTRUCTIONS_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/printer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Base class for instructions with a dimensions vector.
class HloDimensionsInstruction : public HloInstruction {
 public:
  absl::Span<const int64_t> dimensions() const override { return dimensions_; }

  std::vector<int64_t>* mutable_dimensions() override { return &dimensions_; }

  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    switch (hlo->opcode()) {
      case HloOpcode::kBroadcast:
      case HloOpcode::kConcatenate:
      case HloOpcode::kReduce:
      case HloOpcode::kReverse:
      case HloOpcode::kSort:
      case HloOpcode::kTranspose:
        return true;
      default:
        return false;
    }
  }

 protected:
  HloDimensionsInstruction(HloOpcode opcode, const Shape& shape,
                           absl::Span<const int64_t> dimensions)
      : HloInstruction(opcode, shape),
        dimensions_(dimensions.begin(), dimensions.end()) {}
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;

  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  std::vector<int64_t> dimensions_;
};

class HloBatchNormInstruction : public HloInstruction {
 public:
  // Returns feature_index field associated with the instruction. The index
  // represents the index of the feature dimension.
  int64_t feature_index() const { return feature_index_; }

  // Returns a epsilon value associated with the instruction. The is a small
  // number added to the variance to avoid divide-by-zero error.
  float epsilon() const { return epsilon_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    switch (hlo->opcode()) {
      case HloOpcode::kBatchNormGrad:
      case HloOpcode::kBatchNormInference:
      case HloOpcode::kBatchNormTraining:
        return true;
      default:
        return false;
    }
  }

 protected:
  explicit HloBatchNormInstruction(HloOpcode opcode, const Shape& shape,
                                   HloInstruction* operand,
                                   HloInstruction* scale, float epsilon,
                                   int64_t feature_index);

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // A small float number added to the variance to avoid divide-by-zero error.
  float epsilon_ = 0.0f;

  // An integer value representing the index of the feature dimension.
  int64_t feature_index_ = -1;
};

class HloBatchNormTrainingInstruction : public HloBatchNormInstruction {
 public:
  explicit HloBatchNormTrainingInstruction(
      const Shape& shape, HloInstruction* operand, HloInstruction* scale,
      HloInstruction* offset, float epsilon, int64_t feature_index);

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kBatchNormTraining;
  }

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
      float epsilon, int64_t feature_index);

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kBatchNormInference;
  }

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
      HloInstruction* grad_output, float epsilon, int64_t feature_index);

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kBatchNormGrad;
  }

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
                             absl::Span<const int64_t> fft_length);
  FftType fft_type() const { return fft_type_; }

  const std::vector<int64_t>& fft_length() const { return fft_length_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kFft;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Describes FFT type for an FFT instruction.
  FftType fft_type_ = FftType::FFT;

  // Indicates the FFT length for an FFT instruction.
  std::vector<int64_t> fft_length_;
};

class HloAsyncInstruction : public HloInstruction {
 public:
  HloAsyncInstruction(
      HloOpcode opcode, const Shape& shape,
      absl::Span<HloInstruction* const> operands,
      HloComputation* async_computation,
      std::optional<int64_t> async_group_id = std::nullopt,
      absl::string_view async_execution_thread = kMainExecutionThread);
  HloAsyncInstruction(
      HloOpcode opcode, const Shape& shape, HloInstruction* operand,
      HloComputation* async_computation,
      std::optional<int64_t> async_group_id = std::nullopt,
      absl::string_view async_execution_thread = kMainExecutionThread);

  ~HloAsyncInstruction() override;
  // When an async instruction is being destructed, remove it from the vector of
  // pointers of its called computation, to avoid referencing freed memory.
  void ClearAsyncComputationInstruction();

  HloInstruction* async_wrapped_instruction() const;
  HloOpcode async_wrapped_opcode() const;

  // Async group id is a unique id given to a group of async operations that
  // consist of one async start, one async done, and zero or more async update
  // operations. The async group participates in a single async operation. The
  // async operation canonicalizer pass assigns async group ids.
  std::optional<int64_t> async_group_id() const { return async_group_id_; }

  // Async thread name is a unique thread name for one or more async groups.
  // Typically one HLO module contains a main thread as well as one or more
  // parallel threads.
  absl::string_view async_execution_thread() const {
    return async_execution_thread_;
  }
  void set_async_group_id(std::optional<int64_t> async_group_id);
  void set_async_execution_thread(absl::string_view async_execution_thread);
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    switch (hlo->opcode()) {
      case HloOpcode::kAsyncStart:
      case HloOpcode::kAsyncUpdate:
      case HloOpcode::kAsyncDone:
        return true;
      default:
        return false;
    }
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  std::optional<int64_t> async_group_id_;
  std::string async_execution_thread_ = kMainExecutionThread;
};

class HloCopyStartInstruction : public HloInstruction {
 public:
  explicit HloCopyStartInstruction(
      const Shape& shape, HloInstruction* operand,
      std::optional<int> cross_program_prefetch_index);

  std::optional<int> cross_program_prefetch_index() const {
    return cross_program_prefetch_index_;
  }
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kCopyStart;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Each cross program prefetched buffer has a unique index. The indices are
  // assigned contiguously starting from zero in
  // AlternateMemoryBestFitHeap::AllocateCrossProgramPrefetchBuffer. This value
  // is used during codegen to determine which buffer is being speculated at
  // runtime. One possible implementation is to initialize an array with boolean
  // values indicating whether the cross program prefetch succeeds or fails for
  // each buffer.
  std::optional<int> cross_program_prefetch_index_;
};

class HloCompareInstruction : public HloInstruction {
 public:
  explicit HloCompareInstruction(const Shape& shape, HloInstruction* lhs,
                                 HloInstruction* rhs,
                                 ComparisonDirection direction,
                                 std::optional<Comparison::Type> type);
  ComparisonDirection direction() const { return compare_.GetDirection(); }
  ComparisonOrder order() const { return compare_.GetOrder(); }
  Comparison::Type type() const { return compare_.GetType(); }
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kCompare;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  Comparison compare_;
};

class HloTriangularSolveInstruction : public HloInstruction {
 public:
  explicit HloTriangularSolveInstruction(const Shape& shape, HloInstruction* a,
                                         HloInstruction* b,
                                         const TriangularSolveOptions& options);
  const TriangularSolveOptions& triangular_solve_options() const {
    return triangular_solve_options_;
  }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kTriangularSolve;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  TriangularSolveOptions triangular_solve_options_;
};

class HloCholeskyInstruction : public HloInstruction {
 public:
  explicit HloCholeskyInstruction(const Shape& shape, HloInstruction* a,
                                  const CholeskyOptions& options);
  const CholeskyOptions& cholesky_options() const { return cholesky_options_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kCholesky;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  CholeskyOptions cholesky_options_;
};

// Class that represents instructions that synchronize and transfer data between
// partitioned devices. Send/Recv and collective instructions (AllReduce,
// AllToAll, CollectivePermute) belong to this instruction type. A group of
// instructions (of the same opcode) with the same channel_id communicate during
// execution.
class HloChannelInstruction : public HloInstruction {
 public:
  // Returns the channel id associated with the instruction. The id is
  // shared between each Send/Recv pair or a group of collective instructions
  // and is globally unique to identify each channel.
  std::optional<int64_t> channel_id() const { return channel_id_; }
  void set_channel_id(const std::optional<int64_t>& channel_id);

  // Whether this instruction is identical to `other` except for the values of
  // channel IDs, as long as both have channel IDs or neither has a channel ID.
  virtual bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const {
    return channel_id_.has_value() == other.channel_id().has_value();
  }

  static bool ClassOf(const HloInstruction* hlo);

 protected:
  explicit HloChannelInstruction(HloOpcode opcode, const Shape& shape,
                                 const std::optional<int64_t>& channel_id);

  HloInstructionProto ToProto() const override;

  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;

  // Do not override IdenticalSlowPath(). Override
  // IdenticalSlowPathIgnoringChannelIdValues() instead.
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const final;

  std::optional<int64_t> channel_id_;
};

class HloSendRecvInstruction : public HloChannelInstruction {
 public:
  // Returns whether this send/recv instruction sends data to/from the host.
  bool is_host_transfer() const { return is_host_transfer_; }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    switch (hlo->opcode()) {
      case HloOpcode::kSend:
      case HloOpcode::kSendDone:
      case HloOpcode::kRecv:
      case HloOpcode::kRecvDone:
        return true;
      default:
        return false;
    }
  }

 protected:
  explicit HloSendRecvInstruction(HloOpcode opcode, const Shape& shape,
                                  int64_t channel_id, bool is_host_transfer);

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Whether this send/recv instruction sends data to/from the host.
  bool is_host_transfer_;
};

class HloSendInstruction : public HloSendRecvInstruction {
 public:
  explicit HloSendInstruction(HloInstruction* operand, HloInstruction* token,
                              int64_t channel_id, bool is_host_transfer);

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kSend;
  }

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

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kSendDone;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloRecvInstruction : public HloSendRecvInstruction {
 public:
  explicit HloRecvInstruction(const Shape& shape, HloInstruction* token,
                              int64_t channel_id, bool is_host_transfer);

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kRecv;
  }

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

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kRecvDone;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloCollectiveInstruction : public HloChannelInstruction {
 public:
  const std::vector<ReplicaGroup>& replica_groups() const {
    return replica_groups_;
  }

  // Returns true if the layout of the AllReduce is enforced by XLA client (as
  // the layout set in the shape). The only reason for the client to set the
  // layout is to separately compile computations that communicate with
  // AllReduce. Since this field is only set `true` by the client, the compiler
  // only needs to propagate existing values (e.g., Clone, X64Rewriter) or set
  // `false` for all other cases.
  //
  // When this is `true`, there may be communication endpoints outside the
  // current compilation unit, so the compiler considers this AllReduce as
  // side-effecting to disable compiler transformations. The compiler is free to
  // transform unconstrained AllReduces differently across compilation units.
  // It is an error for an HloModule to have a mix of constrained and
  // unconstrained AllReduce instructions (checked by HloVerifier).
  bool constrain_layout() const { return constrain_layout_; }

  static bool ClassOf(const HloInstruction* hlo);

 protected:
  explicit HloCollectiveInstruction(
      HloOpcode opcode, const Shape& shape,
      absl::Span<HloInstruction* const> operands,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id);

  HloInstructionProto ToProto() const override;

  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  std::vector<ReplicaGroup> replica_groups_;
  bool constrain_layout_;
};

class HloAllGatherInstruction : public HloCollectiveInstruction {
 public:
  explicit HloAllGatherInstruction(
      HloOpcode opcode, const Shape& shape,
      absl::Span<HloInstruction* const> operands, int64_t all_gather_dimension,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id, bool use_global_device_ids);
  // Same as HloAllReduceInstruction::use_global_device_ids.
  bool use_global_device_ids() const { return use_global_device_ids_; }

  // The dimension on which data from different participants are concatenated.
  int64_t all_gather_dimension() const { return all_gather_dimension_; }
  absl::Span<const int64_t> dimensions() const override {
    return absl::MakeConstSpan(&all_gather_dimension_, 1);
  }

  void set_all_gather_dimension(int64_t dim) { all_gather_dimension_ = dim; }

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kAllGather ||
           hlo->opcode() == HloOpcode::kAllGatherStart;
  }

 protected:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  HloInstructionProto ToProto() const override;

 private:
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t all_gather_dimension_;
  bool use_global_device_ids_;
};

// Base class for all-reduce and all-reduce scatter instructions.
class HloAllReduceInstructionBase : public HloCollectiveInstruction {
 public:
  explicit HloAllReduceInstructionBase(
      HloOpcode opcode, const Shape& shape,
      absl::Span<HloInstruction* const> operands,
      HloComputation* reduce_computation,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id, bool use_global_device_ids);

  // Returns true if the ids in the ReplicaGroup config represent a global id of
  // (replica_id * partition_count + partition_id) instead of a replica id.
  // This enables more flexible grouping of devices if this all-reduce is both
  // cross-partition and cross-replica.
  //
  // For example with 2 replicas and 4 partitions,
  // replica_groups={{0,1,4,5},{2,3,6,7}}, use_global_device_ids=true means that
  // group[0] = (0,0), (0,1), (1,0), (1,1)
  // group[1] = (0,2), (0,3), (1,2), (1,3)
  // where each pair is (replica_id, partition_id).
  bool use_global_device_ids() const { return use_global_device_ids_; }
  void set_use_global_device_ids(bool value) { use_global_device_ids_ = value; }

  static bool ClassOf(const HloInstruction* hlo);

 protected:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  HloInstructionProto ToProto() const override;

  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

 private:
  bool use_global_device_ids_;
};

class HloAllReduceInstruction : public HloAllReduceInstructionBase {
 public:
  using HloAllReduceInstructionBase::HloAllReduceInstructionBase;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kAllReduce ||
           hlo->opcode() == HloOpcode::kAllReduceStart;
  }

  // Returns true if the AllReduce does no communication, so it's equivalent
  // to a mem copy.
  bool IsNoop() const;

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloReduceScatterInstruction : public HloAllReduceInstructionBase {
 public:
  explicit HloReduceScatterInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      HloComputation* reduce_computation,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id, bool use_global_device_ids,
      int64_t scatter_dimension);

  // The dimension on which reduced data is scattered to different participants.
  int64_t scatter_dimension() const { return scatter_dimension_; }
  absl::Span<const int64_t> dimensions() const override {
    return absl::MakeConstSpan(&scatter_dimension_, 1);
  }

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kReduceScatter;
  }

 protected:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  HloInstructionProto ToProto() const override;

 private:
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t scatter_dimension_;
};

class HloAllToAllInstruction : public HloCollectiveInstruction {
 public:
  explicit HloAllToAllInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      absl::Span<const ReplicaGroup> replica_groups, bool constrain_layout,
      const std::optional<int64_t>& channel_id,
      const std::optional<int64_t>& split_dimension);

  // AllToAll can optionally take a split dimension, which means that this
  // AllToAll takes a single (flattened) array operand and produces an array
  // output (instead of taking a list of operands and producing a tuple).
  //
  // split_dimension specifies which dimension in the operand is split across
  // devices in each replica_group, and also means the concatenated dimension
  // on the output (i.e., input and the output shapes are the same).
  std::optional<int64_t> split_dimension() const { return split_dimension_; }
  void set_split_dimension(int64_t dim) { split_dimension_ = dim; }

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kAllToAll;
  }

 protected:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  HloInstructionProto ToProto() const override;

 private:
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::optional<int64_t> split_dimension_;
};

class HloCollectivePermuteInstruction : public HloChannelInstruction {
 public:
  explicit HloCollectivePermuteInstruction(
      HloOpcode opcode, const Shape& shape, HloInstruction* operand,
      const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs,
      const std::optional<int64_t>& channel_id);

  explicit HloCollectivePermuteInstruction(
      HloOpcode opcode, const Shape& shape, HloInstruction* input,
      HloInstruction* output, HloInstruction* input_start_indices,
      HloInstruction* output_start_indices,
      absl::Span<const std::pair<int64_t, int64_t>> source_target_pairs,
      absl::Span<const std::vector<int64_t>> slice_sizes,
      const std::optional<int64_t>& channel_id);

  const std::vector<std::pair<int64_t, int64_t>>& source_target_pairs() const {
    return source_target_pairs_;
  }

  const std::vector<std::vector<int64_t>>& dynamic_slice_sizes_list() const {
    return slice_sizes_;
  }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kCollectivePermute ||
           hlo->opcode() == HloOpcode::kCollectivePermuteStart;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPathIgnoringChannelIdValues(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  const std::vector<std::pair<int64_t, int64_t>> source_target_pairs_;
  const std::vector<std::vector<int64_t>> slice_sizes_;
};

inline bool HloAllReduceInstructionBase::ClassOf(const HloInstruction* hlo) {
  return HloAllReduceInstruction::ClassOf(hlo) ||
         hlo->opcode() == HloOpcode::kReduceScatter;
}

inline bool HloCollectiveInstruction::ClassOf(const HloInstruction* hlo) {
  return HloAllReduceInstructionBase::ClassOf(hlo) ||
         HloAllGatherInstruction::ClassOf(hlo) ||
         HloAllToAllInstruction::ClassOf(hlo);
}

inline bool HloChannelInstruction::ClassOf(const HloInstruction* hlo) {
  return HloCollectiveInstruction::ClassOf(hlo) ||
         HloCollectivePermuteInstruction::ClassOf(hlo) ||
         HloSendRecvInstruction::ClassOf(hlo);
}

class HloReverseInstruction : public HloDimensionsInstruction {
 public:
  explicit HloReverseInstruction(const Shape& shape, HloInstruction* operand,
                                 absl::Span<const int64_t> dimensions);

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kReverse;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloConcatenateInstruction : public HloDimensionsInstruction {
 public:
  explicit HloConcatenateInstruction(const Shape& shape,
                                     absl::Span<HloInstruction* const> operands,
                                     int64_t dimension);
  // Accessor for the dimension in which a concatenate HLO should occur.
  int64_t concatenate_dimension() const override {
    return HloInstruction::dimensions(0);
  }

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kConcatenate;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloReduceInstruction : public HloDimensionsInstruction {
 public:
  explicit HloReduceInstruction(const Shape& shape,
                                absl::Span<HloInstruction* const> args,
                                absl::Span<const int64_t> dimensions_to_reduce,
                                HloComputation* reduce_computation);

  // Returns the number of input arrays (and, consequentially, the number of
  // init values) this reduce has.
  int64_t input_count() const { return operand_count() / 2; }

  // Returns the input tensors to be reduced.
  absl::Span<HloInstruction* const> inputs() const {
    return absl::MakeSpan(operands()).subspan(0, input_count());
  }

  // Returns the init values of the reduction.
  absl::Span<HloInstruction* const> init_values() const {
    return absl::MakeSpan(operands()).subspan(input_count(), operand_count());
  }

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kReduce;
  }

 private:
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloSortInstruction : public HloDimensionsInstruction {
 public:
  explicit HloSortInstruction(const Shape& shape, int64_t dimension,
                              absl::Span<HloInstruction* const> operands,
                              HloComputation* compare, bool is_stable);
  // Returns the sort dimension for this instruction
  int64_t sort_dimension() const { return HloInstruction::dimensions(0); }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;
  // Returns the key operand to this instruction.
  const HloInstruction* keys() const { return operand(0); }
  HloInstruction* mutable_keys() { return mutable_operand(0); }
  // Returns the number of value operands.
  int64_t values_count() const { return operand_count() - 1; }
  bool is_stable() const { return is_stable_; }

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kSort;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  bool is_stable_;
};

class HloTransposeInstruction : public HloDimensionsInstruction {
 public:
  explicit HloTransposeInstruction(const Shape& shape, HloInstruction* operand,
                                   absl::Span<const int64_t> dimensions);
  // Returns whether this instruction does a rank-2 transposition.
  bool IsRank2Transpose() const;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kTranspose;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloBroadcastInstruction : public HloDimensionsInstruction {
 public:
  explicit HloBroadcastInstruction(
      const Shape& shape, HloInstruction* operand,
      absl::Span<const int64_t> broadcast_dimension);

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kBroadcast;
  }

 private:
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
};

class HloDynamicReshapeInstruction : public HloInstruction {
 public:
  explicit HloDynamicReshapeInstruction(
      const Shape& shape, HloInstruction* data_operand,
      absl::Span<HloInstruction* const> dim_sizes);

  // Returns the input dim sizes dimensions, which is operands[1:]
  absl::Span<HloInstruction* const> dim_sizes() const {
    return absl::MakeSpan(operands()).subspan(1, operand_count());
  }

  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Returns the input dim size dimension, which is operands[1+i]
  HloInstruction* dim_sizes(int64_t i) const { return operands()[i + 1]; }

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kDynamicReshape;
  }
};

class HloReshapeInstruction : public HloInstruction {
 public:
  explicit HloReshapeInstruction(const Shape& shape, HloInstruction* operand,
                                 int64_t inferred_dimension);
  int64_t inferred_dimension() const { return inferred_dimension_; }
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kReshape;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  int64_t inferred_dimension_;
};

class HloMapInstruction : public HloInstruction {
 public:
  explicit HloMapInstruction(const Shape& shape,
                             absl::Span<HloInstruction* const> operands,
                             HloComputation* map_computation);
  // Returns the dimension sizes or numbers associated with this instruction.
  absl::Span<const int64_t> dimensions() const override { return dimensions_; }

  std::vector<int64_t>* mutable_dimensions() override { return &dimensions_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kMap;
  }

 private:
  bool IsElementwiseImpl(
      const std::optional<int64_t>& operand_idx) const override;
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::vector<int64_t> dimensions_;
};

class HloSliceInstruction : public HloInstruction {
 public:
  explicit HloSliceInstruction(const Shape& shape, HloInstruction* operand,
                               absl::Span<const int64_t> start_indices,
                               absl::Span<const int64_t> limit_indices,
                               absl::Span<const int64_t> strides);

  HloInstructionProto ToProto() const override;

  // Returns the start index in the given dimension for a slice node.
  int64_t slice_starts(int64_t dimension) const {
    return slice_starts_[dimension];
  }
  const std::vector<int64_t>& slice_starts() const { return slice_starts_; }
  std::vector<int64_t>* mutable_slice_starts() { return &slice_starts_; }

  // Returns the (exclusive) limit index in the given dimension for a slice
  // node.
  int64_t slice_limits(int64_t dimension) const {
    return slice_limits_[dimension];
  }
  const std::vector<int64_t>& slice_limits() const { return slice_limits_; }
  std::vector<int64_t>* mutable_slice_limits() { return &slice_limits_; }

  // Returns the stride in the given dimension for a slice node.
  int64_t slice_strides(int64_t dimension) const {
    return slice_strides_[dimension];
  }
  const std::vector<int64_t>& slice_strides() const { return slice_strides_; }
  std::vector<int64_t>* mutable_slice_strides() { return &slice_strides_; }

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kSlice;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Describes the [begin, end) index range for a slice.
  std::vector<int64_t> slice_starts_;
  std::vector<int64_t> slice_limits_;
  std::vector<int64_t> slice_strides_;
};

class HloConstantInstruction : public HloInstruction {
 public:
  explicit HloConstantInstruction(Literal literal);
  explicit HloConstantInstruction(Literal literal, const Shape& shape);
  // Used when the literal is too large and dropped.
  explicit HloConstantInstruction(const Shape& shape);
  // Returns the literal associated with this instruction.
  const Literal& literal() const { return *literal_; }
  // Returns the (mutable) literal associated with this instruction.
  Literal* mutable_literal() { return &literal_.value(); }
  // Returns whether there is literal associated with this instruction.
  bool HasLiteral() const { return literal_.has_value(); }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Change the layout for an Constant Hlo instruction to match new_layout.  For
  // tuple shaped constants shape_index is the path to the internal array
  // subshape whose layout needs to be changed.
  void RelayoutConstant(const Layout& new_layout,
                        const ShapeIndex& shape_index = {});

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kConstant;
  }

 private:
  bool IsElementwiseImpl(
      const std::optional<int64_t>& operand_idx) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  void PrintOperandsWithCanonicalNameMap(
      Printer* printer, const HloPrintOptions& options,
      CanonicalNameMap* canonical_name_map) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  std::optional<Literal> literal_;
};

// Abstract class that represents an HLO instruction that "calls" a computation.
// Fusion and Call HLOs inherit from this class.
class HloCallableInstruction : public HloInstruction {
 public:
  HloCallableInstruction(HloOpcode opcode, const Shape& shape);

  HloCallableInstruction(HloOpcode opcode, const Shape& shape,
                         absl::Span<HloInstruction* const> operands);

  HloCallableInstruction(HloOpcode opcode, const Shape& shape,
                         absl::Span<HloInstruction* const> operands,
                         HloComputation* called_computation,
                         absl::string_view prefix = "");

  HloCallableInstruction(HloOpcode opcode, const Shape& shape,
                         absl::Span<HloInstruction* const> operands,
                         absl::Span<HloComputation* const> called_computations);

  ~HloCallableInstruction() override;

  // Adds a new operand to the callable instruction.
  HloInstruction* AddCallOperand(HloInstruction* new_operand);

  // Appends (fuses) the given instruction into this callable instruction.
  // instruction_to_append is cloned and the clone is placed in the callable
  // instruction.  The users of instruction_to_append will be redirected to this
  // callable instruction. instruction_to_append is unchanged otherwise. When
  // add_output is true, a clone of the instruction_to_append will be added as
  // additional output resulting in a multi-output callable instruction.
  HloInstruction* AppendInstructionIntoCalledComputation(
      HloInstruction* instruction_to_append, bool add_output = false);
  // Clones the given instruction_to_append and inserts the clone into this
  // callable instruction. If add_output is true, a clone of
  // instruction_to_append will be in the output of the this callable
  // instruction (part of the tuple of the callable root).
  HloInstruction* CloneAndAppendInstructionIntoCalledComputation(
      HloInstruction* instruction_to_append, bool add_output = false);

  // Retrieves the called computations of an HloCallableInstruction that is
  // being cloned. If the called computations have not yet been cloned, then
  // they are first cloned and added to the context.
  absl::InlinedVector<HloComputation*, 1> GetOrCloneCalledComputations(
      HloCloneContext* context) const;

  HloComputation* called_computation() const;

  HloInstruction* called_computation_root() const;

  // Recursively sets all nested called computation to have thread name as
  // `execution_thread`. if `skip_async_execution_thread_overwrite` is true,
  // skip overwrite async instruction and its comptuations thread name
  // overwriting.
  void RecursivelySetComputationsThreadName(
      absl::string_view execution_thread,
      bool skip_async_execution_thread_overwrite);

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kFusion ||
           hlo->opcode() == HloOpcode::kCall ||
           hlo->opcode() == HloOpcode::kCustomCall;
  }

  // Gets a list of output/operand buffer pairs that alias each other, where the
  // output buffer is represented as a ShapeIndex, and the operand buffer is
  // represented as the operand index and the ShapeIndex. By default this list
  // is empty.
  const std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>&
  output_to_operand_aliasing() const {
    return output_to_operand_aliasing_;
  }
  // Sets the list of output/operand buffer pairs that alias each other.
  void set_output_to_operand_aliasing(
      std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
          aliasing) {
    output_to_operand_aliasing_ = std::move(aliasing);
  }

 protected:
  // Returns the default called computation name.
  virtual std::string default_called_computation_name() const = 0;

 private:
  // A list of output/operand buffer pairs that alias each other. See comment of
  // output_to_operand_aliasing().
  std::vector<std::pair<ShapeIndex, std::pair<int64_t, ShapeIndex>>>
      output_to_operand_aliasing_;
};

class HloFusionInstruction : public HloCallableInstruction {
 public:
  explicit HloFusionInstruction(const Shape& shape, FusionKind fusion_kind,
                                HloInstruction* fused_root);

  explicit HloFusionInstruction(const Shape& shape, FusionKind fusion_kind,
                                absl::Span<HloInstruction* const> operands,
                                HloComputation* fusion_computation,
                                absl::string_view prefix = "");

  ~HloFusionInstruction() override;

  void ClearCalledComputations() override;

  // When a fusion instruction is being destructed, clear the back pointer of
  // its fusion computation, to avoid referencing freed memory.
  void ClearFusionComputationInstruction();

  std::string ToCategory() const override;
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Adds a new operand the fusion instruction.
  HloInstruction* AddFusionOperand(HloInstruction* new_operand);

  // Merges the fused instructions from 'instruction_to_merge' into the
  // fused instruction set of 'this', updating operands as necessary.
  //
  // Precondition: 'instruction_to_merge' must be an operand of 'this'.
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
    CHECK(instruction_to_fuse->IsFusible()) << instruction_to_fuse->ToString();
    return AppendInstructionIntoCalledComputation(instruction_to_fuse);
  }

  // Fuses the given instruction in this fusion instruction and generates a
  // multioutput fusion instruction. A clone of the instruction_to_fuse will
  // be part of the output of fusion instructions. The users of
  // instruction_to_fuse will be redirected to this fusion instructions.
  // instruction_to_fuse is unchanged otherwise.
  HloInstruction* FuseInstructionIntoMultiOutput(
      HloInstruction* instruction_to_fuse) {
    return AppendInstructionIntoCalledComputation(instruction_to_fuse,
                                                  /*add_output=*/true);
  }

  // Returns the computation for this fused instruction.
  HloComputation* fused_instructions_computation() const;

  // Returns the root instruction of the fused expression contained within this
  // fusion instruction.
  HloInstruction* fused_expression_root() const;

  // Returns the list of fused instructions inside this fusion instruction.  The
  // returned type is a range of HloInstruction*s.
  const tsl::gtl::iterator_range<UnwrappingIterator<
      std::list<std::unique_ptr<HloInstruction>>::const_iterator>>
  fused_instructions() const;

  const tsl::gtl::iterator_range<
      UnwrappingIterator<std::list<std::unique_ptr<HloInstruction>>::iterator>>
  fused_instructions();

  // Gets the number of instructions inside this fusion instruction.
  int64_t fused_instruction_count() const;

  // Returns the fused parameter instruction in this fusion instruction
  // corresponding to the given parameter number.
  HloInstruction* fused_parameter(int64_t parameter_number) const;

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

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kFusion;
  }

 protected:
  std::string default_called_computation_name() const override {
    return "fused_computation";
  }

 private:
  bool IsElementwiseImpl(
      const std::optional<int64_t>& operand_idx) const override;
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;

  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // The type of the fusion.
  FusionKind fusion_kind_;
};

class HloCallInstruction : public HloCallableInstruction {
 public:
  HloCallInstruction(const Shape& shape,
                     HloInstruction* called_computation_root);

  HloCallInstruction(const Shape& shape,
                     absl::Span<HloInstruction* const> operands,
                     HloComputation* called_computation);

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kCall;
  }

 protected:
  std::string default_called_computation_name() const override {
    return "called_computation";
  }
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

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kRng;
  }

 private:
  bool IsElementwiseImpl(
      const std::optional<int64_t>& operand_idx) const override;
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
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
  explicit HloParameterInstruction(int64_t parameter_number, const Shape& shape,
                                   const std::string& name);
  int64_t parameter_number() const { return parameter_number_; }

  // Sets and gets the whether all replicas will receive the same parameter data
  // for each leaf buffer in data parallelism.
  void set_parameter_replicated_at_leaf_buffers(
      absl::Span<const bool> parameter_replicated_at_leaf_buffers) {
    CHECK_EQ(ShapeUtil::GetLeafCount(shape()),
             parameter_replicated_at_leaf_buffers.size());
    parameter_replicated_at_leaf_buffers_.emplace(
        parameter_replicated_at_leaf_buffers.begin(),
        parameter_replicated_at_leaf_buffers.end());
  }
  void set_parameter_replicated_at_leaf_buffers(
      const std::vector<bool>& parameter_replicated_at_leaf_buffers) {
    CHECK_EQ(ShapeUtil::GetLeafCount(shape()),
             parameter_replicated_at_leaf_buffers.size());
    parameter_replicated_at_leaf_buffers_ =
        parameter_replicated_at_leaf_buffers;
  }
  const std::optional<std::vector<bool>>& parameter_replicated_at_leaf_buffers()
      const {
    return parameter_replicated_at_leaf_buffers_;
  }

  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kParameter;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  void PrintOperandsWithCanonicalNameMap(
      Printer* printer, const HloPrintOptions& options,
      CanonicalNameMap* canonical_name_map) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t parameter_number_ = 0;

  // Specifies whether each buffer has the same parameter value on all replicas
  // in data parallelism.
  std::optional<std::vector<bool>> parameter_replicated_at_leaf_buffers_;
};

class HloGetTupleElementInstruction : public HloInstruction {
 public:
  explicit HloGetTupleElementInstruction(const Shape& shape,
                                         HloInstruction* operand,
                                         int64_t index);
  // Returns the tuple index associated with this instruction.
  int64_t tuple_index() const { return tuple_index_; }
  // Sets the tuple index associated with this instruction.
  void set_tuple_index(int64_t new_tuple_index) {
    tuple_index_ = new_tuple_index;
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kGetTupleElement;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t tuple_index_ = -1;
};

class HloReducePrecisionInstruction : public HloInstruction {
 public:
  explicit HloReducePrecisionInstruction(const Shape& shape,
                                         HloInstruction* operand,
                                         const int exponent_bits,
                                         const int mantissa_bits);
  // Returns the number of exponent bits for a reduce-precision node.
  int32_t exponent_bits() const { return exponent_bits_; }
  // Returns the number of mantissa bits for a reduce-precision node.
  int32_t mantissa_bits() const { return mantissa_bits_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kReducePrecision;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // The bit sizes for a reduce-precision operation.
  int32_t exponent_bits_ = 0;
  int32_t mantissa_bits_ = 0;
};

class HloInfeedInstruction : public HloInstruction {
 public:
  explicit HloInfeedInstruction(const Shape& infeed_shape,
                                HloInstruction* token_operand,
                                const std::string& config);
  // Returns the infeed configuration string. The infeed configuration includes
  // any metadata needed for the backend compiler (e.g., infeed buffer address)
  // and is target-dependent.
  std::string infeed_config() const { return infeed_config_; }
  void set_infeed_config(const std::string& config) { infeed_config_ = config; }
  // Returns the shape of the data received by the infeed. This is not the same
  // as the shape of the infeed instruction which produces a tuple containing
  // the infeed data shape and a TOKEN.
  const Shape& infeed_shape() const {
    TF_DCHECK_OK(ShapeUtil::ValidateShapeWithOptionalLayout(shape()));
    return ShapeUtil::GetSubshape(shape(), {0});
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kInfeed;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // The string representation of the infeed configuration.
  std::string infeed_config_;
};

class HloOutfeedInstruction : public HloInstruction {
 public:
  explicit HloOutfeedInstruction(const Shape& outfeed_shape,
                                 HloInstruction* operand,
                                 HloInstruction* token_operand,
                                 absl::string_view outfeed_config);
  // Returns the shape for the Outfeed instruction.
  const Shape& outfeed_shape() const { return outfeed_shape_; }
  // Returns the mutable shape for the Outfeed instruction.
  Shape* mutable_outfeed_shape() { return &outfeed_shape_; }
  // Returns the config for the Outfeed instruction.
  const std::string& outfeed_config() const { return outfeed_config_; }
  void set_outfeed_config(const std::string& config) {
    outfeed_config_ = config;
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kOutfeed;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Shape of outfeed request.
  Shape outfeed_shape_;
  // Outfeed configuration information, only present for kOutfeed.
  std::string outfeed_config_;
};

class HloConvolutionInstruction : public HloInstruction {
 public:
  explicit HloConvolutionInstruction(
      const Shape& shape, HloInstruction* lhs, HloInstruction* rhs,
      int64_t feature_group_count, int64_t batch_group_count,
      const Window& window,
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
  int64_t feature_group_count() const { return feature_group_count_; }
  void set_feature_group_count(int64_t num_feature_groups) {
    feature_group_count_ = num_feature_groups;
  }
  // The number of batch groups. Must be a divisor of the input batch dimension.
  int64_t batch_group_count() const { return batch_group_count_; }
  void set_batch_group_count(int64_t num_batch_groups) {
    batch_group_count_ = num_batch_groups;
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

  std::string ToCategory() const override;
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kConvolution;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  // The number of feature groups. Must be a divisor of the input feature
  // dimension and output feature dimension.
  int64_t feature_group_count_;
  // The number of batch groups. Must be a divisor of the input batch dimension.
  int64_t batch_group_count_;
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
  explicit HloReduceWindowInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      absl::Span<HloInstruction* const> init_values, const Window& window,
      HloComputation* reduce_computation);
  const Window& window() const override { return window_; }
  void set_window(const Window& window) override { window_ = window; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;
  // Returns the number of input arrays (and, consequentially, the number of
  // init values) this reduce has.
  int64_t input_count() const { return operand_count() / 2; }
  // Returns the input tensors to be reduced.
  absl::Span<HloInstruction* const> inputs() const {
    return absl::MakeSpan(operands()).subspan(0, input_count());
  }
  // Returns the init values of the reduction.
  absl::Span<HloInstruction* const> init_values() const {
    return absl::MakeSpan(operands()).subspan(input_count(), operand_count());
  }
  // Returns the shapes of input tensors to be reduced.
  absl::InlinedVector<const Shape*, 2> input_shapes() const {
    absl::InlinedVector<const Shape*, 2> shapes;
    for (const auto* op : inputs()) {
      VLOG(2) << "Pushing input array shape for: " << op->ToString() << "\n";
      shapes.push_back(&op->shape());
      VLOG(2) << "Pushed shape: " << shapes.back()->ToString() << "\n";
    }
    return shapes;
  }
  // Returns the init values of the reduction.
  absl::InlinedVector<const Shape*, 2> init_value_shapes() const {
    absl::InlinedVector<const Shape*, 2> shapes;
    for (const auto* op : init_values()) {
      shapes.push_back(&op->shape());
    }
    return shapes;
  }
  // Returns the shapes of the reduced output tensors.
  absl::InlinedVector<const Shape*, 2> output_shapes() const {
    absl::InlinedVector<const Shape*, 2> shapes;
    if (shape().IsArray()) {
      shapes.push_back(&shape());
    } else {
      for (const Shape& tuple_element_shape : shape().tuple_shapes()) {
        shapes.push_back(&tuple_element_shape);
      }
    }
    return shapes;
  }

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kReduceWindow;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
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

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kSelectAndScatter;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  Window window_;
};

class HloCustomCallInstruction : public HloCallableInstruction {
 public:
  HloCustomCallInstruction(const Shape& shape,
                           absl::Span<HloInstruction* const> operands,
                           absl::string_view custom_call_target,
                           std::string opaque,
                           CustomCallApiVersion api_version);

  // Constructor for a custom call with constrained layout. 'shape' and
  // 'operands_with_layout' must all have layouts.
  HloCustomCallInstruction(const Shape& shape,
                           absl::Span<HloInstruction* const> operands,
                           absl::string_view custom_call_target,
                           std::string opaque,
                           absl::Span<const Shape> operand_shapes_with_layout,
                           CustomCallApiVersion api_version);

  // Constructor for a custom call with a to_apply computation.
  HloCustomCallInstruction(const Shape& shape,
                           absl::Span<HloInstruction* const> operands,
                           HloComputation* to_apply,
                           absl::string_view custom_call_target,
                           std::string opaque,
                           CustomCallApiVersion api_version);

  // Constructor for a custom call with multiple computations.
  HloCustomCallInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> operands,
      absl::Span<HloComputation* const> called_computations,
      absl::string_view custom_call_target, std::string opaque,
      CustomCallApiVersion api_version);

  const Window& window() const override {
    CHECK(window_ != nullptr);
    return *window_;
  }

  void set_window(const Window& window) override {
    window_ = std::make_unique<Window>(window);
  }

  const ConvolutionDimensionNumbers& convolution_dimension_numbers() const {
    CHECK(convolution_dimension_numbers_ != nullptr);
    return *convolution_dimension_numbers_;
  }

  void set_convolution_dimension_numbers(
      const ConvolutionDimensionNumbers& dnums) {
    convolution_dimension_numbers_ =
        std::make_unique<ConvolutionDimensionNumbers>(dnums);
  }
  // TODO(jpienaar): Remove this accessor in the follow up.
  const std::string& opaque() const { return raw_backend_config_string(); }
  const std::string& custom_call_target() const { return custom_call_target_; }
  void set_custom_call_target(absl::string_view target) {
    custom_call_target_ = std::string(target);
  }
  void set_feature_group_count(int64_t feature_group_count) {
    feature_group_count_ = feature_group_count;
  }
  void set_batch_group_count(int64_t batch_group_count) {
    batch_group_count_ = batch_group_count;
  }
  // Sets whether this custom call has a side-effect - by default a custom call
  // has no side-effects.
  void set_custom_call_has_side_effect(bool custom_call_has_side_effect) {
    custom_call_has_side_effect_ = custom_call_has_side_effect;
  }
  int64_t feature_group_count() const { return feature_group_count_; }
  int64_t batch_group_count() const { return batch_group_count_; }
  bool custom_call_has_side_effect() const {
    return custom_call_has_side_effect_;
  }
  // Returns padding type used for ops like convolution.
  PaddingType padding_type() const { return padding_type_; }

  void set_padding_type(PaddingType padding_type) {
    padding_type_ = padding_type;
  }

  // Returns the literal associated with this instruction.
  const Literal& literal() const { return *literal_; }
  // Set the value of literal to a new one.
  void set_literal(Literal&& literal) { literal_.emplace(std::move(literal)); }
  // Returns whether there is literal associated with this instruction.
  bool HasLiteral() const { return literal_.has_value(); }

  const PrecisionConfig& precision_config() const { return precision_config_; }
  PrecisionConfig* mutable_precision_config() { return &precision_config_; }

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
  void set_custom_call_schedule(CustomCallSchedule custom_call_schedule) {
    custom_call_schedule_ = custom_call_schedule;
  }
  CustomCallSchedule custom_call_schedule() const {
    return custom_call_schedule_;
  }
  void set_api_version(CustomCallApiVersion api_version) {
    api_version_ = api_version;
  }
  CustomCallApiVersion api_version() const { return api_version_; }

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kCustomCall;
  }

 protected:
  std::string default_called_computation_name() const override {
    return "custom_call_computation";
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;
  // Name of a global symbol to call.
  std::string custom_call_target_;
  // Describes the window in a windowed operation such as convolution.
  std::unique_ptr<Window> window_;
  // Describes the dimension numbers used for a convolution.
  std::unique_ptr<ConvolutionDimensionNumbers> convolution_dimension_numbers_;
  // The number of feature groups. This is used for grouped convolutions.
  int64_t feature_group_count_;
  int64_t batch_group_count_;
  // Whether the result and operand layouts are constrained.
  bool layout_constrained_;
  // Information used to communicate to the implementation about the algorithm
  // used to produce results for convolution instructions.
  PrecisionConfig precision_config_;
  // Describes the padding type for convolution instructions.
  PaddingType padding_type_;
  // For layout-constrained custom calls, this vector holds the shape with
  // layout for each operand.
  std::vector<Shape> operand_shapes_with_layout_;
  // Whether this custom call has a side-effect.
  bool custom_call_has_side_effect_;
  std::optional<Literal> literal_;
  // A custom-call schedule hint.
  CustomCallSchedule custom_call_schedule_;
  // The version of the API used by the custom call function.
  // TODO(b/189822916): Remove this field when all clients are migrated to the
  // status-returning API.
  CustomCallApiVersion api_version_;
};

class HloPadInstruction : public HloInstruction {
 public:
  explicit HloPadInstruction(const Shape& shape, HloInstruction* operand,
                             HloInstruction* padding_value,
                             const PaddingConfig& padding_config);
  // Returns the padding configuration for a pad node.
  const PaddingConfig& padding_config() const { return padding_config_; }
  PaddingConfig* mutable_padding_config() { return &padding_config_; }
  // Returns the operand being padded.
  const HloInstruction* padded_operand() const { return operand(0); }
  HloInstruction* mutable_padded_operand() { return mutable_operand(0); }
  // Returns the padding value.
  const HloInstruction* padding_value() const { return operand(1); }
  HloInstruction* mutable_padding_value() { return mutable_operand(1); }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kPad;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // The padding configuration that describes the edge padding and interior
  // padding of this pad instruction.
  PaddingConfig padding_config_;
};

class HloDynamicIndexInstruction : public HloInstruction {
 public:
  explicit HloDynamicIndexInstruction(HloOpcode opcode, const Shape& shape)
      : HloInstruction(opcode, shape) {}
  virtual int64_t first_index_operand_number() const = 0;

  // Returns a subspan of operands which represent the start indices.
  absl::Span<HloInstruction* const> index_operands() const {
    return absl::MakeSpan(operands()).subspan(first_index_operand_number());
  }

  // Returns the shapes of the index operands.
  std::vector<Shape> index_shapes() const {
    std::vector<Shape> shapes;
    auto indices = index_operands();
    for (const HloInstruction* index : indices) {
      shapes.push_back(index->shape());
    }
    return shapes;
  }

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kDynamicSlice ||
           hlo->opcode() == HloOpcode::kDynamicUpdateSlice;
  }
};

class HloDynamicSliceInstruction : public HloDynamicIndexInstruction {
 public:
  explicit HloDynamicSliceInstruction(const Shape& shape,
                                      HloInstruction* operand,
                                      HloInstruction* start_indices,
                                      absl::Span<const int64_t> slice_sizes);
  explicit HloDynamicSliceInstruction(
      const Shape& shape, HloInstruction* operand,
      absl::Span<HloInstruction* const> start_indices,
      absl::Span<const int64_t> slice_sizes);
  // Old methods kept for smooth subclassing transition END.
  // Returns the size of the slice in the given dimension for a dynamic
  // slice node.
  int64_t slice_sizes(int64_t dimension) const {
    return dynamic_slice_sizes_[dimension];
  }
  const std::vector<int64_t>& dynamic_slice_sizes() const {
    return dynamic_slice_sizes_;
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  int64_t first_index_operand_number() const override { return 1; }
  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kDynamicSlice;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  // Describes the [start, start + size) range size for a dynamic slice
  // ('start' is specified dynamically in the second operand of the operation).
  std::vector<int64_t> dynamic_slice_sizes_;
};

class HloDynamicUpdateSliceInstruction : public HloDynamicIndexInstruction {
 public:
  explicit HloDynamicUpdateSliceInstruction(const Shape& shape,
                                            HloInstruction* operand,
                                            HloInstruction* update,
                                            HloInstruction* start_indices);
  explicit HloDynamicUpdateSliceInstruction(
      const Shape& shape, HloInstruction* operand, HloInstruction* update,
      absl::Span<HloInstruction* const> start_indices);

  int64_t first_index_operand_number() const override { return 2; }

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kDynamicUpdateSlice;
  }
};

class HloGatherInstruction : public HloInstruction {
 public:
  explicit HloGatherInstruction(
      const Shape& shape, HloInstruction* operand,
      HloInstruction* start_indices,
      const GatherDimensionNumbers& gather_dim_numbers,
      absl::Span<const int64_t> slice_sizes, bool indices_are_sorted);
  const GatherDimensionNumbers& gather_dimension_numbers() const {
    CHECK(gather_dimension_numbers_ != nullptr);
    return *gather_dimension_numbers_;
  }
  absl::Span<const int64_t> gather_slice_sizes() const {
    return gather_slice_sizes_;
  }
  bool indices_are_sorted() const { return indices_are_sorted_; }
  void set_indices_are_sorted(bool indices_are_sorted) {
    indices_are_sorted_ = indices_are_sorted;
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  // Creates an instance of GatherDimensionNumbers.
  static GatherDimensionNumbers MakeGatherDimNumbers(
      absl::Span<const int64_t> offset_dims,
      absl::Span<const int64_t> collapsed_slice_dims,
      absl::Span<const int64_t> start_index_map, int64_t index_vector_dim);
  // Returns the dump string of the given gather dimension numbers.
  static std::string GatherDimensionNumbersToString(
      const GatherDimensionNumbers& gather_dimension_numbers);

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kGather;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::unique_ptr<GatherDimensionNumbers> gather_dimension_numbers_;
  std::vector<int64_t> gather_slice_sizes_;
  bool indices_are_sorted_;
};

class HloScatterInstruction : public HloInstruction {
 public:
  explicit HloScatterInstruction(
      const Shape& shape, absl::Span<HloInstruction* const> args,
      HloComputation* update_computation,
      const ScatterDimensionNumbers& scatter_dim_numbers,
      bool indices_are_sorted, bool unique_indices);
  const ScatterDimensionNumbers& scatter_dimension_numbers() const {
    CHECK(scatter_dimension_numbers_ != nullptr);
    return *scatter_dimension_numbers_;
  }
  bool indices_are_sorted() const { return indices_are_sorted_; }
  void set_indices_are_sorted(bool indices_are_sorted) {
    indices_are_sorted_ = indices_are_sorted;
  }
  bool unique_indices() const override { return unique_indices_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;
  int64_t scatter_operand_count() const { return operand_count() / 2; }
  absl::Span<HloInstruction* const> scatter_operands() const {
    return absl::MakeConstSpan(operands()).first(scatter_operand_count());
  }
  absl::Span<HloInstruction* const> scatter_updates() const {
    return absl::MakeConstSpan(operands()).last(scatter_operand_count());
  }
  const HloInstruction* scatter_indices() const {
    return operand(scatter_operand_count());
  }
  HloInstruction* scatter_indices() {
    return mutable_operand(scatter_operand_count());
  }

  // Creates an instance of ScatterDimensionNumbers.
  static ScatterDimensionNumbers MakeScatterDimNumbers(
      absl::Span<const int64_t> update_window_dims,
      absl::Span<const int64_t> inserted_window_dims,
      absl::Span<const int64_t> scatter_dims_to_operand_dims,
      int64_t index_vector_dim);
  // Returns the dump string of the given scatter dimension numbers.
  static std::string ScatterDimensionNumbersToString(
      const ScatterDimensionNumbers& scatter_dimension_numbers);

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kScatter;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  std::unique_ptr<ScatterDimensionNumbers> scatter_dimension_numbers_;
  bool indices_are_sorted_;
  bool unique_indices_;
};

class HloIotaInstruction : public HloInstruction {
 public:
  explicit HloIotaInstruction(const Shape& shape, int64_t iota_dimension);

  // Returns the dimension sizes or numbers associated with this instruction.
  int64_t iota_dimension() const { return iota_dimension_; }
  absl::Span<const int64_t> dimensions() const override {
    return absl::MakeConstSpan(&iota_dimension_, 1);
  }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kIota;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t iota_dimension_;
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

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kDot;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

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

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kDomain;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
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
                                          int64_t dimension);

  // Returns the dimension sizes or numbers associated with this instruction.
  int64_t dimension() const { return dimension_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kGetDimensionSize;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t dimension_;
};

class HloSetDimensionSizeInstruction : public HloInstruction {
 public:
  explicit HloSetDimensionSizeInstruction(const Shape& shape,
                                          HloInstruction* operand,
                                          HloInstruction* val,
                                          int64_t dimension);

  // Returns the dimension sizes or numbers associated with this instruction.
  int64_t dimension() const { return dimension_; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kSetDimensionSize;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t dimension_;
};

class HloRngGetAndUpdateStateInstruction : public HloInstruction {
 public:
  explicit HloRngGetAndUpdateStateInstruction(const Shape& shape,
                                              int64_t delta);

  // Returns the delta value.
  int64_t delta() const { return delta_; }
  void set_delta(int64_t delta) { delta_ = delta; }
  // Returns a serialized representation of this instruction.
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kRngGetAndUpdateState;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  // Implementation for non-common logic of CloneWithNewOperands.
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  int64_t delta_;
};

class HloRngBitGeneratorInstruction : public HloInstruction {
 public:
  HloRngBitGeneratorInstruction(const Shape& shape, HloInstruction* state,
                                RandomAlgorithm algorithm);

  RandomAlgorithm algorithm() const { return algorithm_; }
  HloInstructionProto ToProto() const override;

  static bool ClassOf(const HloInstruction* hlo) {
    return hlo->opcode() == HloOpcode::kRngBitGenerator;
  }

 private:
  std::vector<std::string> ExtraAttributesToStringImpl(
      const HloPrintOptions& options) const override;
  bool IdenticalSlowPath(
      const HloInstruction& other,
      absl::FunctionRef<bool(const HloComputation*, const HloComputation*)>
          eq_computations) const override;
  std::unique_ptr<HloInstruction> CloneWithNewOperandsImpl(
      const Shape& shape, absl::Span<HloInstruction* const> new_operands,
      HloCloneContext* context) const override;

  RandomAlgorithm algorithm_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_HLO_IR_HLO_INSTRUCTIONS_H_
