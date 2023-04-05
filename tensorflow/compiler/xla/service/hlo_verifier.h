/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VERIFIER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VERIFIER_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// Callback to return shape size, in bytes.
using ShapeSizeFn = std::function<int64_t(const Shape&)>;

struct HloVerifierOpts {
  HloVerifierOpts&& MakeLayoutSensitive() {
    layout_sensitive = true;
    return std::move(*this);
  }

  HloVerifierOpts&& WithLayoutSensitive(bool layout_sensitive_p) {
    layout_sensitive = layout_sensitive_p;
    return std::move(*this);
  }

  HloVerifierOpts&& WithAllowMixedPrecision(bool allow_mixed_precision_p) {
    allow_mixed_precision = allow_mixed_precision_p;
    return std::move(*this);
  }

  HloVerifierOpts&& AllowMixedPrecision() {
    allow_mixed_precision = true;
    return std::move(*this);
  }

  HloVerifierOpts&& VerifyBroadcastDimensionsOrder() {
    verify_broadcast_dimensions_order = true;
    return std::move(*this);
  }

  HloVerifierOpts&& VerifyReshapeIsBitcast() {
    verify_reshape_is_bitcast = true;
    return std::move(*this);
  }

  HloVerifierOpts&& VerifyCustomCallNestedComputationThreadName() {
    verify_custom_call_nested_computation_thread_name = true;
    return std::move(*this);
  }

  HloVerifierOpts&& WithAllowBitcastToHaveDifferentSize(bool allow) {
    allow_bitcast_to_have_different_size = allow;
    return std::move(*this);
  }

  HloVerifierOpts&& WithInstructionCanChangeLayout(
      const HloPredicate& instruction_can_change_layout_p) {
    instruction_can_change_layout = instruction_can_change_layout_p;
    return std::move(*this);
  }

  HloVerifierOpts&& WithCustomShapeSize(const ShapeSizeFn& shape_size_p) {
    shape_size = shape_size_p;
    return std::move(*this);
  }

  HloVerifierOpts&& WithVerifyShardingDeviceNumbers(bool verify) {
    verify_sharding_device_numbers = verify;
    return std::move(*this);
  }

  bool IsLayoutSensitive() const { return layout_sensitive; }

  bool AllowMixedPrecision() const { return allow_mixed_precision; }

  const HloPredicate& InstructionCanChangeLayout() const {
    return instruction_can_change_layout;
  }

  bool InstructionCanChangeLayout(const HloInstruction* instruction) const {
    return !instruction_can_change_layout ||
           instruction_can_change_layout(instruction);
  }

  int64_t ShapeSize(const Shape& shape) const { return shape_size(shape); }

  // If the verifier is layout-sensitive, shapes must be equal to what's
  // expected.  Otherwise, the shapes must simply be compatible.
  bool layout_sensitive = false;

  // Whether the inputs and output of an instruction can contain both F32s and
  // BF16s. Tuples that include both F32s and BF16s are allowed regardless of
  // this flag.
  bool allow_mixed_precision = false;

  // Check that `dimensions` attribute of broadcast is sorted.
  bool verify_broadcast_dimensions_order = false;

  // Check that reshape is a physical bitcast.
  bool verify_reshape_is_bitcast = false;

  // Check that custom call's called computations have same thread name as
  // parent computation.
  bool verify_custom_call_nested_computation_thread_name = true;

  // Check device numbers in sharding verification.
  bool verify_sharding_device_numbers = true;

  // Whether bitcast should have the same size, including all paddings.
  bool allow_bitcast_to_have_different_size = false;

  HloPredicate instruction_can_change_layout;

  // Returns a target-specific shape size.
  ShapeSizeFn shape_size = [](const Shape& shape) {
    return ShapeUtil::ByteSizeOf(shape);
  };
};

// Visitor which verifies that the output shape is correctly set. Verifies
// against the inferred shape for the instruction.
class ShapeVerifier : public DfsHloVisitor {
 public:
  explicit ShapeVerifier(const HloVerifierOpts& opts) : opts_(opts) {}

  // Verifies that entry computation layout matches parameters and root shape of
  // the module's entry computation.
  virtual Status VerifyEntryComputationLayout(const HloModule& module);

  Status Preprocess(HloInstruction* hlo) override;

  Status HandleElementwiseUnary(HloInstruction* hlo) override;
  Status HandleElementwiseBinary(HloInstruction* hlo) override;
  Status HandleClamp(HloInstruction* clamp) override;
  Status HandleSelect(HloInstruction* select) override;
  Status HandleConcatenate(HloInstruction* concatenate) override;
  Status HandleIota(HloInstruction* hlo) override;
  Status HandleConvert(HloInstruction* convert) override;
  Status HandleBitcastConvert(HloInstruction* convert) override;
  Status HandleStochasticConvert(HloInstruction* convert) override;
  Status HandleCopy(HloInstruction* copy) override;
  Status HandleDot(HloInstruction* dot) override;
  Status HandleConvolution(HloInstruction* convolution) override;
  Status HandleFft(HloInstruction* fft) override;
  Status HandleCholesky(HloInstruction* hlo) override;
  Status HandleTriangularSolve(HloInstruction* hlo) override;
  Status HandleAllGather(HloInstruction* hlo) override;
  Status HandleAllGatherStart(HloInstruction* hlo) override;
  Status HandleAllGatherDone(HloInstruction* hlo) override;
  Status HandleAllReduce(HloInstruction* hlo) override;
  Status HandleAllReduceStart(HloInstruction* hlo) override;
  Status HandleAllReduceDone(HloInstruction* hlo) override;
  Status HandleAllToAll(HloInstruction* hlo) override;
  Status HandleCollectivePermute(HloInstruction* hlo) override;
  Status HandleCollectivePermuteStart(HloInstruction* hlo) override;
  Status HandleCollectivePermuteDone(HloInstruction* hlo) override;
  Status HandlePartitionId(HloInstruction* hlo) override;
  Status HandleReplicaId(HloInstruction* hlo) override;
  Status HandleReducePrecision(HloInstruction* reduce_precision) override;
  Status HandleInfeed(HloInstruction*) override;
  Status HandleOptimizationBarrier(HloInstruction* hlo) override;
  Status HandleOutfeed(HloInstruction*) override;
  Status HandleRng(HloInstruction*) override;
  Status HandleRngBitGenerator(HloInstruction*) override;
  Status HandleRngGetAndUpdateState(HloInstruction*) override;
  Status HandleReverse(HloInstruction* reverse) override;
  Status HandleSort(HloInstruction* hlo) override;
  Status HandleConstant(HloInstruction* constant) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;
  Status HandleReduce(HloInstruction* reduce) override;
  Status HandleBitcast(HloInstruction* bitcast) override;
  Status HandleBroadcast(HloInstruction* broadcast) override;
  Status HandleReshape(HloInstruction* reshape) override;
  Status HandleDynamicReshape(HloInstruction* dynamic_reshape) override;
  Status HandleTranspose(HloInstruction* transpose) override;
  Status HandleParameter(HloInstruction*) override;
  Status HandleFusion(HloInstruction*) override;
  Status HandleCall(HloInstruction* call) override;
  Status HandleCustomCall(HloInstruction*) override;
  Status HandleSlice(HloInstruction* slice) override;
  Status HandleDynamicSlice(HloInstruction* dynamic_slice) override;
  Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override;
  Status HandleTuple(HloInstruction* tuple) override;
  Status HandleMap(HloInstruction* map) override;
  Status HandleReduceScatter(HloInstruction* hlo) override;
  Status HandleReduceWindow(HloInstruction* reduce_window) override;
  Status HandleSelectAndScatter(HloInstruction* instruction) override;
  Status HandleWhile(HloInstruction* xla_while) override;
  Status HandleConditional(HloInstruction* conditional) override;
  Status HandlePad(HloInstruction* pad) override;
  Status HandleAsyncStart(HloInstruction* async_start) override;
  Status HandleAsyncUpdate(HloInstruction* async_update) override;
  Status HandleAsyncDone(HloInstruction* async_done) override;
  Status HandleCopyStart(HloInstruction* copy_start) override;
  Status HandleCopyDone(HloInstruction* copy_done) override;
  Status HandleSend(HloInstruction* send) override;
  Status HandleSendDone(HloInstruction* send_done) override;
  Status HandleRecv(HloInstruction* recv) override;
  Status HandleRecvDone(HloInstruction* recv_done) override;
  Status HandleBatchNormTraining(HloInstruction* batch_norm_training) override;
  Status HandleBatchNormInference(
      HloInstruction* batch_norm_inference) override;
  Status HandleBatchNormGrad(HloInstruction* batch_norm_grad) override;
  Status HandleGather(HloInstruction* gather) override;
  Status HandleScatter(HloInstruction* scatter) override;
  Status HandleAfterAll(HloInstruction* token) override;
  Status HandleGetDimensionSize(HloInstruction* get_size) override;
  Status HandleSetDimensionSize(HloInstruction* set_size) override;
  Status HandleAddDependency(HloInstruction* add_dependency) override;

  Status FinishVisit(HloInstruction*) override { return OkStatus(); }

 protected:
  // Check the instruction's shape against the shape given by ShapeInference
  // and return an appropriate error if there is a mismatch.
  Status CheckShape(const HloInstruction* instruction,
                    const Shape& inferred_shape,
                    bool only_compare_minor_to_major_in_layout = false);

  // Overload which takes a StatusOr to reduce boilerplate in the caller.
  Status CheckShape(const HloInstruction* instruction,
                    const StatusOr<Shape>& inferred_shape_status);

  // Check a unary (binary, etc) instruction's shape against the inferred shape.
  Status CheckUnaryShape(const HloInstruction* instruction);
  Status CheckBinaryShape(const HloInstruction* instruction);
  Status CheckTernaryShape(const HloInstruction* instruction);
  Status CheckVariadicShape(const HloInstruction* instruction);

 private:
  // Helpers that switch on layout_sensitive_.
  bool ShapesSame(const Shape& a, const Shape& b,
                  bool minor_to_major_only = false,
                  bool ignore_memory_space = false) {
    if (!opts_.layout_sensitive) {
      return ShapeUtil::Compatible(a, b);
    }
    Shape::Equal equal;
    if (ignore_memory_space) {
      equal.IgnoreMemorySpaceInLayout();
    }
    if (minor_to_major_only) {
      equal.MinorToMajorOnlyInLayout();
    }
    return equal(a, b);
  }

  bool ShapesSameIgnoringFpPrecision(const Shape& a, const Shape& b,
                                     bool minor_to_major_only = false) {
    if (!opts_.layout_sensitive) {
      return ShapeUtil::CompatibleIgnoringFpPrecision(a, b);
    }
    Shape::Equal equal;
    if (minor_to_major_only) {
      equal.MinorToMajorOnlyInLayout();
    }
    equal.IgnoreFpPrecision();
    return equal(a, b);
  }

  std::string StringifyShape(const Shape& s) {
    return opts_.layout_sensitive ? ShapeUtil::HumanStringWithLayout(s)
                                  : ShapeUtil::HumanString(s);
  }

  // Helpers that switch on allow_mixed_precision_.
  bool SameElementType(const Shape& a, const Shape& b) {
    return opts_.allow_mixed_precision
               ? ShapeUtil::SameElementTypeIgnoringFpPrecision(a, b)
               : ShapeUtil::SameElementType(a, b);
  }

  // Checks that the given operand of the given instruction is of type TOKEN.
  Status CheckIsTokenOperand(const HloInstruction* instruction,
                             int64_t operand_no);

  // Checks that the shape of the given operand of the given instruction matches
  // the given parameter of the given computation.
  Status CheckOperandAndParameter(const HloInstruction* instruction,
                                  int64_t operand_number,
                                  const HloComputation* computation,
                                  int64_t parameter_number);

  // Returns true if the shapes of the two operands have the same element type,
  // and the result shape either has the same element type as the operand shapes
  // or mixed precision is allowed and the result shape and the operand shapes
  // have floating point element types.
  bool HasCompatibleElementTypes(const Shape& shape_0, const Shape& shape_1,
                                 const Shape& result_shape);

  const HloVerifierOpts& opts_;
};

// An interface used to encapsulate target-specific verification quirks.
class TargetVerifierMetadata {
 public:
  explicit TargetVerifierMetadata(HloVerifierOpts&& opts) : opts_(opts) {
    CHECK(opts.instruction_can_change_layout == nullptr ||
          opts.layout_sensitive);
  }

  virtual std::unique_ptr<ShapeVerifier> GetVerifier() const = 0;

  TargetVerifierMetadata() = default;
  virtual ~TargetVerifierMetadata() = default;

  TargetVerifierMetadata(const TargetVerifierMetadata&) = delete;
  TargetVerifierMetadata& operator=(const TargetVerifierMetadata&) = delete;

  const HloVerifierOpts& GetVerifierOpts() const { return opts_; }

 private:
  HloVerifierOpts opts_;
};

// The default implementation of TargetVerifierMetadata, used unless the target
// needs to override it.
class DefaultVerifierMetadata : public TargetVerifierMetadata {
 public:
  explicit DefaultVerifierMetadata(HloVerifierOpts&& opts)
      : TargetVerifierMetadata(std::move(opts)) {}

  // Creates a ShapeVerifier that checks that shapes match inferred
  // expectations. This creates a new verifier every time because ShapeVerifier,
  // being a DfsHloVisitor, is stateful. We want a clean object for each run of
  // the verifier.
  std::unique_ptr<ShapeVerifier> GetVerifier() const override {
    return std::make_unique<ShapeVerifier>(GetVerifierOpts());
  }
};

// HLO pass that verifies invariants of HLO instructions for each computation in
// the module.
class HloVerifier : public HloModulePass {
 public:
  HloVerifier(
      bool layout_sensitive, bool allow_mixed_precision,
      HloPredicate instruction_can_change_layout_func = {},
      std::function<int64_t(const Shape&)> shape_size_func =
          [](const Shape& shape) { return ShapeUtil::ByteSizeOf(shape); })
      : HloVerifier(HloVerifierOpts{}
                        .WithLayoutSensitive(layout_sensitive)
                        .WithAllowMixedPrecision(allow_mixed_precision)
                        .WithInstructionCanChangeLayout(
                            instruction_can_change_layout_func)
                        .WithCustomShapeSize(shape_size_func)) {}

  explicit HloVerifier(HloVerifierOpts&& opts)
      : target_metadata_(
            std::make_unique<DefaultVerifierMetadata>(std::move(opts))),
        context_("Unknown") {}

  // Uses custom target metadata
  explicit HloVerifier(std::unique_ptr<TargetVerifierMetadata> target_metadata,
                       absl::string_view context = "Unknown")
      : target_metadata_(std::move(target_metadata)), context_(context) {}

  ~HloVerifier() override = default;
  absl::string_view name() const override { return "hlo-verifier"; }

  // Never returns true; no instructions are ever modified by this pass.
  using HloPassInterface::Run;
  using HloPassInterface::RunOnModuleGroup;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Owns verifier config.
  std::unique_ptr<TargetVerifierMetadata> target_metadata_;

  // The hlo pass when the verifier is invoked.
  std::string context_;
};

// Tracks debug metadata coverage on HLO Ops and reports the results as an INFO
// log starting with a `prefix` passed to the ctor.
// TODO(b/261216447): Remove once the work on debug metadata is finished.
class MetadataTracker : public DfsHloVisitorWithDefault {
 public:
  explicit MetadataTracker(absl::string_view prefix);
  ~MetadataTracker() override;
  Status DefaultAction(HloInstruction* instruction) override;
  void HandleMetadata(const OpMetadata& metadata);

 private:
  const std::string prefix_;
  int64_t instruction_count_ = 0;
  int64_t has_op_type_count_ = 0;
  int64_t has_op_name_count_ = 0;
  int64_t has_source_file_count_ = 0;
  int64_t has_source_line_count_ = 0;
  int64_t has_creation_pass_id_count_ = 0;
  int64_t has_logical_creation_pass_id_count_ = 0;
  int64_t has_size_of_generated_code_in_bytes_count_ = 0;
  int64_t has_size_of_memory_working_set_in_bytes_count_ = 0;
  int64_t has_profile_info_count_ = 0;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VERIFIER_H_
