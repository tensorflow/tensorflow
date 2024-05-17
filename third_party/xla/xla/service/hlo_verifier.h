/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_HLO_VERIFIER_H_
#define XLA_SERVICE_HLO_VERIFIER_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/service/hlo_pass_interface.h"

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

  HloVerifierOpts&& WithVerifyS4U4Usage(bool verify) {
    return std::move(*this);
  }

  HloVerifierOpts&& WithAllowUnboundedDynamism(bool allow) {
    allow_unbounded_dynamism = allow;
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

  // Whether unbounded dynamic sizes should be allowed for shapes.
  bool allow_unbounded_dynamism = false;

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
  virtual absl::Status VerifyEntryComputationLayout(const HloModule& module);

  absl::Status Preprocess(HloInstruction* hlo) override;

  absl::Status HandleElementwiseUnary(HloInstruction* hlo) override;
  absl::Status HandleElementwiseBinary(HloInstruction* hlo) override;
  absl::Status HandleClamp(HloInstruction* clamp) override;
  absl::Status HandleSelect(HloInstruction* select) override;
  absl::Status HandleConcatenate(HloInstruction* concatenate) override;
  absl::Status HandleIota(HloInstruction* hlo) override;
  absl::Status HandleConvert(HloInstruction* convert) override;
  absl::Status HandleBitcastConvert(HloInstruction* convert) override;
  absl::Status HandleStochasticConvert(HloInstruction* convert) override;
  absl::Status HandleCopy(HloInstruction* copy) override;
  absl::Status HandleDot(HloInstruction* dot) override;
  absl::Status HandleConvolution(HloInstruction* convolution) override;
  absl::Status HandleFft(HloInstruction* fft) override;
  absl::Status HandleCholesky(HloInstruction* hlo) override;
  absl::Status HandleTriangularSolve(HloInstruction* hlo) override;
  absl::Status HandleAllGather(HloInstruction* hlo) override;
  absl::Status HandleAllGatherStart(HloInstruction* hlo) override;
  absl::Status HandleAllGatherDone(HloInstruction* hlo) override;
  absl::Status HandleAllReduce(HloInstruction* hlo) override;
  absl::Status HandleAllReduceStart(HloInstruction* hlo) override;
  absl::Status HandleAllReduceDone(HloInstruction* hlo) override;
  absl::Status HandleAllToAll(HloInstruction* hlo) override;
  absl::Status HandleCollectiveBroadcast(HloInstruction* hlo) override;
  absl::Status HandleCollectivePermute(HloInstruction* hlo) override;
  absl::Status HandleCollectivePermuteStart(HloInstruction* hlo) override;
  absl::Status HandleCollectivePermuteDone(HloInstruction* hlo) override;
  absl::Status HandlePartitionId(HloInstruction* hlo) override;
  absl::Status HandleReplicaId(HloInstruction* hlo) override;
  absl::Status HandleReducePrecision(HloInstruction* reduce_precision) override;
  absl::Status HandleInfeed(HloInstruction*) override;
  absl::Status HandleOptimizationBarrier(HloInstruction* hlo) override;
  absl::Status HandleOutfeed(HloInstruction*) override;
  absl::Status HandleRng(HloInstruction*) override;
  absl::Status HandleRngBitGenerator(HloInstruction*) override;
  absl::Status HandleRngGetAndUpdateState(HloInstruction*) override;
  absl::Status HandleReverse(HloInstruction* reverse) override;
  absl::Status HandleSort(HloInstruction* hlo) override;
  absl::Status HandleTopK(HloInstruction* hlo) override;
  absl::Status HandleConstant(HloInstruction* constant) override;
  absl::Status HandleGetTupleElement(
      HloInstruction* get_tuple_element) override;
  absl::Status HandleReduce(HloInstruction* reduce) override;
  absl::Status HandleBitcast(HloInstruction* bitcast) override;
  absl::Status HandleBroadcast(HloInstruction* broadcast) override;
  absl::Status HandleReshape(HloInstruction* reshape) override;
  absl::Status HandleDynamicReshape(HloInstruction* dynamic_reshape) override;
  absl::Status HandleTranspose(HloInstruction* transpose) override;
  absl::Status HandleParameter(HloInstruction*) override;
  absl::Status HandleFusion(HloInstruction*) override;
  absl::Status HandleCall(HloInstruction* call) override;
  absl::Status HandleCustomCall(HloInstruction*) override;
  absl::Status HandleSlice(HloInstruction* slice) override;
  absl::Status HandleDynamicSlice(HloInstruction* dynamic_slice) override;
  absl::Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override;
  absl::Status HandleTuple(HloInstruction* tuple) override;
  absl::Status HandleMap(HloInstruction* map) override;
  absl::Status HandleReduceScatter(HloInstruction* hlo) override;
  absl::Status HandleReduceWindow(HloInstruction* reduce_window) override;
  absl::Status HandleSelectAndScatter(HloInstruction* instruction) override;
  absl::Status HandleWhile(HloInstruction* xla_while) override;
  absl::Status HandleConditional(HloInstruction* conditional) override;
  absl::Status HandlePad(HloInstruction* pad) override;
  absl::Status HandleAsyncStart(HloInstruction* async_start) override;
  absl::Status HandleAsyncUpdate(HloInstruction* async_update) override;
  absl::Status HandleAsyncDone(HloInstruction* async_done) override;
  absl::Status HandleCopyStart(HloInstruction* copy_start) override;
  absl::Status HandleCopyDone(HloInstruction* copy_done) override;
  absl::Status HandleSend(HloInstruction* send) override;
  absl::Status HandleSendDone(HloInstruction* send_done) override;
  absl::Status HandleRecv(HloInstruction* recv) override;
  absl::Status HandleRecvDone(HloInstruction* recv_done) override;
  absl::Status HandleBatchNormTraining(
      HloInstruction* batch_norm_training) override;
  absl::Status HandleBatchNormInference(
      HloInstruction* batch_norm_inference) override;
  absl::Status HandleBatchNormGrad(HloInstruction* batch_norm_grad) override;
  absl::Status HandleGather(HloInstruction* gather) override;
  absl::Status HandleScatter(HloInstruction* scatter) override;
  absl::Status HandleAfterAll(HloInstruction* token) override;
  absl::Status HandleGetDimensionSize(HloInstruction* get_size) override;
  absl::Status HandleSetDimensionSize(HloInstruction* set_size) override;
  absl::Status HandleAddDependency(HloInstruction* add_dependency) override;

  absl::Status FinishVisit(HloInstruction*) override { return OkStatus(); }

 protected:
  // Helpers that switch on layout_sensitive_.
  bool ShapesSame(const Shape& a, const Shape& b, Shape::Equal equal = {});

  // Check the instruction's shape against the shape given by ShapeInference
  // and return an appropriate error if there is a mismatch.
  absl::Status CheckShape(const HloInstruction* instruction,
                          const Shape& inferred_shape,
                          bool only_compare_minor_to_major_in_layout = false);

  // Overload which takes a absl::StatusOr to reduce boilerplate in the caller.
  absl::Status CheckShape(const HloInstruction* instruction,
                          const absl::StatusOr<Shape>& inferred_shape_status);

  static absl::Status CheckParameterCount(
      const HloInstruction* calling_instruction,
      const HloComputation* computation, int expected);

  // Check a unary (binary, etc) instruction's shape against the inferred shape.
  absl::Status CheckUnaryShape(const HloInstruction* instruction);
  absl::Status CheckBinaryShape(const HloInstruction* instruction);
  absl::Status CheckTernaryShape(const HloInstruction* instruction);
  absl::Status CheckVariadicShape(const HloInstruction* instruction);

 private:
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
  absl::Status CheckIsTokenOperand(const HloInstruction* instruction,
                                   int64_t operand_no);

  // Checks that the shape of the given operand of the given instruction matches
  // the given parameter of the given computation.
  absl::Status CheckOperandAndParameter(const HloInstruction* instruction,
                                        int64_t operand_number,
                                        const HloComputation* computation,
                                        int64_t parameter_number);

  // Checks that the shape of async op operands and results match the called
  // computation parameters and root.
  absl::Status CheckAsyncOpComputationShapes(const HloInstruction* async_op,
                                             const Shape& async_shape);

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
  absl::StatusOr<bool> Run(
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
  absl::Status DefaultAction(HloInstruction* instruction) override;
  void HandleMetadata(const OpMetadata& metadata);

 private:
  const std::string prefix_;
  int64_t instruction_count_ = 0;
  int64_t has_op_type_count_ = 0;
  int64_t has_op_name_count_ = 0;
  int64_t has_source_file_count_ = 0;
  int64_t has_dummy_source_file_count_ = 0;
  int64_t has_source_line_count_ = 0;
  int64_t has_creation_pass_id_count_ = 0;
  int64_t has_logical_creation_pass_id_count_ = 0;
  int64_t has_size_of_generated_code_in_bytes_count_ = 0;
  int64_t has_size_of_memory_working_set_in_bytes_count_ = 0;
  int64_t has_profile_info_count_ = 0;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_VERIFIER_H_
