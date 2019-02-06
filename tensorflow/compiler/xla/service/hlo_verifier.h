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

#include <memory>
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"

namespace xla {

// Visitor which verifies that the output shape is correctly set. Verifies
// against the inferred shape for the instruction.
// TODO(b/26024837): Check output shape for all instruction types.
class ShapeVerifier : public DfsHloVisitor {
 public:
  ShapeVerifier(bool layout_sensitive, bool allow_mixed_precision)
      : layout_sensitive_(layout_sensitive),
        allow_mixed_precision_(allow_mixed_precision) {}

  // Verifies that entry computation layout matches parameters and root shape of
  // the module's entry computation.
  virtual Status VerifyEntryComputationLayout(const HloModule& module);

  Status Preprocess(HloInstruction* hlo) override;

  Status HandleElementwiseUnary(HloInstruction* hlo) override;
  Status HandleElementwiseBinary(HloInstruction* hlo) override;
  Status HandleClamp(HloInstruction* clamp) override;
  Status HandleSelect(HloInstruction* select) override;
  Status HandleTupleSelect(HloInstruction* tuple_select) override;
  Status HandleConcatenate(HloInstruction* concatenate) override;
  Status HandleIota(HloInstruction* iota) override;
  Status HandleConvert(HloInstruction* convert) override;
  Status HandleBitcastConvert(HloInstruction* convert) override;
  Status HandleCopy(HloInstruction* copy) override;
  Status HandleDot(HloInstruction* dot) override;
  Status HandleConvolution(HloInstruction* convolution) override;
  Status HandleFft(HloInstruction* fft) override;
  Status HandleAllReduce(HloInstruction* crs) override;
  Status HandleAllToAll(HloInstruction* hlo) override;
  Status HandleCollectivePermute(HloInstruction* hlo) override;
  Status HandleReplicaId(HloInstruction* hlo) override;
  Status HandleReducePrecision(HloInstruction* reduce_precision) override;
  Status HandleInfeed(HloInstruction*) override;
  Status HandleOutfeed(HloInstruction*) override;
  Status HandleRng(HloInstruction*) override;
  Status HandleReverse(HloInstruction* reverse) override;
  Status HandleSort(HloInstruction* sort) override;
  Status HandleConstant(HloInstruction* constant) override;
  Status HandleGetTupleElement(HloInstruction* get_tuple_element) override;
  Status HandleReduce(HloInstruction* reduce) override;
  Status HandleBitcast(HloInstruction* bitcast) override;
  Status HandleBroadcast(HloInstruction* broadcast) override;
  Status HandleReshape(HloInstruction* reshape) override;
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
  Status HandleReduceWindow(HloInstruction* reduce_window) override;
  Status HandleSelectAndScatter(HloInstruction* instruction) override;
  Status HandleWhile(HloInstruction* xla_while) override;
  Status HandleConditional(HloInstruction* conditional) override;
  Status HandlePad(HloInstruction* pad) override;
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
  Status HandleAddDependency(HloInstruction* add_dependency) override;

  Status FinishVisit(HloInstruction*) override { return Status::OK(); }

 protected:
  // Check the instruction's shape against the shape given by ShapeInference
  // and return an appropriate error if there is a mismatch.
  Status CheckShape(const HloInstruction* instruction,
                    const Shape& inferred_shape);

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
  bool ShapesSame(const Shape& a, const Shape& b) {
    return layout_sensitive_ ? ShapeUtil::Equal(a, b)
                             : ShapeUtil::Compatible(a, b);
  }
  bool ShapesSameIgnoringFpPrecision(const Shape& a, const Shape& b) {
    return layout_sensitive_ ? ShapeUtil::EqualIgnoringFpPrecision(a, b)
                             : ShapeUtil::CompatibleIgnoringFpPrecision(a, b);
  }
  string StringifyShape(const Shape& s) {
    return layout_sensitive_ ? ShapeUtil::HumanStringWithLayout(s)
                             : ShapeUtil::HumanString(s);
  }

  // Helpers that switch on allow_mixed_precision_.
  bool SameElementType(const Shape& a, const Shape& b) {
    return allow_mixed_precision_
               ? ShapeUtil::SameElementTypeIgnoringFpPrecision(a, b)
               : ShapeUtil::SameElementType(a, b);
  }

  // Checks that the given operand of the given instruction is of type TOKEN.
  Status CheckIsTokenOperand(const HloInstruction* instruction,
                             int64 operand_no);

  // Checks that the shape of the given operand of the given instruction matches
  // the given parameter of the given computation.
  Status CheckOperandAndParameter(const HloInstruction* instruction,
                                  int64 operand_number,
                                  const HloComputation* computation,
                                  int64 parameter_number);

  // Returns true if the shapes of the two operands have the same element type,
  // and the result shape either has the same element type as the operand shapes
  // or mixed precision is allowed and the result shape and the operand shapes
  // have floating point element types.
  bool HasCompatibleElementTypes(const Shape& shape_0, const Shape& shape_1,
                                 const Shape& result_shape);

  // If the verifier is layout-sensitive, shapes must be equal to what's
  // expected.  Otherwise, the shapes must simply be compatible.
  bool layout_sensitive_;

  // Whether the inputs and output of an instruction can contain both F32s and
  // BF16s. Tuples that include both F32s and BF16s are allowed regardless of
  // this flag.
  bool allow_mixed_precision_;
};

// An interface used to encapsulate target-specific verification quirks.
class TargetVerifierMetadata {
 public:
  TargetVerifierMetadata(std::function<int64(const Shape&)> shape_size_function)
      : shape_size_function_(shape_size_function) {}

  // Returns a target-specific shape size.
  int64 ShapeSize(const Shape& shape) const {
    return shape_size_function_(shape);
  }

  virtual std::unique_ptr<ShapeVerifier> GetVerifier() const = 0;

  TargetVerifierMetadata() {}
  virtual ~TargetVerifierMetadata() {}

  TargetVerifierMetadata(const TargetVerifierMetadata&) = delete;
  TargetVerifierMetadata& operator=(const TargetVerifierMetadata&) = delete;

 private:
  // Returns a target-specific shape size.
  std::function<int64(const Shape&)> shape_size_function_;
};

// The default implementation of TargetVerifierMetadata, used unless the target
// needs to override it.
class DefaultVerifierMetadata : public TargetVerifierMetadata {
 public:
  DefaultVerifierMetadata(
      bool layout_sensitive, bool allow_mixed_precision,
      std::function<int64(const Shape&)> shape_size_function)
      : TargetVerifierMetadata(shape_size_function),
        layout_sensitive_(layout_sensitive),
        allow_mixed_precision_(allow_mixed_precision) {}

  // Creates a ShapeVerifier that checks that shapes match inferred
  // expectations. This creates a new verifier every time because ShapeVerifier,
  // being a DfsHloVisitor, is stateful. We want a clean object for each run of
  // the verifier.
  std::unique_ptr<ShapeVerifier> GetVerifier() const override {
    return absl::make_unique<ShapeVerifier>(layout_sensitive_,
                                            allow_mixed_precision_);
  }

 private:
  bool layout_sensitive_;
  bool allow_mixed_precision_;
};

// HLO pass that verifies invariants of HLO instructions for each computation in
// the module.
class HloVerifier : public HloModulePass {
 public:
  explicit HloVerifier(
      bool layout_sensitive, bool allow_mixed_precision,
      std::function<bool(const HloInstruction*)>
          instruction_can_change_layout_func = {},
      std::function<int64(const Shape&)> shape_size_func =
          [](const Shape& shape) { return ShapeUtil::ByteSizeOf(shape); })
      : target_metadata_(absl::make_unique<DefaultVerifierMetadata>(
            layout_sensitive, allow_mixed_precision, shape_size_func)),
        instruction_can_change_layout_func_(
            std::move(instruction_can_change_layout_func)) {
    CHECK(instruction_can_change_layout_func_ == nullptr || layout_sensitive);
  }

  // Uses custom target metadata
  explicit HloVerifier(std::unique_ptr<TargetVerifierMetadata> target_metadata)
      : target_metadata_(std::move(target_metadata)) {}

  ~HloVerifier() override = default;
  absl::string_view name() const override { return "verifier"; }

  // Never returns true; no instructions are ever modified by this pass.
  StatusOr<bool> Run(HloModule* module) override;

 private:
  std::unique_ptr<TargetVerifierMetadata> target_metadata_;

  // Determines whether an instruction can change layouts.
  std::function<bool(const HloInstruction*)>
      instruction_can_change_layout_func_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_VERIFIER_H_
