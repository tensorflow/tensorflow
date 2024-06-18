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

#ifndef XLA_HLO_IR_DFS_HLO_VISITOR_WITH_DEFAULT_H_
#define XLA_HLO_IR_DFS_HLO_VISITOR_WITH_DEFAULT_H_

#include <memory>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "tsl/platform/status.h"

namespace xla {

// DfsHloVisitor with default action based on the HloInstruction being visited.
// Users should not use this class directly, but use the type aliases
// DfsHloVisitorWithDefault/ConstDfsHloVisitorWithDefault instead.
//
// Do *not* add an override to this class if the opcode is covered by
// HandleElementwiseUnary/Binary. These opcode handlers dispatch to
// HandleElementwiseUnary/Binary in DfsHloVisitorBase. Adding such a handler
// here will break passes which rely on the HandleElementwiseUnary/Binary
// handling these opcodes.
template <typename HloInstructionPtr>
class DfsHloVisitorWithDefaultBase
    : public DfsHloVisitorBase<HloInstructionPtr> {
 public:
  DfsHloVisitorWithDefaultBase() = default;
  ~DfsHloVisitorWithDefaultBase() override = default;

  // Default action performed on HloInstruction.
  virtual absl::Status DefaultAction(HloInstructionPtr hlo_instruction) = 0;

  absl::Status HandleElementwiseUnary(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleElementwiseBinary(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }

  absl::Status HandleBatchNormTraining(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }

  absl::Status HandleBatchNormInference(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }

  absl::Status HandleBatchNormGrad(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }

  absl::Status HandleClamp(HloInstructionPtr clamp) override {
    return DefaultAction(clamp);
  }
  absl::Status HandleConcatenate(HloInstructionPtr concatenate) override {
    return DefaultAction(concatenate);
  }
  absl::Status HandleSelect(HloInstructionPtr select) override {
    return DefaultAction(select);
  }
  absl::Status HandleDot(HloInstructionPtr dot) override {
    return DefaultAction(dot);
  }
  absl::Status HandleConvolution(HloInstructionPtr convolution) override {
    return DefaultAction(convolution);
  }
  absl::Status HandleFft(HloInstructionPtr fft) override {
    return DefaultAction(fft);
  }
  absl::Status HandleTriangularSolve(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleCholesky(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleOptimizationBarrier(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleAllGather(HloInstructionPtr crs) override {
    return DefaultAction(crs);
  }
  absl::Status HandleAllGatherStart(HloInstructionPtr crs) override {
    return DefaultAction(crs);
  }
  absl::Status HandleAllGatherDone(HloInstructionPtr crs) override {
    return DefaultAction(crs);
  }
  absl::Status HandleAllReduce(HloInstructionPtr crs) override {
    return DefaultAction(crs);
  }
  absl::Status HandleReduceScatter(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleAllReduceStart(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleAllReduceDone(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleAllToAll(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleCollectiveBroadcast(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleCollectivePermute(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleCollectivePermuteStart(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleCollectivePermuteDone(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleReplicaId(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandlePartitionId(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  absl::Status HandleRng(HloInstructionPtr random) override {
    return DefaultAction(random);
  }
  absl::Status HandleRngBitGenerator(HloInstructionPtr random) override {
    return DefaultAction(random);
  }
  absl::Status HandleRngGetAndUpdateState(HloInstructionPtr random) override {
    return DefaultAction(random);
  }
  absl::Status HandleInfeed(HloInstructionPtr infeed) override {
    return DefaultAction(infeed);
  }
  absl::Status HandleOutfeed(HloInstructionPtr outfeed) override {
    return DefaultAction(outfeed);
  }
  absl::Status HandleReverse(HloInstructionPtr reverse) override {
    return DefaultAction(reverse);
  }
  absl::Status HandleSort(HloInstructionPtr sort) override {
    return DefaultAction(sort);
  }
  absl::Status HandleConstant(HloInstructionPtr constant) override {
    return DefaultAction(constant);
  }
  absl::Status HandleIota(HloInstructionPtr iota) override {
    return DefaultAction(iota);
  }
  absl::Status HandleGetTupleElement(
      HloInstructionPtr get_tuple_element) override {
    return DefaultAction(get_tuple_element);
  }
  absl::Status HandleParameter(HloInstructionPtr parameter) override {
    return DefaultAction(parameter);
  }
  absl::Status HandleFusion(HloInstructionPtr fusion) override {
    return DefaultAction(fusion);
  }
  absl::Status HandleCall(HloInstructionPtr call) override {
    return DefaultAction(call);
  }
  absl::Status HandleCustomCall(HloInstructionPtr custom_call) override {
    return DefaultAction(custom_call);
  }
  absl::Status HandleSlice(HloInstructionPtr slice) override {
    return DefaultAction(slice);
  }
  absl::Status HandleDynamicSlice(HloInstructionPtr dynamic_slice) override {
    return DefaultAction(dynamic_slice);
  }
  absl::Status HandleDynamicUpdateSlice(
      HloInstructionPtr dynamic_update_slice) override {
    return DefaultAction(dynamic_update_slice);
  }
  absl::Status HandleTuple(HloInstructionPtr tuple) override {
    return DefaultAction(tuple);
  }
  absl::Status HandleMap(HloInstructionPtr map) override {
    return DefaultAction(map);
  }
  absl::Status HandleReduce(HloInstructionPtr reduce) override {
    return DefaultAction(reduce);
  }
  absl::Status HandleReduceWindow(HloInstructionPtr reduce_window) override {
    return DefaultAction(reduce_window);
  }
  absl::Status HandleSelectAndScatter(
      HloInstructionPtr select_and_scatter) override {
    return DefaultAction(select_and_scatter);
  }
  absl::Status HandleBitcast(HloInstructionPtr bitcast) override {
    return DefaultAction(bitcast);
  }
  absl::Status HandleBroadcast(HloInstructionPtr broadcast) override {
    return DefaultAction(broadcast);
  }
  absl::Status HandlePad(HloInstructionPtr pad) override {
    return DefaultAction(pad);
  }
  absl::Status HandleDynamicReshape(
      HloInstructionPtr dynamic_reshape) override {
    return DefaultAction(dynamic_reshape);
  }
  absl::Status HandleReshape(HloInstructionPtr reshape) override {
    return DefaultAction(reshape);
  }
  absl::Status HandleTranspose(HloInstructionPtr transpose) override {
    return DefaultAction(transpose);
  }
  absl::Status HandleWhile(HloInstructionPtr xla_while) override {
    return DefaultAction(xla_while);
  }
  absl::Status HandleConditional(HloInstructionPtr conditional) override {
    return DefaultAction(conditional);
  }
  absl::Status HandleAsyncStart(HloInstructionPtr async_start) override {
    return DefaultAction(async_start);
  }
  absl::Status HandleAsyncUpdate(HloInstructionPtr async_update) override {
    return DefaultAction(async_update);
  }
  absl::Status HandleAsyncDone(HloInstructionPtr async_done) override {
    return DefaultAction(async_done);
  }
  absl::Status HandleCopyStart(HloInstructionPtr copy_start) override {
    return DefaultAction(copy_start);
  }
  absl::Status HandleCopyDone(HloInstructionPtr copy_done) override {
    return DefaultAction(copy_done);
  }
  absl::Status HandleRecv(HloInstructionPtr recv) override {
    return DefaultAction(recv);
  }
  absl::Status HandleRecvDone(HloInstructionPtr recv_done) override {
    return DefaultAction(recv_done);
  }
  absl::Status HandleSend(HloInstructionPtr send) override {
    return DefaultAction(send);
  }
  absl::Status HandleTopK(HloInstructionPtr topk) override {
    return DefaultAction(topk);
  }
  absl::Status HandleSendDone(HloInstructionPtr send_done) override {
    return DefaultAction(send_done);
  }
  absl::Status HandleGather(HloInstructionPtr gather) override {
    return DefaultAction(gather);
  }
  absl::Status HandleScatter(HloInstructionPtr scatter) override {
    return DefaultAction(scatter);
  }
  absl::Status HandleAfterAll(HloInstructionPtr token) override {
    return DefaultAction(token);
  }
  absl::Status HandleGetDimensionSize(HloInstructionPtr get_size) override {
    return DefaultAction(get_size);
  }
  absl::Status HandleSetDimensionSize(HloInstructionPtr get_size) override {
    return DefaultAction(get_size);
  }
  absl::Status HandleAddDependency(HloInstructionPtr add_dependency) override {
    return DefaultAction(add_dependency);
  }

  // Invoked to inform the visitor that the traversal has completed, and that
  // the root was "root".
  absl::Status FinishVisit(HloInstructionPtr /*root*/) override {
    return absl::OkStatus();
  }

 private:
  DfsHloVisitorWithDefaultBase(const DfsHloVisitorWithDefaultBase&) = delete;
  DfsHloVisitorWithDefaultBase& operator=(const DfsHloVisitorWithDefaultBase&) =
      delete;
};

// Users should use one of these two type aliases, which are the only two valid
// instantiations of DfsHloVisitorWithDefaultBase.
using DfsHloVisitorWithDefault = DfsHloVisitorWithDefaultBase<HloInstruction*>;
using ConstDfsHloVisitorWithDefault =
    DfsHloVisitorWithDefaultBase<const HloInstruction*>;

// A common base class for visitors performing rewriting operation.
//
// Subclasses call ReplaceWithNewInstruction and ReplaceInstruction while
// visiting.
class DfsHloRewriteVisitor : public DfsHloVisitorWithDefault {
 public:
  // Runs a visitor on the module and returns whether the module has changed.
  absl::StatusOr<bool> RunOnModule(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {}) {
    absl::Status status;
    for (HloComputation* computation :
         module->MakeNonfusionComputations(execution_threads)) {
      status = computation->Accept(this);
      if (ABSL_PREDICT_FALSE(!status.ok())) return status;
    }
    return changed();
  }

  // Default visitor action is to do nothing and return OK.
  absl::Status DefaultAction(HloInstruction* /*hlo_instruction*/) override {
    return absl::OkStatus();
  }

  bool changed() const { return changed_; }

 protected:
  // Replaces the existing HLO instruction old_instruction, with
  // new_instruction, and marks the optimizer status as changed.
  // Returns the absl::Status representing the result of the replace operation.
  absl::Status ReplaceWithNewInstruction(
      HloInstruction* old_instruction,
      std::unique_ptr<HloInstruction> new_instruction) {
    VLOG(3) << "Replacing instruction:" << "\n  old: "
            << old_instruction->ToString()
            << "\n  new: " << new_instruction->ToString();
    absl::Status status = old_instruction->parent()->ReplaceWithNewInstruction(
        old_instruction, std::move(new_instruction));
    if (ABSL_PREDICT_TRUE(status.ok())) {
      changed_ = true;
    }
    return status;
  }

  // Replaces the existing HLO instruction old_instruction, with
  // new_instruction, and marks the optimizer status as changed.
  // Returns the absl::Status representing the result of the replace operation.
  absl::StatusOr<bool> ReplaceInstruction(HloInstruction* old_instruction,
                                          HloInstruction* new_instruction,
                                          bool preserve_sharding) {
    VLOG(3) << "Replacing instruction:" << "\n  old: "
            << old_instruction->ToString()
            << "\n  new: " << new_instruction->ToString();
    absl::StatusOr<bool> changed_or =
        old_instruction->parent()->ReplaceInstruction(
            old_instruction, new_instruction, preserve_sharding);
    if (ABSL_PREDICT_TRUE(changed_or.ok())) {
      changed_ |= changed_or.value();
    }
    return changed_or;
  }

  absl::Status ReplaceInstruction(HloInstruction* old_instruction,
                                  HloInstruction* new_instruction) {
    absl::StatusOr<bool> changed_or =
        ReplaceInstruction(old_instruction, new_instruction,
                           /*preserve_sharding=*/false);
    if (ABSL_PREDICT_TRUE(changed_or.ok())) {
      DCHECK(changed_or.value());
    }
    return changed_or.status();
  }

  // Mark the computation as having changed.
  void MarkAsChanged() { changed_ = true; }

 private:
  bool changed_ = false;
};

// (Const)FunctionVisitor lets you transform an
// std::function<absl::Status((const) HloInstruction*)> into a
// (Const)DfsHloVisitor.
//
// This is useful if you have code that needs to handle visitors in the form of
// both std::function and DfsHloVisitor.  You can wrap the function in a
// FunctionVisitor and then treat it like any other DfsHloVisitor.
template <typename HloInstructionPtr>
class FunctionVisitorBase
    : public DfsHloVisitorWithDefaultBase<HloInstructionPtr> {
 public:
  explicit FunctionVisitorBase(
      std::function<absl::Status(HloInstructionPtr)> visitor_func)
      : visitor_func_(std::move(visitor_func)) {}

  absl::Status DefaultAction(HloInstructionPtr hlo_instruction) override {
    return visitor_func_(hlo_instruction);
  }

 private:
  FunctionVisitorBase(const FunctionVisitorBase&) = delete;
  FunctionVisitorBase& operator=(const FunctionVisitorBase&) = delete;

  std::function<absl::Status(HloInstructionPtr)> visitor_func_;
};

using FunctionVisitor = FunctionVisitorBase<HloInstruction*>;
using ConstFunctionVisitor = FunctionVisitorBase<const HloInstruction*>;

}  // namespace xla

#endif  // XLA_HLO_IR_DFS_HLO_VISITOR_WITH_DEFAULT_H_
