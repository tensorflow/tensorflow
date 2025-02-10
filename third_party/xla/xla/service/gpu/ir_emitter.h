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

#ifndef XLA_SERVICE_GPU_IR_EMITTER_H_
#define XLA_SERVICE_GPU_IR_EMITTER_H_

#include <vector>

#include "absl/status/status.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/AtomicOrdering.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/hlo_to_ir_bindings.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/ir_builder_mixin.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

// Abstract base class for translating HLO graphs to LLVM IR for a GPU.
//
// There are two concrete subclasses of IrEmitter: IrEmitterNested and
// IrEmitterUnnested.  In the unnested variety, each HLO gets its own kernel
// function, whereas in the nested version the whole computation is emitted as
// one *non-kernel* function.
//
// In XLA, kernel functions never call other kernel functions.  This means that
// if we have a kernel -- e.g. implementing a kReduce HLO -- that wants to use
// an HLO computation as a "subroutine" -- e.g. the HLO computation that
// specifies how to reduce two elements -- then the subroutine computation must
// be emitted using IrEmitterNested.
//
// Fusion nodes are a special case.  A fusion node is emitted using
// IrEmitterUnnested, but the code is generated using FusedIrEmitter, which is
// not a subclass of gpu::IrEmitter, and in fact is better understood as an IR
// generator generator.  See comments on that class.
class IrEmitter : public DfsHloVisitorWithDefault,
                  public IrBuilderMixin<IrEmitter> {
 public:
  IrEmitter(const IrEmitter&) = delete;
  IrEmitter& operator=(const IrEmitter&) = delete;

  absl::Status DefaultAction(HloInstruction* hlo) override;
  absl::Status HandleConstant(HloInstruction* constant) override;
  absl::Status HandleGetTupleElement(
      HloInstruction* get_tuple_element) override;
  absl::Status HandleConvolution(HloInstruction* convolution) override;
  absl::Status HandleFft(HloInstruction* fft) override;
  absl::Status HandleAllReduce(HloInstruction* crs) override;
  absl::Status HandleInfeed(HloInstruction* infeed) override;
  absl::Status HandleOutfeed(HloInstruction* outfeed) override;
  absl::Status HandleSend(HloInstruction* send) override;
  absl::Status HandleSendDone(HloInstruction* send_done) override;
  absl::Status HandleRecv(HloInstruction* recv) override;
  absl::Status HandleRecvDone(HloInstruction* recv_done) override;
  absl::Status HandleParameter(HloInstruction* parameter) override;
  absl::Status HandleTuple(HloInstruction* tuple) override;
  absl::Status HandleScatter(HloInstruction* scatter) override;
  absl::Status HandleCall(HloInstruction* call) override;
  absl::Status HandleCustomCall(HloInstruction* custom_call) override;
  absl::Status HandleBatchNormInference(HloInstruction* batch_norm) override;
  absl::Status HandleBatchNormTraining(HloInstruction* batch_norm) override;
  absl::Status HandleBatchNormGrad(HloInstruction* batch_norm) override;
  absl::Status HandleAddDependency(HloInstruction* add_dependency) override;

  absl::Status FinishVisit(HloInstruction* root) override {
    return absl::OkStatus();
  }

  llvm::IRBuilderBase* builder() { return &b_; }

 protected:
  // Constructs an IrEmitter with the given IrEmitter context.
  // ir_emitter_context is owned by the caller and should outlive the IrEmitter
  // object.
  explicit IrEmitter(IrEmitterContext* ir_emitter_context, bool is_nested);

  // Helper for calling HloToIrBindings::GetIrArray.
  //
  // Gets the IrArray which contains inst.  This array has metadata that makes
  // it valid only within the IR that implements consumer.  If you are
  // implementing an HLO and want to get its own output buffer, call
  // GetIrArray(hlo, hlo).
  llvm_ir::IrArray GetIrArray(const HloInstruction& inst,
                              const HloInstruction& consumer,
                              const ShapeIndex& shape_index = {}) {
    return bindings_.GetIrArray(inst, consumer, shape_index);
  }
  // A convenient helper for calling HloToIrBindings::GetBasePointer.
  llvm::Value* GetBasePointer(const HloInstruction& inst,
                              ShapeIndexView shape_index = {}) const {
    return bindings_.GetBasePointer(inst, shape_index);
  }

  // Generates the IrArray for each output of an hlo instruction and returns
  // a vector containing such IrArrays.
  std::vector<llvm_ir::IrArray> ConstructIrArrayForOutputs(
      const HloInstruction& hlo);

  // Emit a single-threaded or multi-threaded loop that computes every element
  // in the result of the given HLO instruction. This produces a series of
  // nested loops (e.g. one for each dimension of the `hlo`'s shape). The body
  // of the inner-most loop is provided by the body_emitter function.
  virtual absl::Status EmitTargetElementLoop(
      const HloInstruction& hlo,
      const llvm_ir::ElementGenerator& body_emitter) = 0;

  IrEmitterContext* ir_emitter_context_;
  llvm::Module* module_;

  // The following fields track the IR emission state. According to LLVM memory
  // management rules, their memory is owned by the module.
  llvm::IRBuilder<> b_;

  // Mapping from HLO to its underlying LLVM value.
  HloToIrBindings bindings_;

  // Bind all argument IrArrays of `fusion` to `fused_emitter`.
  void BindFusionArguments(const HloInstruction* fusion,
                           FusedIrEmitter* fused_emitter);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_IR_EMITTER_H_
