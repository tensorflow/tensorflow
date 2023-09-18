/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <string>

#include "deallocation/transforms/analysis.h"
#include "deallocation/transforms/passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace deallocation {
namespace {

#define GEN_PASS_DEF_ANNOTATEDEALLOCATIONPASS
#include "deallocation/transforms/passes.h.inc"

std::string getDebugString(AsmState& state, DeallocationAnalysis& analysis,
                           Value value) {
  std::string out;
  llvm::raw_string_ostream os(out);
  llvm::interleaveComma(analysis.getBackingMemory(value), os,
                        [&](Value v) { v.printAsOperand(os, state); });
  return out;
}

Attribute getDebugAttribute(AsmState& state, DeallocationAnalysis& analysis,
                            Region& region) {
  mlir::OpBuilder b(region.getContext());
  return b.getArrayAttr(llvm::to_vector(
      llvm::map_range(region.getArguments(), [&](Value arg) -> Attribute {
        return b.getStringAttr(getDebugString(state, analysis, arg));
      })));
}

struct AnnotatePass : public impl::AnnotateDeallocationPassBase<AnnotatePass> {
  void runOnOperation() override {
    DeallocationAnalysis analysis;
    AsmState state(getOperation());
    mlir::OpBuilder b(getOperation());
    getOperation().walk([&](Operation* op) {
      std::string out;
      llvm::raw_string_ostream os(out);
      if (op->getNumRegions() > 0) {
        op->setAttr("deallocation.region_args_backing_memory",
                    b.getArrayAttr(llvm::to_vector(
                        llvm::map_range(op->getRegions(), [&](Region& region) {
                          return getDebugAttribute(state, analysis, region);
                        }))));
      }

      if (op->getNumResults() > 0) {
        op->setAttr("deallocation.result_backing_memory",
                    b.getArrayAttr(llvm::to_vector(llvm::map_range(
                        op->getResults(), [&](Value result) -> Attribute {
                          return b.getStringAttr(
                              getDebugString(state, analysis, result));
                        }))));
      }
    });
  }
};

}  // namespace

// Pass to annotate ops with debug information.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createDeallocationAnnotationPass() {
  return std::make_unique<AnnotatePass>();
}

}  // namespace deallocation
}  // namespace mlir
