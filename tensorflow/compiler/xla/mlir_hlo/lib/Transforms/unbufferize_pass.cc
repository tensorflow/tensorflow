/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This files implements a pass that changes a partially bufferized function
// back to tensor arguments and return values.

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Value.h"

namespace mlir {

#define GEN_PASS_DEF_UNBUFFERIZEPASS
#include "mlir-hlo/Transforms/passes.h.inc"

using ::mlir::func::FuncOp;

namespace {
class UnbufferizePass : public impl::UnbufferizePassBase<UnbufferizePass> {
 public:
  using UnbufferizePassBase<UnbufferizePass>::UnbufferizePassBase;

 private:
  void runOnOperation() override;
};
}  // namespace

void UnbufferizePass::runOnOperation() {
  FuncOp funcOp = getOperation();
  IRRewriter rewriter(funcOp.getContext());
  BitVector argsToErase(funcOp.getNumArguments());
  BlockAndValueMapping mapping;
  llvm::SmallDenseSet<BlockArgument> insertedArgs;
  funcOp->walk([&](bufferization::ToTensorOp op) {
    auto arg = op.getMemref().dyn_cast<BlockArgument>();
    if (!arg) return;
    Value newValue = mapping.lookupOrNull(arg);
    if (newValue == nullptr) {
      auto attrs = funcOp.getArgAttrDict(arg.getArgNumber());
      funcOp.insertArgument(funcOp.getNumArguments(), op.getType(), attrs,
                            arg.getLoc());
      newValue = funcOp.getArguments().back();
      mapping.map(arg, newValue);
    }
    rewriter.replaceOp(op, newValue);
    argsToErase.set(arg.getArgNumber());
  });
  SmallVector<Value> results;
  SmallVector<DictionaryAttr> resultAttrs;
  funcOp->walk([&](memref::TensorStoreOp op) {
    auto arg = op.getMemref().dyn_cast<BlockArgument>();
    if (!arg) return;
    argsToErase.set(arg.getArgNumber());
    results.push_back(op.getTensor());
    resultAttrs.push_back(funcOp.getArgAttrDict(arg.getArgNumber()));
    rewriter.eraseOp(op);
  });
  argsToErase.resize(funcOp.getNumArguments());
  funcOp.eraseArguments(argsToErase);
  auto resultIndices = llvm::to_vector(llvm::seq<unsigned>(0, results.size()));
  funcOp.insertResults(resultIndices, TypeRange(ValueRange(results)),
                       resultAttrs);
  Operation *terminator = funcOp.getBody().back().getTerminator();
  rewriter.setInsertionPoint(terminator);
  rewriter.replaceOpWithNewOp<func::ReturnOp>(terminator, results);
}

std::unique_ptr<OperationPass<func::FuncOp>> hlo::createUnbufferizePass() {
  return std::make_unique<UnbufferizePass>();
}

}  // namespace mlir
