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

#include <memory>
#include <string>

#include "absl/strings/str_format.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_NAMEANONYMOUSITERATORSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct NameAnonymousIteratorsPass
    : public impl::NameAnonymousIteratorsPassBase<NameAnonymousIteratorsPass> {
  void runOnOperation() override;
};

template <typename OP>
int replace(OP op, int count) {
  OpBuilder builder(op);
  std::string name = absl::StrFormat("_iterator%d", count++);

  auto new_op = builder.create<TF::IteratorOp>(
      op->getLoc(), op->getResultTypes()[0], name, /*container=*/"",
      op.getOutputTypes(), op.getOutputShapes());
  op->getResults()[0].replaceAllUsesWith(new_op->getResults()[0]);
  if (op->use_empty()) op->erase();
  return count;
}

void NameAnonymousIteratorsPass::runOnOperation() {
  int count = 1;
  getOperation().walk(
      [&](TF::AnonymousIteratorOp op) { count = replace(op, count); });
  getOperation().walk(
      [&](TF::AnonymousIteratorV2Op op) { count = replace(op, count); });
  getOperation().walk(
      [&](TF::AnonymousIteratorV3Op op) { count = replace(op, count); });
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateNameAnonymousIteratorsPass() {
  return std::make_unique<NameAnonymousIteratorsPass>();
}

}  // namespace TF
}  // namespace mlir
