/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for fusing linalg ops obtained after LHLO
// lowering.

#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "absl/memory/memory.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir

namespace mlir {
namespace xla_lhlo {
namespace {

using linalg::LinalgOp;

struct LhloFuseLinalg : public FunctionPass<LhloFuseLinalg> {
  void runOnFunction() override {
    auto func = getFunction();

    // TODO(pifon): Remove assumption that the function has a single block.
    if (func.getBlocks().size() != 1) {
      emitError(func.getLoc(), "The function needs to have a single block.");
      signalPassFailure();
      return;
    }

    // The fusion in Linalg is currently possible only when the consumer op is
    // tiled. In order to greedily fuse the ops, we have to start from the tiled
    // root linalg ops, i.e. linalg ops that write to output buffers of the
    // function.
    llvm::SmallDenseSet<Value*> func_args;
    for (auto func_arg : func.getArguments()) {
      func_args.insert(func_arg);
    }
    OpBuilder b(func);
    OperationFolder folder(func.getContext());
    func.walk([&](linalg::GenericOp generic_op) {
      const SmallVector<int64_t, 2> tile_sizes(
          generic_op.getNumInputsAndOutputs(), 1);
      auto op = cast<LinalgOp>(generic_op.getOperation());
      for (const Value* result : op.getOutputs()) {
        if (!func_args.count(result)) continue;
        if (linalg::tileLinalgOp(b, op, tile_sizes, /*permutation=*/{},
                                 &folder)) {
          generic_op.erase();
          return;
        }
      }
    });

    // Fuse producers of tiled linalg ops.
    llvm::SmallDenseSet<Operation*> erase_set;
    SmallVector<Operation*, 8> linalg_ops;
    func.walk([&](LinalgOp op) { linalg_ops.push_back(op); });
    linalg::Aliases aliases;
    linalg::LinalgDependenceGraph graph(aliases, linalg_ops);
    for (auto* op : llvm::reverse(linalg_ops)) {
      for (unsigned id = 0, e = LinalgOp(op).getNumInputs(); id < e; ++id) {
        if (auto info = fuseProducerOf(b, op, id, graph, &folder)) {
          erase_set.insert(info->originalProducer.getOperation());
        }
      }
    }
    for (auto* e : erase_set) e->erase();
  }
};

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> createLhloFuseLinalg() {
  return absl::make_unique<LhloFuseLinalg>();
}

static PassRegistration<LhloFuseLinalg> legalize_pass(
    "lhlo-fuse-linalg",
    "Greedily fuse linalg ops obtained after LHLO lowering.");

}  // namespace xla_lhlo
}  // namespace mlir
