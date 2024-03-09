/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_TF_DATAFLOW_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_TF_DATAFLOW_H_

#include "llvm/ADT/STLExtras.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TF {

template <typename L>
class TensorflowDataflowAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<dataflow::Lattice<L>> {
 public:
  using StateT = dataflow::Lattice<L>;
  using dataflow::SparseForwardDataFlowAnalysis<
      StateT>::SparseForwardDataFlowAnalysis;
  using dataflow::SparseForwardDataFlowAnalysis<StateT>::getLatticeElement;
  ~TensorflowDataflowAnalysis() override = default;

  bool ForwardThroughTFOperation(Operation *op,
                                 ArrayRef<const StateT *> operands,
                                 ArrayRef<StateT *> results) {
    if (auto cast = dyn_cast<TF::CastOp>(op)) {
      this->join(results[0], *operands[0]);
    } else if (auto while_op = dyn_cast<TF::WhileRegionOp>(op)) {
      for (auto &region : while_op->getRegions()) {
        for (auto [arg, value] :
             llvm::zip(region.getArguments(), while_op->getOperands())) {
          this->join(getLatticeElement(arg), *getLatticeElement(value));
        }
      }
    } else if (auto while_op = dyn_cast<TF::WhileOp>(op)) {
      func::FuncOp cond = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          while_op, while_op.getCondAttr());
      func::FuncOp body = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          while_op, while_op.getBodyAttr());
      for (auto &arg : while_op->getOpOperands()) {
        BlockArgument cond_arg = cond.getArgument(arg.getOperandNumber());
        this->join(getLatticeElement(cond_arg), *getLatticeElement(arg.get()));
        BlockArgument body_arg = body.getArgument(arg.getOperandNumber());
        this->join(getLatticeElement(body_arg), *getLatticeElement(arg.get()));
      }
    } else if (auto graph = dyn_cast<tf_executor::GraphOp>(op)) {
      for (auto &arg : graph.GetFetch()->getOpOperands()) {
        if (arg.getOperandNumber() < graph.getNumResults()) {
          auto result = graph.getResult(arg.getOperandNumber());
          this->join(getLatticeElement(result), *getLatticeElement(arg.get()));
        }
      }
    } else if (auto island = dyn_cast<tf_executor::IslandOp>(op)) {
      for (auto &arg : island.GetYield()->getOpOperands()) {
        auto result = island.getResult(arg.getOperandNumber());
        this->join(getLatticeElement(result), *getLatticeElement(arg.get()));
      }
    } else {
      return false;
    }
    return true;
  }

  void setToEntryState(StateT *lattice) override {
    this->propagateIfChanged(lattice,
                             lattice->join(L::EntryState(lattice->getPoint())));
  }
};

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_ANALYSIS_TF_DATAFLOW_H_
