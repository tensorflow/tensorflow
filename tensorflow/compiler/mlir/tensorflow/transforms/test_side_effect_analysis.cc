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

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

namespace mlir {
namespace tf_executor {

namespace {

// A pass that adds "Predecessors" and "Successors" remarks for each op based on
// SideEffectAnalysis result. For testing purpose only.
struct TestSideEffectAnalysis
    : public mlir::PassWrapper<TestSideEffectAnalysis, FunctionPass> {
  void runOnFunction() override {
    int64_t next_id = 0;
    llvm::SmallDenseMap<Operation*, int64_t, 8> ids;
    getFunction().walk([&](Operation* op) {
      ids[op] = next_id++;
      op->emitRemark("ID: ") << ids[op];
    });
    auto join_ids = [&](const llvm::ArrayRef<Operation*> ops) {
      llvm::SmallVector<std::string, 8> id_vec;
      id_vec.reserve(ops.size());
      for (auto op : ops) id_vec.push_back(std::to_string(ids[op]));
      return llvm::join(id_vec, ",");
    };
    auto& analysis = getAnalysis<TF::SideEffectAnalysis>();
    getFunction().walk([&](Operation* op) {
      if (!analysis.DirectControlPredecessors(op).empty()) {
        op->emitRemark("Predecessors: ")
            << "{" << join_ids(analysis.DirectControlPredecessors(op)) << "}";
      }
      if (!analysis.DirectControlSuccessors(op).empty()) {
        op->emitRemark("Successors: ")
            << "{" << join_ids(analysis.DirectControlSuccessors(op)) << "}";
      }
    });
  }
};

static mlir::PassRegistration<TestSideEffectAnalysis> pass(
    "tf-test-side-effect-analysis",
    "Add remarks based on side-effect analysis result, for testing purpose.");

}  // anonymous namespace

}  // namespace tf_executor
}  // namespace mlir
