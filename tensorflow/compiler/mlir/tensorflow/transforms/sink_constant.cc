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
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassManager.h"  // TF:local_config_mlir
#include "mlir/Support/LLVM.h"  // TF:local_config_mlir
#include "mlir/Transforms/Passes.h"  // TF:local_config_mlir
#include "mlir/Transforms/RegionUtils.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"

#define DEBUG_TYPE "tf-executor-sink-constant"

namespace mlir {
namespace tf_executor {

namespace {
using ::mlir::TF::ConstOp;

class ExecutorConstantSinking
    : public mlir::FunctionPass<ExecutorConstantSinking> {
  void runOnFunction() override {
    getFunction().walk([](tf_device::LaunchOp launch) {
      LLVM_DEBUG(llvm::dbgs() << "Visit " << *launch.getOperation() << "\n");
      // For each launch op, we find the values used that come from a constant
      // defined above and sink these constants in the region body.
      // The sunk_constant map keeps a mapping from a ConstOp defined above to
      // a sunk clone of it. This allows for reusing a sunk constant with
      // multiple uses in the region.
      llvm::DenseMap<Value, TF::ConstOp> sunk_constant;
      Region &body = launch.body();
      visitUsedValuesDefinedAbove(body, [&](OpOperand *use) {
        Value constant = use->get();
        auto const_op =
            dyn_cast_or_null<TF::ConstOp>(constant->getDefiningOp());
        if (!const_op) return;

        // We found a constant, try to insert it in the map and re-use its
        // cloned value if any.
        auto map_entry = sunk_constant.try_emplace(constant, nullptr);
        if (!map_entry.second) {
          // This constant has already been cloned into the region, reuse it.
          use->set(map_entry.first->getSecond().getResult());
          LLVM_DEBUG(llvm::dbgs() << "Re-use sunk constant " << *use->get()
                                  << "\n     in " << *use->get() << "\n");
          if (constant->use_empty()) const_op.erase();
          return;
        }
        if (constant->hasOneUse()) {
          LLVM_DEBUG(llvm::dbgs() << "Moved constant " << *constant << "\n");
          const_op.getOperation()->moveBefore(&body.begin()->front());
          return;
        }
        map_entry.first->getSecond() = const_op.clone();
        body.begin()->getOperations().insert(body.begin()->begin(),
                                             map_entry.first->getSecond());
        use->set(map_entry.first->getSecond().getResult());
        LLVM_DEBUG(llvm::dbgs() << "Sunk cloned constant " << *use->get()
                                << "\n     in " << *use->get() << "\n");
      });
    });
  }
};

static mlir::PassRegistration<ExecutorConstantSinking> pass(
    "tf-device-constant-sinking",
    "Sink constants implicitly captured in a tf_device.launch region. This "
    "reduces the number of arguments when outlining later.");

}  // anonymous namespace

std::unique_ptr<OpPassBase<FuncOp>> CreateTFExecutorConstantSinkingPass() {
  return std::make_unique<ExecutorConstantSinking>();
}

}  // namespace tf_executor
}  // namespace mlir
