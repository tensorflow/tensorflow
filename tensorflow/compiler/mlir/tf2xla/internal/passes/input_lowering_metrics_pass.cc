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

#include <memory>

#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/legalization_op_config.h"
#include "tensorflow/core/lib/monitoring/counter.h"

namespace tensorflow {
namespace tf2xla {
namespace internal {

namespace {

using mlir::Operation;
using mlir::WalkResult;

#define GEN_PASS_DEF_INPUTLOWERINGMETRICSPASS
#include "tensorflow/compiler/mlir/tf2xla/internal/passes/lowering_passes.h.inc"

auto* dynamism_op_counter = tensorflow::monitoring::Counter<1>::New(
    "/tensorflow/core/tf2xla/api/v2/dynamism_op_counter",
    "Counts how many ops are dynamic", "op_name");

auto* dynamism_function_counter = tensorflow::monitoring::Counter<1>::New(
    "/tensorflow/core/tf2xla/api/v2/dynamism_function_counter",
    "Counts how many functions are dynamic", "has_dynamism");

constexpr char kNotDynamicFunctionName[] = "kNotDynamicFunction";
constexpr char kDynamicFunctionName[] = "kDynamicFunction";

class InputMetricsLoweringPass
    : public impl::InputLoweringMetricsPassBase<InputMetricsLoweringPass> {
 public:
  void runOnOperation() override;
};

void InputMetricsLoweringPass::runOnOperation() {
  bool has_dynamic_op = false;
  Operation* func_op = getOperation();

  func_op->walk([&](Operation* op) {
    auto abstractOp = op->getRegisteredInfo();
    if (!abstractOp) return WalkResult::advance();

    if (mlir::hlo::IsDynamicPadderOp(abstractOp->getTypeID())) {
      has_dynamic_op = true;
      dynamism_op_counter->GetCell(op->getName().getStringRef().str())
          ->IncrementBy(1);
    }

    return WalkResult::advance();
  });

  if (has_dynamic_op) {
    dynamism_function_counter->GetCell(kDynamicFunctionName)->IncrementBy(1);
  } else {
    dynamism_function_counter->GetCell(kNotDynamicFunctionName)->IncrementBy(1);
  }
}
}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateInputLoweringMetricsPass() {
  return std::make_unique<InputMetricsLoweringPass>();
}

}  // namespace internal
}  // namespace tf2xla
}  // namespace tensorflow
