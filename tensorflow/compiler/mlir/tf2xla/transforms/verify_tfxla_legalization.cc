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

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/xla_legalize_targets.h"
#include "tensorflow/core/lib/monitoring/counter.h"

namespace mlir {
namespace mhlo {

namespace {

#define GEN_PASS_DEF_VERIFYTFXLALEGALIZATION
#include "tensorflow/compiler/mlir/tf2xla/transforms/xla_legalize_tf_passes.h.inc"

auto* mlir_failed_legalization_op_count =
    tensorflow::monitoring::Counter<1>::New(
        "/tensorflow/core/tf2xla/"
        "mlir_second_phase_failed_legalization_op_count",
        "Counts which op fails to legalize", "op_name");

class VerifyTFXLALegalization
    : public impl::VerifyTFXLALegalizationBase<VerifyTFXLALegalization> {
 public:
  explicit VerifyTFXLALegalization(bool legalize_chlo) {
    legalize_chlo_ = legalize_chlo_;
  }

  void runOnOperation() override;
};

void VerifyTFXLALegalization::runOnOperation() {
  Operation* func_op = getOperation();
  ConversionTarget default_conversion_target =
      GetDefaultLegalConversionTargets(getContext(), legalize_chlo_);

  auto walk_result = func_op->walk([&](Operation* op) {
    if (default_conversion_target.isLegal(op)) {
      return WalkResult::advance();
    }

    emitError(op->getLoc()) << "Could not legalize op: " << op->getName();
    mlir_failed_legalization_op_count
        ->GetCell(op->getName().getStringRef().str())
        ->IncrementBy(1);
    return WalkResult::interrupt();
  });

  if (walk_result.wasInterrupted()) signalPassFailure();
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateVerifyTFXLALegalizationPass(bool legalize_chlo) {
  return std::make_unique<VerifyTFXLALegalization>(legalize_chlo);
}

}  // namespace mhlo
}  // namespace mlir
