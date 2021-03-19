/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for some optimizations to reduce size on export.

#include <memory>

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/xla_passes_detail.h"

#define DEBUG_TYPE "xla-prepare-for-export"

namespace mlir {
namespace mhlo {
namespace {

// Prepare module for export to XLA HLO.
struct PrepareForExportPass : PrepareForExportPassBase<PrepareForExportPass> {
  void runOnFunction() override;
};

static PassRegistration<PrepareForExportPass> registration(
    "xla-prepare-for-export", "Prepare for XLA export");

}  // end namespace

void PrepareForExportPass::runOnFunction() {
  getFunction().walk([&](Operation *op) {
    mlir::SplatElementsAttr attr;
    if (!matchPattern(op, m_Constant(&attr))) return;
    // Only consider int or floats for now.
    if (!attr.getType().getElementType().isIntOrFloat()) return;
    // Arbitrarialy chosen "small" number. This could be chosen based on the
    // proto size too.
    if (attr.getNumElements() < 32) return;
    ShapedType return_type = op->getResultTypes().front().cast<ShapedType>();
    ImplicitLocOpBuilder b(op->getLoc(), op);
    auto cst = b.create<::mlir::mhlo::ConstOp>(attr.getSplatValue());
    auto broadcast = b.create<::mlir::mhlo::BroadcastInDimOp>(
        return_type, cst, b.getI64TensorAttr({}));
    op->replaceAllUsesWith(broadcast);
    op->erase();
  });
}

std::unique_ptr<OperationPass<FuncOp>> CreatePrepareForExport() {
  return std::make_unique<PrepareForExportPass>();
}

}  // end namespace mhlo
}  // end namespace mlir
