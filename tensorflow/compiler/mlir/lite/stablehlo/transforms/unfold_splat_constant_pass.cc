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

#include <cstdint>
#include <memory>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {
namespace {

#define DEBUG_TYPE "unfold-splat-constant-pass"

#define GEN_PASS_DEF_UNFOLDSPLATCONSTANTPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h.inc"

// Undo the MHLO::BroadcastInDimOp folding pattern on splat tensor.
// TODO(b/295966255): Remove this pass after moving MHLO folders to a separate
// pass and folders are not applied by default.
class UnfoldSplatConstantPass
    : public impl::UnfoldSplatConstantPassBase<UnfoldSplatConstantPass> {
 public:
  void runOnOperation() override {
    auto module = getOperation();

    mlir::OpBuilder op_builder(&module.getBodyRegion());
    // Cannot use the pattern rewriter because the driver applies folders by
    // default.
    module.walk([&](mhlo::ConstantOp const_op) {
      UnfoldSplatConstant(&op_builder, const_op);
    });
  }

 private:
  void UnfoldSplatConstant(mlir::OpBuilder* op_builder,
                           mhlo::ConstantOp const_op) const {
    auto splat_elements_attr =
        const_op.getValue().dyn_cast<SplatElementsAttr>();
    if (!splat_elements_attr) {
      return;
    }
    if (splat_elements_attr.getNumElements() == 1) {
      return;
    }
    auto element_type = splat_elements_attr.getType().getElementType();
    if (element_type.isa<ComplexType>() ||
        element_type.isa<quant::QuantizedType>()) {
      return;
    }
    op_builder->setInsertionPoint(const_op);
    Value scalar = op_builder->create<mhlo::ConstantOp>(
        const_op->getLoc(),
        DenseElementsAttr::get(
            RankedTensorType::get(/*shape=*/{}, element_type),
            splat_elements_attr.getSplatValue<Attribute>()));
    auto broadcast_dims = DenseIntElementsAttr::get(
        RankedTensorType::get(/*shape=*/{0}, op_builder->getI64Type()),
        llvm::SmallVector<int64_t>{});
    mhlo::BroadcastInDimOp broadcast_in_dim_op =
        op_builder->create<mhlo::BroadcastInDimOp>(
            const_op->getLoc(), splat_elements_attr.getType(), scalar,
            broadcast_dims);
    const_op->replaceAllUsesWith(broadcast_in_dim_op);
    const_op->erase();
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateUnfoldSplatConstantPass() {
  return std::make_unique<UnfoldSplatConstantPass>();
}

static PassRegistration<UnfoldSplatConstantPass> pass;

}  // namespace odml
}  // namespace mlir
