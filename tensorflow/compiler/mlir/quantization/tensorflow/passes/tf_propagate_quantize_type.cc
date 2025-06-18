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

#include <algorithm>
#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Rewrite/FrozenRewritePatternSet.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/tf_attrs_and_constraints.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/ops/temp_tf_op_quant_spec.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
namespace mlir {
namespace tf_quant {
namespace {

constexpr StringRef kDequantizeFunctionName = "composite_dequantize";

class PropagateQuantizeType
    : public PassWrapper<PropagateQuantizeType, OperationPass<ModuleOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PropagateQuantizeType)

  // Constructor used by the PassRegistration. This will remove the adaptor ops.
  explicit PropagateQuantizeType() = default;

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tf-quant-propagate-quantize-type";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Propagate quantized type through allowed ops.";
  }

  void runOnOperation() override;
};

// Propagate dequantize op if the next op supports the data type.
// Given the below graph,
// op_before_dequantize -> dequantize_op -> user_op -> rest_op
// the transformation is applied to result the following graph:
// op_before_dequantize -> user_op -> new_dequantize_op -> rest_op
class PropagateDequantizeOpIfAllowed
    : public OpRewritePattern<TF::PartitionedCallOp> {
 public:
  explicit PropagateDequantizeOpIfAllowed(MLIRContext* context)
      : OpRewritePattern<TF::PartitionedCallOp>(context) {}

  // Create a new dequantize op that is propagated.
  void createNewDequantizeOp(PatternRewriter& rewriter,
                             TF::PartitionedCallOp original_dequantize_op,
                             Operation* user_op, int user_idx,
                             Type new_user_op_type) const {
    auto op_before_dequantize = original_dequantize_op.getOperand(0);

    // Create a new dequantize op that is propagated.
    rewriter.setInsertionPointAfter(user_op);
    TF::PartitionedCallOp new_dequantize_op =
        cast<TF::PartitionedCallOp>(rewriter.clone(*original_dequantize_op));

    // Skip the original dequant op and connect the op before dequantize to the
    // user op.
    user_op->setOperand(user_idx, op_before_dequantize);

    // Wire input/output nodes.
    new_dequantize_op->setOperand(0, user_op->getResult(0));
    new_dequantize_op->getResult(0).setType(user_op->getResult(0).getType());
    user_op->getResult(0).replaceAllUsesExcept(new_dequantize_op->getResult(0),
                                               new_dequantize_op);
    user_op->getResult(0).setType(new_user_op_type);
  }

  LogicalResult matchAndRewrite(TF::PartitionedCallOp op,
                                PatternRewriter& rewriter) const override {
    const auto f_attr = mlir::dyn_cast<FlatSymbolRefAttr>(op.getFAttr());
    StringRef function_name = f_attr.getValue();
    if (!function_name.starts_with(kDequantizeFunctionName)) return failure();

    llvm::SmallVector<Operation*> users(op->getUsers().begin(),
                                        op->getUsers().end());

    bool changed = false;
    for (auto& use : op->getUses()) {
      Operation* user_op = use.getOwner();
      int user_idx = use.getOperandNumber();
      if (!IsOpWithInt8TypeOperand(user_op)) continue;
      // If the next op is terminator, function type needs to be changed so
      // handle this case separately when propagating for function op is
      // added.
      if (std::any_of(user_op->getResult(0).getUsers().begin(),
                      user_op->getResult(0).getUsers().end(), [](Operation* y) {
                        return y->hasTrait<OpTrait::IsTerminator>();
                      }))
        continue;
      if (IsOpWithDataMovementTrait(user_op)) {
        auto op_before_dequantize = op.getOperand(0);
        // New user op type needs to be set since user_op can output integer
        // type for the data movement case.
        auto original_result_type = user_op->getResult(0).getType();
        auto new_user_op_type = CloneTypeWithNewElementType(
            original_result_type,
            mlir::cast<ShapedType>(op_before_dequantize.getType())
                .getElementType());
        createNewDequantizeOp(rewriter, op, user_op, user_idx,
                              new_user_op_type);
      } else {
        createNewDequantizeOp(rewriter, op, user_op, user_idx,
                              user_op->getResult(0).getType());
      }
      changed = true;
    }
    return changed ? success() : failure();
  }
};

void PropagateQuantizeType::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto module_op = getOperation();
  MLIRContext* ctx = &getContext();

  patterns.add<PropagateDequantizeOpIfAllowed>(ctx);
  FrozenRewritePatternSet frozen_patterns(std::move(patterns));
  // Propagation can happen recursively with multiple functions so keep this
  // module level.
  for (auto func : module_op.getOps<func::FuncOp>()) {
    if (failed(applyPatternsGreedily(func, frozen_patterns))) {
      func.emitError() << "tf-quant-propagate-quantize-type failed.";
      signalPassFailure();
    }
  }
}

}  // namespace

// Creates an instance of the TensorFlow dialect PropagateQuantizeType pass.
std::unique_ptr<OperationPass<ModuleOp>> CreatePropagateQuantizeTypePass() {
  return std::make_unique<PropagateQuantizeType>();
}

static PassRegistration<PropagateQuantizeType> pass;

}  // namespace tf_quant
}  // namespace mlir
