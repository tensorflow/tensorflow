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

// This transformation pass takes operations in TensorFlowLite dialect and
// optimizes them to resulting operations in TensorFlowLite dialect.

#include <climits>

#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Support/Functional.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual Optimize Pass.
namespace {

// Optimize TFLite operations in functions.
struct Optimize : public FunctionPass<Optimize> {
  void runOnFunction() override;
};

// Returns whether the given `a` and `b` ElementsAttr have broadcast-compatible
// types.
bool IsBroadcastableElementsAttrs(Attribute a, Attribute b) {
  return OpTrait::util::getBroadcastedType(a.getType(), b.getType()) != Type();
}

#include "tensorflow/compiler/mlir/lite/transforms/generated_optimize.inc"
// Fuse Add with FullyConnected.
// Note that this assumes that the bias in the fullyConnected
// is always None.
// TODO(b/136285429): Move to tablegen when variadic is supported
// and add support for bias with noneType type.
struct FuseFullyConnectedAndAdd : public RewritePattern {
  explicit FuseFullyConnectedAndAdd(MLIRContext *context)
      : RewritePattern(TFL::AddOp::getOperationName(),
                       {"tfl.fully_connected", "tfl.add", "std.constant"}, 4,
                       context) {}

  PatternMatchResult matchAndRewrite(Operation *add_op,
                                     PatternRewriter &rewriter) const override {
    // Fully Connected.
    Operation *fully_connected = add_op->getOperand(0)->getDefiningOp();
    if (!fully_connected || !isa<TFL::FullyConnectedOp>(fully_connected))
      return matchFailure();
    TFL::FullyConnectedOp fully_connected_op =
        llvm::cast<TFL::FullyConnectedOp>(fully_connected);
    Value *input = fully_connected_op.input();
    Value *filter = fully_connected_op.filter();

    // Make sure the bias is None.
    // TODO(karimnosseir): Support non None case.
    Operation *bias_op = fully_connected_op.bias()->getDefiningOp();
    if (!bias_op || !isa<ConstantOp>(bias_op)) return matchFailure();
    if (!fully_connected_op.bias()->getType().isa<NoneType>())
      return matchFailure();

    auto activation_func = fully_connected_op.getAttrOfType<StringAttr>(
        "fused_activation_function");
    if (!activation_func) return matchFailure();
    if (activation_func.cast<StringAttr>().getValue() != "NONE")
      return matchFailure();

    auto weight_format =
        fully_connected_op.getAttrOfType<StringAttr>("weights_format");
    if (!weight_format) return matchFailure();

    auto keep_num_dims =
        fully_connected_op.getAttrOfType<BoolAttr>("keep_num_dims");
    if (!keep_num_dims) return matchFailure();

    auto constant_op = add_op->getOperand(1)->getDefiningOp();
    if (!constant_op) return matchFailure();
    if (!isa<ConstantOp>(constant_op)) return matchFailure();

    auto add_value = constant_op->getAttrOfType<Attribute>("value");
    if (!add_value) return matchFailure();
    if (!((add_value.cast<ElementsAttr>().getType().getElementType().isF32())))
      return matchFailure();

    auto fused_activation_func =
        add_op->getAttrOfType<StringAttr>("fused_activation_function");
    if (!fused_activation_func) return matchFailure();

    // Rewrite
    // TODO(karimnosseir): Check what constraints needed to apply.
    // TODO(b/136171362): Check for single output consumer.
    rewriter.replaceOpWithNewOp<TFL::FullyConnectedOp>(
        add_op, add_op->getResult(0)->getType(),
        /*input=*/input,
        /*filter=*/filter,
        /*bias=*/add_op->getOperand(1),
        /*fused_activation_function=*/fused_activation_func,
        /*weights_format=*/weight_format,
        /*keep_num_dims=*/keep_num_dims);

    return matchSuccess();
  }
};

void Optimize::runOnFunction() {
  OwningRewritePatternList patterns;
  auto func = getFunction();
  // Add the generated patterns to the list.
  TFL::populateWithGenerated(&getContext(), &patterns);
  patterns.push_back(
      llvm::make_unique<FuseFullyConnectedAndAdd>(&getContext()));
  applyPatternsGreedily(func, std::move(patterns));
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect Optimize pass.
FunctionPassBase *CreateOptimizePass() { return new Optimize(); }

static PassRegistration<Optimize> pass(
    "tfl-optimize", "Optimize within the TensorFlow Lite dialect");

}  // namespace TFL
}  // namespace mlir
