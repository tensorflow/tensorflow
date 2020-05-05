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

// This file implements the optimzation passe on TFRT CoreRuntime dialect.
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/passes.h"
#include "tfrt/core_runtime/opdefs/core_runtime.h"

namespace tensorflow {
namespace {

// Implement a constant fold pattern for corert dialect. The following pattern
// will be matched:
//
// %0 = corert.executeop(%cpu) "tf.Const"()
//  {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : 1
// %1 = corert.executeop(%cpu) "tf.Transpose"(%arg, %0)
//  {T = f32, Tperm = i32} : 1
//
// And it will converted to:
//
// %1 = corert.executeop(%cpu) "_tf.Transpose"(%arg)
//  {T = f32, Tperm = i32, perm = dense<[0, 3, 1, 2]> : tensor<4xi32>} : 1
//
class CoreRTExecuteOpRewritePattern
    : public mlir::OpRewritePattern<tfrt::corert::ExecuteOp> {
 public:
  CoreRTExecuteOpRewritePattern(
      mlir::MLIRContext *context,
      ArrayRef<std::pair<StringRef, ArrayRef<StringRef>>> ops_to_attrs)
      : OpRewritePattern(context),
        ops_to_attrs_(ops_to_attrs.begin(), ops_to_attrs.end()) {}

  mlir::LogicalResult matchAndRewrite(
      tfrt::corert::ExecuteOp op,
      mlir::PatternRewriter &rewriter) const override {
    auto attr_names = ops_to_attrs_.lookup(op.op_name());
    if (attr_names.empty()) return failure();

    SmallVector<mlir::Value, 4> new_operands;
    SmallVector<std::pair<StringRef, Attribute>, 4> new_attributes;
    op.getOpAttrs(&new_attributes);
    assert(op.operands().size() == attr_names.size());
    for (const auto &iter : llvm::zip(op.operands(), attr_names)) {
      mlir::Value arg = std::get<0>(iter);
      StringRef name = std::get<1>(iter);

      Attribute const_attr;
      if (!name.empty() && matchPattern(arg, m_Constant(&const_attr))) {
        // Convert the folded argument to an attribute.
        new_attributes.push_back({name, const_attr});
      } else {
        // Keep the argument that is not folded.
        new_operands.push_back(arg);
      }
    }

    if (new_operands.size() == op.operands().size()) return failure();

    SmallString<32> new_op_name{"_"};
    new_op_name += op.op_name();

    rewriter.replaceOpWithNewOp<tfrt::corert::ExecuteOp>(
        op, op.getResultTypes(), op.device(), new_operands, new_attributes,
        new_op_name);

    return success();
  }

 private:
  // Map from op_name to attr_names. The attr_names indicates the name of the
  // attribute to which each constant-folded argument is converted. An empty
  // string means this argument should not be folded.
  llvm::DenseMap<StringRef, ArrayRef<StringRef>> ops_to_attrs_;
};

struct CoreRTOptimizePass
    : public mlir::PassWrapper<CoreRTOptimizePass, FunctionPass> {
  void runOnFunction() override {
    mlir::OwningRewritePatternList patterns;
    auto func = getFunction();

    static constexpr StringRef kMeanAttrs[] = {"", "reduction_indices"};
    static constexpr StringRef kPadAttrs[] = {"", "paddings"};
    static constexpr StringRef kTransposeAttrs[] = {"", "perm"};

    static constexpr std::pair<StringRef, ArrayRef<StringRef>> kOpsToAttrs[] = {
        {"tf.Mean", kMeanAttrs},
        {"tf.Pad", kPadAttrs},
        {"tf.Transpose", kTransposeAttrs},
    };

    patterns.insert<CoreRTExecuteOpRewritePattern>(&getContext(), kOpsToAttrs);

    mlir::applyPatternsAndFoldGreedily(func, patterns);
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> CreateCoreRTOptimizePass() {
  return std::make_unique<CoreRTOptimizePass>();
}

static mlir::PassRegistration<CoreRTOptimizePass> pass("corert-optimize",
                                                       "Optimizes corert.");

}  // namespace tensorflow
