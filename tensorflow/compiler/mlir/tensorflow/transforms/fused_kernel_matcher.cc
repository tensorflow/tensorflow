/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdio>
#include <iostream>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {

namespace TF {

namespace {

// Note: This implements the fusions performed in the old Remapper Grappler
// pass. That pass has specific cases for GPU and based on different
// target configurations on both CPU and GPU (Intel MKL, ROCm, etc.). This MLIR
// pass covers the general CPU case and at the moment does not account for any
// specific target configurations.
// TODO(b/158265178): Support GPU-specific fusions.
// TODO(b/158266710): Support CPU MKL configurations.

// Optimizes TF computations by fusing subgraphs/nodes onto more efficient
// implementations to decrease the number of operations needed to perform a
// computation.
struct FusedKernelMatcherPass
    : public PassWrapper<FusedKernelMatcherPass, FunctionPass> {
  void runOnFunction() override;
};

// Returns an op's name with the dialect prefix stripped off.
StringRef GetOpNameWithoutDialect(Operation *op) {
  return op->getName().getStringRef().split(".").second;
}

bool IsActivationFunction(Operation *op) {
  return isa<EluOp>(op) || isa<ReluOp>(op) || isa<Relu6Op>(op);
}

// Finds and returns an activation op that uses the result of `op`. If there are
// multiple such activations, one is returned (with no guarantee as to which
// one). If there are no activation functions that use the output, returns
// nullptr.
Operation *GetActivation(Value op) {
  for (auto &use : op.getUses()) {
    if (IsActivationFunction(use.getOwner())) return use.getOwner();
  }
  return nullptr;
}

// Finds and returns a BiasAdd that uses the result of `op` as the `value`
// input. If there are multiple such BiasAdds, one is returned (with no
// guarantee as to which one). If there are no BiasAdds that use the output,
// returns a null BiasAddOp.
BiasAddOp GetBiasAdd(Value op) {
  for (auto &use : op.getUses()) {
    auto bias_add = dyn_cast_or_null<BiasAddOp>(use.getOwner());
    // If it's a BiasAdd, check that the conv op is the first input.
    if (bias_add && bias_add.value() == op) return bias_add;
  }
  // No BiasAddOps found among uses.
  return BiasAddOp();
}

// Performs a fusion of the following pattern(s), if possible:
//   Conv2D + BiasAdd + <Activation> -> _FusedConv2D
//
// Note that fusion with activation is preferred, but a Conv2D and BiasAdd can
// also be replaced by a _FusedConv2D if there is no other activation function.
// i.e., this class also supports the following fusion:
//   Conv2D + BiasAdd -> _FusedConv2D
//
// TODO(b/158266331): Support fusing Conv2D + BiasAdd + a chain of activations.
class FuseConv2DBiasAdd : public OpRewritePattern<Conv2DOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(Conv2DOp op,
                                PatternRewriter &rewriter) const override {
    // If the convolution is used in multiple places, fusing it will only create
    // more convolutions, which is slower.
    if (!op.getResult().hasOneUse())
      return rewriter.notifyMatchFailure(op, "result is used by multiple ops");

    BiasAddOp bias_add = GetBiasAdd(op);
    if (!bias_add) {
      return rewriter.notifyMatchFailure(
          op, "does not feed into a tf.BiasAdd/tf.BiasAddV1 op");
    }

    // Check that Conv and BiasAdd formats match.
    if (op.data_format() != bias_add.data_format()) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "data format does not match Conv2D data format ("
             << bias_add.data_format() << " vs " << op.data_format() << ")";
      });
    }

    SmallVector<Location, 3> locations{op.getLoc(), bias_add.getLoc()};
    SmallVector<Attribute, 2> fused_ops{StringAttr::get(
        GetOpNameWithoutDialect(bias_add), rewriter.getContext())};
    Type result_type;

    // BiasAdd may or may not feed into an activation function.
    auto activation = GetActivation(bias_add);

    // If there is an activation, only fuse it if this is the only op to use the
    // result of the BiasAdd.
    bool fuse_activation = activation && bias_add.output().hasOneUse();

    // Include info about the activation function if applicable.
    if (fuse_activation) {
      locations.push_back(activation->getLoc());
      fused_ops.push_back(StringAttr::get(GetOpNameWithoutDialect(activation),
                                          rewriter.getContext()));
      result_type = activation->getResultTypes().front();
    } else {
      result_type = bias_add.getResult().getType();
    }

    auto loc = rewriter.getFusedLoc(locations);
    ArrayAttr fused_ops_attr = ArrayAttr::get(fused_ops, rewriter.getContext());
    // Epsilon is used only in fusions with the BatchNorm op.
    APFloat epsilon = APFloat(0.0f);
    auto fused_op = rewriter.create<_FusedConv2DOp>(
        loc, result_type, op.input(), op.filter(), bias_add.bias(),
        op.strides(), op.padding(), op.explicit_paddings(), op.data_format(),
        op.dilations(), op.use_cudnn_on_gpu(), fused_ops_attr, epsilon);
    auto op_to_replace = fuse_activation ? activation : bias_add;
    rewriter.replaceOp(op_to_replace, {fused_op});
    return success();
  }
};

void FusedKernelMatcherPass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto func = getFunction();
  patterns.insert<FuseConv2DBiasAdd>(&getContext());

  applyPatternsAndFoldGreedily(func, patterns);
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateFusedKernelMatcherPass() {
  return std::make_unique<FusedKernelMatcherPass>();
}

static PassRegistration<FusedKernelMatcherPass> pass(
    "tf-fused-kernel-matcher",
    "Matches computations corresponding to optimized fused kernels");

}  // namespace TF

}  // namespace mlir
