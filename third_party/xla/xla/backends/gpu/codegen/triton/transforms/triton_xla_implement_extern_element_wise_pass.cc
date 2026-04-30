/* Copyright 2026 The OpenXLA Authors.

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

// CUDA-specific implementation of extern_elementwise atomic functions.
// This pass runs in the Triton CUDA pipeline and inlines the implementations
// of custom atomic functions by replacing llvm.call operations with LLVM
// intrinsics.

#include <memory>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/extern_function_helper.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_TRITONXLAIMPLEMENTEXTERNELEMENTWISEPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

// Pattern to rewrite llvm.call operations to XLA extern functions
class RewriteExternCallPattern : public OpRewritePattern<LLVM::CallOp> {
 public:
  using OpRewritePattern<LLVM::CallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallOp call_op,
                                PatternRewriter& rewriter) const override {
    // Check if this is a call to one of our extern functions
    std::optional<llvm::StringRef> callee = call_op.getCallee();
    if (!callee) {
      return failure();
    }

    llvm::StringRef callee_name = *callee;

    // Parse the function name to get the instruction
    absl::StatusOr<ExternFunctionInstruction> parsed =
        ParseExternFunctionName(callee_name);
    if (!parsed.ok()) {
      return rewriter.notifyMatchFailure(
          call_op,
          absl::StrFormat("Failed to parse extern function name: %s - %s",
                          callee_name, parsed.status().ToString()));
    }

    absl::Status validation = ValidateMemorySemantic(*parsed);
    if (!validation.ok()) {
      return rewriter.notifyMatchFailure(
          call_op,
          absl::StrFormat("Invalid memory semantic for function: %s - %s",
                          callee_name, validation.ToString()));
    }

    LLVMOpCreationParams params{/*.builder=*/rewriter,
                                /*.loc=*/call_op.getLoc(),
                                /*.target=*/TargetBackend::CUDA,
                                /*.operands=*/call_op.getOperands()};

    mlir::Value result = CreateLLVMOpsForInstruction(*parsed, params);

    // Find the function definition.
    LLVM::LLVMFuncOp func_op =
        call_op->getParentOfType<ModuleOp>().lookupSymbol<LLVM::LLVMFuncOp>(
            callee_name);
    rewriter.replaceOp(call_op, result);
    // If the function is now unused, erase it.
    if (func_op && func_op.isExternal() && func_op.use_empty()) {
      rewriter.eraseOp(func_op);
    }
    return success();
  }
};

// MLIR pass that inlines extern function calls with LLVM intrinsics
class TritonXLAImplementExternElementWisePass
    : public impl::TritonXLAImplementExternElementWisePassBase<
          TritonXLAImplementExternElementWisePass> {
 public:
  using Base::Base;

 private:
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext* context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<RewriteExternCallPattern>(context);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateTritonXLAImplementExternElementWisePass() {
  return std::make_unique<TritonXLAImplementExternElementWisePass>();
}

}  // namespace mlir::triton::xla
