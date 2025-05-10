/* Copyright 2025 The OpenXLA Authors.

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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

namespace mlir::triton::xla {
namespace {

#define GEN_PASS_DEF_EXTRACTTMAINFOPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

struct ExtractTmaInfoPass
    : public impl::ExtractTmaInfoPassBase<ExtractTmaInfoPass> {
  void runOnOperation() override {
    mlir::ModuleOp mod = getOperation();
    mod.walk([&](mlir::triton::FuncOp func_op) -> void {
      for (auto arg : func_op.getArguments()) {
        if (!isa<TensorDescType>(arg.getType())) {
          continue;
        }

        auto tma_descriptor_attr = func_op.getArgAttrOfType<TmaDescriptorAttr>(
            arg.getArgNumber(), "tt.tma_descriptor");
        if (!tma_descriptor_attr) {
          emitError(arg.getLoc(),
                    "Argument of type tt.tensordesc must have attribute "
                    "tt.tma_descriptor");
          return signalPassFailure();
        }

        auto swizzle_mode = nvidia_gpu::getTMASwizzleMode(
            /*op=*/nullptr, mlir::cast<TensorDescType>(arg.getType()));
        if (!swizzle_mode.has_value()) {
          emitError(
              arg.getLoc(),
              "Unable to determine swizzle mode from tt.tensordesc's layout");
          return signalPassFailure();
        }
        SwizzleMode swizzle_mode_enum;
        switch (swizzle_mode.value()) {
          case 0:
            swizzle_mode_enum = SwizzleMode::kNone;
            break;
          case 1:
            swizzle_mode_enum = SwizzleMode::k32b;
            break;
          case 2:
            swizzle_mode_enum = SwizzleMode::k64b;
            break;
          case 3:
            swizzle_mode_enum = SwizzleMode::k128b;
            break;
          default:
            emitError(arg.getLoc(),
                      "Unable to determine swizzle mode from triton's "
                      "getTMASwizzleMode");
            return signalPassFailure();
        }

        IRRewriter rewriter(&getContext());
        func_op.setArgAttr(
            arg.getArgNumber(), "tt.tma_descriptor",
            rewriter.getAttr<TmaDescriptorAttr>(
                tma_descriptor_attr.getGlobalShape(),
                tma_descriptor_attr.getBlockShape(),
                tma_descriptor_attr.getElementByteSize(),
                SwizzleModeAttr::get(&getContext(), swizzle_mode_enum)));
      }
    });
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateExtractTmaInfoPass() {
  return std::make_unique<ExtractTmaInfoPass>();
}

}  // namespace mlir::triton::xla
