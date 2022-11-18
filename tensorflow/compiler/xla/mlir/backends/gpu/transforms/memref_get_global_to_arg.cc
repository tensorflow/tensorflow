/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/gpu/transforms/passes.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_CONVERTMEMREFGETGLOBALTOARGPASS
#include "tensorflow/compiler/xla/mlir/backends/gpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

class ConvertMemrefGetGlobalToArgPass
    : public impl::ConvertMemrefGetGlobalToArgPassBase<
          ConvertMemrefGetGlobalToArgPass> {
 public:
  ConvertMemrefGetGlobalToArgPass() = default;

  explicit ConvertMemrefGetGlobalToArgPass(int64_t min_num_elements) {
    this->min_num_elements_ = min_num_elements;
  }

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<memref::MemRefDialect>();
  }
};

//===----------------------------------------------------------------------===//

using GlobalConstantsArgs =
    llvm::DenseMap<func::FuncOp, llvm::StringMap<Value>>;

// Returns a mapping from a global constant name to the function argument.
//
// Example:
//
//   memref.global "private" constant @cst : memref<2x3xf32>
//   func @get_global(%arg0: memref<24xi8> {lmhlo.constant_name = "cst"})
//
// All memref.get_global operations will be replaced by constant arguments
// corresponding to the global constant.
static GlobalConstantsArgs GetConstantArgs(ModuleOp m) {
  GlobalConstantsArgs mapping;

  m.walk([&](func::FuncOp func) {
    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      auto cst = func.getArgAttrOfType<StringAttr>(i, "lmhlo.constant_name");
      if (cst) mapping[func][cst] = func.getArgument(i);
    }
  });

  return mapping;
}

class GetGlobalOpLowering : public OpRewritePattern<memref::GetGlobalOp> {
 public:
  GetGlobalOpLowering(MLIRContext* ctx, const GlobalConstantsArgs& cst_args)
      : OpRewritePattern<memref::GetGlobalOp>(ctx), cst_args_(cst_args) {}

  LogicalResult matchAndRewrite(memref::GetGlobalOp op,
                                PatternRewriter& rewriter) const override {
    // Find global constants mapping for the parent function.
    auto func_mapping = cst_args_.find(op->getParentOfType<func::FuncOp>());
    if (func_mapping == cst_args_.end()) return failure();

    // Check if the global operation correposponds to the LMHLO constant arg.
    auto arg = func_mapping->second.find(op.getName());
    if (arg == func_mapping->second.end()) return failure();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    MemRefType memref = op->getResult(0).getType().cast<MemRefType>();

    // For identity layouts we can replace all loads from a global with the
    // corresponding argument.
    if (memref.getLayout().isIdentity()) {
      Value c0 = b.create<arith::ConstantOp>(rewriter.getIndexAttr(0));
      rewriter.replaceOpWithNewOp<memref::ViewOp>(op, memref, arg->second, c0,
                                                  ValueRange());
      return success();
    }

    // For non-identity type we first view constant argument as a flat memref
    // with the correct element type, and then cast it to the strided memref
    // corresponding to the original memref layout.

    // Get the strides and offset from the original memref type.
    int64_t offset;
    llvm::SmallVector<int64_t> strides;
    if (failed(getStridesAndOffset(memref, strides, offset)))
      return op.emitOpError("failed to compute strides and offset");

    // Create a 1d view into the corresponding argument.
    Value c0 = b.create<arith::ConstantOp>(rewriter.getIndexAttr(0));
    Value flat_view = b.create<memref::ViewOp>(
        MemRefType::get({memref.getNumElements()}, memref.getElementType()),
        arg->second, c0, ValueRange());

    // Cast flat memref view into the original memref type.
    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, memref, flat_view, offset, memref.getShape(), strides);

    return success();
  }

 private:
  const GlobalConstantsArgs& cst_args_;
};

void ConvertMemrefGetGlobalToArgPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext* ctx = module.getContext();

  // Replace memref loads from globals corresponding to the constant arguments.
  RewritePatternSet patterns(ctx);
  GlobalConstantsArgs cst_args = GetConstantArgs(module);
  patterns.insert<GetGlobalOpLowering>(ctx, cst_args);

  // Set up conversion target to rewrite only GetGlobalOp larger than the
  // threshold and avoid any other canonicalizations that can break later
  // passes.
  ConversionTarget target(*ctx);
  target.addDynamicallyLegalOp<memref::GetGlobalOp>(
      [&](memref::GetGlobalOp op) {
        auto memref = op.getType();
        return memref.getNumElements() < min_num_elements_;
      });
  target.addLegalOp<arith::ConstantOp, memref::ViewOp,
                    memref::ReinterpretCastOp>();

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertMemrefGetGlobalToArgPass() {
  return std::make_unique<ConvertMemrefGetGlobalToArgPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConvertMemrefGetGlobalToArgPass(int64_t min_num_elements) {
  return std::make_unique<ConvertMemrefGetGlobalToArgPass>(min_num_elements);
}

}  // namespace gpu
}  // namespace xla
