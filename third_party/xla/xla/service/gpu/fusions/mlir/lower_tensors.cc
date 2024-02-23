
/* Copyright 2024 The OpenXLA Authors.
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
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"  // from @llvm-project
#include "mlir/Dialect/Affine/IR/AffineOps.h"  // from @llvm-project
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"  // from @llvm-project
#include "mlir/Dialect/SCF/IR/SCF.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"  // from @llvm-project
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "xla/layout_util.h"
#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_ops.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DEF_LOWERTENSORSPASS
#include "xla/service/gpu/fusions/mlir/passes.h.inc"

namespace {

using mlir::failure;
using mlir::success;

struct RewriteFunctionSignatures : mlir::OpRewritePattern<mlir::func::FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::func::FuncOp op, mlir::PatternRewriter& rewriter) const override {
    auto is_tensor = [](mlir::Type ty) {
      return ty.isa<mlir::RankedTensorType>();
    };
    if (!llvm::any_of(op.getFunctionType().getInputs(), is_tensor)) {
      return rewriter.notifyMatchFailure(op,
                                         "the function has no input tensors");
    }

    bool some_tensor_result =
        llvm::any_of(op.getFunctionType().getResults(), is_tensor);
    bool all_tensor_results =
        llvm::all_of(op.getFunctionType().getResults(), is_tensor);
    if (some_tensor_result && !all_tensor_results) {
      op->emitOpError("function has a mix of tensor and non-tensor results");
      return failure();
    }

    mlir::TypeRange new_results = op.getFunctionType().getResults();
    if (some_tensor_result) {
      new_results = {};
      auto terminator = op.getFunctionBody().front().getTerminator();
      rewriter.setInsertionPoint(terminator);
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(terminator);
    }

    llvm::SmallVector<mlir::Type> new_operands(
        op.getFunctionType().getInputs());
    for (auto&& [index, operand] : llvm::enumerate(new_operands)) {
      if (is_tensor(operand)) {
        rewriter.setInsertionPointToStart(&op.getBody().front());
        auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
            op.getLoc(), operand, op.getArgument(index));
        op.getArgument(index).replaceAllUsesExcept(cast.getResult(0), cast);
        operand = mlir::LLVM::LLVMPointerType::get(op.getContext());
      }
    }

    op.setFunctionType(rewriter.getFunctionType(new_operands, new_results));
    auto& entry = op->getRegion(0).front();
    for (auto [arg, arg_type] : llvm::zip(entry.getArguments(), new_operands)) {
      arg.setType(arg_type);
    }

    return success();
  }
};

mlir::Value CreateGep(mlir::Operation* op,
                      mlir::TypedValue<mlir::RankedTensorType> tensor,
                      mlir::ValueRange indices,
                      mlir::PatternRewriter& rewriter) {
  auto ptr = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  auto byte_shape = ShapeUtil::MakeShape(U8, tensor.getType().getShape());
  if (auto encoding = tensor.getType().getEncoding()) {
    *byte_shape.mutable_layout() = LayoutUtil::MakeLayout(llvm::to_vector(
        encoding.cast<mlir::DenseElementsAttr>().getValues<int64_t>()));
  }
  auto linearize_map = mlir::getAffineConstantExpr(0, rewriter.getContext());
  for (auto [dim, stride] :
       llvm::enumerate(*ShapeUtil::ByteStrides(byte_shape))) {
    linearize_map = linearize_map +
                    mlir::getAffineDimExpr(dim, rewriter.getContext()) * stride;
  }

  rewriter.setInsertionPoint(op);
  mlir::Value index = rewriter.create<mlir::affine::AffineApplyOp>(
      tensor.getLoc(), linearize_map, indices);
  auto index_ty =
      ShapeUtil::ElementsIn(byte_shape) < std::numeric_limits<int32_t>::max()
          ? rewriter.getI32Type()
          : rewriter.getI64Type();
  index = rewriter.create<mlir::arith::IndexCastUIOp>(tensor.getLoc(), index_ty,
                                                      index);

  auto tensor_ptr = rewriter
                        .create<mlir::UnrealizedConversionCastOp>(
                            tensor.getLoc(), ptr, tensor)
                        .getResult(0);
  auto gep = rewriter.create<mlir::LLVM::GEPOp>(
      tensor.getLoc(), ptr, tensor.getType().getElementType(), tensor_ptr,
      index);
  gep.setInbounds(true);
  return gep;
}

struct RewriteTensorExtract : mlir::OpRewritePattern<mlir::tensor::ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::ExtractOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto gep = CreateGep(op, op.getTensor(), op.getIndices(), rewriter);
    rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, op.getType(), gep);
    return success();
  }
};

struct RewriteTensorInsert : mlir::OpRewritePattern<mlir::tensor::InsertOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::InsertOp op,
      mlir::PatternRewriter& rewriter) const override {
    mlir::Value dest = op.getDest();
    while (dest.getDefiningOp()) {
      int result_number = dest.cast<mlir::OpResult>().getResultNumber();
      if (auto insert = dest.getDefiningOp<mlir::tensor::InsertOp>()) {
        dest = insert.getDest();
      } else if (auto scf_if = dest.getDefiningOp<mlir::scf::IfOp>()) {
        // Pick one of the branches, they're required to yield the same buffers.
        dest = scf_if.getThenRegion().front().getTerminator()->getOperand(
            result_number);
      } else if (auto scf_for = dest.getDefiningOp<mlir::scf::ForOp>()) {
        dest = scf_for.getInitArgs()[result_number];
      }
    }

    auto gep =
        CreateGep(op, dest.cast<mlir::TypedValue<mlir::RankedTensorType>>(),
                  op.getIndices(), rewriter);
    rewriter.create<mlir::LLVM::StoreOp>(gep.getLoc(), op.getScalar(), gep);

    op.replaceAllUsesWith(op.getDest());
    op.erase();
    return success();
  }
};

struct RewriteCall : mlir::OpRewritePattern<mlir::func::CallOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::func::CallOp op, mlir::PatternRewriter& rewriter) const override {
    if (!llvm::any_of(op->getOperandTypes(), [](mlir::Type ty) {
          return ty.isa<mlir::RankedTensorType>();
        })) {
      return rewriter.notifyMatchFailure(op, "the call has no input tensors");
    }

    for (const auto&& [index, arg] : llvm::enumerate(op.getOperands())) {
      if (arg.getType().isa<mlir::RankedTensorType>()) {
        op.setOperand(
            index,
            rewriter
                .create<mlir::UnrealizedConversionCastOp>(
                    op.getLoc(),
                    mlir::LLVM::LLVMPointerType::get(op.getContext()), arg)
                .getResult(0));
      }
    }
    return success();
  }
};

struct RewriteAllocateShared : mlir::OpRewritePattern<AllocateSharedOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      AllocateSharedOp op, mlir::PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    auto shaped_ty = op.getResult().getType().cast<mlir::ShapedType>();
    constexpr int kGPUSharedMemoryAddrSpace = 3;
    auto array_ty = mlir::LLVM::LLVMArrayType::get(shaped_ty.getElementType(),
                                                   shaped_ty.getNumElements());

    std::string name;
    int index = 0;
    do {
      name = absl::StrCat("shared_", index);
      ++index;
    } while (module.lookupSymbol(name));

    rewriter.setInsertionPointToStart(module.getBody());
    auto global = rewriter.create<mlir::LLVM::GlobalOp>(
        op.getLoc(), array_ty, /*isConstant=*/false,
        /*linkage=*/mlir::LLVM::Linkage::Private, name,
        /*value=*/mlir::Attribute{},
        /*alignment=*/0, kGPUSharedMemoryAddrSpace);

    rewriter.setInsertionPoint(op);
    auto addr = rewriter.create<mlir::LLVM::AddressOfOp>(op.getLoc(), global);
    rewriter.replaceOpWithNewOp<mlir::UnrealizedConversionCastOp>(
        op, op.getResult().getType(),
        rewriter
            .create<mlir::LLVM::AddrSpaceCastOp>(
                op.getLoc(), mlir::LLVM::LLVMPointerType::get(op.getContext()),
                addr)
            .getResult());
    return mlir::success();
  }
};

class LowerTensorsPass : public impl::LowerTensorsPassBase<LowerTensorsPass> {
 public:
  void runOnOperation() override;
};

void LowerTensorsPass::runOnOperation() {
  mlir::RewritePatternSet tensor_patterns(&getContext());
  tensor_patterns
      .add<RewriteTensorExtract, RewriteTensorInsert, RewriteAllocateShared>(
          &getContext());
  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
          getOperation(), std::move(tensor_patterns)))) {
    signalPassFailure();
  }

  mlir::RewritePatternSet function_patterns(&getContext());
  function_patterns.add<RewriteFunctionSignatures, RewriteCall>(&getContext());
  mlir::scf::ForOp::getCanonicalizationPatterns(function_patterns,
                                                &getContext());
  mlir::scf::IfOp::getCanonicalizationPatterns(function_patterns,
                                               &getContext());
  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
          getOperation(), std::move(function_patterns)))) {
    signalPassFailure();
  }

  getOperation()->walk([this](mlir::LLVM::LoadOp load) {
    mlir::Value addr = load.getAddr();
    if (auto gep = load.getAddr().getDefiningOp<mlir::LLVM::GEPOp>()) {
      addr = gep.getBase();
    }
    if (addr.getDefiningOp<mlir::LLVM::AddrSpaceCastOp>()) {
      // Shared memory - no need to annotate anything.
      return;
    }
    if (auto base = mlir::dyn_cast<mlir::BlockArgument>(addr)) {
      if (auto func = mlir::dyn_cast<mlir::func::FuncOp>(
              base.getOwner()->getParentOp())) {
        if (func.getArgAttr(base.getArgNumber(), "xla.invariant")) {
          load.setInvariant(true);
        }
        return;
      }
    }
    load.emitOpError("load op address is not (a GEP of) a function argument");
    signalPassFailure();
  });
}

}  // namespace

std::unique_ptr<::mlir::Pass> CreateLowerTensorsPass() {
  return std::make_unique<LowerTensorsPass>();
}

}  // namespace gpu
}  // namespace xla
