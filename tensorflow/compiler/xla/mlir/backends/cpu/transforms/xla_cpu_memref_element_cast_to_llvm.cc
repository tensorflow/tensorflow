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

#include <memory>
#include <utility>

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"  // from @llvm-project
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir/xla_cpu/ir/xla_cpu.h"

namespace xla {
namespace cpu {
namespace {

#define GEN_PASS_DEF_CONVERTXLACPUMEMREFELEMENTCASTTOLLVMPASS
#include "tensorflow/compiler/xla/mlir/backends/cpu/transforms/passes.h.inc"

using namespace mlir;  // NOLINT

struct MemRefElementCastOpLowering
    : public ConvertOpToLLVMPattern<xla_cpu::MemRefElementCastOp> {
  using ConvertOpToLLVMPattern<
      xla_cpu::MemRefElementCastOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      xla_cpu::MemRefElementCastOp cast_op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto target_memref_ty = cast_op.getDst().getType().cast<MemRefType>();

    LLVMTypeConverter type_converter = *getTypeConverter();
    auto target_desc_ty = type_converter.convertType(target_memref_ty)
                              .dyn_cast_or_null<LLVM::LLVMStructType>();
    if (!target_desc_ty) {
      return failure();
    }

    // Unpack the descriptor into the list of its fields.
    Location loc = cast_op.getLoc();
    Type src_type = cast_op.getSrc().getType();

    SmallVector<Value> desc_fields;
    MemRefDescriptor::unpack(rewriter, loc, adaptor.getSrc(),
                             src_type.cast<MemRefType>(), desc_fields);

    // Bitcast allocated and aligned pointers.
    auto dst_elem_ty =
        typeConverter->convertType(cast_op.getType().getElementType());
    auto dst_elem_ptr_ty = LLVM::LLVMPointerType::get(
        dst_elem_ty, cast_op.getType().getMemorySpaceAsInt());
    desc_fields[0] =
        rewriter.create<LLVM::BitcastOp>(loc, dst_elem_ptr_ty, desc_fields[0]);
    desc_fields[1] =
        rewriter.create<LLVM::BitcastOp>(loc, dst_elem_ptr_ty, desc_fields[1]);

    // Create descriptor.
    auto dst_desc = MemRefDescriptor::pack(rewriter, loc, type_converter,
                                           cast_op.getType(), desc_fields);
    rewriter.replaceOp(cast_op, {dst_desc});
    return success();
  }
};

struct ConvertXlaCpuMemRefElementCastToLLVMPass
    : public impl::ConvertXlaCpuMemRefElementCastToLLVMPassBase<
          ConvertXlaCpuMemRefElementCastToLLVMPass> {
  ConvertXlaCpuMemRefElementCastToLLVMPass() = default;

  void runOnOperation() override {
    Operation *op = getOperation();
    const auto &data_layout_analysis = getAnalysis<DataLayoutAnalysis>();
    LowerToLLVMOptions options(&getContext(),
                               data_layout_analysis.getAtOrAbove(op));
    // TODO(b/267828330): Migrate to opaque pointers.
    options.useOpaquePointers = false;

    LLVMTypeConverter type_converter(&getContext(), options,
                                     &data_layout_analysis);
    RewritePatternSet patterns(&getContext());
    patterns.add<MemRefElementCastOpLowering>(type_converter);

    LLVMConversionTarget target(getContext());
    target.addLegalOp<func::FuncOp>();
    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createConvertXlaCpuMemRefElementCastToLLVMPass() {
  return std::make_unique<ConvertXlaCpuMemRefElementCastToLLVMPass>();
}

}  // namespace cpu
}  // namespace xla
