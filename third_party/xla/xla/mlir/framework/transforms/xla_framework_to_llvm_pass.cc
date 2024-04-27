/* Copyright 2022 The OpenXLA Authors.

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

#include <array>
#include <memory>
#include <stdexcept>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Constants.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"  // from @llvm-project
#include "mlir/Conversion/LLVMCommon/Pattern.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/IRMapping.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "xla/mlir/framework/ir/xla_framework.h"
#include "xla/mlir/framework/transforms/passes.h"

namespace mlir {
namespace xla_framework {
namespace {

// Create a memref descriptor given a pointer and memref type information.
struct XLABufferToMemOpConversion
    : public ConvertOpToLLVMPattern<::mlir::xla_framework::XLABufferToMemOp> {
  using ConvertOpToLLVMPattern<
      ::mlir::xla_framework::XLABufferToMemOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      ::mlir::xla_framework::XLABufferToMemOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto mem_ref_type = op.getType();

    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value size_bytes;
    this->getMemRefDescriptorSizes(loc, mem_ref_type, ValueRange(), rewriter,
                                   sizes, strides, size_bytes);

    Value ptr = adaptor.getBuffer();
    Value result = this->createMemRefDescriptor(loc, mem_ref_type, ptr, ptr,
                                                sizes, strides, rewriter);
    rewriter.replaceOp(op, {result});
    return success();
  }
};

// Convert to the expected function signature and offer unwrapping for each of
// the original arguments.
struct BarePtrFuncOpConversion : public ConvertOpToLLVMPattern<func::FuncOp> {
  using ConvertOpToLLVMPattern<func::FuncOp>::ConvertOpToLLVMPattern;

  Value LoadValue(ConversionPatternRewriter &rewriter, Location loc,
                  Value pointer, Value index) const {
    auto ptr = LLVM::LLVMPointerType::get(rewriter.getContext());
    return rewriter.create<LLVM::LoadOp>(
        loc, ptr, rewriter.create<LLVM::GEPOp>(loc, ptr, ptr, pointer, index));
  }

  mlir::func::FuncOp convertFuncOpToLLVMFuncOp(
      func::FuncOp funcOp, ConversionPatternRewriter &rewriter) const {
    auto loc = funcOp.getLoc();

    // This signature is predetermined by
    // tensorflow/compiler/xla/service/cpu/ir_function.cc
    //
    // This only works for the global function version that tf.compile uses.
    // Local functions will only be called by MLIR compiled code, so we can
    // ignore them.
    auto ptr = LLVM::LLVMPointerType::get(rewriter.getContext());
    std::array<Type, 6> arg_types = {ptr, ptr, ptr, ptr, ptr, ptr};
    auto llvm_type =
        mlir::FunctionType::get(rewriter.getContext(), arg_types, {});

    if (!llvm_type) return nullptr;

    rewriter.setInsertionPoint(funcOp);
    auto new_func_op = rewriter.create<mlir::func::FuncOp>(
        loc, funcOp.getName(), llvm_type, llvm::SmallVector<NamedAttribute>());
    auto locs = llvm::SmallVector<mlir::Location>(arg_types.size(), loc);
    Block *new_entry =
        rewriter.createBlock(&new_func_op.getBody(), {}, arg_types, locs);

    // This assertion might change but is in place for the current
    // implementation.
    assert(funcOp.getFunctionType().getNumResults() == 0 &&
           "xla_entry function lowered with result values when memrefs should "
           "be caller supplied");

    IRMapping mapping;
    auto num_refs = funcOp.getFunctionType().getNumInputs();
    auto result_index = 0;
    for (unsigned i = 0; i < num_refs; ++i) {
      if (funcOp.getArgAttr(i, "xla_framework.input_mapping")) {
        Value index = rewriter.create<LLVM::ConstantOp>(
            loc, typeConverter->convertType(rewriter.getIntegerType(32)),
            funcOp.getArgAttrOfType<mlir::IntegerAttr>(
                i, "xla_framework.input_mapping"));

        Value ptr = LoadValue(rewriter, loc, new_entry->getArgument(3), index);
        mapping.map(funcOp.front().getArgument(i), ptr);
      } else {
        Value index = rewriter.create<LLVM::ConstantOp>(
            loc, typeConverter->convertType(rewriter.getIntegerType(32)),
            funcOp->getAttrOfType<mlir::IntegerAttr>(
                "xla_framework.result_mapping"));
        Value first_load =
            LoadValue(rewriter, loc, new_entry->getArgument(3), index);

        // Handle multi-value results which are wrapped in a tuple.
        if (funcOp->hasAttr("xla_framework.result_inner_mapping")) {
          auto current_index = result_index++;
          Value inner_index = rewriter.create<LLVM::ConstantOp>(
              loc, typeConverter->convertType(rewriter.getIntegerType(32)),
              rewriter.getI32IntegerAttr(static_cast<int32_t>(
                  mlir::cast<mlir::IntegerAttr>(
                      funcOp
                          ->getAttrOfType<mlir::ArrayAttr>(
                              "xla_framework.result_inner_mapping")
                          .getValue()[current_index])
                      .getValue()
                      .getSExtValue())));

          auto ptr =
              LoadValue(rewriter, loc, new_entry->getArgument(3), inner_index);
          mapping.map(funcOp.front().getArgument(i), ptr);

          auto ptr_type = LLVM::LLVMPointerType::get(rewriter.getContext());
          Value second_index = rewriter.create<LLVM::ConstantOp>(
              loc, typeConverter->convertType(rewriter.getIntegerType(32)),
              rewriter.getI32IntegerAttr(current_index));
          rewriter.create<LLVM::StoreOp>(
              loc, ptr,
              rewriter.create<LLVM::GEPOp>(loc, ptr_type, ptr_type, first_load,
                                           llvm::ArrayRef(second_index)));

        } else {
          // Non tuple outputs can be simply mapped to the first load op.
          mapping.map(funcOp.front().getArgument(i), first_load);
        }
      }
    }

    // Clone the region and handle ReturnOps specially as there will be no
    // return values now.
    for (auto &op : funcOp.front()) {
      if (isa<mlir::func::ReturnOp>(op)) {
        rewriter.create<mlir::func::ReturnOp>(loc, ValueRange());
      } else {
        rewriter.clone(op, mapping);
      }
    }

    return new_func_op;
  }

  LogicalResult matchAndRewrite(
      func::FuncOp funcOp, OpAdaptor,
      ConversionPatternRewriter &rewriter) const override {
    // Only outline functions that are globally available.
    if (!funcOp->hasAttr("xla_entry")) return failure();

    // Store the type of memref-typed arguments before the conversion so that we
    // can promote them to MemRef descriptor at the beginning of the function.
    convertFuncOpToLLVMFuncOp(funcOp, rewriter);

    rewriter.eraseOp(funcOp);
    return success();
  }
};

#define GEN_PASS_DEF_LEGALIZEXLAFRAMEWORKTOLLVM
#include "xla/mlir/framework/transforms/passes.h.inc"

class LegalizeXLAFrameworkToLLVMPass
    : public impl::LegalizeXLAFrameworkToLLVMBase<
          LegalizeXLAFrameworkToLLVMPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, LLVM::LLVMDialect,
                    xla_framework::XLAFrameworkDialect>();
  }

 public:
  explicit LegalizeXLAFrameworkToLLVMPass() {}

  void runOnOperation() override {
    ModuleOp m = getOperation();

    // Populate type conversions.
    MLIRContext *ctx = m.getContext();
    LLVMTypeConverter type_converter(ctx);
    type_converter.addConversion([&](::mlir::xla_framework::BufferType) {
      return LLVM::LLVMPointerType::get(ctx);
    });

    // Populate patterns.
    RewritePatternSet patterns(&getContext());
    patterns.add<XLABufferToMemOpConversion, BarePtrFuncOpConversion>(
        type_converter, 2);
    //  Set target.
    ConversionTarget target(*ctx);
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalDialect<xla_framework::XLAFrameworkDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([](func::FuncOp op) {
      if (llvm::any_of(llvm::concat<const Type>(op.getArgumentTypes(),
                                                op.getResultTypes()),
                       [](Type type) {
                         return mlir::isa<xla_framework::BufferType>(type);
                       }))
        return false;
      return true;
    });

    if (failed(applyFullConversion(m, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateLegalizeXLAFrameworkToLLVMPass() {
  return std::make_unique<LegalizeXLAFrameworkToLLVMPass>();
}

}  // namespace xla_framework
}  // namespace mlir
