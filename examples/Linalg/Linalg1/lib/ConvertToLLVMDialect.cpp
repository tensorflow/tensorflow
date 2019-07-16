//===- ConvertToLLVMDialect.cpp - conversion from Linalg to LLVM dialect --===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/Conversion/ControlFlowToCFG/ConvertControlFlowToCFG.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LowerAffine.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

#include "linalg1/Common.h"
#include "linalg1/ConvertToLLVMDialect.h"
#include "linalg1/LLVMIntrinsics.h"
#include "linalg1/Ops.h"
#include "linalg1/Passes.h"

using namespace mlir;

// Convert the given type to the LLVM IR Dialect type.  The following
// conversions are supported:
//   - an Index type is converted into an LLVM integer type with pointer
//     bitwidth (analogous to intptr_t in C);
//   - an Integer type is converted into an LLVM integer type of the same width;
//   - an F32 type is converted into an LLVM float type
//   - a Range or View is converted into an LLVM structure type containing the
//     respective dynamic values.
Type linalg::convertLinalgType(Type t) {
  auto *context = t.getContext();
  auto *dialect = context->getRegisteredDialect<LLVM::LLVMDialect>();

  // Simple conversions.
  if (t.isa<IndexType>()) {
    int width = dialect->getLLVMModule().getDataLayout().getPointerSizeInBits();
    return LLVM::LLVMType::getIntNTy(dialect, width);
  }
  if (auto intTy = t.dyn_cast<IntegerType>())
    return LLVM::LLVMType::getIntNTy(dialect, intTy.getWidth());
  if (t.isF32())
    return LLVM::LLVMType::getFloatTy(dialect);
  if (t.isF64())
    return LLVM::LLVMType::getDoubleTy(dialect);

  // Range descriptor contains the range bounds and the step as 64-bit integers.
  //
  // struct {
  //   int64_t min;
  //   int64_t max;
  //   int64_t step;
  // };
  if (auto rangeTy = t.dyn_cast<linalg::RangeType>()) {
    auto int64Ty = LLVM::LLVMType::getInt64Ty(dialect);
    return LLVM::LLVMType::getStructTy(int64Ty, int64Ty, int64Ty);
  }

  // View descriptor contains the pointer to the data buffer, followed by a
  // 64-bit integer containing the distance between the beginning of the buffer
  // and the first element to be accessed through the view, followed by two
  // arrays, each containing as many 64-bit integers as the rank of the View.
  // The first array represents the size, in number of original elements, of the
  // view along the given dimension.  When taking the view, the size is the
  // difference between the upper and the lower bound of the range.  The second
  // array represents the "stride" (in tensor abstraction sense), i.e. the
  // number of consecutive elements of the underlying buffer that separate two
  // consecutive elements addressable through the view along the given
  // dimension.  When taking the view, the strides are constructed as products
  // of the original sizes along the trailing dimensions, multiplied by the view
  // step.  For example, a view of a MxN memref with ranges {0:M:1}, {0:N:1},
  // i.e. the view of a complete memref, will have strides N and 1.  A view with
  // ranges {0:M:2}, {0:N:3} will have strides 2*N and 3.
  //
  // template <typename Elem, size_t Rank>
  // struct {
  //   Elem *ptr;
  //   int64_t offset;
  //   int64_t sizes[Rank];
  //   int64_t strides[Rank];
  // };
  if (auto viewTy = t.dyn_cast<linalg::ViewType>()) {
    auto elemTy = linalg::convertLinalgType(viewTy.getElementType())
                      .cast<LLVM::LLVMType>()
                      .getPointerTo();
    auto int64Ty = LLVM::LLVMType::getInt64Ty(dialect);
    auto arrayTy = LLVM::LLVMType::getArrayTy(int64Ty, viewTy.getRank());
    return LLVM::LLVMType::getStructTy(elemTy, int64Ty, arrayTy, arrayTy);
  }

  // All other types are kept as is.
  return t;
}

// Create an array attribute containing integer attributes with values provided
// in `position`.
static ArrayAttr makePositionAttr(OpBuilder &builder, ArrayRef<int> position) {
  SmallVector<Attribute, 4> attrs;
  attrs.reserve(position.size());
  for (auto p : position)
    attrs.push_back(builder.getI64IntegerAttr(p));
  return builder.getArrayAttr(attrs);
}

// RangeOp creates a new range descriptor.
class RangeOpConversion : public ConversionPattern {
public:
  explicit RangeOpConversion(MLIRContext *context)
      : ConversionPattern(linalg::RangeOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                                     PatternRewriter &rewriter) const override {
    auto rangeOp = cast<linalg::RangeOp>(op);
    auto rangeDescriptorType =
        linalg::convertLinalgType(rangeOp.getResult()->getType());

    using namespace intrinsics;
    edsc::ScopedContext context(rewriter, op->getLoc());

    // Fill in an aggregate value of the descriptor.
    Value *rangeDescriptor = undef(rangeDescriptorType);
    rangeDescriptor = insertvalue(rangeDescriptorType, rangeDescriptor,
                                  operands[0], makePositionAttr(rewriter, 0));
    rangeDescriptor = insertvalue(rangeDescriptorType, rangeDescriptor,
                                  operands[1], makePositionAttr(rewriter, 1));
    rangeDescriptor = insertvalue(rangeDescriptorType, rangeDescriptor,
                                  operands[2], makePositionAttr(rewriter, 2));
    rewriter.replaceOp(op, rangeDescriptor);
    return matchSuccess();
  }
};

class ViewOpConversion : public ConversionPattern {
public:
  explicit ViewOpConversion(MLIRContext *context)
      : ConversionPattern(linalg::ViewOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                                     PatternRewriter &rewriter) const override {
    auto viewOp = cast<linalg::ViewOp>(op);
    auto viewDescriptorType = linalg::convertLinalgType(viewOp.getViewType());
    auto memrefType =
        viewOp.getSupportingMemRef()->getType().cast<MemRefType>();
    auto int64Ty = linalg::convertLinalgType(rewriter.getIntegerType(64));

    // Helper function to create an integer array attribute out of a list of
    // values.
    auto pos = [&rewriter](ArrayRef<int> values) {
      return makePositionAttr(rewriter, values);
    };

    // Helper function to emit an LLVMIR Dialect 64-bit integer constant given
    // its value.
    auto i64cst = [&rewriter, int64Ty](int64_t value) {
      return intrinsics::constant(
          int64Ty, IntegerAttr::get(rewriter.getIndexType(), value));
    };

    // Helper function to obtain the size of the given `memref` along the
    // dimension `dim`.  For static dimensions, emits a constant; for dynamic
    // dimensions, extracts the size from the memref descriptor.
    auto memrefSize = [int64Ty, pos, i64cst](MemRefType type, Value *memref,
                                             int dim) -> Value * {
      assert(dim < type.getRank());
      if (type.getShape()[dim] != -1) {
        return i64cst(type.getShape()[dim]);
      }
      int dynamicDimPos = 0;
      for (int i = 0; i < dim; ++i)
        if (type.getShape()[i] == -1)
          ++dynamicDimPos;
      return intrinsics::extractvalue(int64Ty, memref, pos(1 + dynamicDimPos));
    };

    // Helper function to obtain the data pointer of the given `memref`.
    auto memrefPtr = [pos](MemRefType type, Value *memref) -> Value * {
      if (type.hasStaticShape())
        return memref;

      auto elementTy = linalg::convertLinalgType(type.getElementType())
                           .cast<LLVM::LLVMType>()
                           .getPointerTo();
      return intrinsics::extractvalue(elementTy, memref, pos(0));
    };

    using namespace intrinsics;
    edsc::ScopedContext context(rewriter, op->getLoc());

    // Declare the view descriptor.
    Value *viewDescriptor = undef(viewDescriptorType);
    // Insert the data pointer.
    Value *bufferPtr = memrefPtr(memrefType, operands[0]);
    viewDescriptor =
        insertvalue(viewDescriptorType, viewDescriptor, bufferPtr, pos(0));

    // Collect all memref sizes but the first, which are needed for further
    // computation.
    SmallVector<Value *, 4> trueSizes(memrefType.getRank());
    for (int i = 1, e = memrefType.getRank(); i < e; ++i) {
      trueSizes[i] = memrefSize(memrefType, operands[0], i);
    }

    // Compute all strides of the memref.
    SmallVector<Value *, 4> trueStrides(memrefType.getRank());
    if (viewOp.getRank() != 0)
      trueStrides[memrefType.getRank() - 1] = i64cst(1);
    for (int i = memrefType.getRank() - 2; i >= 0; --i)
      trueStrides[i] = mul(trueStrides[i + 1], trueSizes[i + 1]);

    // Compute and insert the base offset.
    Value *baseOffset = i64cst(0);
    for (int j = 0, e = memrefType.getRank(); j < e; ++j) {
      Value *indexing = operands[1 + j];
      Value *min = viewOp.getIndexing(j)->getType().isa<linalg::RangeType>()
                       ? (Value *)extractvalue(int64Ty, indexing, pos(0))
                       : indexing;
      Value *product = mul(min, trueStrides[j]);
      baseOffset = add(baseOffset, product);
    }
    viewDescriptor =
        insertvalue(viewDescriptorType, viewDescriptor, baseOffset, pos(1));

    // Compute and insert view sizes (max - min along the range).  Skip the
    // non-range operands as they will be projected away from the view.
    int i = 0;
    for (Value *index : viewOp.getIndexings()) {
      if (!index->getType().isa<linalg::RangeType>())
        continue;

      Value *rangeDescriptor = operands[1 + i];
      Value *min = extractvalue(int64Ty, rangeDescriptor, pos(0));
      Value *max = extractvalue(int64Ty, rangeDescriptor, pos(1));
      Value *size = sub(max, min);

      viewDescriptor =
          insertvalue(viewDescriptorType, viewDescriptor, size, pos({2, i}));
      ++i;
    }

    // Compute and insert view strides.  Step over the strides that correspond
    // to non-range operands as they are projected away from the view.
    i = 0;
    for (int j = 0, e = trueStrides.size(); j < e; ++j) {
      if (!viewOp.getIndexing(j)->getType().isa<linalg::RangeType>())
        continue;
      Value *step = extractvalue(int64Ty, operands[1 + j], pos(2));
      Value *stride = mul(trueStrides[j], step);
      viewDescriptor =
          insertvalue(viewDescriptorType, viewDescriptor, stride, pos({3, i}));
      ++i;
    }

    rewriter.replaceOp(op, viewDescriptor);
    return matchSuccess();
  }
};

class SliceOpConversion : public ConversionPattern {
public:
  explicit SliceOpConversion(MLIRContext *context)
      : ConversionPattern(linalg::SliceOp::getOperationName(), 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                                     PatternRewriter &rewriter) const override {
    auto sliceOp = cast<linalg::SliceOp>(op);
    auto newViewDescriptorType =
        linalg::convertLinalgType(sliceOp.getViewType());
    auto elementType = linalg::convertLinalgType(sliceOp.getElementType())
                           .cast<LLVM::LLVMType>()
                           .getPointerTo();
    auto int64Ty = linalg::convertLinalgType(rewriter.getIntegerType(64));

    auto pos = [&rewriter](ArrayRef<int> values) {
      return makePositionAttr(rewriter, values);
    };

    // First operand to `slice` is the old view descriptor.
    Value *oldViewDescriptor = operands[0];

    // Properties of the slice.
    bool isRankDecreasing = sliceOp.isRankDecreasing();
    int dim = sliceOp.getSlicingDim();
    assert(isRankDecreasing ^
           sliceOp.getIndexing()->getType().isa<linalg::RangeType>());

    // Declare the descriptor of the new view.
    using namespace intrinsics;
    edsc::ScopedContext context(rewriter, op->getLoc());
    Value *newViewDescriptor = undef(newViewDescriptorType);

    // Copy the buffer pointer from the old descriptor to the new one.
    Value *buffer = extractvalue(elementType, oldViewDescriptor, pos(0));
    newViewDescriptor =
        insertvalue(newViewDescriptorType, newViewDescriptor, buffer, pos(0));

    // Update the base offset:
    //   base_offset' = base_offset + min_d * stride_d
    // where d is the dimension being sliced, min_d is the minimum value of the
    // range (in case of a single-value slice, that value), stride_d is the
    // stride along this dimension.
    Value *baseOffset = extractvalue(int64Ty, oldViewDescriptor, pos(1));
    Value *slicingValue = operands[1];
    // If `slice` is not rank-decreasing, we need to extract the "min" value
    // from the range descriptor.  Otherwise, we take the value directly.
    Value *min = !isRankDecreasing
                     ? (Value *)extractvalue(int64Ty, slicingValue, pos(0))
                     : slicingValue;
    Value *stride = extractvalue(int64Ty, oldViewDescriptor, pos({3, dim}));
    baseOffset = add(baseOffset, mul(min, stride));
    newViewDescriptor = insertvalue(newViewDescriptorType, newViewDescriptor,
                                    baseOffset, pos(1));

    // Copy the sizes and strides into the new descriptor, updating or dropping
    // the affected dimension.  If the `slice` is rank-decreasing, the resulting
    // view will no longer one of the dimensions, its size and stride become
    // unnecessary and can be dropped.  Otherwise, the size of the affected
    // updated to the size of the range and its stride is multiplied with the
    // step of the range.
    for (int i = 0, e = sliceOp.getRank(); i < e; ++i) {
      int originalPos = (isRankDecreasing && i >= dim) ? i + 1 : i;
      Value *size;
      Value *stride;
      if (!isRankDecreasing && i == dim) {
        Value *upper = extractvalue(int64Ty, slicingValue, pos(1));
        Value *lower = extractvalue(int64Ty, slicingValue, pos(0));
        size = sub(upper, lower);

        Value *previousStride =
            extractvalue(int64Ty, oldViewDescriptor, pos({3, originalPos}));
        Value *step = extractvalue(int64Ty, slicingValue, pos(2));
        stride = mul(previousStride, step);
      } else {
        size = extractvalue(int64Ty, oldViewDescriptor, pos({2, originalPos}));
        stride =
            extractvalue(int64Ty, oldViewDescriptor, pos({3, originalPos}));
      }
      newViewDescriptor = insertvalue(newViewDescriptorType, newViewDescriptor,
                                      size, pos({2, i}));
      newViewDescriptor = insertvalue(newViewDescriptorType, newViewDescriptor,
                                      stride, pos({3, i}));
    }

    rewriter.replaceOp(op, newViewDescriptor);
    return matchSuccess();
  }
};

// When converting the "some_consumer" operation, don't emit anything and
// effectively drop it.
class DropConsumer : public ConversionPattern {
public:
  explicit DropConsumer(MLIRContext *context)
      : ConversionPattern("some_consumer", 1, context) {}

  PatternMatchResult matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                                     PatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, llvm::None);
    return matchSuccess();
  }
};

void linalg::populateLinalg1ToLLVMConversionPatterns(
    mlir::OwningRewritePatternList &patterns, mlir::MLIRContext *context) {
  RewriteListBuilder<DropConsumer, RangeOpConversion, SliceOpConversion,
                     ViewOpConversion>::build(patterns, context);
}

namespace {
/// A type conversion class that converts Linalg and Std types to LLVM.
struct LinalgTypeConverter : public LLVMTypeConverter {
  using LLVMTypeConverter::LLVMTypeConverter;

  // This gets called for block and region arguments, and attributes.
  Type convertType(Type t) override {
    if (auto result = LLVMTypeConverter::convertType(t))
      return result;
    return linalg::convertLinalgType(t);
  }
};
} // end anonymous namespace

LogicalResult linalg::convertToLLVM(mlir::ModuleOp module) {
  // Convert Linalg ops to the LLVM IR dialect using the converter defined
  // above.
  LinalgTypeConverter converter(module.getContext());
  OwningRewritePatternList patterns;
  populateAffineToStdConversionPatterns(patterns, module.getContext());
  populateLoopToStdConversionPatterns(patterns, module.getContext());
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateLinalg1ToLLVMConversionPatterns(patterns, module.getContext());

  ConversionTarget target(*module.getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  return applyFullConversion(module, target, converter, std::move(patterns));
}

namespace {
struct LowerLinalgToLLVMPass : public ModulePass<LowerLinalgToLLVMPass> {
  void runOnModule() {
    if (failed(linalg::convertToLLVM(getModule())))
      signalPassFailure();
  }
};
} // namespace

ModulePassBase *linalg::createLowerLinalgToLLVMPass() {
  return new LowerLinalgToLLVMPass();
}

static PassRegistration<LowerLinalgToLLVMPass>
    pass("lower-linalg-to-llvm",
         "Lower the operations from the linalg dialect into the LLVM dialect");
