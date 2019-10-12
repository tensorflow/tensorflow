//===- LowerToLLVMDialect.cpp - conversion from Linalg to LLVM dialect ----===//
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

#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Intrinsics.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LowerAffine.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;
using namespace mlir::LLVM;
using namespace mlir::linalg;
using namespace mlir::linalg::intrinsics;

using add = ValueBuilder<mlir::LLVM::AddOp>;
using addi = ValueBuilder<mlir::AddIOp>;
using bitcast = ValueBuilder<mlir::LLVM::BitcastOp>;
using cmpi = ValueBuilder<mlir::CmpIOp>;
using constant = ValueBuilder<mlir::LLVM::ConstantOp>;
using extractvalue = ValueBuilder<mlir::LLVM::ExtractValueOp>;
using gep = ValueBuilder<mlir::LLVM::GEPOp>;
using insertvalue = ValueBuilder<mlir::LLVM::InsertValueOp>;
using llvm_call = OperationBuilder<mlir::LLVM::CallOp>;
using llvm_icmp = ValueBuilder<LLVM::ICmpOp>;
using llvm_load = ValueBuilder<LLVM::LoadOp>;
using llvm_store = OperationBuilder<LLVM::StoreOp>;
using llvm_select = ValueBuilder<LLVM::SelectOp>;
using mul = ValueBuilder<mlir::LLVM::MulOp>;
using ptrtoint = ValueBuilder<mlir::LLVM::PtrToIntOp>;
using sub = ValueBuilder<mlir::LLVM::SubOp>;
using llvm_undef = ValueBuilder<mlir::LLVM::UndefOp>;
using urem = ValueBuilder<mlir::LLVM::URemOp>;
using llvm_alloca = ValueBuilder<LLVM::AllocaOp>;
using llvm_return = OperationBuilder<LLVM::ReturnOp>;

template <typename T>
static LLVMType getPtrToElementType(T containerType,
                                    LLVMTypeConverter &lowering) {
  return lowering.convertType(containerType.getElementType())
      .template cast<LLVMType>()
      .getPointerTo();
}

// Convert the given type to the LLVM IR Dialect type.  The following
// conversions are supported:
//   - an Index type is converted into an LLVM integer type with pointer
//     bitwidth (analogous to intptr_t in C);
//   - an Integer type is converted into an LLVM integer type of the same width;
//   - an F32 type is converted into an LLVM float type
//   - a Buffer, Range or View is converted into an LLVM structure type
//     containing the respective dynamic values.
static Type convertLinalgType(Type t, LLVMTypeConverter &lowering) {
  auto *context = t.getContext();
  auto int64Ty = lowering.convertType(IntegerType::get(64, context))
                     .cast<LLVM::LLVMType>();

  // A buffer descriptor contains the pointer to a flat region of storage and
  // the size of the region.
  //
  // template <typename Elem, size_t Rank>
  // struct {
  //   void *baseAlloc;
  //   Elem *ptr;
  //   int64_t size;
  // };
  if (auto bufferType = t.dyn_cast<BufferType>()) {
    auto voidPtrTy = LLVMType::getInt8Ty(lowering.getDialect()).getPointerTo();
    auto ptrTy = getPtrToElementType(bufferType, lowering);
    return LLVMType::getStructTy(voidPtrTy, ptrTy, int64Ty);
  }

  // Range descriptor contains the range bounds and the step as 64-bit integers.
  //
  // struct {
  //   int64_t min;
  //   int64_t max;
  //   int64_t step;
  // };
  if (t.isa<RangeType>())
    return LLVMType::getStructTy(int64Ty, int64Ty, int64Ty);

  return Type();
}

static constexpr int kBasePtrPosInBuffer = 0;
static constexpr int kPtrPosInBuffer = 1;
static constexpr int kSizePosInBuffer = 2;
static constexpr int kPtrPosInView = 0;
static constexpr int kOffsetPosInView = 1;
static constexpr int kSizePosInView = 2;
static constexpr int kStridePosInView = 3;

namespace {
/// Factor out the common information for all view conversions:
///   1. common types in (standard and LLVM dialects)
///   2. `pos` method
///   3. view descriptor construction `desc`.
class BaseViewConversionHelper {
public:
  BaseViewConversionHelper(Location loc, MemRefType memRefType,
                           ConversionPatternRewriter &rewriter,
                           LLVMTypeConverter &lowering)
      : zeroDMemRef(memRefType.getRank() == 0),
        elementTy(getPtrToElementType(memRefType, lowering)),
        int64Ty(
            lowering.convertType(rewriter.getIntegerType(64)).cast<LLVMType>()),
        desc(nullptr), rewriter(rewriter) {
    assert(isStrided(memRefType) && "expected strided memref type");
    viewDescriptorTy = lowering.convertType(memRefType).cast<LLVMType>();
    desc = rewriter.create<LLVM::UndefOp>(loc, viewDescriptorTy);
  }

  ArrayAttr pos(ArrayRef<int64_t> values) const {
    return rewriter.getI64ArrayAttr(values);
  };

  bool zeroDMemRef;
  LLVMType elementTy, int64Ty, viewDescriptorTy;
  Value *desc;
  ConversionPatternRewriter &rewriter;
};
} // namespace

// BufferAllocOp creates a new `!linalg.buffer` value.
class BufferAllocOpConversion : public LLVMOpLowering {
public:
  explicit BufferAllocOpConversion(MLIRContext *context,
                                   LLVMTypeConverter &lowering_)
      : LLVMOpLowering(BufferAllocOp::getOperationName(), context, lowering_) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto indexType = IndexType::get(op->getContext());
    auto voidPtrTy =
        LLVM::LLVMType::getInt8Ty(lowering.getDialect()).getPointerTo();
    auto int64Ty = lowering.convertType(rewriter.getIntegerType(64))
                       .cast<LLVM::LLVMType>();
    // Insert the `malloc` declaration if it is not already present.
    auto module = op->getParentOfType<ModuleOp>();
    auto mallocFunc = module.lookupSymbol<LLVMFuncOp>("malloc");
    if (!mallocFunc) {
      OpBuilder moduleBuilder(op->getParentOfType<ModuleOp>().getBodyRegion());
      mallocFunc = moduleBuilder.create<LLVMFuncOp>(
          rewriter.getUnknownLoc(), "malloc",
          LLVM::LLVMType::getFunctionTy(voidPtrTy, int64Ty,
                                        /*isVarArg=*/false));
    }

    // Get MLIR types for injecting element pointer.
    auto allocOp = cast<BufferAllocOp>(op);
    auto elementType = allocOp.getElementType();
    uint64_t elementSize = 0;
    if (auto vectorType = elementType.dyn_cast<VectorType>())
      elementSize = vectorType.getNumElements() *
                    llvm::divideCeil(vectorType.getElementTypeBitWidth(), 8);
    else
      elementSize = llvm::divideCeil(elementType.getIntOrFloatBitWidth(), 8);
    auto bufferType = allocOp.getBufferType();
    auto elementPtrType = getPtrToElementType(bufferType, lowering);
    auto bufferDescriptorTy = convertLinalgType(bufferType, lowering);

    // Emit IR for creating a new buffer descriptor with an underlying malloc.
    edsc::ScopedContext context(rewriter, op->getLoc());
    auto constantSize = bufferType.getBufferSize();
    Value *size =
        constantSize
            ? constant(int64Ty, IntegerAttr::get(indexType, *constantSize))
                  .getValue()
            : operands[0];
    Value *allocSize =
        mul(size, constant(int64Ty, IntegerAttr::get(indexType, elementSize)));
    Value *one = nullptr, *align = nullptr;
    if (allocOp.alignment().hasValue()) {
      one = constant(int64Ty, IntegerAttr::get(indexType, 1));
      align =
          constant(int64Ty, rewriter.getIntegerAttr(
                                rewriter.getIndexType(),
                                allocOp.alignment().getValue().getSExtValue()));
      allocSize = sub(add(allocSize, align), one);
    }

    Value *allocated =
        llvm_call(voidPtrTy, rewriter.getSymbolRefAttr(mallocFunc), allocSize)
            .getOperation()
            ->getResult(0);
    Value *data = allocated;
    if (allocOp.alignment().hasValue()) {
      // offset = (align - (ptr % align))% align
      Value *offset =
          urem(sub(align, urem(ptrtoint(int64Ty, allocated), align)), align);
      data = gep(voidPtrTy, allocated, offset);
    }
    data = bitcast(elementPtrType, data);
    Value *desc = llvm_undef(bufferDescriptorTy);
    desc = insertvalue(bufferDescriptorTy, desc, allocated,
                       rewriter.getI64ArrayAttr(kBasePtrPosInBuffer));
    desc = insertvalue(bufferDescriptorTy, desc, data,
                       rewriter.getI64ArrayAttr(kPtrPosInBuffer));
    desc = insertvalue(bufferDescriptorTy, desc, size,
                       rewriter.getI64ArrayAttr(kSizePosInBuffer));
    rewriter.replaceOp(op, desc);
    return matchSuccess();
  }
};

// BufferDeallocOp creates no value.
class BufferDeallocOpConversion : public LLVMOpLowering {
public:
  explicit BufferDeallocOpConversion(MLIRContext *context,
                                     LLVMTypeConverter &lowering_)
      : LLVMOpLowering(BufferDeallocOp::getOperationName(), context,
                       lowering_) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto voidTy = LLVM::LLVMType::getVoidTy(lowering.getDialect());
    auto voidPtrTy =
        LLVM::LLVMType::getInt8Ty(lowering.getDialect()).getPointerTo();
    // Insert the `free` declaration if it is not already present.
    auto module = op->getParentOfType<ModuleOp>();
    auto freeFunc = module.lookupSymbol<LLVMFuncOp>("free");
    if (!freeFunc) {
      OpBuilder moduleBuilder(op->getParentOfType<ModuleOp>().getBodyRegion());
      freeFunc = moduleBuilder.create<LLVMFuncOp>(
          rewriter.getUnknownLoc(), "free",
          LLVM::LLVMType::getFunctionTy(voidTy, voidPtrTy,
                                        /*isVarArg=*/false));
    }

    // Emit MLIR for buffer_dealloc.
    BufferDeallocOpOperandAdaptor adaptor(operands);
    edsc::ScopedContext context(rewriter, op->getLoc());
    Value *base = extractvalue(voidPtrTy, adaptor.buffer(),
                               rewriter.getI64ArrayAttr(kBasePtrPosInBuffer));
    llvm_call(ArrayRef<Type>(), rewriter.getSymbolRefAttr(freeFunc), base);
    rewriter.replaceOp(op, llvm::None);
    return matchSuccess();
  }
};

// BufferSizeOp creates a new `index` value.
class BufferSizeOpConversion : public LLVMOpLowering {
public:
  BufferSizeOpConversion(MLIRContext *context, LLVMTypeConverter &lowering_)
      : LLVMOpLowering(BufferSizeOp::getOperationName(), context, lowering_) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto int64Ty = lowering.convertType(rewriter.getIntegerType(64));
    edsc::ScopedContext context(rewriter, op->getLoc());
    BufferSizeOpOperandAdaptor adaptor(operands);
    rewriter.replaceOp(
        op, {extractvalue(int64Ty, adaptor.buffer(),
                          rewriter.getI64ArrayAttr(kSizePosInBuffer))});
    return matchSuccess();
  }
};

// RangeOp creates a new range descriptor.
class RangeOpConversion : public LLVMOpLowering {
public:
  explicit RangeOpConversion(MLIRContext *context, LLVMTypeConverter &lowering_)
      : LLVMOpLowering(RangeOp::getOperationName(), context, lowering_) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto rangeOp = cast<RangeOp>(op);
    auto rangeDescriptorTy =
        convertLinalgType(rangeOp.getResult()->getType(), lowering);

    edsc::ScopedContext context(rewriter, op->getLoc());

    // Fill in an aggregate value of the descriptor.
    RangeOpOperandAdaptor adaptor(operands);
    Value *desc = llvm_undef(rangeDescriptorTy);
    desc = insertvalue(desc, adaptor.min(), rewriter.getI64ArrayAttr(0));
    desc = insertvalue(desc, adaptor.max(), rewriter.getI64ArrayAttr(1));
    desc = insertvalue(desc, adaptor.step(), rewriter.getI64ArrayAttr(2));
    rewriter.replaceOp(op, desc);
    return matchSuccess();
  }
};

/// Conversion pattern that transforms a linalg.slice op into:
///   1. A function entry `alloca` operation to allocate a ViewDescriptor.
///   2. A load of the ViewDescriptor from the pointer allocated in 1.
///   3. Updates to the ViewDescriptor to introduce the data ptr, offset, size
///      and stride corresponding to the region of memory within the bounds of
///      the parent view.
///   4. A store of the resulting ViewDescriptor to the alloca'ed pointer.
/// The linalg.slice op is replaced by the alloca'ed pointer.
class SliceOpConversion : public LLVMOpLowering {
public:
  explicit SliceOpConversion(MLIRContext *context, LLVMTypeConverter &lowering_)
      : LLVMOpLowering(SliceOp::getOperationName(), context, lowering_) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SliceOpOperandAdaptor adaptor(operands);
    Value *baseDesc = adaptor.view();

    auto sliceOp = cast<SliceOp>(op);
    auto memRefType = sliceOp.getBaseViewType();

    BaseViewConversionHelper helper(op->getLoc(), sliceOp.getViewType(),
                                    rewriter, lowering);
    LLVMType elementTy = helper.elementTy, int64Ty = helper.int64Ty;
    Value *desc = helper.desc;

    edsc::ScopedContext context(rewriter, op->getLoc());

    // TODO(ntv): extract sizes and emit asserts.
    SmallVector<Value *, 4> strides(memRefType.getRank());
    for (int i = 0, e = memRefType.getRank(); i < e; ++i)
      strides[i] =
          extractvalue(int64Ty, baseDesc, helper.pos({kStridePosInView, i}));

    // Compute base offset.
    Value *baseOffset =
        extractvalue(int64Ty, baseDesc, helper.pos(kOffsetPosInView));
    for (int i = 0, e = memRefType.getRank(); i < e; ++i) {
      Value *indexing = adaptor.indexings()[i];
      Value *min = indexing;
      if (sliceOp.indexing(i)->getType().isa<RangeType>())
        min = extractvalue(int64Ty, indexing, helper.pos(0));
      baseOffset = add(baseOffset, mul(min, strides[i]));
    }

    // Insert base pointer.
    auto ptrPos = helper.pos(kPtrPosInView);
    desc = insertvalue(desc, extractvalue(elementTy, baseDesc, ptrPos), ptrPos);

    // Insert base offset.
    desc = insertvalue(desc, baseOffset, helper.pos(kOffsetPosInView));

    // Corner case, no sizes or strides: early return the descriptor.
    if (helper.zeroDMemRef)
      return rewriter.replaceOp(op, desc), matchSuccess();

    Value *zero =
        constant(int64Ty, rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    // Compute and insert view sizes (max - min along the range) and strides.
    // Skip the non-range operands as they will be projected away from the view.
    int numNewDims = 0;
    for (auto en : llvm::enumerate(sliceOp.indexings())) {
      Value *indexing = en.value();
      if (indexing->getType().isa<RangeType>()) {
        int rank = en.index();
        Value *rangeDescriptor = adaptor.indexings()[rank];
        Value *min = extractvalue(int64Ty, rangeDescriptor, helper.pos(0));
        Value *max = extractvalue(int64Ty, rangeDescriptor, helper.pos(1));
        Value *step = extractvalue(int64Ty, rangeDescriptor, helper.pos(2));
        Value *baseSize =
            extractvalue(int64Ty, baseDesc, helper.pos({kSizePosInView, rank}));
        // Bound upper by base view upper bound.
        max = llvm_select(llvm_icmp(ICmpPredicate::slt, max, baseSize), max,
                          baseSize);
        Value *size = sub(max, min);
        // Bound lower by zero.
        size =
            llvm_select(llvm_icmp(ICmpPredicate::slt, size, zero), zero, size);
        Value *stride = mul(strides[rank], step);
        desc =
            insertvalue(desc, size, helper.pos({kSizePosInView, numNewDims}));
        desc = insertvalue(desc, stride,
                           helper.pos({kStridePosInView, numNewDims}));
        ++numNewDims;
      }
    }

    rewriter.replaceOp(op, desc);
    return matchSuccess();
  }
};

/// Conversion pattern that transforms a linalg.transpose op into:
///   1. A function entry `alloca` operation to allocate a ViewDescriptor.
///   2. A load of the ViewDescriptor from the pointer allocated in 1.
///   3. Updates to the ViewDescriptor to introduce the data ptr, offset, size
///      and stride. Size and stride are permutations of the original values.
///   4. A store of the resulting ViewDescriptor to the alloca'ed pointer.
/// The linalg.transpose op is replaced by the alloca'ed pointer.
class TransposeOpConversion : public LLVMOpLowering {
public:
  explicit TransposeOpConversion(MLIRContext *context,
                                 LLVMTypeConverter &lowering_)
      : LLVMOpLowering(TransposeOp::getOperationName(), context, lowering_) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Initialize the common boilerplate and alloca at the top of the FuncOp.
    TransposeOpOperandAdaptor adaptor(operands);
    Value *baseDesc = adaptor.view();

    auto tranposeOp = cast<TransposeOp>(op);
    // No permutation, early exit.
    if (tranposeOp.permutation().isIdentity())
      return rewriter.replaceOp(op, baseDesc), matchSuccess();

    BaseViewConversionHelper helper(op->getLoc(), tranposeOp.getViewType(),
                                    rewriter, lowering);
    LLVMType elementTy = helper.elementTy, int64Ty = helper.int64Ty;
    Value *desc = helper.desc;

    edsc::ScopedContext context(rewriter, op->getLoc());
    // Copy the base pointer from the old descriptor to the new one.
    ArrayAttr ptrPos = helper.pos(kPtrPosInView);
    desc = insertvalue(desc, extractvalue(elementTy, baseDesc, ptrPos), ptrPos);

    // Copy the offset pointer from the old descriptor to the new one.
    ArrayAttr offPos = helper.pos(kOffsetPosInView);
    desc = insertvalue(desc, extractvalue(int64Ty, baseDesc, offPos), offPos);

    // Iterate over the dimensions and apply size/stride permutation.
    for (auto en : llvm::enumerate(tranposeOp.permutation().getResults())) {
      int sourcePos = en.index();
      int targetPos = en.value().cast<AffineDimExpr>().getPosition();
      Value *size = extractvalue(int64Ty, baseDesc,
                                 helper.pos({kSizePosInView, sourcePos}));
      desc = insertvalue(desc, size, helper.pos({kSizePosInView, targetPos}));
      Value *stride = extractvalue(int64Ty, baseDesc,
                                   helper.pos({kStridePosInView, sourcePos}));
      desc =
          insertvalue(desc, stride, helper.pos({kStridePosInView, targetPos}));
    }

    rewriter.replaceOp(op, desc);
    return matchSuccess();
  }
};

/// Conversion pattern that transforms a linalg.view op into:
///   1. A function entry `alloca` operation to allocate a ViewDescriptor.
///   2. A load of the ViewDescriptor from the pointer allocated in 1.
///   3. Updates to the ViewDescriptor to introduce the data ptr, offset, size
///      and stride.
///   4. A store of the resulting ViewDescriptor to the alloca'ed pointer.
/// The linalg.view op is replaced by the alloca'ed pointer.
class ViewOpConversion : public LLVMOpLowering {
public:
  explicit ViewOpConversion(MLIRContext *context, LLVMTypeConverter &lowering_)
      : LLVMOpLowering(ViewOp::getOperationName(), context, lowering_) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    ViewOpOperandAdaptor adaptor(operands);

    auto viewOp = cast<ViewOp>(op);
    BaseViewConversionHelper helper(op->getLoc(), viewOp.getViewType(),
                                    rewriter, lowering);
    LLVMType elementTy = helper.elementTy, int64Ty = helper.int64Ty;
    Value *desc = helper.desc;

    Value *bufferDescriptor = adaptor.buffer();
    auto bufferTy = getPtrToElementType(
        viewOp.buffer()->getType().cast<BufferType>(), lowering);

    edsc::ScopedContext context(rewriter, op->getLoc());

    // Copy the buffer pointer from the old descriptor to the new one.
    Value *bufferAsViewElementType =
        bitcast(elementTy, extractvalue(bufferTy, bufferDescriptor,
                                        helper.pos(kPtrPosInBuffer)));
    desc =
        insertvalue(desc, bufferAsViewElementType, helper.pos(kPtrPosInView));

    // Zero base offset.
    auto indexTy = rewriter.getIndexType();
    Value *baseOffset = constant(int64Ty, IntegerAttr::get(indexTy, 0));
    desc = insertvalue(desc, baseOffset, helper.pos(kOffsetPosInView));

    // Corner case, no sizes or stride: early return the descriptor.
    if (helper.zeroDMemRef) {
      rewriter.replaceOp(op, desc);
      return matchSuccess();
    }

    // Compute and insert view sizes (max - min along the range).
    int numRanges = llvm::size(viewOp.ranges());
    Value *runningStride = constant(int64Ty, IntegerAttr::get(indexTy, 1));
    for (int i = numRanges - 1; i >= 0; --i) {
      // Update stride.
      Value *rangeDescriptor = operands[1 + i];
      Value *step = extractvalue(int64Ty, rangeDescriptor, helper.pos(2));
      Value *stride = mul(runningStride, step);
      desc = insertvalue(desc, stride, helper.pos({kStridePosInView, i}));
      // Update size.
      Value *min = extractvalue(int64Ty, rangeDescriptor, helper.pos(0));
      Value *max = extractvalue(int64Ty, rangeDescriptor, helper.pos(1));
      Value *size = sub(max, min);
      desc = insertvalue(desc, size, helper.pos({kSizePosInView, i}));
      // Update stride for the next dimension.
      if (i > 0)
        runningStride = mul(runningStride, max);
    }

    rewriter.replaceOp(op, desc);
    return matchSuccess();
  }
};

// YieldOp produces and LLVM::ReturnOp.
class YieldOpConversion : public LLVMOpLowering {
public:
  explicit YieldOpConversion(MLIRContext *context, LLVMTypeConverter &lowering_)
      : LLVMOpLowering(YieldOp::getOperationName(), context, lowering_) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, operands);
    return matchSuccess();
  }
};

// Get a SymbolRefAttr containing the library function name for the LinalgOp.
// If the library function does not exist, insert a declaration.
template <typename LinalgOp>
static SymbolRefAttr getLibraryCallSymbolRef(Operation *op,
                                             PatternRewriter &rewriter) {
  auto linalgOp = cast<LinalgOp>(op);
  auto fnName = linalgOp.getLibraryCallName();
  if (fnName.empty()) {
    op->emitWarning("No library call defined for: ") << *op;
    return {};
  }

  // fnName is a dynamic std::String, unique it via a SymbolRefAttr.
  SymbolRefAttr fnNameAttr = rewriter.getSymbolRefAttr(fnName);
  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnName)) {
    return fnNameAttr;
  }

  SmallVector<Type, 4> inputTypes(op->getOperandTypes());
  assert(op->getNumResults() == 0 &&
         "Library call for linalg operation can be generated only for ops that "
         "have void return types");
  auto libFnType = FunctionType::get(inputTypes, {}, rewriter.getContext());

  OpBuilder::InsertionGuard guard(rewriter);
  // Insert before module terminator.
  rewriter.setInsertionPoint(module.getBody(),
                             std::prev(module.getBody()->end()));
  rewriter.create<FuncOp>(op->getLoc(), fnNameAttr.getValue(), libFnType,
                          ArrayRef<NamedAttribute>{});
  return fnNameAttr;
}

namespace {
// The conversion class from Linalg to LLVMIR.
class LinalgTypeConverter : public LLVMTypeConverter {
  using LLVMTypeConverter::LLVMTypeConverter;

public:
  Type convertType(Type t) override {
    if (auto result = LLVMTypeConverter::convertType(t))
      return result;
    return convertLinalgType(t, *this);
  }
};
} // end anonymous namespace

// LinalgOpConversion<LinalgOp> creates a new call to the
// `LinalgOp::getLibraryCallName()` function.
// The implementation of the function can be either in the same module or in an
// externally linked library.
template <typename LinalgOp>
class LinalgOpConversion : public OpRewritePattern<LinalgOp> {
public:
  using OpRewritePattern<LinalgOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(LinalgOp op,
                                     PatternRewriter &rewriter) const override {
    auto libraryCallName = getLibraryCallSymbolRef<LinalgOp>(op, rewriter);
    if (!libraryCallName)
      return this->matchFailure();

    SmallVector<Value *, 4> operands(op.getOperands().begin(),
                                     op.getOperands().end());
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, libraryCallName.getValue(),
                                              ArrayRef<Type>{}, operands);
    return this->matchSuccess();
  }
};

/// Conversion pattern specialization for CopyOp. This kicks in when both input
/// and output permutations are left unspecified or are the identity.
template <> class LinalgOpConversion<CopyOp> : public OpRewritePattern<CopyOp> {
public:
  using OpRewritePattern<CopyOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(CopyOp op,
                                     PatternRewriter &rewriter) const override {
    auto inputPerm = op.inputPermutation();
    if (inputPerm.hasValue() && !inputPerm->isIdentity())
      return matchFailure();
    auto outputPerm = op.outputPermutation();
    if (outputPerm.hasValue() && !outputPerm->isIdentity())
      return matchFailure();

    auto libraryCallName = getLibraryCallSymbolRef<CopyOp>(op, rewriter);
    if (!libraryCallName)
      return matchFailure();

    SmallVector<Value *, 4> operands(op.getOperands().begin(),
                                     op.getOperands().end());
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, libraryCallName.getValue(),
                                              ArrayRef<Type>{}, operands);
    return matchSuccess();
  }
};

/// A non-conversion rewrite pattern kicks in to convert CopyOp with
/// permutations into a sequence of TransposeOp and permutation-free CopyOp.
/// This interplays together with TransposeOpConversion and
/// LinalgConversion<CopyOp> to create a path to the LLVM dialect.
class CopyTransposeConversion : public OpRewritePattern<CopyOp> {
public:
  using OpRewritePattern<CopyOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(CopyOp op,
                                     PatternRewriter &rewriter) const override {
    Value *in = op.input(), *out = op.output();

    // If either inputPerm or outputPerm are non-identities, insert transposes.
    auto inputPerm = op.inputPermutation();
    if (inputPerm.hasValue() && !inputPerm->isIdentity())
      in = rewriter.create<linalg::TransposeOp>(op.getLoc(), in,
                                                AffineMapAttr::get(*inputPerm));
    auto outputPerm = op.outputPermutation();
    if (outputPerm.hasValue() && !outputPerm->isIdentity())
      out = rewriter.create<linalg::TransposeOp>(
          op.getLoc(), out, AffineMapAttr::get(*outputPerm));

    // If nothing was transposed, fail and let the conversion kick in.
    if (in == op.input() && out == op.output())
      return matchFailure();

    rewriter.replaceOpWithNewOp<CopyOp>(op, in, out);
    return matchSuccess();
  }
};

/// A non-conversion rewrite pattern kicks in to convert SubViewOp into RangeOps
/// and SliceOps.
class SubViewOpConversion : public OpRewritePattern<SubViewOp> {
public:
  using OpRewritePattern<SubViewOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(SubViewOp op,
                                     PatternRewriter &rewriter) const override {
    auto *view = op.getView();
    SmallVector<Value *, 8> ranges;
    for (auto sliceRange : op.getRanges())
      ranges.push_back(rewriter.create<RangeOp>(
          op.getLoc(), sliceRange.min, sliceRange.max, sliceRange.step));
    rewriter.replaceOpWithNewOp<SliceOp>(op, view, ranges);
    return matchSuccess();
  }
};

/// Populate the given list with patterns that convert from Linalg to Standard.
static void
populateLinalgToStandardConversionPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *ctx) {
  // TODO(ntv) ConvOp conversion needs to export a descriptor with relevant
  // attribute values such as kernel striding and dilation.
  patterns.insert<CopyTransposeConversion, LinalgOpConversion<CopyOp>,
                  LinalgOpConversion<DotOp>, LinalgOpConversion<FillOp>,
                  LinalgOpConversion<MatvecOp>, LinalgOpConversion<MatmulOp>,
                  LinalgOpConversion<ConvOp>, LinalgOpConversion<GenericOp>,
                  SubViewOpConversion>(ctx);
}

/// Populate the given list with patterns that convert from Linalg to LLVM.
static void
populateLinalgToLLVMConversionPatterns(LinalgTypeConverter &converter,
                                       OwningRewritePatternList &patterns,
                                       MLIRContext *ctx) {
  patterns.insert<BufferAllocOpConversion, BufferDeallocOpConversion,
                  BufferSizeOpConversion, RangeOpConversion, SliceOpConversion,
                  TransposeOpConversion, ViewOpConversion, YieldOpConversion>(
      ctx, converter);
}

namespace {
struct LowerLinalgToLLVMPass : public ModulePass<LowerLinalgToLLVMPass> {
  void runOnModule() override;
};
} // namespace

void LowerLinalgToLLVMPass::runOnModule() {
  auto module = getModule();

  // Convert to the LLVM IR dialect using the converter defined above.
  OwningRewritePatternList patterns;
  LinalgTypeConverter converter(&getContext());
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateLinalgToStandardConversionPatterns(patterns, &getContext());
  populateLinalgToLLVMConversionPatterns(converter, patterns, &getContext());

  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
  target.addLegalOp<ModuleOp, ModuleTerminatorOp>();
  if (failed(applyFullConversion(module, target, patterns, &converter)))
    signalPassFailure();
}

std::unique_ptr<OpPassBase<ModuleOp>>
mlir::linalg::createLowerLinalgToLLVMPass() {
  return std::make_unique<LowerLinalgToLLVMPass>();
}

static PassRegistration<LowerLinalgToLLVMPass>
    pass("linalg-convert-to-llvm",
         "Lower the operations from the linalg dialect into the LLVM dialect");
