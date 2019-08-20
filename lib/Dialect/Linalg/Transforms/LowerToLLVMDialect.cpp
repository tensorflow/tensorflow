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

#include "mlir/Conversion/ControlFlowToCFG/ConvertControlFlowToCFG.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Intrinsics.h"
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
using undef = ValueBuilder<mlir::LLVM::UndefOp>;
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

  // A linalg.view type converts to a *pointer to* a view descriptor. The view
  // descriptor contains the pointer to the data buffer, followed by a 64-bit
  // integer containing the distance between the beginning of the buffer and the
  // first element to be accessed through the view, followed by two arrays, each
  // containing as many 64-bit integers as the rank of the View. The first array
  // represents the size, in number of original elements, of the view along the
  // given dimension.  When taking the view, the size is the difference between
  // the upper and the lower bound of the range. The second array represents the
  // "stride" (in tensor abstraction sense), i.e. the number of consecutive
  // elements of the underlying buffer that separate two consecutive elements
  // addressable through the view along the given dimension.  When taking the
  // view, the strides are constructed as products of the original sizes along
  // the trailing dimensions, multiplied by the view step.  For example, a view
  // of a MxN memref with ranges {0:M:1}, {0:N:1}, i.e. the view of a complete
  // memref, will have strides N and 1.  A view with ranges {0:M:2}, {0:N:3}
  // will have strides 2*N and 3.
  //
  // template <typename Elem, size_t Rank>
  // struct {
  //   Elem *ptr;
  //   int64_t offset;
  //   int64_t sizes[Rank];
  //   int64_t strides[Rank];
  // } *;
  if (auto viewType = t.dyn_cast<ViewType>()) {
    auto ptrTy = getPtrToElementType(viewType, lowering);
    auto arrayTy = LLVMType::getArrayTy(int64Ty, viewType.getRank());
    return LLVMType::getStructTy(ptrTy, int64Ty, arrayTy, arrayTy)
        .getPointerTo();
  }

  return Type();
}

static constexpr int kBasePtrPosInBuffer = 0;
static constexpr int kPtrPosInBuffer = 1;
static constexpr int kSizePosInBuffer = 2;
static constexpr int kPtrPosInView = 0;
static constexpr int kOffsetPosInView = 1;
static constexpr int kSizePosInView = 2;
static constexpr int kStridePosInView = 3;

// Create an array attribute containing integer attributes with values provided
// in `position`.
static ArrayAttr positionAttr(Builder &builder, ArrayRef<int> position) {
  SmallVector<Attribute, 4> attrs;
  attrs.reserve(position.size());
  for (auto p : position)
    attrs.push_back(builder.getI64IntegerAttr(p));
  return builder.getArrayAttr(attrs);
}

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
    auto int64Ty = lowering.convertType(rewriter.getIntegerType(64));
    // Insert the `malloc` declaration if it is not already present.
    auto module = op->getParentOfType<ModuleOp>();
    FuncOp mallocFunc = module.lookupSymbol<FuncOp>("malloc");
    if (!mallocFunc) {
      auto mallocType = rewriter.getFunctionType(int64Ty, voidPtrTy);
      mallocFunc =
          FuncOp::create(rewriter.getUnknownLoc(), "malloc", mallocType);
      module.push_back(mallocFunc);
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
      one = constant(int64Ty,
                     rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
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
    Value *desc = undef(bufferDescriptorTy);
    desc = insertvalue(bufferDescriptorTy, desc, allocated,
                       positionAttr(rewriter, kBasePtrPosInBuffer));
    desc = insertvalue(bufferDescriptorTy, desc, data,
                       positionAttr(rewriter, kPtrPosInBuffer));
    desc = insertvalue(bufferDescriptorTy, desc, size,
                       positionAttr(rewriter, kSizePosInBuffer));
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
    auto voidPtrTy =
        LLVM::LLVMType::getInt8Ty(lowering.getDialect()).getPointerTo();
    // Insert the `free` declaration if it is not already present.
    auto module = op->getParentOfType<ModuleOp>();
    FuncOp freeFunc = module.lookupSymbol<FuncOp>("free");
    if (!freeFunc) {
      auto freeType = rewriter.getFunctionType(voidPtrTy, {});
      freeFunc = FuncOp::create(rewriter.getUnknownLoc(), "free", freeType);
      module.push_back(freeFunc);
    }

    // Emit MLIR for buffer_dealloc.
    BufferDeallocOpOperandAdaptor adaptor(operands);
    edsc::ScopedContext context(rewriter, op->getLoc());
    Value *base = extractvalue(voidPtrTy, adaptor.buffer(),
                               positionAttr(rewriter, kBasePtrPosInBuffer));
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
                          positionAttr(rewriter, kSizePosInBuffer))});
    return matchSuccess();
  }
};

// DimOp creates a new `index` value.
class DimOpConversion : public LLVMOpLowering {
public:
  explicit DimOpConversion(MLIRContext *context, LLVMTypeConverter &lowering_)
      : LLVMOpLowering(linalg::DimOp::getOperationName(), context, lowering_) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto dimOp = cast<linalg::DimOp>(op);
    auto indexTy = lowering.convertType(rewriter.getIndexType());
    edsc::ScopedContext context(rewriter, op->getLoc());
    auto pos = positionAttr(
        rewriter, {kSizePosInView, static_cast<int>(dimOp.getIndex())});
    linalg::DimOpOperandAdaptor adaptor(operands);
    Value *viewDescriptor = llvm_load(adaptor.view());
    rewriter.replaceOp(op, {extractvalue(indexTy, viewDescriptor, pos)});
    return matchSuccess();
  }
};

namespace {
// Common functionality for Linalg LoadOp and StoreOp conversion to the
// LLVM IR Dialect.
template <typename Op> class LoadStoreOpConversion : public LLVMOpLowering {
public:
  explicit LoadStoreOpConversion(MLIRContext *context,
                                 LLVMTypeConverter &lowering_)
      : LLVMOpLowering(Op::getOperationName(), context, lowering_) {}
  using Base = LoadStoreOpConversion<Op>;

  // Compute the pointer to an element of the buffer underlying the view given
  // current view indices.  Use the base offset and strides stored in the view
  // descriptor to emit IR iteratively computing the actual offset, followed by
  // a getelementptr. This must be called under an edsc::ScopedContext.
  Value *obtainDataPtr(Operation *op, Value *viewDescriptorPtr,
                       ArrayRef<Value *> indices,
                       ConversionPatternRewriter &rewriter) const {
    auto loadOp = cast<Op>(op);
    auto elementTy = getPtrToElementType(loadOp.getViewType(), lowering);
    auto int64Ty = lowering.convertType(rewriter.getIntegerType(64));
    auto pos = [&rewriter](ArrayRef<int> values) {
      return positionAttr(rewriter, values);
    };

    // Linearize subscripts as:
    //   base_offset + SUM_i index_i * stride_i.
    Value *viewDescriptor = llvm_load(viewDescriptorPtr);
    Value *base = extractvalue(elementTy, viewDescriptor, pos(kPtrPosInView));
    Value *offset =
        extractvalue(int64Ty, viewDescriptor, pos(kOffsetPosInView));
    for (int i = 0, e = loadOp.getRank(); i < e; ++i) {
      Value *stride =
          extractvalue(int64Ty, viewDescriptor, pos({kStridePosInView, i}));
      Value *additionalOffset = mul(indices[i], stride);
      offset = add(offset, additionalOffset);
    }
    return gep(elementTy, base, offset);
  }
};
} // namespace

// A load is converted into the actual address computation, getelementptr and
// an LLVM IR load.
class LoadOpConversion : public LoadStoreOpConversion<linalg::LoadOp> {
  using Base::Base;
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext edscContext(rewriter, op->getLoc());
    auto elementTy = lowering.convertType(*op->result_type_begin());
    linalg::LoadOpOperandAdaptor adaptor(operands);
    auto ptr = obtainDataPtr(op, adaptor.view(), adaptor.indices(), rewriter);
    rewriter.replaceOp(op, {llvm_load(elementTy, ptr)});
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
    Value *desc = undef(rangeDescriptorTy);
    desc = insertvalue(desc, adaptor.min(), positionAttr(rewriter, 0));
    desc = insertvalue(desc, adaptor.max(), positionAttr(rewriter, 1));
    desc = insertvalue(desc, adaptor.step(), positionAttr(rewriter, 2));
    rewriter.replaceOp(op, desc);
    return matchSuccess();
  }
};

/// Conversion pattern that transforms a linalg.slice op into:
///   1. A function entry `alloca` operation to allocate a ViewDescriptor.
///   2. A load of the ViewDescriptor from the pointer allocated in 1.
///   3. Updates to the ViewDescriptor to introduce the data ptr, offset, size
///      and stride corresponding to the
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
    auto sliceOp = cast<SliceOp>(op);
    auto viewDescriptorPtrTy =
        convertLinalgType(sliceOp.getViewType(), lowering);
    auto viewType = sliceOp.getBaseViewType();
    auto int64Ty = lowering.convertType(rewriter.getIntegerType(64));

    // Helper function to create an integer array attribute out of a list of
    // values.
    auto pos = [&rewriter](ArrayRef<int> values) {
      return positionAttr(rewriter, values);
    };

    edsc::ScopedContext context(rewriter, op->getLoc());
    // Declare the view descriptor and insert data ptr *at the entry block of
    // the function*, which is the preferred location for LLVM's analyses.
    auto ip = rewriter.getInsertionPoint();
    auto ib = rewriter.getInsertionBlock();
    rewriter.setInsertionPointToStart(
        &op->getParentOfType<FuncOp>().getBlocks().front());
    Value *one =
        constant(int64Ty, rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
    // Alloca with proper alignment.
    Value *allocatedDesc =
        llvm_alloca(viewDescriptorPtrTy, one, /*alignment=*/8);
    Value *desc = llvm_load(allocatedDesc);
    rewriter.setInsertionPoint(ib, ip);

    Value *baseDesc = llvm_load(adaptor.view());

    auto ptrPos = pos(kPtrPosInView);
    auto elementTy = getPtrToElementType(sliceOp.getViewType(), lowering);
    desc = insertvalue(desc, extractvalue(elementTy, baseDesc, ptrPos), ptrPos);

    // TODO(ntv): extract sizes and emit asserts.
    SmallVector<Value *, 4> strides(viewType.getRank());
    for (int i = 0, e = viewType.getRank(); i < e; ++i) {
      strides[i] = extractvalue(int64Ty, baseDesc, pos({kStridePosInView, i}));
    }

    // Compute and insert base offset.
    Value *baseOffset = extractvalue(int64Ty, baseDesc, pos(kOffsetPosInView));
    for (int i = 0, e = viewType.getRank(); i < e; ++i) {
      Value *indexing = adaptor.indexings()[i];
      Value *min =
          sliceOp.indexing(i)->getType().isa<RangeType>()
              ? static_cast<Value *>(extractvalue(int64Ty, indexing, pos(0)))
              : indexing;
      Value *product = mul(min, strides[i]);
      baseOffset = add(baseOffset, product);
    }
    desc = insertvalue(desc, baseOffset, pos(kOffsetPosInView));

    // Compute and insert view sizes (max - min along the range) and strides.
    // Skip the non-range operands as they will be projected away from the view.
    int numNewDims = 0;
    for (auto en : llvm::enumerate(sliceOp.indexings())) {
      Value *indexing = en.value();
      if (indexing->getType().isa<RangeType>()) {
        int i = en.index();
        Value *rangeDescriptor = adaptor.indexings()[i];
        Value *min = extractvalue(int64Ty, rangeDescriptor, pos(0));
        Value *max = extractvalue(int64Ty, rangeDescriptor, pos(1));
        Value *step = extractvalue(int64Ty, rangeDescriptor, pos(2));
        Value *size = sub(max, min);
        Value *stride = mul(strides[i], step);
        desc = insertvalue(desc, size, pos({kSizePosInView, numNewDims}));
        desc = insertvalue(desc, stride, pos({kStridePosInView, numNewDims}));
        ++numNewDims;
      }
    }

    // Store back in alloca'ed region.
    llvm_store(desc, allocatedDesc);
    rewriter.replaceOp(op, allocatedDesc);
    return matchSuccess();
  }
};

// A store is converted into the actual address computation, getelementptr and
// an LLVM IR store.
class StoreOpConversion : public LoadStoreOpConversion<linalg::StoreOp> {
  using Base::Base;
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    edsc::ScopedContext edscContext(rewriter, op->getLoc());
    linalg::StoreOpOperandAdaptor adaptor(operands);
    Value *ptr = obtainDataPtr(op, adaptor.view(), adaptor.indices(), rewriter);
    llvm_store(adaptor.value(), ptr);
    rewriter.replaceOp(op, llvm::None);
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
    auto viewOp = cast<ViewOp>(op);
    ViewOpOperandAdaptor adaptor(operands);
    auto viewDescriptorPtrTy =
        convertLinalgType(viewOp.getViewType(), lowering);
    auto elementTy = getPtrToElementType(viewOp.getViewType(), lowering);
    auto int64Ty = lowering.convertType(rewriter.getIntegerType(64));

    auto pos = [&rewriter](ArrayRef<int> values) {
      return positionAttr(rewriter, values);
    };

    Value *bufferDescriptor = adaptor.buffer();
    auto bufferTy = getPtrToElementType(
        viewOp.buffer()->getType().cast<BufferType>(), lowering);

    // Declare the descriptor of the view.
    edsc::ScopedContext context(rewriter, op->getLoc());
    auto ip = rewriter.getInsertionPoint();
    auto ib = rewriter.getInsertionBlock();
    rewriter.setInsertionPointToStart(
        &op->getParentOfType<FuncOp>().getBlocks().front());
    Value *one =
        constant(int64Ty, rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
    // Alloca for proper alignment.
    Value *allocatedDesc =
        llvm_alloca(viewDescriptorPtrTy, one, /*alignment=*/8);
    Value *desc = llvm_load(allocatedDesc);
    rewriter.setInsertionPoint(ib, ip);

    // Copy the buffer pointer from the old descriptor to the new one.
    Value *bufferAsViewElementType =
        bitcast(elementTy,
                extractvalue(bufferTy, bufferDescriptor, pos(kPtrPosInBuffer)));
    desc = insertvalue(desc, bufferAsViewElementType, pos(kPtrPosInView));

    // Zero base offset.
    auto indexTy = rewriter.getIndexType();
    Value *baseOffset = constant(int64Ty, IntegerAttr::get(indexTy, 0));
    desc = insertvalue(desc, baseOffset, pos(kOffsetPosInView));

    // Compute and insert view sizes (max - min along the range).
    int numRanges = llvm::size(viewOp.ranges());
    Value *runningStride = constant(int64Ty, IntegerAttr::get(indexTy, 1));
    for (int i = numRanges - 1; i >= 0; --i) {
      // Update stride.
      Value *rangeDescriptor = operands[1 + i];
      Value *step = extractvalue(int64Ty, rangeDescriptor, pos(2));
      Value *stride = mul(runningStride, step);
      desc = insertvalue(desc, stride, pos({kStridePosInView, i}));
      // Update size.
      Value *min = extractvalue(int64Ty, rangeDescriptor, pos(0));
      Value *max = extractvalue(int64Ty, rangeDescriptor, pos(1));
      Value *size = sub(max, min);
      desc = insertvalue(desc, size, pos({kSizePosInView, i}));
      // Update stride for the next dimension.
      if (i > 0)
        runningStride = mul(runningStride, max);
    }

    // Store back in alloca'ed region.
    llvm_store(desc, allocatedDesc);
    rewriter.replaceOp(op, allocatedDesc);
    return matchSuccess();
  }
};

// Create a function definition which takes as argument pointers to the input
// types and returns pointers to the output types.
static FuncOp getLLVMLibraryCallImplDefinition(FuncOp libFn) {
  auto implFnName = (libFn.getName().str() + "_impl");
  auto module = libFn.getParentOfType<ModuleOp>();
  if (auto f = module.lookupSymbol<FuncOp>(implFnName)) {
    return f;
  }
  SmallVector<Type, 4> fnArgTypes;
  for (auto t : libFn.getType().getInputs()) {
    assert(t && t.isa<LLVMType>() &&
           "Expected LLVM Type for argument while generating library Call "
           "Implementation Definition");
    fnArgTypes.push_back(t.cast<LLVMType>().getPointerTo());
  }
  auto implFnType = FunctionType::get(fnArgTypes, {}, libFn.getContext());

  // Insert the implementation function definition.
  auto implFnDefn = FuncOp::create(libFn.getLoc(), implFnName, implFnType);
  module.push_back(implFnDefn);
  return implFnDefn;
}

// Get function definition for the LinalgOp. If it doesn't exist, insert a
// definition.
template <typename LinalgOp>
static FuncOp
getLLVMLibraryCallDeclaration(Operation *op, LLVMTypeConverter &lowering,
                              ConversionPatternRewriter &rewriter) {
  auto linalgOp = cast<LinalgOp>(op);
  auto fnName = linalgOp.getLibraryCallName();
  if (fnName.empty()) {
    op->emitWarning("No library call defined for: ") << *op;
    return FuncOp();
  }
  auto module = op->getParentOfType<ModuleOp>();
  if (auto f = module.lookupSymbol<FuncOp>(fnName)) {
    return f;
  }

  // Get the Function type consistent with LLVM Lowering.
  SmallVector<Type, 4> inputTypes;
  for (auto operand : op->getOperands())
    inputTypes.push_back(lowering.convertType(operand->getType()));
  assert(op->getNumResults() == 0 &&
         "Library call for linalg operation can be generated only for ops that "
         "have void return types");
  auto libFnType = FunctionType::get(inputTypes, {}, op->getContext());
  auto libFn = FuncOp::create(op->getLoc(), fnName, libFnType);
  module.push_back(libFn);
  // Return after creating the function definition. The body will be created
  // later.
  return libFn;
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
template <typename LinalgOp> class LinalgOpConversion : public LLVMOpLowering {
public:
  explicit LinalgOpConversion(MLIRContext *context,
                              LinalgTypeConverter &lowering_)
      : LLVMOpLowering(LinalgOp::getOperationName(), context, lowering_) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Only emit library call declaration. Fill in the body later.
    auto f = getLLVMLibraryCallDeclaration<LinalgOp>(op, lowering, rewriter);
    if (!f)
      return matchFailure();

    auto fAttr = rewriter.getSymbolRefAttr(f);
    auto named = rewriter.getNamedAttr("callee", fAttr);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, operands,
                                              ArrayRef<NamedAttribute>{named});
    return matchSuccess();
  }
};

/// Populate the given list with patterns that convert from Linalg to LLVM.
static void
populateLinalgToLLVMConversionPatterns(LinalgTypeConverter &converter,
                                       OwningRewritePatternList &patterns,
                                       MLIRContext *ctx) {
  patterns
      .insert<BufferAllocOpConversion, BufferDeallocOpConversion,
              BufferSizeOpConversion, DimOpConversion,
              LinalgOpConversion<DotOp>, LinalgOpConversion<FillOp>,
              LinalgOpConversion<MatmulOp>, LoadOpConversion, RangeOpConversion,
              SliceOpConversion, StoreOpConversion, ViewOpConversion>(
          ctx, converter);
}

namespace {
struct LowerLinalgToLLVMPass : public ModulePass<LowerLinalgToLLVMPass> {
  void runOnModule();
};
} // namespace

// This is currently written as a standalone function because the lowering to
// affine will look different than lowering to LLVM and it is still unclear how
// everything will be eventually structured.
static void lowerLinalgSubViewOps(FuncOp &f) {
  f.walk<SubViewOp>([&](SubViewOp op) {
    OpBuilder b(op);
    ScopedContext scope(b, op.getLoc());
    auto *view = op.getView();
    SmallVector<Value *, 8> ranges;
    for (auto en : llvm::enumerate(op.getRanges())) {
      using edsc::op::operator<;
      using linalg::intrinsics::dim;
      unsigned rank = en.index();
      auto sliceRange = en.value();
      auto size = dim(view, rank);
      ValueHandle ub(sliceRange.max);
      auto max = edsc::intrinsics::select(size < ub, size, ub);
      ranges.push_back(range(sliceRange.min, max, sliceRange.step));
    }
    op.replaceAllUsesWith(slice(view, ranges));
    op.erase();
  });
}

void LowerLinalgToLLVMPass::runOnModule() {
  auto module = getModule();

  for (auto f : module.getOps<FuncOp>())
    lowerLinalgSubViewOps(f);

  // Convert to the LLVM IR dialect using the converter defined above.
  OwningRewritePatternList patterns;
  LinalgTypeConverter converter(&getContext());
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateLinalgToLLVMConversionPatterns(converter, patterns, &getContext());

  ConversionTarget target(getContext());
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
  if (failed(applyPartialConversion(module, target, patterns, &converter))) {
    signalPassFailure();
  }
}

std::unique_ptr<ModulePassBase> mlir::linalg::createLowerLinalgToLLVMPass() {
  return std::make_unique<LowerLinalgToLLVMPass>();
}

static PassRegistration<LowerLinalgToLLVMPass>
    pass("linalg-lower-to-llvm-dialect",
         "Lower the operations from the linalg dialect into the LLVM dialect");
