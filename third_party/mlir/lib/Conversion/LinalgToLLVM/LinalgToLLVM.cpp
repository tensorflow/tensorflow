//===- LinalgToLLVM.cpp - conversion from Linalg to LLVM dialect ----------===//
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

#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
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

namespace {
/// EDSC-compatible wrapper for MemRefDescriptor.
class BaseViewConversionHelper {
public:
  BaseViewConversionHelper(Type type)
      : d(MemRefDescriptor::undef(rewriter(), loc(), type)) {}

  BaseViewConversionHelper(Value *v) : d(v) {}

  /// Wrappers around MemRefDescriptor that use EDSC builder and location.
  Value *allocatedPtr() { return d.allocatedPtr(rewriter(), loc()); }
  void setAllocatedPtr(Value *v) { d.setAllocatedPtr(rewriter(), loc(), v); }
  Value *alignedPtr() { return d.alignedPtr(rewriter(), loc()); }
  void setAlignedPtr(Value *v) { d.setAlignedPtr(rewriter(), loc(), v); }
  Value *offset() { return d.offset(rewriter(), loc()); }
  void setOffset(Value *v) { d.setOffset(rewriter(), loc(), v); }
  Value *size(unsigned i) { return d.size(rewriter(), loc(), i); }
  void setSize(unsigned i, Value *v) { d.setSize(rewriter(), loc(), i, v); }
  Value *stride(unsigned i) { return d.stride(rewriter(), loc(), i); }
  void setStride(unsigned i, Value *v) { d.setStride(rewriter(), loc(), i, v); }

  operator Value *() { return d; }

private:
  OpBuilder &rewriter() { return ScopedContext::getBuilder(); }
  Location loc() { return ScopedContext::getLocation(); }

  MemRefDescriptor d;
};
} // namespace

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
    edsc::ScopedContext context(rewriter, op->getLoc());
    SliceOpOperandAdaptor adaptor(operands);
    BaseViewConversionHelper baseDesc(adaptor.view());

    auto sliceOp = cast<SliceOp>(op);
    auto memRefType = sliceOp.getBaseViewType();
    auto int64Ty = lowering.convertType(rewriter.getIntegerType(64))
                       .cast<LLVM::LLVMType>();

    BaseViewConversionHelper desc(lowering.convertType(sliceOp.getViewType()));

    // TODO(ntv): extract sizes and emit asserts.
    SmallVector<Value *, 4> strides(memRefType.getRank());
    for (int i = 0, e = memRefType.getRank(); i < e; ++i)
      strides[i] = baseDesc.stride(i);

    auto pos = [&rewriter](ArrayRef<int64_t> values) {
      return rewriter.getI64ArrayAttr(values);
    };

    // Compute base offset.
    Value *baseOffset = baseDesc.offset();
    for (int i = 0, e = memRefType.getRank(); i < e; ++i) {
      Value *indexing = adaptor.indexings()[i];
      Value *min = indexing;
      if (sliceOp.indexing(i)->getType().isa<RangeType>())
        min = extractvalue(int64Ty, indexing, pos(0));
      baseOffset = add(baseOffset, mul(min, strides[i]));
    }

    // Insert the base and aligned pointers.
    desc.setAllocatedPtr(baseDesc.allocatedPtr());
    desc.setAlignedPtr(baseDesc.alignedPtr());

    // Insert base offset.
    desc.setOffset(baseOffset);

    // Corner case, no sizes or strides: early return the descriptor.
    if (sliceOp.getViewType().getRank() == 0)
      return rewriter.replaceOp(op, {desc}), matchSuccess();

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
        Value *min = extractvalue(int64Ty, rangeDescriptor, pos(0));
        Value *max = extractvalue(int64Ty, rangeDescriptor, pos(1));
        Value *step = extractvalue(int64Ty, rangeDescriptor, pos(2));
        Value *baseSize = baseDesc.size(rank);

        // Bound upper by base view upper bound.
        max = llvm_select(llvm_icmp(ICmpPredicate::slt, max, baseSize), max,
                          baseSize);
        Value *size = sub(max, min);
        // Bound lower by zero.
        size =
            llvm_select(llvm_icmp(ICmpPredicate::slt, size, zero), zero, size);
        Value *stride = mul(strides[rank], step);
        desc.setSize(numNewDims, size);
        desc.setStride(numNewDims, stride);
        ++numNewDims;
      }
    }

    rewriter.replaceOp(op, {desc});
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
    edsc::ScopedContext context(rewriter, op->getLoc());
    TransposeOpOperandAdaptor adaptor(operands);
    BaseViewConversionHelper baseDesc(adaptor.view());

    auto transposeOp = cast<TransposeOp>(op);
    // No permutation, early exit.
    if (transposeOp.permutation().isIdentity())
      return rewriter.replaceOp(op, {baseDesc}), matchSuccess();

    BaseViewConversionHelper desc(
        lowering.convertType(transposeOp.getViewType()));

    // Copy the base and aligned pointers from the old descriptor to the new
    // one.
    desc.setAllocatedPtr(baseDesc.allocatedPtr());
    desc.setAlignedPtr(baseDesc.alignedPtr());

    // Copy the offset pointer from the old descriptor to the new one.
    desc.setOffset(baseDesc.offset());

    // Iterate over the dimensions and apply size/stride permutation.
    for (auto en : llvm::enumerate(transposeOp.permutation().getResults())) {
      int sourcePos = en.index();
      int targetPos = en.value().cast<AffineDimExpr>().getPosition();
      desc.setSize(targetPos, baseDesc.size(sourcePos));
      desc.setStride(targetPos, baseDesc.stride(sourcePos));
    }

    rewriter.replaceOp(op, {desc});
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

template <typename LinalgOp>
static SmallVector<Type, 4> ExtractOperandTypes(Operation *op) {
  return SmallVector<Type, 4>{op->getOperandTypes()};
}

template <>
SmallVector<Type, 4> ExtractOperandTypes<IndexedGenericOp>(Operation *op) {
  auto ctx = op->getContext();
  auto indexedGenericOp = cast<IndexedGenericOp>(op);
  auto numLoops = indexedGenericOp.getNumLoops();

  SmallVector<Type, 4> result;
  result.reserve(numLoops + op->getNumOperands());
  for (unsigned i = 0; i < numLoops; ++i) {
    result.push_back(IndexType::get(ctx));
  }
  for (auto type : op->getOperandTypes()) {
    result.push_back(type);
  }
  return result;
}

// Get a SymbolRefAttr containing the library function name for the LinalgOp.
// If the library function does not exist, insert a declaration.
template <typename LinalgOp>
static FlatSymbolRefAttr getLibraryCallSymbolRef(Operation *op,
                                                 PatternRewriter &rewriter) {
  auto linalgOp = cast<LinalgOp>(op);
  auto fnName = linalgOp.getLibraryCallName();
  if (fnName.empty()) {
    op->emitWarning("No library call defined for: ") << *op;
    return {};
  }

  // fnName is a dynamic std::String, unique it via a SymbolRefAttr.
  FlatSymbolRefAttr fnNameAttr = rewriter.getSymbolRefAttr(fnName);
  auto module = op->getParentOfType<ModuleOp>();
  if (module.lookupSymbol(fnName)) {
    return fnNameAttr;
  }

  SmallVector<Type, 4> inputTypes(ExtractOperandTypes<LinalgOp>(op));
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

Type LinalgTypeConverter::convertType(Type t) {
  if (auto result = LLVMTypeConverter::convertType(t))
    return result;
  return convertLinalgType(t, *this);
}

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

    rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, libraryCallName.getValue(), ArrayRef<Type>{}, op.getOperands());
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

    rewriter.replaceOpWithNewOp<mlir::CallOp>(
        op, libraryCallName.getValue(), ArrayRef<Type>{}, op.getOperands());
    return matchSuccess();
  }
};

/// Conversion pattern specialization for IndexedGenericOp.
template <>
class LinalgOpConversion<IndexedGenericOp>
    : public OpRewritePattern<IndexedGenericOp> {
public:
  using OpRewritePattern<IndexedGenericOp>::OpRewritePattern;

  PatternMatchResult matchAndRewrite(IndexedGenericOp op,
                                     PatternRewriter &rewriter) const override {
    auto libraryCallName =
        getLibraryCallSymbolRef<IndexedGenericOp>(op, rewriter);
    if (!libraryCallName)
      return this->matchFailure();

    // TODO(pifon, ntv): Use induction variables values instead of zeros, when
    // IndexedGenericOp is tiled.
    auto zero = rewriter.create<mlir::ConstantOp>(
        op.getLoc(), rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    auto indexedGenericOp = cast<IndexedGenericOp>(op);
    auto numLoops = indexedGenericOp.getNumLoops();
    SmallVector<Value *, 4> operands;
    operands.reserve(numLoops + op.getNumOperands());
    for (unsigned i = 0; i < numLoops; ++i) {
      operands.push_back(zero);
    }
    for (auto operand : op.getOperands()) {
      operands.push_back(operand);
    }
    rewriter.replaceOpWithNewOp<mlir::CallOp>(op, libraryCallName.getValue(),
                                              ArrayRef<Type>{}, operands);
    return this->matchSuccess();
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

/// Populate the given list with patterns that convert from Linalg to Standard.
static void
populateLinalgToStandardConversionPatterns(OwningRewritePatternList &patterns,
                                           MLIRContext *ctx) {
  // TODO(ntv) ConvOp conversion needs to export a descriptor with relevant
  // attribute values such as kernel striding and dilation.
  patterns.insert<CopyTransposeConversion, LinalgOpConversion<ConvOp>,
                  LinalgOpConversion<CopyOp>, LinalgOpConversion<DotOp>,
                  LinalgOpConversion<FillOp>, LinalgOpConversion<GenericOp>,
                  LinalgOpConversion<IndexedGenericOp>,
                  LinalgOpConversion<MatmulOp>, LinalgOpConversion<MatvecOp>>(
      ctx);
}

/// Populate the given list with patterns that convert from Linalg to LLVM.
void mlir::populateLinalgToLLVMConversionPatterns(
    LinalgTypeConverter &converter, OwningRewritePatternList &patterns,
    MLIRContext *ctx) {
  patterns.insert<RangeOpConversion, SliceOpConversion, TransposeOpConversion,
                  YieldOpConversion>(ctx, converter);
}

namespace {
struct ConvertLinalgToLLVMPass : public ModulePass<ConvertLinalgToLLVMPass> {
  void runOnModule() override;
};
} // namespace

void ConvertLinalgToLLVMPass::runOnModule() {
  auto module = getModule();

  // Convert to the LLVM IR dialect using the converter defined above.
  OwningRewritePatternList patterns;
  LinalgTypeConverter converter(&getContext());
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns);
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
mlir::linalg::createConvertLinalgToLLVMPass() {
  return std::make_unique<ConvertLinalgToLLVMPass>();
}

static PassRegistration<ConvertLinalgToLLVMPass> pass(
    "convert-linalg-to-llvm",
    "Convert the operations from the linalg dialect into the LLVM dialect");
