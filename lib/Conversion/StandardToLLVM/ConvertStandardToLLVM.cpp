//===- ConvertStandardToLLVM.cpp - Standard to LLVM dialect conversion-----===//
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
//
// This file implements a pass to convert MLIR standard and builtin dialects
// into the LLVM IR dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/ControlFlowToCFG/ConvertControlFlowToCFG.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/StandardOps/Ops.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"

using namespace mlir;

LLVMTypeConverter::LLVMTypeConverter(MLIRContext *ctx)
    : llvmDialect(ctx->getRegisteredDialect<LLVM::LLVMDialect>()) {
  assert(llvmDialect && "LLVM IR dialect is not registered");
  module = &llvmDialect->getLLVMModule();
}

// Get the LLVM context.
llvm::LLVMContext &LLVMTypeConverter::getLLVMContext() {
  return module->getContext();
}

// Extract an LLVM IR type from the LLVM IR dialect type.
LLVM::LLVMType LLVMTypeConverter::unwrap(Type type) {
  if (!type)
    return nullptr;
  auto *mlirContext = type.getContext();
  auto wrappedLLVMType = type.dyn_cast<LLVM::LLVMType>();
  if (!wrappedLLVMType)
    emitError(UnknownLoc::get(mlirContext),
              "conversion resulted in a non-LLVM type");
  return wrappedLLVMType;
}

LLVM::LLVMType LLVMTypeConverter::getIndexType() {
  return LLVM::LLVMType::getIntNTy(
      llvmDialect, module->getDataLayout().getPointerSizeInBits());
}

Type LLVMTypeConverter::convertIndexType(IndexType type) {
  return getIndexType();
}

Type LLVMTypeConverter::convertIntegerType(IntegerType type) {
  return LLVM::LLVMType::getIntNTy(llvmDialect, type.getWidth());
}

Type LLVMTypeConverter::convertFloatType(FloatType type) {
  switch (type.getKind()) {
  case mlir::StandardTypes::F32:
    return LLVM::LLVMType::getFloatTy(llvmDialect);
  case mlir::StandardTypes::F64:
    return LLVM::LLVMType::getDoubleTy(llvmDialect);
  case mlir::StandardTypes::F16:
    return LLVM::LLVMType::getHalfTy(llvmDialect);
  case mlir::StandardTypes::BF16: {
    auto *mlirContext = llvmDialect->getContext();
    return emitError(UnknownLoc::get(mlirContext), "unsupported type: BF16"),
           Type();
  }
  default:
    llvm_unreachable("non-float type in convertFloatType");
  }
}

// Function types are converted to LLVM Function types by recursively converting
// argument and result types.  If MLIR Function has zero results, the LLVM
// Function has one VoidType result.  If MLIR Function has more than one result,
// they are into an LLVM StructType in their order of appearance.
Type LLVMTypeConverter::convertFunctionType(FunctionType type) {
  // Convert argument types one by one and check for errors.
  SmallVector<LLVM::LLVMType, 8> argTypes;
  for (auto t : type.getInputs()) {
    auto converted = convertType(t);
    if (!converted)
      return {};
    argTypes.push_back(unwrap(converted));
  }

  // If function does not return anything, create the void result type,
  // if it returns on element, convert it, otherwise pack the result types into
  // a struct.
  LLVM::LLVMType resultType =
      type.getNumResults() == 0
          ? LLVM::LLVMType::getVoidTy(llvmDialect)
          : unwrap(packFunctionResults(type.getResults()));
  if (!resultType)
    return {};
  return LLVM::LLVMType::getFunctionTy(resultType, argTypes, /*isVarArg=*/false)
      .getPointerTo();
}

// Convert a MemRef to an LLVM type. If the memref is statically-shaped, then
// we return a pointer to the converted element type. Otherwise we return an
// LLVM stucture type, where the first element of the structure type is a
// pointer to the elemental type of the MemRef and the following N elements are
// values of the Index type, one for each of N dynamic dimensions of the MemRef.
Type LLVMTypeConverter::convertMemRefType(MemRefType type) {
  LLVM::LLVMType elementType = unwrap(convertType(type.getElementType()));
  if (!elementType)
    return {};
  auto ptrType = elementType.getPointerTo();

  // Extra value for the memory space.
  unsigned numDynamicSizes = type.getNumDynamicDims();
  // If memref is statically-shaped we return the underlying pointer type.
  if (numDynamicSizes == 0)
    return ptrType;

  SmallVector<LLVM::LLVMType, 8> types(numDynamicSizes + 1, getIndexType());
  types.front() = ptrType;

  return LLVM::LLVMType::getStructTy(llvmDialect, types);
}

// Convert a 1D vector type to an LLVM vector type.
Type LLVMTypeConverter::convertVectorType(VectorType type) {
  if (type.getRank() != 1) {
    auto *mlirContext = llvmDialect->getContext();
    emitError(UnknownLoc::get(mlirContext), "only 1D vectors are supported");
    return {};
  }

  LLVM::LLVMType elementType = unwrap(convertType(type.getElementType()));
  return elementType
             ? LLVM::LLVMType::getVectorTy(elementType, type.getShape().front())
             : Type();
}

// Dispatch based on the actual type.  Return null type on error.
Type LLVMTypeConverter::convertStandardType(Type type) {
  if (auto funcType = type.dyn_cast<FunctionType>())
    return convertFunctionType(funcType);
  if (auto intType = type.dyn_cast<IntegerType>())
    return convertIntegerType(intType);
  if (auto floatType = type.dyn_cast<FloatType>())
    return convertFloatType(floatType);
  if (auto indexType = type.dyn_cast<IndexType>())
    return convertIndexType(indexType);
  if (auto memRefType = type.dyn_cast<MemRefType>())
    return convertMemRefType(memRefType);
  if (auto vectorType = type.dyn_cast<VectorType>())
    return convertVectorType(vectorType);
  if (auto llvmType = type.dyn_cast<LLVM::LLVMType>())
    return llvmType;

  return {};
}

// Convert the element type of the memref `t` to to an LLVM type using
// `lowering`, get a pointer LLVM type pointing to the converted `t`, wrap it
// into the MLIR LLVM dialect type and return.
static Type getMemRefElementPtrType(MemRefType t, LLVMTypeConverter &lowering) {
  auto elementType = t.getElementType();
  auto converted = lowering.convertType(elementType);
  if (!converted)
    return {};
  return converted.cast<LLVM::LLVMType>().getPointerTo();
}

LLVMOpLowering::LLVMOpLowering(StringRef rootOpName, MLIRContext *context,
                               LLVMTypeConverter &lowering_)
    : ConversionPattern(rootOpName, /*benefit=*/1, context),
      lowering(lowering_) {}

namespace {
// Base class for Standard to LLVM IR op conversions.  Matches the Op type
// provided as template argument.  Carries a reference to the LLVM dialect in
// case it is necessary for rewriters.
template <typename SourceOp>
class LLVMLegalizationPattern : public LLVMOpLowering {
public:
  // Construct a conversion pattern.
  explicit LLVMLegalizationPattern(LLVM::LLVMDialect &dialect_,
                                   LLVMTypeConverter &lowering_)
      : LLVMOpLowering(SourceOp::getOperationName(), dialect_.getContext(),
                       lowering_),
        dialect(dialect_) {}

  // Get the LLVM IR dialect.
  LLVM::LLVMDialect &getDialect() const { return dialect; }
  // Get the LLVM context.
  llvm::LLVMContext &getContext() const { return dialect.getLLVMContext(); }
  // Get the LLVM module in which the types are constructed.
  llvm::Module &getModule() const { return dialect.getLLVMModule(); }

  // Get the MLIR type wrapping the LLVM integer type whose bit width is defined
  // by the pointer size used in the LLVM module.
  LLVM::LLVMType getIndexType() const {
    return LLVM::LLVMType::getIntNTy(
        &dialect, getModule().getDataLayout().getPointerSizeInBits());
  }

  // Get the MLIR type wrapping the LLVM i8* type.
  LLVM::LLVMType getVoidPtrType() const {
    return LLVM::LLVMType::getInt8PtrTy(&dialect);
  }

  // Create an LLVM IR pseudo-operation defining the given index constant.
  Value *createIndexConstant(ConversionPatternRewriter &builder, Location loc,
                             uint64_t value) const {
    auto attr = builder.getIntegerAttr(builder.getIndexType(), value);
    return builder.create<LLVM::ConstantOp>(loc, getIndexType(), attr);
  }

  // Get the array attribute named "position" containing the given list of
  // integers as integer attribute elements.
  static ArrayAttr getIntegerArrayAttr(ConversionPatternRewriter &builder,
                                       ArrayRef<int64_t> values) {
    SmallVector<Attribute, 4> attrs;
    attrs.reserve(values.size());
    for (int64_t pos : values)
      attrs.push_back(builder.getIntegerAttr(builder.getIndexType(), pos));
    return builder.getArrayAttr(attrs);
  }

  // Extract raw data pointer value from a value representing a memref.
  static Value *extractMemRefElementPtr(ConversionPatternRewriter &builder,
                                        Location loc,
                                        Value *convertedMemRefValue,
                                        Type elementTypePtr,
                                        bool hasStaticShape) {
    Value *buffer;
    if (hasStaticShape)
      return convertedMemRefValue;
    else
      return builder.create<LLVM::ExtractValueOp>(
          loc, elementTypePtr, convertedMemRefValue,
          getIntegerArrayAttr(builder, 0));
    return buffer;
  }

protected:
  LLVM::LLVMDialect &dialect;
};

struct FuncOpConversion : public LLVMLegalizationPattern<FuncOp> {
  using LLVMLegalizationPattern<FuncOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto funcOp = cast<FuncOp>(op);
    FunctionType type = funcOp.getType();

    // Convert the original function arguments.
    TypeConverter::SignatureConversion result(type.getNumInputs());
    for (unsigned i = 0, e = type.getNumInputs(); i != e; ++i)
      if (failed(lowering.convertSignatureArg(i, type.getInput(i), result)))
        return matchFailure();

    // Pack the result types into a struct.
    Type packedResult;
    if (type.getNumResults() != 0) {
      if (!(packedResult = lowering.packFunctionResults(type.getResults())))
        return matchFailure();
    }

    // Create a new function with an updated signature.
    auto newFuncOp = rewriter.cloneWithoutRegions(funcOp);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    newFuncOp.setType(FunctionType::get(
        result.getConvertedTypes(),
        packedResult ? ArrayRef<Type>(packedResult) : llvm::None,
        funcOp.getContext()));

    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&newFuncOp.getBody(), result);
    rewriter.replaceOp(op, llvm::None);
    return matchSuccess();
  }
};

// Basic lowering implementation for one-to-one rewriting from Standard Ops to
// LLVM Dialect Ops.
template <typename SourceOp, typename TargetOp>
struct OneToOneLLVMOpLowering : public LLVMLegalizationPattern<SourceOp> {
  using LLVMLegalizationPattern<SourceOp>::LLVMLegalizationPattern;
  using Super = OneToOneLLVMOpLowering<SourceOp, TargetOp>;

  // Convert the type of the result to an LLVM type, pass operands as is,
  // preserve attributes.
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numResults = op->getNumResults();

    Type packedType;
    if (numResults != 0) {
      packedType = this->lowering.packFunctionResults(
          llvm::to_vector<4>(op->getResultTypes()));
      assert(packedType && "type conversion failed, such operation should not "
                           "have been matched");
    }

    auto newOp = rewriter.create<TargetOp>(op->getLoc(), packedType, operands,
                                           op->getAttrs());

    // If the operation produced 0 or 1 result, return them immediately.
    if (numResults == 0)
      return rewriter.replaceOp(op, llvm::None), this->matchSuccess();
    if (numResults == 1)
      return rewriter.replaceOp(op, newOp.getOperation()->getResult(0)),
             this->matchSuccess();

    // Otherwise, it had been converted to an operation producing a structure.
    // Extract individual results from the structure and return them as list.
    SmallVector<Value *, 4> results;
    results.reserve(numResults);
    for (unsigned i = 0; i < numResults; ++i) {
      auto type = this->lowering.convertType(op->getResult(i)->getType());
      results.push_back(rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), type, newOp.getOperation()->getResult(0),
          this->getIntegerArrayAttr(rewriter, i)));
    }
    rewriter.replaceOp(op, results);
    return this->matchSuccess();
  }
};

// Specific lowerings.
// FIXME: this should be tablegen'ed.
struct AddIOpLowering : public OneToOneLLVMOpLowering<AddIOp, LLVM::AddOp> {
  using Super::Super;
};
struct SubIOpLowering : public OneToOneLLVMOpLowering<SubIOp, LLVM::SubOp> {
  using Super::Super;
};
struct MulIOpLowering : public OneToOneLLVMOpLowering<MulIOp, LLVM::MulOp> {
  using Super::Super;
};
struct DivISOpLowering : public OneToOneLLVMOpLowering<DivISOp, LLVM::SDivOp> {
  using Super::Super;
};
struct DivIUOpLowering : public OneToOneLLVMOpLowering<DivIUOp, LLVM::UDivOp> {
  using Super::Super;
};
struct RemISOpLowering : public OneToOneLLVMOpLowering<RemISOp, LLVM::SRemOp> {
  using Super::Super;
};
struct RemIUOpLowering : public OneToOneLLVMOpLowering<RemIUOp, LLVM::URemOp> {
  using Super::Super;
};
struct AndOpLowering : public OneToOneLLVMOpLowering<AndOp, LLVM::AndOp> {
  using Super::Super;
};
struct OrOpLowering : public OneToOneLLVMOpLowering<OrOp, LLVM::OrOp> {
  using Super::Super;
};
struct XOrOpLowering : public OneToOneLLVMOpLowering<XOrOp, LLVM::XOrOp> {
  using Super::Super;
};
struct AddFOpLowering : public OneToOneLLVMOpLowering<AddFOp, LLVM::FAddOp> {
  using Super::Super;
};
struct SubFOpLowering : public OneToOneLLVMOpLowering<SubFOp, LLVM::FSubOp> {
  using Super::Super;
};
struct MulFOpLowering : public OneToOneLLVMOpLowering<MulFOp, LLVM::FMulOp> {
  using Super::Super;
};
struct DivFOpLowering : public OneToOneLLVMOpLowering<DivFOp, LLVM::FDivOp> {
  using Super::Super;
};
struct RemFOpLowering : public OneToOneLLVMOpLowering<RemFOp, LLVM::FRemOp> {
  using Super::Super;
};
struct SelectOpLowering
    : public OneToOneLLVMOpLowering<SelectOp, LLVM::SelectOp> {
  using Super::Super;
};
struct CallOpLowering : public OneToOneLLVMOpLowering<CallOp, LLVM::CallOp> {
  using Super::Super;
};
struct CallIndirectOpLowering
    : public OneToOneLLVMOpLowering<CallIndirectOp, LLVM::CallOp> {
  using Super::Super;
};
struct ConstLLVMOpLowering
    : public OneToOneLLVMOpLowering<ConstantOp, LLVM::ConstantOp> {
  using Super::Super;
};

// Check if the MemRefType `type` is supported by the lowering. We currently do
// not support memrefs with affine maps and non-default memory spaces.
static bool isSupportedMemRefType(MemRefType type) {
  if (!type.getAffineMaps().empty())
    return false;
  if (type.getMemorySpace() != 0)
    return false;
  return true;
}

// An `alloc` is converted into a definition of a memref descriptor value and
// a call to `malloc` to allocate the underlying data buffer.  The memref
// descriptor is of the LLVM structure type where the first element is a pointer
// to the (typed) data buffer, and the remaining elements serve to store
// dynamic sizes of the memref using LLVM-converted `index` type.
struct AllocOpLowering : public LLVMLegalizationPattern<AllocOp> {
  using LLVMLegalizationPattern<AllocOp>::LLVMLegalizationPattern;

  PatternMatchResult match(Operation *op) const override {
    MemRefType type = cast<AllocOp>(op).getType();
    return isSupportedMemRefType(type) ? matchSuccess() : matchFailure();
  }

  void rewrite(Operation *op, ArrayRef<Value *> operands,
               ConversionPatternRewriter &rewriter) const override {
    auto allocOp = cast<AllocOp>(op);
    MemRefType type = allocOp.getType();

    // Get actual sizes of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.  In case of
    // zero-dimensional memref, assume a scalar (size 1).
    SmallVector<Value *, 4> sizes;
    auto numOperands = allocOp.getNumOperands();
    sizes.reserve(numOperands);
    unsigned i = 0;
    for (int64_t s : type.getShape())
      sizes.push_back(s == -1 ? operands[i++]
                              : createIndexConstant(rewriter, op->getLoc(), s));
    if (sizes.empty())
      sizes.push_back(createIndexConstant(rewriter, op->getLoc(), 1));

    // Compute the total number of memref elements.
    Value *cumulativeSize = sizes.front();
    for (unsigned i = 1, e = sizes.size(); i < e; ++i)
      cumulativeSize = rewriter.create<LLVM::MulOp>(
          op->getLoc(), getIndexType(),
          ArrayRef<Value *>{cumulativeSize, sizes[i]});

    // Compute the total amount of bytes to allocate.
    auto elementType = type.getElementType();
    assert((elementType.isIntOrFloat() || elementType.isa<VectorType>()) &&
           "invalid memref element type");
    uint64_t elementSize = 0;
    if (auto vectorType = elementType.dyn_cast<VectorType>())
      elementSize = vectorType.getNumElements() *
                    llvm::divideCeil(vectorType.getElementTypeBitWidth(), 8);
    else
      elementSize = llvm::divideCeil(elementType.getIntOrFloatBitWidth(), 8);
    cumulativeSize = rewriter.create<LLVM::MulOp>(
        op->getLoc(), getIndexType(),
        ArrayRef<Value *>{
            cumulativeSize,
            createIndexConstant(rewriter, op->getLoc(), elementSize)});

    // Insert the `malloc` declaration if it is not already present.
    auto module = op->getParentOfType<ModuleOp>();
    FuncOp mallocFunc = module.lookupSymbol<FuncOp>("malloc");
    if (!mallocFunc) {
      auto mallocType =
          rewriter.getFunctionType(getIndexType(), getVoidPtrType());
      mallocFunc =
          FuncOp::create(rewriter.getUnknownLoc(), "malloc", mallocType);
      module.push_back(mallocFunc);
    }

    // Allocate the underlying buffer and store a pointer to it in the MemRef
    // descriptor.
    Value *allocated =
        rewriter
            .create<LLVM::CallOp>(op->getLoc(), getVoidPtrType(),
                                  rewriter.getSymbolRefAttr(mallocFunc),
                                  cumulativeSize)
            .getResult(0);
    auto structElementType = lowering.convertType(elementType);
    auto elementPtrType =
        structElementType.cast<LLVM::LLVMType>().getPointerTo();
    allocated = rewriter.create<LLVM::BitcastOp>(op->getLoc(), elementPtrType,
                                                 ArrayRef<Value *>(allocated));

    // Deal with static memrefs
    if (numOperands == 0)
      return rewriter.replaceOp(op, allocated);

    // Create the MemRef descriptor.
    auto structType = lowering.convertType(type);
    Value *memRefDescriptor = rewriter.create<LLVM::UndefOp>(
        op->getLoc(), structType, ArrayRef<Value *>{});

    memRefDescriptor = rewriter.create<LLVM::InsertValueOp>(
        op->getLoc(), structType, memRefDescriptor, allocated,
        getIntegerArrayAttr(rewriter, 0));

    // Store dynamically allocated sizes in the descriptor.  Dynamic sizes are
    // passed in as operands.
    for (auto indexedSize : llvm::enumerate(operands)) {
      memRefDescriptor = rewriter.create<LLVM::InsertValueOp>(
          op->getLoc(), structType, memRefDescriptor, indexedSize.value(),
          getIntegerArrayAttr(rewriter, 1 + indexedSize.index()));
    }

    // Return the final value of the descriptor.
    rewriter.replaceOp(op, memRefDescriptor);
  }
};

// A `dealloc` is converted into a call to `free` on the underlying data buffer.
// The memref descriptor being an SSA value, there is no need to clean it up
// in any way.
struct DeallocOpLowering : public LLVMLegalizationPattern<DeallocOp> {
  using LLVMLegalizationPattern<DeallocOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 1 && "dealloc takes one operand");
    OperandAdaptor<DeallocOp> transformed(operands);

    // Insert the `free` declaration if it is not already present.
    FuncOp freeFunc =
        op->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>("free");
    if (!freeFunc) {
      auto freeType = rewriter.getFunctionType(getVoidPtrType(), {});
      freeFunc = FuncOp::create(rewriter.getUnknownLoc(), "free", freeType);
      op->getParentOfType<ModuleOp>().push_back(freeFunc);
    }

    auto type = transformed.memref()->getType().cast<LLVM::LLVMType>();
    auto hasStaticShape = type.getUnderlyingType()->isPointerTy();
    Type elementPtrType = hasStaticShape ? type : type.getStructElementType(0);
    Value *bufferPtr =
        extractMemRefElementPtr(rewriter, op->getLoc(), transformed.memref(),
                                elementPtrType, hasStaticShape);
    Value *casted = rewriter.create<LLVM::BitcastOp>(
        op->getLoc(), getVoidPtrType(), bufferPtr);
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(
        op, ArrayRef<Type>(), rewriter.getSymbolRefAttr(freeFunc), casted);
    return matchSuccess();
  }
};

struct MemRefCastOpLowering : public LLVMLegalizationPattern<MemRefCastOp> {
  using LLVMLegalizationPattern<MemRefCastOp>::LLVMLegalizationPattern;

  PatternMatchResult match(Operation *op) const override {
    auto memRefCastOp = cast<MemRefCastOp>(op);
    MemRefType sourceType =
        memRefCastOp.getOperand()->getType().cast<MemRefType>();
    MemRefType targetType = memRefCastOp.getType();
    return (isSupportedMemRefType(targetType) &&
            isSupportedMemRefType(sourceType))
               ? matchSuccess()
               : matchFailure();
  }

  void rewrite(Operation *op, ArrayRef<Value *> operands,
               ConversionPatternRewriter &rewriter) const override {
    auto memRefCastOp = cast<MemRefCastOp>(op);
    OperandAdaptor<MemRefCastOp> transformed(operands);
    auto targetType = memRefCastOp.getType();
    auto sourceType = memRefCastOp.getOperand()->getType().cast<MemRefType>();

    // Copy the data buffer pointer.
    auto elementTypePtr = getMemRefElementPtrType(targetType, lowering);
    Value *buffer =
        extractMemRefElementPtr(rewriter, op->getLoc(), transformed.source(),
                                elementTypePtr, sourceType.hasStaticShape());
    // Account for static memrefs as target types
    if (targetType.hasStaticShape())
      return rewriter.replaceOp(op, buffer);

    // Create the new MemRef descriptor.
    auto structType = lowering.convertType(targetType);
    Value *newDescriptor = rewriter.create<LLVM::UndefOp>(
        op->getLoc(), structType, ArrayRef<Value *>{});
    // Otherwise target type is dynamic memref, so create a proper descriptor.
    newDescriptor = rewriter.create<LLVM::InsertValueOp>(
        op->getLoc(), structType, newDescriptor, buffer,
        getIntegerArrayAttr(rewriter, 0));

    // Fill in the dynamic sizes of the new descriptor.  If the size was
    // dynamic, copy it from the old descriptor.  If the size was static, insert
    // the constant.  Note that the positions of dynamic sizes in the
    // descriptors start from 1 (the buffer pointer is at position zero).
    int64_t sourceDynamicDimIdx = 1;
    int64_t targetDynamicDimIdx = 1;
    for (int i = 0, e = sourceType.getRank(); i < e; ++i) {
      // Ignore new static sizes (they will be known from the type).  If the
      // size was dynamic, update the index of dynamic types.
      if (targetType.getShape()[i] != -1) {
        if (sourceType.getShape()[i] == -1)
          ++sourceDynamicDimIdx;
        continue;
      }

      auto sourceSize = sourceType.getShape()[i];
      Value *size =
          sourceSize == -1
              ? rewriter.create<LLVM::ExtractValueOp>(
                    op->getLoc(), getIndexType(),
                    transformed.source(), // NB: dynamic memref
                    getIntegerArrayAttr(rewriter, sourceDynamicDimIdx++))
              : createIndexConstant(rewriter, op->getLoc(), sourceSize);
      newDescriptor = rewriter.create<LLVM::InsertValueOp>(
          op->getLoc(), structType, newDescriptor, size,
          getIntegerArrayAttr(rewriter, targetDynamicDimIdx++));
    }
    assert(sourceDynamicDimIdx - 1 == sourceType.getNumDynamicDims() &&
           "source dynamic dimensions were not processed");
    assert(targetDynamicDimIdx - 1 == targetType.getNumDynamicDims() &&
           "target dynamic dimensions were not set up");

    rewriter.replaceOp(op, newDescriptor);
  }
};

// A `dim` is converted to a constant for static sizes and to an access to the
// size stored in the memref descriptor for dynamic sizes.
struct DimOpLowering : public LLVMLegalizationPattern<DimOp> {
  using LLVMLegalizationPattern<DimOp>::LLVMLegalizationPattern;

  PatternMatchResult match(Operation *op) const override {
    auto dimOp = cast<DimOp>(op);
    MemRefType type = dimOp.getOperand()->getType().cast<MemRefType>();
    return isSupportedMemRefType(type) ? matchSuccess() : matchFailure();
  }

  void rewrite(Operation *op, ArrayRef<Value *> operands,
               ConversionPatternRewriter &rewriter) const override {
    auto dimOp = cast<DimOp>(op);
    OperandAdaptor<DimOp> transformed(operands);
    MemRefType type = dimOp.getOperand()->getType().cast<MemRefType>();

    auto shape = type.getShape();
    uint64_t index = dimOp.getIndex();
    // Extract dynamic size from the memref descriptor and define static size
    // as a constant.
    if (shape[index] == -1) {
      // Find the position of the dynamic dimension in the list of dynamic sizes
      // by counting the number of preceding dynamic dimensions.  Start from 1
      // because the buffer pointer is at position zero.
      int64_t position = 1;
      for (uint64_t i = 0; i < index; ++i) {
        if (shape[i] == -1)
          ++position;
      }
      rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
          op, getIndexType(), transformed.memrefOrTensor(),
          getIntegerArrayAttr(rewriter, position));
    } else {
      rewriter.replaceOp(
          op, createIndexConstant(rewriter, op->getLoc(), shape[index]));
    }
  }
};

// Common base for load and store operations on MemRefs.  Restricts the match
// to supported MemRef types.  Provides functionality to emit code accessing a
// specific element of the underlying data buffer.
template <typename Derived>
struct LoadStoreOpLowering : public LLVMLegalizationPattern<Derived> {
  using LLVMLegalizationPattern<Derived>::LLVMLegalizationPattern;
  using Base = LoadStoreOpLowering<Derived>;

  PatternMatchResult match(Operation *op) const override {
    MemRefType type = cast<Derived>(op).getMemRefType();
    return isSupportedMemRefType(type) ? this->matchSuccess()
                                       : this->matchFailure();
  }

  // Given subscript indices and array sizes in row-major order,
  //   i_n, i_{n-1}, ..., i_1
  //   s_n, s_{n-1}, ..., s_1
  // obtain a value that corresponds to the linearized subscript
  //   \sum_k i_k * \prod_{j=1}^{k-1} s_j
  // by accumulating the running linearized value.
  // Note that `indices` and `allocSizes` are passed in the same order as they
  // appear in load/store operations and memref type declarations.
  Value *linearizeSubscripts(ConversionPatternRewriter &builder, Location loc,
                             ArrayRef<Value *> indices,
                             ArrayRef<Value *> allocSizes) const {
    assert(indices.size() == allocSizes.size() &&
           "mismatching number of indices and allocation sizes");
    assert(!indices.empty() && "cannot linearize a 0-dimensional access");

    Value *linearized = indices.front();
    for (int i = 1, nSizes = allocSizes.size(); i < nSizes; ++i) {
      linearized = builder.create<LLVM::MulOp>(
          loc, this->getIndexType(),
          ArrayRef<Value *>{linearized, allocSizes[i]});
      linearized = builder.create<LLVM::AddOp>(
          loc, this->getIndexType(), ArrayRef<Value *>{linearized, indices[i]});
    }
    return linearized;
  }

  // Given the MemRef type, a descriptor and a list of indices, extract the data
  // buffer pointer from the descriptor, convert multi-dimensional subscripts
  // into a linearized index (using dynamic size data from the descriptor if
  // necessary) and get the pointer to the buffer element identified by the
  // indices.
  Value *getElementPtr(Location loc, Type elementTypePtr,
                       ArrayRef<int64_t> shape, Value *memRefDescriptor,
                       ArrayRef<Value *> indices,
                       ConversionPatternRewriter &rewriter) const {
    // Get the list of MemRef sizes.  Static sizes are defined as constants.
    // Dynamic sizes are extracted from the MemRef descriptor, where they start
    // from the position 1 (the buffer is at position 0).
    SmallVector<Value *, 4> sizes;
    unsigned dynamicSizeIdx = 1;
    for (int64_t s : shape) {
      if (s == -1) {
        Value *size = rewriter.create<LLVM::ExtractValueOp>(
            loc, this->getIndexType(), memRefDescriptor,
            this->getIntegerArrayAttr(rewriter, dynamicSizeIdx++));
        sizes.push_back(size);
      } else {
        sizes.push_back(this->createIndexConstant(rewriter, loc, s));
      }
    }

    // The second and subsequent operands are access subscripts.  Obtain the
    // linearized address in the buffer.
    Value *subscript = linearizeSubscripts(rewriter, loc, indices, sizes);

    Value *dataPtr = rewriter.create<LLVM::ExtractValueOp>(
        loc, elementTypePtr, memRefDescriptor,
        this->getIntegerArrayAttr(rewriter, 0));
    return rewriter.create<LLVM::GEPOp>(loc, elementTypePtr,
                                        ArrayRef<Value *>{dataPtr, subscript},
                                        ArrayRef<NamedAttribute>{});
  }
  // This is a getElementPtr variant, where the value is a direct raw pointer.
  // If a shape is empty, we are dealing with a zero-dimensional memref. Return
  // the pointer unmodified in this case.  Otherwise, linearize subscripts to
  // obtain the offset with respect to the base pointer.  Use this offset to
  // compute and return the element pointer.
  Value *getRawElementPtr(Location loc, Type elementTypePtr,
                          ArrayRef<int64_t> shape, Value *rawDataPtr,
                          ArrayRef<Value *> indices,
                          ConversionPatternRewriter &rewriter) const {
    if (shape.empty())
      return rawDataPtr;

    SmallVector<Value *, 4> sizes;
    for (int64_t s : shape) {
      sizes.push_back(this->createIndexConstant(rewriter, loc, s));
    }

    Value *subscript = linearizeSubscripts(rewriter, loc, indices, sizes);
    return rewriter.create<LLVM::GEPOp>(
        loc, elementTypePtr, ArrayRef<Value *>{rawDataPtr, subscript},
        ArrayRef<NamedAttribute>{});
  }

  Value *getDataPtr(Location loc, MemRefType type, Value *dataPtr,
                    ArrayRef<Value *> indices,
                    ConversionPatternRewriter &rewriter,
                    llvm::Module &module) const {
    auto ptrType = getMemRefElementPtrType(type, this->lowering);
    auto shape = type.getShape();
    if (type.hasStaticShape()) {
      // NB: If memref was statically-shaped, dataPtr is pointer to raw data.
      return getRawElementPtr(loc, ptrType, shape, dataPtr, indices, rewriter);
    }
    return getElementPtr(loc, ptrType, shape, dataPtr, indices, rewriter);
  }
};

// Load operation is lowered to obtaining a pointer to the indexed element
// and loading it.
struct LoadOpLowering : public LoadStoreOpLowering<LoadOp> {
  using Base::Base;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loadOp = cast<LoadOp>(op);
    OperandAdaptor<LoadOp> transformed(operands);
    auto type = loadOp.getMemRefType();

    Value *dataPtr = getDataPtr(op->getLoc(), type, transformed.memref(),
                                transformed.indices(), rewriter, getModule());
    auto elementType = lowering.convertType(type.getElementType());

    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, elementType,
                                              ArrayRef<Value *>{dataPtr});
    return matchSuccess();
  }
};

// Store opreation is lowered to obtaining a pointer to the indexed element,
// and storing the given value to it.
struct StoreOpLowering : public LoadStoreOpLowering<StoreOp> {
  using Base::Base;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = cast<StoreOp>(op).getMemRefType();
    OperandAdaptor<StoreOp> transformed(operands);

    Value *dataPtr = getDataPtr(op->getLoc(), type, transformed.memref(),
                                transformed.indices(), rewriter, getModule());
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, transformed.value(),
                                               dataPtr);
    return matchSuccess();
  }
};

// The lowering of index_cast becomes an integer conversion since index becomes
// an integer.  If the bit width of the source and target integer types is the
// same, just erase the cast.  If the target type is wider, sign-extend the
// value, otherwise truncate it.
struct IndexCastOpLowering : public LLVMLegalizationPattern<IndexCastOp> {
  using LLVMLegalizationPattern<IndexCastOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    IndexCastOpOperandAdaptor transformed(operands);
    auto indexCastOp = cast<IndexCastOp>(op);

    auto targetType =
        this->lowering.convertType(indexCastOp.getResult()->getType())
            .cast<LLVM::LLVMType>();
    auto sourceType = transformed.in()->getType().cast<LLVM::LLVMType>();
    unsigned targetBits = targetType.getUnderlyingType()->getIntegerBitWidth();
    unsigned sourceBits = sourceType.getUnderlyingType()->getIntegerBitWidth();

    if (targetBits == sourceBits)
      rewriter.replaceOp(op, transformed.in());
    else if (targetBits < sourceBits)
      rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, targetType,
                                                 transformed.in());
    else
      rewriter.replaceOpWithNewOp<LLVM::SExtOp>(op, targetType,
                                                transformed.in());
    return matchSuccess();
  }
};

// Convert std.cmpi predicate into the LLVM dialect ICmpPredicate.  The two
// enums share the numerical values so just cast.
static LLVM::ICmpPredicate convertCmpIPredicate(CmpIPredicate pred) {
  return static_cast<LLVM::ICmpPredicate>(pred);
}

struct CmpIOpLowering : public LLVMLegalizationPattern<CmpIOp> {
  using LLVMLegalizationPattern<CmpIOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto cmpiOp = cast<CmpIOp>(op);
    CmpIOpOperandAdaptor transformed(operands);

    rewriter.replaceOpWithNewOp<LLVM::ICmpOp>(
        op, lowering.convertType(cmpiOp.getResult()->getType()),
        rewriter.getI64IntegerAttr(
            static_cast<int64_t>(convertCmpIPredicate(cmpiOp.getPredicate()))),
        transformed.lhs(), transformed.rhs());

    return matchSuccess();
  }
};

struct SIToFPLowering
    : public OneToOneLLVMOpLowering<SIToFPOp, LLVM::SIToFPOp> {
  using Super::Super;
};

// Base class for LLVM IR lowering terminator operations with successors.
template <typename SourceOp, typename TargetOp>
struct OneToOneLLVMTerminatorLowering
    : public LLVMLegalizationPattern<SourceOp> {
  using LLVMLegalizationPattern<SourceOp>::LLVMLegalizationPattern;
  using Super = OneToOneLLVMTerminatorLowering<SourceOp, TargetOp>;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> properOperands,
                  ArrayRef<Block *> destinations,
                  ArrayRef<ArrayRef<Value *>> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TargetOp>(op, properOperands, destinations,
                                          operands, op->getAttrs());
    return this->matchSuccess();
  }
};

// Special lowering pattern for `ReturnOps`.  Unlike all other operations,
// `ReturnOp` interacts with the function signature and must have as many
// operands as the function has return values.  Because in LLVM IR, functions
// can only return 0 or 1 value, we pack multiple values into a structure type.
// Emit `UndefOp` followed by `InsertValueOp`s to create such structure if
// necessary before returning it
struct ReturnOpLowering : public LLVMLegalizationPattern<ReturnOp> {
  using LLVMLegalizationPattern<ReturnOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numArguments = op->getNumOperands();

    // If ReturnOp has 0 or 1 operand, create it and return immediately.
    if (numArguments == 0) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(
          op, llvm::ArrayRef<Value *>(), llvm::ArrayRef<Block *>(),
          llvm::ArrayRef<llvm::ArrayRef<Value *>>(), op->getAttrs());
      return matchSuccess();
    }
    if (numArguments == 1) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(
          op, llvm::ArrayRef<Value *>(operands.front()),
          llvm::ArrayRef<Block *>(), llvm::ArrayRef<llvm::ArrayRef<Value *>>(),
          op->getAttrs());
      return matchSuccess();
    }

    // Otherwise, we need to pack the arguments into an LLVM struct type before
    // returning.
    auto packedType =
        lowering.packFunctionResults(llvm::to_vector<4>(op->getOperandTypes()));

    Value *packed = rewriter.create<LLVM::UndefOp>(op->getLoc(), packedType);
    for (unsigned i = 0; i < numArguments; ++i) {
      packed = rewriter.create<LLVM::InsertValueOp>(
          op->getLoc(), packedType, packed, operands[i],
          getIntegerArrayAttr(rewriter, i));
    }
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(
        op, llvm::makeArrayRef(packed), llvm::ArrayRef<Block *>(),
        llvm::ArrayRef<llvm::ArrayRef<Value *>>(), op->getAttrs());
    return matchSuccess();
  }
};

// FIXME: this should be tablegen'ed as well.
struct BranchOpLowering
    : public OneToOneLLVMTerminatorLowering<BranchOp, LLVM::BrOp> {
  using Super::Super;
};
struct CondBranchOpLowering
    : public OneToOneLLVMTerminatorLowering<CondBranchOp, LLVM::CondBrOp> {
  using Super::Super;
};

} // namespace

static void ensureDistinctSuccessors(Block &bb) {
  auto *terminator = bb.getTerminator();

  // Find repeated successors with arguments.
  llvm::SmallDenseMap<Block *, llvm::SmallVector<int, 4>> successorPositions;
  for (int i = 0, e = terminator->getNumSuccessors(); i < e; ++i) {
    Block *successor = terminator->getSuccessor(i);
    // Blocks with no arguments are safe even if they appear multiple times
    // because they don't need PHI nodes.
    if (successor->getNumArguments() == 0)
      continue;
    successorPositions[successor].push_back(i);
  }

  // If a successor appears for the second or more time in the terminator,
  // create a new dummy block that unconditionally branches to the original
  // destination, and retarget the terminator to branch to this new block.
  // There is no need to pass arguments to the dummy block because it will be
  // dominated by the original block and can therefore use any values defined in
  // the original block.
  for (const auto &successor : successorPositions) {
    const auto &positions = successor.second;
    // Start from the second occurrence of a block in the successor list.
    for (auto position = std::next(positions.begin()), end = positions.end();
         position != end; ++position) {
      auto *dummyBlock = new Block();
      bb.getParent()->push_back(dummyBlock);
      auto builder = OpBuilder(dummyBlock);
      SmallVector<Value *, 8> operands(
          terminator->getSuccessorOperands(*position));
      builder.create<BranchOp>(terminator->getLoc(), successor.first, operands);
      terminator->setSuccessor(dummyBlock, *position);
      for (int i = 0, e = terminator->getNumSuccessorOperands(*position); i < e;
           ++i)
        terminator->eraseSuccessorOperand(*position, i);
    }
  }
}

void mlir::LLVM::ensureDistinctSuccessors(ModuleOp m) {
  for (auto f : m.getOps<FuncOp>()) {
    for (auto &bb : f.getBlocks()) {
      ::ensureDistinctSuccessors(bb);
    }
  }
}

/// Collect a set of patterns to convert from the Standard dialect to LLVM.
void mlir::populateStdToLLVMConversionPatterns(
    LLVMTypeConverter &converter, OwningRewritePatternList &patterns) {
  // FIXME: this should be tablegen'ed
  patterns.insert<
      AddFOpLowering, AddIOpLowering, AndOpLowering, AllocOpLowering,
      BranchOpLowering, CallIndirectOpLowering, CallOpLowering, CmpIOpLowering,
      CondBranchOpLowering, ConstLLVMOpLowering, DeallocOpLowering,
      DimOpLowering, DivISOpLowering, DivIUOpLowering, DivFOpLowering,
      FuncOpConversion, IndexCastOpLowering, LoadOpLowering,
      MemRefCastOpLowering, MulFOpLowering, MulIOpLowering, OrOpLowering,
      RemISOpLowering, RemIUOpLowering, RemFOpLowering, ReturnOpLowering,
      SelectOpLowering, SIToFPLowering, StoreOpLowering, SubFOpLowering,
      SubIOpLowering, XOrOpLowering>(*converter.getDialect(), converter);
}

// Convert types using the stored LLVM IR module.
Type LLVMTypeConverter::convertType(Type t) { return convertStandardType(t); }

// Create an LLVM IR structure type if there is more than one result.
Type LLVMTypeConverter::packFunctionResults(ArrayRef<Type> types) {
  assert(!types.empty() && "expected non-empty list of type");

  if (types.size() == 1)
    return convertType(types.front());

  SmallVector<LLVM::LLVMType, 8> resultTypes;
  resultTypes.reserve(types.size());
  for (auto t : types) {
    auto converted = convertType(t).dyn_cast<LLVM::LLVMType>();
    if (!converted)
      return {};
    resultTypes.push_back(converted);
  }

  return LLVM::LLVMType::getStructTy(llvmDialect, resultTypes);
}

/// Create an instance of LLVMTypeConverter in the given context.
static std::unique_ptr<LLVMTypeConverter>
makeStandardToLLVMTypeConverter(MLIRContext *context) {
  return llvm::make_unique<LLVMTypeConverter>(context);
}

namespace {
/// A pass converting MLIR operations into the LLVM IR dialect.
struct LLVMLoweringPass : public ModulePass<LLVMLoweringPass> {
  // By default, the patterns are those converting Standard operations to the
  // LLVMIR dialect.
  explicit LLVMLoweringPass(
      LLVMPatternListFiller patternListFiller =
          populateStdToLLVMConversionPatterns,
      LLVMTypeConverterMaker converterBuilder = makeStandardToLLVMTypeConverter)
      : patternListFiller(patternListFiller),
        typeConverterMaker(converterBuilder) {}

  // Run the dialect converter on the module.
  void runOnModule() override {
    if (!typeConverterMaker || !patternListFiller)
      return signalPassFailure();

    ModuleOp m = getModule();
    LLVM::ensureDistinctSuccessors(m);
    std::unique_ptr<LLVMTypeConverter> typeConverter =
        typeConverterMaker(&getContext());
    if (!typeConverter)
      return signalPassFailure();

    OwningRewritePatternList patterns;
    populateLoopToStdConversionPatterns(patterns, m.getContext());
    patternListFiller(*typeConverter, patterns);

    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter->isSignatureLegal(op.getType());
    });
    if (failed(applyPartialConversion(m, target, std::move(patterns),
                                      typeConverter.get())))
      signalPassFailure();
  }

  // Callback for creating a list of patterns.  It is called every time in
  // runOnModule since applyPartialConversion consumes the list.
  LLVMPatternListFiller patternListFiller;

  // Callback for creating an instance of type converter.  The converter
  // constructor needs an MLIRContext, which is not available until runOnModule.
  LLVMTypeConverterMaker typeConverterMaker;
};
} // end namespace

ModulePassBase *mlir::createConvertToLLVMIRPass() {
  return new LLVMLoweringPass;
}

ModulePassBase *
mlir::createConvertToLLVMIRPass(LLVMPatternListFiller patternListFiller,
                                LLVMTypeConverterMaker typeConverterMaker) {
  return new LLVMLoweringPass(patternListFiller, typeConverterMaker);
}

static PassRegistration<LLVMLoweringPass>
    pass("lower-to-llvm", "Convert all functions to the LLVM IR dialect");
