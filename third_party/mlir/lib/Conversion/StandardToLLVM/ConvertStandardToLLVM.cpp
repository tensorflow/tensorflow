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
#include "mlir/Conversion/LoopToStandard/ConvertLoopToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
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

// Except for signatures, MLIR function types are converted into LLVM
// pointer-to-function types.
Type LLVMTypeConverter::convertFunctionType(FunctionType type) {
  SignatureConversion conversion(type.getNumInputs());
  LLVM::LLVMType converted =
      convertFunctionSignature(type, /*isVariadic=*/false, conversion);
  return converted.getPointerTo();
}

// Function types are converted to LLVM Function types by recursively converting
// argument and result types.  If MLIR Function has zero results, the LLVM
// Function has one VoidType result.  If MLIR Function has more than one result,
// they are into an LLVM StructType in their order of appearance.
LLVM::LLVMType LLVMTypeConverter::convertFunctionSignature(
    FunctionType type, bool isVariadic,
    LLVMTypeConverter::SignatureConversion &result) {
  // Convert argument types one by one and check for errors.
  for (auto &en : llvm::enumerate(type.getInputs()))
    if (failed(convertSignatureArg(en.index(), en.value(), result)))
      return {};

  SmallVector<LLVM::LLVMType, 8> argTypes;
  argTypes.reserve(llvm::size(result.getConvertedTypes()));
  for (Type type : result.getConvertedTypes())
    argTypes.push_back(unwrap(type));

  // If function does not return anything, create the void result type,
  // if it returns on element, convert it, otherwise pack the result types into
  // a struct.
  LLVM::LLVMType resultType =
      type.getNumResults() == 0
          ? LLVM::LLVMType::getVoidTy(llvmDialect)
          : unwrap(packFunctionResults(type.getResults()));
  if (!resultType)
    return {};
  return LLVM::LLVMType::getFunctionTy(resultType, argTypes, isVariadic);
}

// Convert a MemRef to an LLVM type. The result is a MemRef descriptor which
// contains:
//   1. the pointer to the data buffer, followed by
//   2.  a lowered `index`-type integer containing the distance between the
//   beginning of the buffer and the first element to be accessed through the
//   view, followed by
//   3. an array containing as many `index`-type integers as the rank of the
//   MemRef: the array represents the size, in number of elements, of the memref
//   along the given dimension. For constant MemRef dimensions, the
//   corresponding size entry is a constant whose runtime value must match the
//   static value, followed by
//   4. a second array containing as many `index`-type integers as the rank of
//   the MemRef: the second array represents the "stride" (in tensor abstraction
//   sense), i.e. the number of consecutive elements of the underlying buffer.
//   TODO(ntv, zinenko): add assertions for the static cases.
//
// template <typename Elem, size_t Rank>
// struct {
//   Elem *ptr;
//   int64_t offset;
//   int64_t sizes[Rank]; // omitted when rank == 0
//   int64_t strides[Rank]; // omitted when rank == 0
// };
static unsigned kPtrPosInMemRefDescriptor = 0;
static unsigned kOffsetPosInMemRefDescriptor = 1;
static unsigned kSizePosInMemRefDescriptor = 2;
static unsigned kStridePosInMemRefDescriptor = 3;
Type LLVMTypeConverter::convertMemRefType(MemRefType type) {
  int64_t offset;
  SmallVector<int64_t, 4> strides;
  bool strideSuccess = succeeded(getStridesAndOffset(type, strides, offset));
  assert(strideSuccess &&
         "Non-strided layout maps must have been normalized away");
  (void)strideSuccess;
  LLVM::LLVMType elementType = unwrap(convertType(type.getElementType()));
  if (!elementType)
    return {};
  auto ptrTy = elementType.getPointerTo(type.getMemorySpace());
  auto indexTy = getIndexType();
  auto rank = type.getRank();
  if (rank > 0) {
    auto arrayTy = LLVM::LLVMType::getArrayTy(indexTy, type.getRank());
    return LLVM::LLVMType::getStructTy(ptrTy, indexTy, arrayTy, arrayTy);
  }
  return LLVM::LLVMType::getStructTy(ptrTy, indexTy);
}

// Convert an n-D vector type to an LLVM vector type via (n-1)-D array type when
// n > 1.
// For example, `vector<4 x f32>` converts to `!llvm.type<"<4 x float>">` and
// `vector<4 x 8 x 16 f32>` converts to `!llvm<"[4 x [8 x <16 x float>]]">`.
Type LLVMTypeConverter::convertVectorType(VectorType type) {
  auto elementType = unwrap(convertType(type.getElementType()));
  if (!elementType)
    return {};
  auto vectorType =
      LLVM::LLVMType::getVectorTy(elementType, type.getShape().back());
  auto shape = type.getShape();
  for (int i = shape.size() - 2; i >= 0; --i)
    vectorType = LLVM::LLVMType::getArrayTy(vectorType, shape[i]);
  return vectorType;
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
  return converted.cast<LLVM::LLVMType>().getPointerTo(t.getMemorySpace());
}

LLVMOpLowering::LLVMOpLowering(StringRef rootOpName, MLIRContext *context,
                               LLVMTypeConverter &lowering_,
                               PatternBenefit benefit)
    : ConversionPattern(rootOpName, benefit, context), lowering(lowering_) {}

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

  LLVM::LLVMType getVoidType() const {
    return LLVM::LLVMType::getVoidTy(&dialect);
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

  // Extract raw data pointer value from a value representing a memref.
  static Value *extractMemRefElementPtr(ConversionPatternRewriter &builder,
                                        Location loc, Value *memref,
                                        Type elementTypePtr) {
    return builder.create<LLVM::ExtractValueOp>(
        loc, elementTypePtr, memref,
        builder.getIndexArrayAttr(kPtrPosInMemRefDescriptor));
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
    // Pack the result types into a struct.
    Type packedResult;
    if (type.getNumResults() != 0)
      if (!(packedResult = lowering.packFunctionResults(type.getResults())))
        return matchFailure();
    LLVM::LLVMType resultType = packedResult
                                    ? packedResult.cast<LLVM::LLVMType>()
                                    : LLVM::LLVMType::getVoidTy(&dialect);

    SmallVector<LLVM::LLVMType, 4> argTypes;
    argTypes.reserve(type.getNumInputs());
    SmallVector<unsigned, 4> promotedArgIndices;
    promotedArgIndices.reserve(type.getNumInputs());

    // Convert the original function arguments. Struct arguments are promoted to
    // pointer to struct arguments to allow calling external functions with
    // various ABIs (e.g. compiled from C/C++ on platform X).
    auto varargsAttr = funcOp.getAttrOfType<BoolAttr>("std.varargs");
    TypeConverter::SignatureConversion result(funcOp.getNumArguments());
    for (auto en : llvm::enumerate(type.getInputs())) {
      auto t = en.value();
      auto converted = lowering.convertType(t).dyn_cast<LLVM::LLVMType>();
      if (!converted)
        return matchFailure();
      if (t.isa<MemRefType>()) {
        converted = converted.getPointerTo();
        promotedArgIndices.push_back(en.index());
      }
      argTypes.push_back(converted);
    }
    for (unsigned idx = 0, e = argTypes.size(); idx < e; ++idx)
      result.addInputs(idx, argTypes[idx]);

    auto llvmType = LLVM::LLVMType::getFunctionTy(
        resultType, argTypes, varargsAttr && varargsAttr.getValue());

    // Only retain those attributes that are not constructed by build.
    SmallVector<NamedAttribute, 4> attributes;
    for (const auto &attr : funcOp.getAttrs()) {
      if (attr.first.is(SymbolTable::getSymbolAttrName()) ||
          attr.first.is(impl::getTypeAttrName()) ||
          attr.first.is("std.varargs"))
        continue;
      attributes.push_back(attr);
    }

    // Create an LLVM funcion.
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        op->getLoc(), funcOp.getName(), llvmType, attributes);
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());

    // Tell the rewriter to convert the region signature.
    rewriter.applySignatureConversion(&newFuncOp.getBody(), result);

    // Insert loads from memref descriptor pointers in function bodies.
    if (!newFuncOp.getBody().empty()) {
      Block *firstBlock = &newFuncOp.getBody().front();
      rewriter.setInsertionPoint(firstBlock, firstBlock->begin());
      for (unsigned idx : promotedArgIndices) {
        BlockArgument *arg = firstBlock->getArgument(idx);
        Value *loaded = rewriter.create<LLVM::LoadOp>(funcOp.getLoc(), arg);
        rewriter.replaceUsesOfBlockArgument(arg, loaded);
      }
    }

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
          rewriter.getIndexArrayAttr(i)));
    }
    rewriter.replaceOp(op, results);
    return this->matchSuccess();
  }
};

// Express `linearIndex` in terms of coordinates of `basis`.
// Returns the empty vector when linearIndex is out of the range [0, P] where
// P is the product of all the basis coordinates.
//
// Prerequisites:
//   Basis is an array of nonnegative integers (signed type inherited from
//   vector shape type).
static SmallVector<int64_t, 4> getCoordinates(ArrayRef<int64_t> basis,
                                              unsigned linearIndex) {
  SmallVector<int64_t, 4> res;
  res.reserve(basis.size());
  for (unsigned basisElement : llvm::reverse(basis)) {
    res.push_back(linearIndex % basisElement);
    linearIndex = linearIndex / basisElement;
  }
  if (linearIndex > 0)
    return {};
  std::reverse(res.begin(), res.end());
  return res;
}

template <typename SourceOp, unsigned OpCount> struct OpCountValidator {
  static_assert(
      std::is_base_of<
          typename OpTrait::NOperands<OpCount>::template Impl<SourceOp>,
          SourceOp>::value,
      "wrong operand count");
};

template <typename SourceOp> struct OpCountValidator<SourceOp, 1> {
  static_assert(std::is_base_of<OpTrait::OneOperand<SourceOp>, SourceOp>::value,
                "expected a single operand");
};

template <typename SourceOp, unsigned OpCount> void ValidateOpCount() {
  OpCountValidator<SourceOp, OpCount>();
}

// Basic lowering implementation for rewriting from Standard Ops to LLVM Dialect
// Ops for N-ary ops with one result. This supports higher-dimensional vector
// types.
template <typename SourceOp, typename TargetOp, unsigned OpCount>
struct NaryOpLLVMOpLowering : public LLVMLegalizationPattern<SourceOp> {
  using LLVMLegalizationPattern<SourceOp>::LLVMLegalizationPattern;
  using Super = NaryOpLLVMOpLowering<SourceOp, TargetOp, OpCount>;

  // Convert the type of the result to an LLVM type, pass operands as is,
  // preserve attributes.
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    ValidateOpCount<SourceOp, OpCount>();
    static_assert(
        std::is_base_of<OpTrait::OneResult<SourceOp>, SourceOp>::value,
        "expected single result op");
    static_assert(std::is_base_of<OpTrait::SameOperandsAndResultType<SourceOp>,
                                  SourceOp>::value,
                  "expected same operands and result type");

    // Cannot convert ops if their operands are not of LLVM type.
    for (Value *operand : operands) {
      if (!operand || !operand->getType().isa<LLVM::LLVMType>())
        return this->matchFailure();
    }

    auto loc = op->getLoc();
    auto llvmArrayTy = operands[0]->getType().cast<LLVM::LLVMType>();

    if (!llvmArrayTy.isArrayTy()) {
      auto newOp = rewriter.create<TargetOp>(
          op->getLoc(), operands[0]->getType(), operands, op->getAttrs());
      rewriter.replaceOp(op, newOp.getResult());
      return this->matchSuccess();
    }

    // Unroll iterated array type until we hit a non-array type.
    auto llvmTy = llvmArrayTy;
    SmallVector<int64_t, 4> arraySizes;
    while (llvmTy.isArrayTy()) {
      arraySizes.push_back(llvmTy.getArrayNumElements());
      llvmTy = llvmTy.getArrayElementType();
    }
    assert(llvmTy.isVectorTy() && "unexpected n-ary op over non-vector type");
    auto llvmVectorTy = llvmTy;

    // Iteratively extract a position coordinates with basis `arraySize` from a
    // `linearIndex` that is incremented at each step. This terminates when
    // `linearIndex` exceeds the range specified by `arraySize`.
    // This has the effect of fully unrolling the dimensions of the n-D array
    // type, getting to the underlying vector element.
    Value *desc = rewriter.create<LLVM::UndefOp>(loc, llvmArrayTy);
    unsigned ub = 1;
    for (auto s : arraySizes)
      ub *= s;
    for (unsigned linearIndex = 0; linearIndex < ub; ++linearIndex) {
      auto coords = getCoordinates(arraySizes, linearIndex);
      // Linear index is out of bounds, we are done.
      if (coords.empty())
        break;

      auto position = rewriter.getIndexArrayAttr(coords);

      // For this unrolled `position` corresponding to the `linearIndex`^th
      // element, extract operand vectors
      SmallVector<Value *, OpCount> extractedOperands;
      for (unsigned i = 0; i < OpCount; ++i) {
        extractedOperands.push_back(rewriter.create<LLVM::ExtractValueOp>(
            loc, llvmVectorTy, operands[i], position));
      }
      Value *newVal = rewriter.create<TargetOp>(
          loc, llvmVectorTy, extractedOperands, op->getAttrs());
      desc = rewriter.create<LLVM::InsertValueOp>(loc, llvmArrayTy, desc,
                                                  newVal, position);
    }
    rewriter.replaceOp(op, desc);
    return this->matchSuccess();
  }
};

template <typename SourceOp, typename TargetOp>
using UnaryOpLLVMOpLowering = NaryOpLLVMOpLowering<SourceOp, TargetOp, 1>;
template <typename SourceOp, typename TargetOp>
using BinaryOpLLVMOpLowering = NaryOpLLVMOpLowering<SourceOp, TargetOp, 2>;

// Specific lowerings.
// FIXME: this should be tablegen'ed.
struct ExpOpLowering : public UnaryOpLLVMOpLowering<ExpOp, LLVM::ExpOp> {
  using Super::Super;
};
struct AddIOpLowering : public BinaryOpLLVMOpLowering<AddIOp, LLVM::AddOp> {
  using Super::Super;
};
struct SubIOpLowering : public BinaryOpLLVMOpLowering<SubIOp, LLVM::SubOp> {
  using Super::Super;
};
struct MulIOpLowering : public BinaryOpLLVMOpLowering<MulIOp, LLVM::MulOp> {
  using Super::Super;
};
struct DivISOpLowering : public BinaryOpLLVMOpLowering<DivISOp, LLVM::SDivOp> {
  using Super::Super;
};
struct DivIUOpLowering : public BinaryOpLLVMOpLowering<DivIUOp, LLVM::UDivOp> {
  using Super::Super;
};
struct RemISOpLowering : public BinaryOpLLVMOpLowering<RemISOp, LLVM::SRemOp> {
  using Super::Super;
};
struct RemIUOpLowering : public BinaryOpLLVMOpLowering<RemIUOp, LLVM::URemOp> {
  using Super::Super;
};
struct AndOpLowering : public BinaryOpLLVMOpLowering<AndOp, LLVM::AndOp> {
  using Super::Super;
};
struct OrOpLowering : public BinaryOpLLVMOpLowering<OrOp, LLVM::OrOp> {
  using Super::Super;
};
struct XOrOpLowering : public BinaryOpLLVMOpLowering<XOrOp, LLVM::XOrOp> {
  using Super::Super;
};
struct AddFOpLowering : public BinaryOpLLVMOpLowering<AddFOp, LLVM::FAddOp> {
  using Super::Super;
};
struct SubFOpLowering : public BinaryOpLLVMOpLowering<SubFOp, LLVM::FSubOp> {
  using Super::Super;
};
struct MulFOpLowering : public BinaryOpLLVMOpLowering<MulFOp, LLVM::FMulOp> {
  using Super::Super;
};
struct DivFOpLowering : public BinaryOpLLVMOpLowering<DivFOp, LLVM::FDivOp> {
  using Super::Super;
};
struct RemFOpLowering : public BinaryOpLLVMOpLowering<RemFOp, LLVM::FRemOp> {
  using Super::Super;
};
struct SelectOpLowering
    : public OneToOneLLVMOpLowering<SelectOp, LLVM::SelectOp> {
  using Super::Super;
};
struct ConstLLVMOpLowering
    : public OneToOneLLVMOpLowering<ConstantOp, LLVM::ConstantOp> {
  using Super::Super;
};

// Check if the MemRefType `type` is supported by the lowering. We currently
// only support memrefs with identity maps.
static bool isSupportedMemRefType(MemRefType type) {
  return type.getAffineMaps().empty() ||
         llvm::all_of(type.getAffineMaps(),
                      [](AffineMap map) { return map.isIdentity(); });
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
    if (isSupportedMemRefType(type))
      return matchSuccess();

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(type, strides, offset);
    if (failed(successStrides))
      return matchFailure();

    // Dynamic strides are ok if they can be deduced from dynamic sizes (which
    // is guaranteed when succeeded(successStrides)). Dynamic offset however can
    // never be alloc'ed.
    if (offset == MemRefType::getDynamicStrideOrOffset())
      return matchFailure();

    return matchSuccess();
  }

  void rewrite(Operation *op, ArrayRef<Value *> operands,
               ConversionPatternRewriter &rewriter) const override {
    auto allocOp = cast<AllocOp>(op);
    MemRefType type = allocOp.getType();

    // Get actual sizes of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.  In case of
    // zero-dimensional memref, assume a scalar (size 1).
    SmallVector<Value *, 4> sizes;
    sizes.reserve(type.getRank());
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

    // Compute the size of an individual element. This emits the MLIR equivalent
    // of the following sizeof(...) implementation in LLVM IR:
    //   %0 = getelementptr %elementType* null, %indexType 1
    //   %1 = ptrtoint %elementType* %0 to %indexType
    // which is a common pattern of getting the size of a type in bytes.
    auto elementType = type.getElementType();
    auto convertedPtrType =
        lowering.convertType(elementType).cast<LLVM::LLVMType>().getPointerTo();
    auto nullPtr =
        rewriter.create<LLVM::NullOp>(op->getLoc(), convertedPtrType);
    auto one = createIndexConstant(rewriter, op->getLoc(), 1);
    auto gep = rewriter.create<LLVM::GEPOp>(op->getLoc(), convertedPtrType,
                                            ArrayRef<Value *>{nullPtr, one});
    auto elementSize =
        rewriter.create<LLVM::PtrToIntOp>(op->getLoc(), getIndexType(), gep);
    cumulativeSize = rewriter.create<LLVM::MulOp>(
        op->getLoc(), getIndexType(),
        ArrayRef<Value *>{cumulativeSize, elementSize});

    // Insert the `malloc` declaration if it is not already present.
    auto module = op->getParentOfType<ModuleOp>();
    auto mallocFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("malloc");
    if (!mallocFunc) {
      OpBuilder moduleBuilder(op->getParentOfType<ModuleOp>().getBodyRegion());
      mallocFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(
          rewriter.getUnknownLoc(), "malloc",
          LLVM::LLVMType::getFunctionTy(getVoidPtrType(), getIndexType(),
                                        /*isVarArg=*/false));
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
    auto elementPtrType = structElementType.cast<LLVM::LLVMType>().getPointerTo(
        type.getMemorySpace());
    allocated = rewriter.create<LLVM::BitcastOp>(op->getLoc(), elementPtrType,
                                                 ArrayRef<Value *>(allocated));

    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(type, strides, offset);
    assert(succeeded(successStrides) && "unexpected non-strided memref");
    (void)successStrides;
    assert(offset != MemRefType::getDynamicStrideOrOffset() &&
           "unexpected dynamic offset");

    // 0-D memref corner case: they have size 1 ...
    assert(((type.getRank() == 0 && strides.empty() && sizes.size() == 1) ||
            (strides.size() == sizes.size())) &&
           "unexpected number of strides");

    // Create the MemRef descriptor.
    auto structType = lowering.convertType(type);
    Value *memRefDescriptor = rewriter.create<LLVM::UndefOp>(
        op->getLoc(), structType, ArrayRef<Value *>{});

    memRefDescriptor = rewriter.create<LLVM::InsertValueOp>(
        op->getLoc(), structType, memRefDescriptor, allocated,
        rewriter.getIndexArrayAttr(kPtrPosInMemRefDescriptor));
    memRefDescriptor = rewriter.create<LLVM::InsertValueOp>(
        op->getLoc(), structType, memRefDescriptor,
        createIndexConstant(rewriter, op->getLoc(), offset),
        rewriter.getIndexArrayAttr(kOffsetPosInMemRefDescriptor));

    if (type.getRank() == 0)
      // No size/stride descriptor in memref, return the descriptor value.
      return rewriter.replaceOp(op, memRefDescriptor);

    // Store all sizes in the descriptor. Only dynamic sizes are passed in as
    // operands to AllocOp.
    Value *runningStride = nullptr;
    // Iterate strides in reverse order, compute runningStride and strideValues.
    auto nStrides = strides.size();
    SmallVector<Value *, 4> strideValues(nStrides, nullptr);
    for (auto indexedStride : llvm::enumerate(llvm::reverse(strides))) {
      int64_t index = nStrides - 1 - indexedStride.index();
      if (strides[index] == MemRefType::getDynamicStrideOrOffset())
        // Identity layout map is enforced in the match function, so we compute:
        //   `runningStride *= sizes[index]`
        runningStride = runningStride
                            ? rewriter.create<LLVM::MulOp>(
                                  op->getLoc(), runningStride, sizes[index])
                            : createIndexConstant(rewriter, op->getLoc(), 1);
      else
        runningStride =
            createIndexConstant(rewriter, op->getLoc(), strides[index]);
      strideValues[index] = runningStride;
    }
    // Fill size and stride descriptors in memref.
    for (auto indexedSize : llvm::enumerate(sizes)) {
      int64_t index = indexedSize.index();
      memRefDescriptor = rewriter.create<LLVM::InsertValueOp>(
          op->getLoc(), structType, memRefDescriptor, indexedSize.value(),
          rewriter.getI64ArrayAttr({kSizePosInMemRefDescriptor, index}));
      memRefDescriptor = rewriter.create<LLVM::InsertValueOp>(
          op->getLoc(), structType, memRefDescriptor, strideValues[index],
          rewriter.getI64ArrayAttr({kStridePosInMemRefDescriptor, index}));
    }

    // Return the final value of the descriptor.
    rewriter.replaceOp(op, memRefDescriptor);
  }
};

// A CallOp automatically promotes MemRefType to a sequence of alloca/store and
// passes the pointer to the MemRef across function boundaries.
template <typename CallOpType>
struct CallOpInterfaceLowering : public LLVMLegalizationPattern<CallOpType> {
  using LLVMLegalizationPattern<CallOpType>::LLVMLegalizationPattern;
  using Super = CallOpInterfaceLowering<CallOpType>;
  using Base = LLVMLegalizationPattern<CallOpType>;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    OperandAdaptor<CallOpType> transformed(operands);
    auto callOp = cast<CallOpType>(op);

    // Pack the result types into a struct.
    Type packedResult;
    unsigned numResults = callOp.getNumResults();
    auto resultTypes = llvm::to_vector<4>(callOp.getResultTypes());
    if (numResults != 0) {
      if (!(packedResult = this->lowering.packFunctionResults(resultTypes)))
        return this->matchFailure();
    }

    SmallVector<Value *, 4> opOperands(op->getOperands());
    auto promoted = this->lowering.promoteMemRefDescriptors(
        op->getLoc(), opOperands, operands, rewriter);
    auto newOp = rewriter.create<LLVM::CallOp>(op->getLoc(), packedResult,
                                               promoted, op->getAttrs());

    // If < 2 results, packing did not do anything and we can just return.
    if (numResults < 2) {
      SmallVector<Value *, 4> results(newOp.getResults());
      rewriter.replaceOp(op, results);
      return this->matchSuccess();
    }

    // Otherwise, it had been converted to an operation producing a structure.
    // Extract individual results from the structure and return them as list.
    // TODO(aminim, ntv, riverriddle, zinenko): this seems like patching around
    // a particular interaction between MemRefType and CallOp lowering. Find a
    // way to avoid special casing.
    SmallVector<Value *, 4> results;
    results.reserve(numResults);
    for (unsigned i = 0; i < numResults; ++i) {
      auto type = this->lowering.convertType(op->getResult(i)->getType());
      results.push_back(rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), type, newOp.getOperation()->getResult(0),
          rewriter.getIndexArrayAttr(i)));
    }
    rewriter.replaceOp(op, results);

    return this->matchSuccess();
  }
};

struct CallOpLowering : public CallOpInterfaceLowering<CallOp> {
  using Super::Super;
};

struct CallIndirectOpLowering : public CallOpInterfaceLowering<CallIndirectOp> {
  using Super::Super;
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
    auto freeFunc =
        op->getParentOfType<ModuleOp>().lookupSymbol<LLVM::LLVMFuncOp>("free");
    if (!freeFunc) {
      OpBuilder moduleBuilder(op->getParentOfType<ModuleOp>().getBodyRegion());
      freeFunc = moduleBuilder.create<LLVM::LLVMFuncOp>(
          rewriter.getUnknownLoc(), "free",
          LLVM::LLVMType::getFunctionTy(getVoidType(), getVoidPtrType(),
                                        /*isVarArg=*/false));
    }

    auto type = transformed.memref()->getType().cast<LLVM::LLVMType>();
    Type elementPtrType = type.getStructElementType(kPtrPosInMemRefDescriptor);
    Value *bufferPtr = extractMemRefElementPtr(
        rewriter, op->getLoc(), transformed.memref(), elementPtrType);
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
    // memref_cast is defined for source and destination memref types with the
    // same element type, same mappings, same address space and same rank.
    // Therefore a simple bitcast suffices. If not it is undefined behavior.
    auto targetStructType = lowering.convertType(memRefCastOp.getType());
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, targetStructType,
                                                 transformed.source());
  }
};

// A `dim` is converted to a constant for static sizes and to an access to the
// size stored in the memref descriptor for dynamic sizes.
struct DimOpLowering : public LLVMLegalizationPattern<DimOp> {
  using LLVMLegalizationPattern<DimOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto dimOp = cast<DimOp>(op);
    OperandAdaptor<DimOp> transformed(operands);
    MemRefType type = dimOp.getOperand()->getType().cast<MemRefType>();

    auto shape = type.getShape();
    int64_t index = dimOp.getIndex();
    // Extract dynamic size from the memref descriptor.
    if (ShapedType::isDynamic(shape[index]))
      rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
          op, getIndexType(), transformed.memrefOrTensor(),
          rewriter.getI64ArrayAttr({kSizePosInMemRefDescriptor, index}));
    else
      // Use constant for static size.
      rewriter.replaceOp(
          op, createIndexConstant(rewriter, op->getLoc(), shape[index]));
    return matchSuccess();
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

  // This is a strided getElementPtr variant that linearizes subscripts as:
  //   `base_offset + index_0 * stride_0 + ... + index_n * stride_n`.
  Value *getStridedElementPtr(Location loc, Type elementTypePtr,
                              Value *memRefDescriptor,
                              ArrayRef<Value *> indices,
                              ArrayRef<int64_t> strides, int64_t offset,
                              ConversionPatternRewriter &rewriter) const {
    auto indexTy = this->getIndexType();
    Value *base = this->extractMemRefElementPtr(rewriter, loc, memRefDescriptor,
                                                elementTypePtr);
    Value *offsetValue =
        offset == MemRefType::getDynamicStrideOrOffset()
            ? rewriter.create<LLVM::ExtractValueOp>(
                  loc, indexTy, memRefDescriptor,
                  rewriter.getIndexArrayAttr(kOffsetPosInMemRefDescriptor))
            : this->createIndexConstant(rewriter, loc, offset);
    for (int i = 0, e = indices.size(); i < e; ++i) {
      Value *stride;
      if (strides[i] != MemRefType::getDynamicStrideOrOffset()) {
        // Use static stride.
        auto attr =
            rewriter.getIntegerAttr(rewriter.getIndexType(), strides[i]);
        stride = rewriter.create<LLVM::ConstantOp>(loc, indexTy, attr);
      } else {
        // Use dynamic stride.
        stride = rewriter.create<LLVM::ExtractValueOp>(
            loc, indexTy, memRefDescriptor,
            rewriter.getIndexArrayAttr({kStridePosInMemRefDescriptor, i}));
      }
      Value *additionalOffset =
          rewriter.create<LLVM::MulOp>(loc, indices[i], stride);
      offsetValue =
          rewriter.create<LLVM::AddOp>(loc, offsetValue, additionalOffset);
    }
    return rewriter.create<LLVM::GEPOp>(loc, elementTypePtr, base, offsetValue);
  }

  Value *getDataPtr(Location loc, MemRefType type, Value *memRefDesc,
                    ArrayRef<Value *> indices,
                    ConversionPatternRewriter &rewriter,
                    llvm::Module &module) const {
    auto ptrType = getMemRefElementPtrType(type, this->lowering);
    int64_t offset;
    SmallVector<int64_t, 4> strides;
    auto successStrides = getStridesAndOffset(type, strides, offset);
    assert(succeeded(successStrides) && "unexpected non-strided memref");
    (void)successStrides;
    return getStridedElementPtr(loc, ptrType, memRefDesc, indices, strides,
                                offset, rewriter);
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
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, dataPtr);
    return matchSuccess();
  }
};

// Store operation is lowered to obtaining a pointer to the indexed element,
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

// Convert std.cmp predicate into the LLVM dialect CmpPredicate.  The two
// enums share the numerical values so just cast.
template <typename LLVMPredType, typename StdPredType>
static LLVMPredType convertCmpPredicate(StdPredType pred) {
  return static_cast<LLVMPredType>(pred);
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
        rewriter.getI64IntegerAttr(static_cast<int64_t>(
            convertCmpPredicate<LLVM::ICmpPredicate>(cmpiOp.getPredicate()))),
        transformed.lhs(), transformed.rhs());

    return matchSuccess();
  }
};

struct CmpFOpLowering : public LLVMLegalizationPattern<CmpFOp> {
  using LLVMLegalizationPattern<CmpFOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto cmpfOp = cast<CmpFOp>(op);
    CmpFOpOperandAdaptor transformed(operands);

    rewriter.replaceOpWithNewOp<LLVM::FCmpOp>(
        op, lowering.convertType(cmpfOp.getResult()->getType()),
        rewriter.getI64IntegerAttr(static_cast<int64_t>(
            convertCmpPredicate<LLVM::FCmpPredicate>(cmpfOp.getPredicate()))),
        transformed.lhs(), transformed.rhs());

    return matchSuccess();
  }
};

struct SIToFPLowering
    : public OneToOneLLVMOpLowering<SIToFPOp, LLVM::SIToFPOp> {
  using Super::Super;
};

struct FPExtLowering : public OneToOneLLVMOpLowering<FPExtOp, LLVM::FPExtOp> {
  using Super::Super;
};

struct FPTruncLowering
    : public OneToOneLLVMOpLowering<FPTruncOp, LLVM::FPTruncOp> {
  using Super::Super;
};

struct SignExtendIOpLowering
    : public OneToOneLLVMOpLowering<SignExtendIOp, LLVM::SExtOp> {
  using Super::Super;
};

struct TruncateIOpLowering
    : public OneToOneLLVMOpLowering<TruncateIOp, LLVM::TruncOp> {
  using Super::Super;
};

struct ZeroExtendIOpLowering
    : public OneToOneLLVMOpLowering<ZeroExtendIOp, LLVM::ZExtOp> {
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
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, llvm::ArrayRef<Value *>(),
                                                  llvm::ArrayRef<Block *>(),
                                                  op->getAttrs());
      return matchSuccess();
    }
    if (numArguments == 1) {
      rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(
          op, llvm::ArrayRef<Value *>(operands.front()),
          llvm::ArrayRef<Block *>(), op->getAttrs());
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
          rewriter.getIndexArrayAttr(i));
    }
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, llvm::makeArrayRef(packed),
                                                llvm::ArrayRef<Block *>(),
                                                op->getAttrs());
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

// The Splat operation is lowered to an insertelement + a shufflevector
// operation. Splat to only 1-d vector result types are lowered.
struct SplatOpLowering : public LLVMLegalizationPattern<SplatOp> {
  using LLVMLegalizationPattern<SplatOp>::LLVMLegalizationPattern;

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto splatOp = cast<SplatOp>(op);
    VectorType resultType = splatOp.getType().dyn_cast<VectorType>();
    if (!resultType || resultType.getRank() != 1)
      return matchFailure();

    // First insert it into an undef vector so we can shuffle it.
    auto vectorType = lowering.convertType(splatOp.getType());
    Value *undef = rewriter.create<LLVM::UndefOp>(op->getLoc(), vectorType);
    auto zero = rewriter.create<LLVM::ConstantOp>(
        op->getLoc(), lowering.convertType(rewriter.getIntegerType(32)),
        rewriter.getZeroAttr(rewriter.getIntegerType(32)));

    auto v = rewriter.create<LLVM::InsertElementOp>(
        op->getLoc(), vectorType, undef, splatOp.getOperand(), zero);

    int64_t width = splatOp.getType().cast<VectorType>().getDimSize(0);
    SmallVector<int32_t, 4> zeroValues(width, 0);

    // Shuffle the value across the desired number of elements.
    ArrayAttr zeroAttrs = rewriter.getI32ArrayAttr(zeroValues);
    rewriter.replaceOpWithNewOp<LLVM::ShuffleVectorOp>(op, v, undef, zeroAttrs);
    return matchSuccess();
  }
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
  // clang-format off
  patterns.insert<
      AddFOpLowering,
      AddIOpLowering,
      AllocOpLowering,
      AndOpLowering,
      BranchOpLowering,
      CallIndirectOpLowering,
      CallOpLowering,
      CmpFOpLowering,
      CmpIOpLowering,
      CondBranchOpLowering,
      ConstLLVMOpLowering,
      DeallocOpLowering,
      DimOpLowering,
      DivFOpLowering,
      DivISOpLowering,
      DivIUOpLowering,
      ExpOpLowering,
      FPExtLowering,
      FPTruncLowering,
      FuncOpConversion,
      IndexCastOpLowering,
      LoadOpLowering,
      MemRefCastOpLowering,
      MulFOpLowering,
      MulIOpLowering,
      OrOpLowering,
      RemFOpLowering,
      RemISOpLowering,
      RemIUOpLowering,
      ReturnOpLowering,
      SIToFPLowering,
      SelectOpLowering,
      SignExtendIOpLowering,
      SplatOpLowering,
      StoreOpLowering,
      SubFOpLowering,
      SubIOpLowering,
      TruncateIOpLowering,
      XOrOpLowering,
      ZeroExtendIOpLowering>(*converter.getDialect(), converter);
  // clang-format on
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

Value *LLVMTypeConverter::promoteOneMemRefDescriptor(Location loc,
                                                     Value *operand,
                                                     OpBuilder &builder) {
  auto *context = builder.getContext();
  auto int64Ty = LLVM::LLVMType::getInt64Ty(getDialect());
  auto indexType = IndexType::get(context);
  // Alloca with proper alignment. We do not expect optimizations of this
  // alloca op and so we omit allocating at the entry block.
  auto ptrType = operand->getType().cast<LLVM::LLVMType>().getPointerTo();
  Value *one = builder.create<LLVM::ConstantOp>(loc, int64Ty,
                                                IntegerAttr::get(indexType, 1));
  Value *allocated =
      builder.create<LLVM::AllocaOp>(loc, ptrType, one, /*alignment=*/0);
  // Store into the alloca'ed descriptor.
  builder.create<LLVM::StoreOp>(loc, operand, allocated);
  return allocated;
}

SmallVector<Value *, 4> LLVMTypeConverter::promoteMemRefDescriptors(
    Location loc, ArrayRef<Value *> opOperands, ArrayRef<Value *> operands,
    OpBuilder &builder) {
  SmallVector<Value *, 4> promotedOperands;
  promotedOperands.reserve(operands.size());
  for (auto it : llvm::zip(opOperands, operands)) {
    auto *operand = std::get<0>(it);
    auto *llvmOperand = std::get<1>(it);
    if (!operand->getType().isa<MemRefType>()) {
      promotedOperands.push_back(operand);
      continue;
    }
    promotedOperands.push_back(
        promoteOneMemRefDescriptor(loc, llvmOperand, builder));
  }
  return promotedOperands;
}

/// Create an instance of LLVMTypeConverter in the given context.
static std::unique_ptr<LLVMTypeConverter>
makeStandardToLLVMTypeConverter(MLIRContext *context) {
  return std::make_unique<LLVMTypeConverter>(context);
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
    if (failed(applyPartialConversion(m, target, patterns, &*typeConverter)))
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

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createLowerToLLVMPass() {
  return std::make_unique<LLVMLoweringPass>();
}

std::unique_ptr<OpPassBase<ModuleOp>>
mlir::createLowerToLLVMPass(LLVMPatternListFiller patternListFiller,
                            LLVMTypeConverterMaker typeConverterMaker) {
  return std::make_unique<LLVMLoweringPass>(patternListFiller,
                                            typeConverterMaker);
}

static PassRegistration<LLVMLoweringPass>
    pass("lower-to-llvm", "Convert all functions to the LLVM IR dialect");
