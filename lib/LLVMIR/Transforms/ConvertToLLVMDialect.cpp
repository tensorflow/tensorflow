//===- ConvertToLLVMDialect.cpp - MLIR to LLVM dialect conversion ---------===//
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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/Support/Functional.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"

using namespace mlir;

namespace {
// Type converter for the LLVM IR dialect.  Converts MLIR standard and builtin
// types into equivalent LLVM IR dialect types.
class TypeConverter {
public:
  // Convert one type `t ` and register it in the `llvmModule`.  The latter may
  // be used to extract information specific to the data layout.
  // Dispatches to the private functions below based on the actual type.
  static Type convert(Type t, llvm::Module &llvmModule);

  // Convert the element type of the memref `t` to to an LLVM type, get a
  // pointer LLVM type pointing to the converted `t`, wrap it into the MLIR LLVM
  // dialect type and return.
  static Type getMemRefElementPtrType(MemRefType t, llvm::Module &llvmModule);

  // Convert a non-empty list of types to an LLVM IR dialect type wrapping an
  // LLVM IR structure type, elements of which are formed by converting
  // individual types in the given list.  Register the type in the `llvmModule`.
  // The module may be also used to query the data layout.
  static Type pack(ArrayRef<Type> types, llvm::Module &llvmModule,
                   MLIRContext &context);

  // Convert a function signature type to the LLVM IR dialect.  The outer
  // function type remains `mlir::FunctionType`.  Argument types are converted
  // to LLVM IR as is.  If the function returns a single result, its type is
  // converted.  Otherwise, the types of results are packed into an LLVM IR
  // structure type.
  static FunctionType convertFunctionSignature(FunctionType t,
                                               llvm::Module &llvmModule);

private:
  // Construct a type converter.
  explicit TypeConverter(llvm::Module &llvmModule, MLIRContext *context)
      : module(llvmModule), llvmContext(llvmModule.getContext()),
        builder(llvmModule.getContext()), mlirContext(context) {}

  // Convert a function type.  The arguments and results are converted one by
  // one.  Additionally, if the function returns more than one value, pack the
  // results into an LLVM IR structure type so that the converted function type
  // returns at most one result.
  Type convertFunctionType(FunctionType type);

  // Convert function type arguments and results without converting the
  // function type itself.
  FunctionType convertFunctionSignatureType(FunctionType type);

  // Convert the index type.  Uses llvmModule data layout to create an integer
  // of the pointer bitwidth.
  Type convertIndexType(IndexType type);

  // Convert an integer type `i*` to `!llvm<"i*">`.
  Type convertIntegerType(IntegerType type);

  // Convert a floating point type: `f16` to `!llvm<"half">`, `f32` to
  // `!llvm<"float">` and `f64` to `!llvm<"double">`.  `bf16` is not supported
  // by LLVM.
  Type convertFloatType(FloatType type);

  // Convert a memref type into an LLVM structure type with:
  //   1. a pointer to the memref element type
  //   2. as many index types as memref has dynamic dimensions.
  Type convertMemRefType(MemRefType type);

  // Convert a 1D vector type into an LLVM vector type.
  Type convertVectorType(VectorType type);

  // Convert a non-empty list of types into an LLVM structure type containing
  // those types.  If the list contains a single element, convert the element
  // directly.
  Type getPackedResultType(ArrayRef<Type> types);

  // Convert a type to the LLVM IR dialect.  Returns a null type in case of
  // error.
  Type convertType(Type type);

  // Get the LLVM representation of the index type based on the bitwidth of the
  // pointer as defined by the data layout of the module.
  llvm::IntegerType *getIndexType();

  // Wrap the given LLVM IR type into an LLVM IR dialect type.
  Type wrap(llvm::Type *llvmType) {
    return LLVM::LLVMType::get(mlirContext, llvmType);
  }

  // Extract an LLVM IR type from the LLVM IR dialect type.
  llvm::Type *unwrap(Type type) {
    if (!type)
      return nullptr;
    auto wrappedLLVMType = type.dyn_cast<LLVM::LLVMType>();
    if (!wrappedLLVMType)
      return mlirContext->emitError(UnknownLoc::get(mlirContext),
                                    "conversion resulted in a non-LLVM type"),
             nullptr;
    return wrappedLLVMType.getUnderlyingType();
  }

  llvm::Module &module;
  llvm::LLVMContext &llvmContext;
  llvm::IRBuilder<> builder;

  MLIRContext *mlirContext;
};
} // end anonymous namespace

llvm::IntegerType *TypeConverter::getIndexType() {
  return builder.getIntNTy(module.getDataLayout().getPointerSizeInBits());
}

Type TypeConverter::convertIndexType(IndexType type) {
  return wrap(getIndexType());
}

Type TypeConverter::convertIntegerType(IntegerType type) {
  return wrap(builder.getIntNTy(type.getWidth()));
}

Type TypeConverter::convertFloatType(FloatType type) {
  MLIRContext *context = type.getContext();
  switch (type.getKind()) {
  case mlir::StandardTypes::F32:
    return wrap(builder.getFloatTy());
  case mlir::StandardTypes::F64:
    return wrap(builder.getDoubleTy());
  case mlir::StandardTypes::F16:
    return wrap(builder.getHalfTy());
  case mlir::StandardTypes::BF16:
    return context->emitError(UnknownLoc::get(context),
                              "unsupported type: BF16"),
           Type();
  default:
    llvm_unreachable("non-float type in convertFloatType");
  }
}

// If `types` has more than one type, pack them into an LLVM StructType,
// otherwise just convert the type.
Type TypeConverter::getPackedResultType(ArrayRef<Type> types) {
  // We don't convert zero-valued functions to one-valued functions returning
  // void yet.
  assert(!types.empty() && "empty type list");

  // Convert result types one by one and check for errors.
  SmallVector<llvm::Type *, 8> resultTypes;
  for (auto t : types) {
    llvm::Type *converted = unwrap(convertType(t));
    if (!converted)
      return {};
    resultTypes.push_back(converted);
  }

  // LLVM does not support tuple returns.  If there are more than 2 results,
  // pack them into an LLVM struct type.
  if (resultTypes.size() == 1)
    return wrap(resultTypes.front());
  return wrap(llvm::StructType::get(llvmContext, resultTypes));
}

// Function types are converted to LLVM Function types by recursively converting
// argument and result types.  If MLIR Function has zero results, the LLVM
// Function has one VoidType result.  If MLIR Function has more than one result,
// they are into an LLVM StructType in their order of appearance.
Type TypeConverter::convertFunctionType(FunctionType type) {
  // Convert argument types one by one and check for errors.
  SmallVector<llvm::Type *, 8> argTypes;
  for (auto t : type.getInputs()) {
    auto converted = convertType(t);
    if (!converted)
      return {};
    argTypes.push_back(unwrap(converted));
  }

  // If function does not return anything, create the void result type,
  // if it returns on element, convert it, otherwise pack the result types into
  // a struct.
  llvm::Type *resultType = type.getNumResults() == 0
                               ? llvm::Type::getVoidTy(llvmContext)
                               : unwrap(getPackedResultType(type.getResults()));
  if (!resultType)
    return {};
  return wrap(llvm::FunctionType::get(resultType, argTypes, /*isVarArg=*/false)
                  ->getPointerTo());
}

FunctionType TypeConverter::convertFunctionSignatureType(FunctionType type) {
  SmallVector<Type, 8> argTypes;
  for (auto t : type.getInputs()) {
    auto converted = convertType(t);
    if (!converted)
      return {};
    argTypes.push_back(converted);
  }

  // If function does not return anything, return immediately.
  if (type.getNumResults() == 0)
    return FunctionType::get(argTypes, {}, type.getContext());

  // Otherwise pack the result types into a struct.
  if (auto result = getPackedResultType(type.getResults()))
    return FunctionType::get(argTypes, {result}, type.getContext());

  return {};
}

// MemRefs are converted into LLVM structure types to accommodate dynamic sizes.
// The first element of a structure is a pointer to the elemental type of the
// MemRef.  The following N elements are values of the Index type, one for each
// of N dynamic dimensions of the MemRef.
Type TypeConverter::convertMemRefType(MemRefType type) {
  llvm::Type *elementType = unwrap(convertType(type.getElementType()));
  if (!elementType)
    return {};
  auto ptrType = elementType->getPointerTo();

  // Extra value for the memory space.
  unsigned numDynamicSizes = type.getNumDynamicDims();
  SmallVector<llvm::Type *, 8> types(numDynamicSizes + 1, getIndexType());
  types.front() = ptrType;

  return wrap(llvm::StructType::get(llvmContext, types));
}

// Convert a 1D vector type to an LLVM vector type.
Type TypeConverter::convertVectorType(VectorType type) {
  if (type.getRank() != 1) {
    MLIRContext *context = type.getContext();
    context->emitError(UnknownLoc::get(context),
                       "only 1D vectors are supported");
    return {};
  }

  llvm::Type *elementType = unwrap(convertType(type.getElementType()));
  return elementType
             ? wrap(llvm::VectorType::get(elementType, type.getShape().front()))
             : Type();
}

// Dispatch based on the actual type.  Return null type on error.
Type TypeConverter::convertType(Type type) {
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

  MLIRContext *context = type.getContext();
  std::string message;
  llvm::raw_string_ostream os(message);
  os << "unsupported type: ";
  type.print(os);
  context->emitError(UnknownLoc::get(context), os.str());
  return {};
}

Type TypeConverter::convert(Type t, llvm::Module &module) {
  return TypeConverter(module, t.getContext()).convertType(t);
}

FunctionType TypeConverter::convertFunctionSignature(FunctionType t,
                                                     llvm::Module &module) {
  return TypeConverter(module, t.getContext()).convertFunctionSignatureType(t);
}

Type TypeConverter::getMemRefElementPtrType(MemRefType t,
                                            llvm::Module &module) {
  auto elementType = t.getElementType();
  auto converted = convert(elementType, module);
  if (!converted)
    return {};
  llvm::Type *llvmType = converted.cast<LLVM::LLVMType>().getUnderlyingType();
  return LLVM::LLVMType::get(t.getContext(), llvmType->getPointerTo());
}

Type TypeConverter::pack(ArrayRef<Type> types, llvm::Module &module,
                         MLIRContext &mlirContext) {
  return TypeConverter(module, &mlirContext).getPackedResultType(types);
}

namespace {
// Base class for Standard to LLVM IR op conversions.  Matches the Op type
// provided as template argument.  Carries a reference to the LLVM dialect in
// case it is necessary for rewriters.
template <typename SourceOp>
class LLVMLegalizationPattern : public DialectOpConversion {
public:
  // Construct a conversion pattern.
  explicit LLVMLegalizationPattern(LLVM::LLVMDialect &dialect)
      : DialectOpConversion(SourceOp::getOperationName(), 1,
                            dialect.getContext()),
        dialect(dialect) {}

  // Match by type.
  PatternMatchResult match(Instruction *op) const override {
    if (op->isa<SourceOp>())
      return this->matchSuccess();
    return this->matchFailure();
  }

  // Get the LLVM IR dialect.
  LLVM::LLVMDialect &getDialect() const { return dialect; }
  // Get the LLVM context.
  llvm::LLVMContext &getContext() const { return dialect.getLLVMContext(); }
  // Get the LLVM module in which the types are constructed.
  llvm::Module &getModule() const { return dialect.getLLVMModule(); }

  // Get the MLIR type wrapping the LLVM integer type whose bit width is defined
  // by the pointer size used in the LLVM module.
  LLVM::LLVMType getIndexType() const {
    llvm::Type *llvmType = llvm::Type::getIntNTy(
        getContext(), getModule().getDataLayout().getPointerSizeInBits());
    return LLVM::LLVMType::get(dialect.getContext(), llvmType);
  }

  // Get the MLIR type wrapping the LLVM i8* type.
  LLVM::LLVMType getVoidPtrType() const {
    return LLVM::LLVMType::get(dialect.getContext(),
                               llvm::Type::getInt8PtrTy(getContext()));
  }

  // Create an LLVM IR pseudo-operation defining the given index constant.
  Value *createIndexConstant(FuncBuilder &builder, Location loc,
                             uint64_t value) const {
    auto attr = builder.getIntegerAttr(builder.getIndexType(), value);
    auto attrId = builder.getIdentifier("value");
    auto namedAttr = NamedAttribute{attrId, attr};
    return builder.create<LLVM::ConstantOp>(
        loc, getIndexType(), ArrayRef<Value *>{},
        ArrayRef<NamedAttribute>{namedAttr});
  }

  // Get the array attribute named "position" containing the given list of
  // integers as integer attribute elements.
  static NamedAttribute getPositionAttribute(FuncBuilder &builder,
                                             ArrayRef<int64_t> positions) {
    SmallVector<Attribute, 4> attrPositions;
    attrPositions.reserve(positions.size());
    for (int64_t pos : positions)
      attrPositions.push_back(
          builder.getIntegerAttr(builder.getIndexType(), pos));
    auto attr = builder.getArrayAttr(attrPositions);
    auto attrId = builder.getIdentifier("position");
    return {attrId, attr};
  }

protected:
  LLVM::LLVMDialect &dialect;
};

// Given a range of MLIR typed objects, return a list of their types.
template <typename T>
SmallVector<Type, 4> getTypes(llvm::iterator_range<T> range) {
  SmallVector<Type, 4> types;
  types.reserve(llvm::size(range));
  for (auto operand : range) {
    types.push_back(operand->getType());
  }
  return types;
}

// Basic lowering implementation for one-to-one rewriting from Standard Ops to
// LLVM Dialect Ops.
template <typename SourceOp, typename TargetOp>
struct OneToOneLLVMOpLowering : public LLVMLegalizationPattern<SourceOp> {
  using LLVMLegalizationPattern<SourceOp>::LLVMLegalizationPattern;
  using Super = OneToOneLLVMOpLowering<SourceOp, TargetOp>;

  // Convert the type of the result to an LLVM type, pass operands as is,
  // preserve attributes.
  SmallVector<Value *, 4> rewrite(Instruction *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    unsigned numResults = op->getNumResults();
    auto *mlirContext = op->getContext();

    // FIXME: using void here because there is a special case in the
    // builder... change this to use an empty type instead.
    auto voidType = LLVM::LLVMType::get(
        mlirContext, llvm::Type::getVoidTy(this->dialect.getLLVMContext()));
    auto packedType =
        numResults == 0
            ? voidType
            : TypeConverter::pack(getTypes(op->getResults()),
                                  this->dialect.getLLVMModule(), *mlirContext);
    assert(
        packedType &&
        "type conversion failed, such operation should not have been matched");

    auto newOp = rewriter.create<TargetOp>(op->getLoc(), packedType, operands,
                                           op->getAttrs());

    // If the operation produced 0 or 1 result, return them immediately.
    if (numResults == 0)
      return {};
    if (numResults == 1)
      return {newOp->getInstruction()->getResult(0)};

    // Otherwise, it had been converted to an operation producing a structure.
    // Extract individual results from the structure and return them as list.
    SmallVector<Value *, 4> results;
    results.reserve(numResults);
    for (unsigned i = 0; i < numResults; ++i) {
      auto positionNamedAttr = this->getPositionAttribute(rewriter, i);
      auto type = TypeConverter::convert(op->getResult(i)->getType(),
                                         this->dialect.getLLVMModule());
      results.push_back(rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), type,
          ArrayRef<Value *>(newOp->getInstruction()->getResult(0)),
          llvm::makeArrayRef(positionNamedAttr)));
    }
    return results;
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
struct CmpIOpLowering : public OneToOneLLVMOpLowering<CmpIOp, LLVM::ICmpOp> {
  using Super::Super;
};
struct SelectOpLowering
    : public OneToOneLLVMOpLowering<SelectOp, LLVM::SelectOp> {
  using Super::Super;
};

// Refine the matcher for call operations that return one result or more.
// Since tablegen'ed MLIR Ops cannot have variadic results, we separate calls
// that have 0 or 1 result (LLVM calls cannot have more than 1).
template <typename SourceOp>
struct NonZeroResultCallLowering
    : public OneToOneLLVMOpLowering<SourceOp, LLVM::CallOp> {
  using OneToOneLLVMOpLowering<SourceOp, LLVM::CallOp>::OneToOneLLVMOpLowering;
  using Super = NonZeroResultCallLowering<SourceOp>;

  PatternMatchResult match(Instruction *op) const override {
    if (op->getNumResults() > 0)
      return OneToOneLLVMOpLowering<SourceOp, LLVM::CallOp>::match(op);
    return this->matchFailure();
  }
};

// Refine the matcher for call operations that return zero results.
// Since tablegen'ed MLIR Ops cannot have variadic results, we separate calls
// that have 0 or 1 result (LLVM calls cannot have more than 1).
template <typename SourceOp>
struct ZeroResultCallLowering
    : public OneToOneLLVMOpLowering<SourceOp, LLVM::Call0Op> {
  using OneToOneLLVMOpLowering<SourceOp, LLVM::Call0Op>::OneToOneLLVMOpLowering;
  using Super = ZeroResultCallLowering<SourceOp>;

  PatternMatchResult match(Instruction *op) const override {
    if (op->getNumResults() == 0)
      return OneToOneLLVMOpLowering<SourceOp, LLVM::Call0Op>::match(op);
    return this->matchFailure();
  }
};

struct Call0OpLowering : public ZeroResultCallLowering<CallOp> {
  using Super::Super;
};
struct CallOpLowering : public NonZeroResultCallLowering<CallOp> {
  using Super::Super;
};
struct CallIndirect0OpLowering : public ZeroResultCallLowering<CallIndirectOp> {
  using Super::Super;
};
struct CallIndirectOpLowering
    : public NonZeroResultCallLowering<CallIndirectOp> {
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

  PatternMatchResult match(Instruction *op) const override {
    if (!LLVMLegalizationPattern<AllocOp>::match(op))
      return matchFailure();
    auto allocOp = op->cast<AllocOp>();
    MemRefType type = allocOp->getType();
    return isSupportedMemRefType(type) ? matchSuccess() : matchFailure();
  }

  SmallVector<Value *, 4> rewrite(Instruction *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    auto allocOp = op->cast<AllocOp>();
    MemRefType type = allocOp->getType();

    // Get actual sizes of the memref as values: static sizes are constant
    // values and dynamic sizes are passed to 'alloc' as operands.
    SmallVector<Value *, 4> sizes;
    sizes.reserve(allocOp->getNumOperands());
    unsigned i = 0;
    for (int64_t s : type.getShape())
      sizes.push_back(s == -1 ? operands[i++]
                              : createIndexConstant(rewriter, op->getLoc(), s));
    assert(!sizes.empty() && "zero-dimensional allocation");

    // Compute the total number of memref elements.
    Value *cumulativeSize = sizes.front();
    for (unsigned i = 1, e = sizes.size(); i < e; ++i)
      cumulativeSize = rewriter.create<LLVM::MulOp>(
          op->getLoc(), getIndexType(),
          ArrayRef<Value *>{cumulativeSize, sizes[i]});

    // Create the MemRef descriptor.
    auto structType = TypeConverter::convert(type, getModule());
    Value *memRefDescriptor = rewriter.create<LLVM::UndefOp>(
        op->getLoc(), structType, ArrayRef<Value *>{});

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
    Function *mallocFunc =
        op->getFunction()->getModule()->getNamedFunction("malloc");
    if (!mallocFunc) {
      auto mallocType =
          rewriter.getFunctionType(getIndexType(), getVoidPtrType());
      mallocFunc = new Function(rewriter.getUnknownLoc(), "malloc", mallocType);
      op->getFunction()->getModule()->getFunctions().push_back(mallocFunc);
    }

    // Allocate the underlying buffer and store a pointer to it in the MemRef
    // descriptor.
    auto mallocNamedAttr = NamedAttribute{rewriter.getIdentifier("callee"),
                                          rewriter.getFunctionAttr(mallocFunc)};
    Value *allocated = rewriter.create<LLVM::CallOp>(
        op->getLoc(), getVoidPtrType(), ArrayRef<Value *>(cumulativeSize),
        llvm::makeArrayRef(mallocNamedAttr));
    auto structElementType = TypeConverter::convert(elementType, getModule());
    auto elementPtrType = LLVM::LLVMType::get(
        op->getContext(), structElementType.cast<LLVM::LLVMType>()
                              .getUnderlyingType()
                              ->getPointerTo());
    allocated = rewriter.create<LLVM::BitcastOp>(op->getLoc(), elementPtrType,
                                                 ArrayRef<Value *>(allocated));
    auto namedPositionAttr = getPositionAttribute(rewriter, 0);
    memRefDescriptor = rewriter.create<LLVM::InsertValueOp>(
        op->getLoc(), structType,
        ArrayRef<Value *>{memRefDescriptor, allocated},
        llvm::makeArrayRef(namedPositionAttr));

    // Store dynamically allocated sizes in the descriptor.  Dynamic sizes are
    // passed in as operands.
    for (auto indexedSize : llvm::enumerate(operands)) {
      auto positionAttr =
          getPositionAttribute(rewriter, 1 + indexedSize.index());
      memRefDescriptor = rewriter.create<LLVM::InsertValueOp>(
          op->getLoc(), structType,
          ArrayRef<Value *>{memRefDescriptor, indexedSize.value()},
          llvm::makeArrayRef(positionAttr));
    }

    // Return the final value of the descriptor.
    return {memRefDescriptor};
  }
};

// A `dealloc` is converted into a call to `free` on the underlying data buffer.
// The memref descriptor being an SSA value, there is no need to clean it up
// in any way.
struct DeallocOpLowering : public LLVMLegalizationPattern<DeallocOp> {
  using LLVMLegalizationPattern<DeallocOp>::LLVMLegalizationPattern;

  SmallVector<Value *, 4> rewrite(Instruction *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    assert(operands.size() == 1 && "dealloc takes one operand");

    // Insert the `free` declaration if it is not already present.
    Function *freeFunc =
        op->getFunction()->getModule()->getNamedFunction("free");
    if (!freeFunc) {
      auto freeType = rewriter.getFunctionType(getVoidPtrType(), {});
      freeFunc = new Function(rewriter.getUnknownLoc(), "free", freeType);
      op->getFunction()->getModule()->getFunctions().push_back(freeFunc);
    }

    // Obtain the MLIR-wrapped LLVM IR element pointer type.
    llvm::Type *structType = cast<llvm::StructType>(
        operands[0]->getType().cast<LLVM::LLVMType>().getUnderlyingType());
    auto elementPtrType =
        rewriter.getType<LLVM::LLVMType>(structType->getStructElementType(0));

    // Extract the pointer to the data buffer and pass it to `free`.
    Value *bufferPtr = rewriter.create<LLVM::ExtractValueOp>(
        op->getLoc(), elementPtrType, operands[0],
        llvm::makeArrayRef(getPositionAttribute(rewriter, 0)));
    Value *casted = rewriter.create<LLVM::BitcastOp>(
        op->getLoc(), getVoidPtrType(), bufferPtr);
    auto freeNamedAttr = NamedAttribute{rewriter.getIdentifier("callee"),
                                        rewriter.getFunctionAttr(freeFunc)};
    rewriter.create<LLVM::Call0Op>(op->getLoc(), casted,
                                   llvm::makeArrayRef(freeNamedAttr));
    return {};
  }
};

struct MemRefCastOpLowering : public LLVMLegalizationPattern<MemRefCastOp> {
  using LLVMLegalizationPattern<MemRefCastOp>::LLVMLegalizationPattern;

  PatternMatchResult match(Instruction *op) const override {
    if (!LLVMLegalizationPattern<MemRefCastOp>::match(op))
      return matchFailure();
    auto memRefCastOp = op->cast<MemRefCastOp>();
    MemRefType sourceType =
        memRefCastOp->getOperand()->getType().cast<MemRefType>();
    MemRefType targetType = memRefCastOp->getType();
    return (isSupportedMemRefType(targetType) &&
            isSupportedMemRefType(sourceType))
               ? matchSuccess()
               : matchFailure();
  }

  SmallVector<Value *, 4> rewrite(Instruction *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    auto memRefCastOp = op->cast<MemRefCastOp>();
    auto targetType = memRefCastOp->getType();
    auto sourceType = memRefCastOp->getOperand()->getType().cast<MemRefType>();

    // Create the new MemRef descriptor.
    auto structType = TypeConverter::convert(targetType, getModule());
    Value *newDescriptor = rewriter.create<LLVM::UndefOp>(
        op->getLoc(), structType, ArrayRef<Value *>{});

    // Copy the data buffer pointer.
    auto elementTypePtr =
        TypeConverter::getMemRefElementPtrType(targetType, getModule());
    Value *oldDescriptor = operands[0];
    Value *buffer = rewriter.create<LLVM::ExtractValueOp>(
        op->getLoc(), elementTypePtr, ArrayRef<Value *>{oldDescriptor},
        getPositionAttribute(rewriter, 0));
    newDescriptor = rewriter.create<LLVM::InsertValueOp>(
        op->getLoc(), structType, ArrayRef<Value *>{newDescriptor, buffer},
        getPositionAttribute(rewriter, 0));

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
                    ArrayRef<Value *>{oldDescriptor},
                    getPositionAttribute(rewriter, sourceDynamicDimIdx++))
              : createIndexConstant(rewriter, op->getLoc(), sourceSize);
      newDescriptor = rewriter.create<LLVM::InsertValueOp>(
          op->getLoc(), structType, ArrayRef<Value *>{newDescriptor, size},
          getPositionAttribute(rewriter, targetDynamicDimIdx++));
    }
    assert(sourceDynamicDimIdx - 1 == sourceType.getNumDynamicDims() &&
           "source dynamic dimensions were not processed");
    assert(targetDynamicDimIdx - 1 == targetType.getNumDynamicDims() &&
           "target dynamic dimensions were not set up");

    return {newDescriptor};
  }
};

// A `dim` is converted to a constant for static sizes and to an access to the
// size stored in the memref descriptor for dynamic sizes.
struct DimOpLowering : public LLVMLegalizationPattern<DimOp> {
  using LLVMLegalizationPattern<DimOp>::LLVMLegalizationPattern;

  PatternMatchResult match(Instruction *op) const override {
    if (!LLVMLegalizationPattern<DimOp>::match(op))
      return this->matchFailure();
    auto dimOp = op->cast<DimOp>();
    MemRefType type = dimOp->getOperand()->getType().cast<MemRefType>();
    return isSupportedMemRefType(type) ? matchSuccess() : matchFailure();
  }

  SmallVector<Value *, 4> rewrite(Instruction *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    assert(operands.size() == 1 && "expected exactly one operand");
    auto dimOp = op->cast<DimOp>();
    MemRefType type = dimOp->getOperand()->getType().cast<MemRefType>();

    SmallVector<Value *, 4> results;
    auto shape = type.getShape();
    uint64_t index = dimOp->getIndex();
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
      results.push_back(rewriter.create<LLVM::ExtractValueOp>(
          op->getLoc(), getIndexType(), operands,
          getPositionAttribute(rewriter, position)));
    } else {
      results.push_back(
          createIndexConstant(rewriter, op->getLoc(), shape[index]));
    }
    return results;
  }
};

// Common base for load and store operations on MemRefs.  Restricts the match
// to supported MemRef types.  Provides functionality to emit code accessing a
// specific element of the underlying data buffer.
template <typename Derived>
struct LoadStoreOpLowering : public LLVMLegalizationPattern<Derived> {
  using LLVMLegalizationPattern<Derived>::LLVMLegalizationPattern;
  using Base = LoadStoreOpLowering<Derived>;

  PatternMatchResult match(Instruction *op) const override {
    if (!LLVMLegalizationPattern<Derived>::match(op))
      return this->matchFailure();
    auto loadOp = op->cast<Derived>();
    MemRefType type = loadOp->getMemRefType();
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
  Value *linearizeSubscripts(FuncBuilder &builder, Location loc,
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
  // indies.
  Value *getElementPtr(Location loc, MemRefType type, Value *memRefDescriptor,
                       ArrayRef<Value *> indices, FuncBuilder &rewriter) const {
    auto elementTypePtr =
        TypeConverter::getMemRefElementPtrType(type, this->getModule());

    // Get the list of MemRef sizes.  Static sizes are defined as constants.
    // Dynamic sizes are extracted from the MemRef descriptor, where they start
    // from the position 1 (the buffer is at position 0).
    SmallVector<Value *, 4> sizes;
    unsigned dynamicSizeIdx = 1;
    for (int64_t s : type.getShape()) {
      if (s == -1) {
        Value *size = rewriter.create<LLVM::ExtractValueOp>(
            loc, this->getIndexType(), ArrayRef<Value *>{memRefDescriptor},
            llvm::makeArrayRef(
                this->getPositionAttribute(rewriter, dynamicSizeIdx++)));
        sizes.push_back(size);
      } else {
        sizes.push_back(this->createIndexConstant(rewriter, loc, s));
      }
    }

    // The second and subsequent operands are access subscripts.  Obtain the
    // linearized address in the buffer.
    Value *subscript = linearizeSubscripts(rewriter, loc, indices, sizes);

    Value *dataPtr = rewriter.create<LLVM::ExtractValueOp>(
        loc, elementTypePtr, ArrayRef<Value *>{memRefDescriptor},
        llvm::makeArrayRef(this->getPositionAttribute(rewriter, 0)));
    return rewriter.create<LLVM::GEPOp>(loc, elementTypePtr,
                                        ArrayRef<Value *>{dataPtr, subscript},
                                        ArrayRef<NamedAttribute>{});
  }
};

// Load operation is lowered to obtaining a pointer to the indexed element
// and loading it.
struct LoadOpLowering : public LoadStoreOpLowering<LoadOp> {
  using Base::Base;

  SmallVector<Value *, 4> rewrite(Instruction *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    auto loadOp = op->cast<LoadOp>();
    auto type = loadOp->getMemRefType();
    auto elementType =
        TypeConverter::convert(type.getElementType(), getModule());
    Value *dataPtr = getElementPtr(op->getLoc(), type, operands.front(),
                                   operands.drop_front(), rewriter);

    SmallVector<Value *, 4> results;
    results.push_back(rewriter.create<LLVM::LoadOp>(
        op->getLoc(), elementType, ArrayRef<Value *>{dataPtr}));
    return results;
  }
};

// Store opreation is lowered to obtaining a pointer to the indexed element,
// and storing the given value to it.
struct StoreOpLowering : public LoadStoreOpLowering<StoreOp> {
  using Base::Base;

  SmallVector<Value *, 4> rewrite(Instruction *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    auto storeOp = op->cast<StoreOp>();
    auto type = storeOp->getMemRefType();
    Value *dataPtr = getElementPtr(op->getLoc(), type, operands[1],
                                   operands.drop_front(2), rewriter);

    rewriter.create<LLVM::StoreOp>(op->getLoc(), operands[0], dataPtr);
    return {};
  }
};

// Base class for LLVM IR lowering terminator operations with successors.
template <typename SourceOp, typename TargetOp>
struct OneToOneLLVMTerminatorLowering
    : public LLVMLegalizationPattern<SourceOp> {
  using LLVMLegalizationPattern<SourceOp>::LLVMLegalizationPattern;
  using Super = OneToOneLLVMTerminatorLowering<SourceOp, TargetOp>;

  void rewriteTerminator(Instruction *op, ArrayRef<Value *> properOperands,
                         ArrayRef<Block *> destinations,
                         ArrayRef<ArrayRef<Value *>> operands,
                         FuncBuilder &rewriter) const override {
    rewriter.create<TargetOp>(op->getLoc(), properOperands, destinations,
                              operands, op->getAttrs());
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

  SmallVector<Value *, 4> rewrite(Instruction *op, ArrayRef<Value *> operands,
                                  FuncBuilder &rewriter) const override {
    unsigned numArguments = op->getNumOperands();

    // If ReturnOp has 0 or 1 operand, create it and return immediately.
    if (numArguments == 0) {
      rewriter.create<LLVM::ReturnOp>(
          op->getLoc(), llvm::ArrayRef<Value *>(), llvm::ArrayRef<Block *>(),
          llvm::ArrayRef<llvm::ArrayRef<Value *>>(), op->getAttrs());
      return {};
    }
    if (numArguments == 1) {
      rewriter.create<LLVM::ReturnOp>(
          op->getLoc(), llvm::ArrayRef<Value *>(operands.front()),
          llvm::ArrayRef<Block *>(), llvm::ArrayRef<llvm::ArrayRef<Value *>>(),
          op->getAttrs());
      return {};
    }

    // Otherwise, we need to pack the arguments into an LLVM struct type before
    // returning.
    auto *mlirContext = op->getContext();
    auto packedType = TypeConverter::pack(
        getTypes(op->getOperands()), dialect.getLLVMModule(), *mlirContext);

    Value *packed = rewriter.create<LLVM::UndefOp>(op->getLoc(), packedType);
    for (unsigned i = 0; i < numArguments; ++i) {
      auto positionNamedAttr = getPositionAttribute(rewriter, i);
      packed = rewriter.create<LLVM::InsertValueOp>(
          op->getLoc(), packedType,
          llvm::ArrayRef<Value *>{packed, operands[i]},
          llvm::makeArrayRef(positionNamedAttr));
    }
    rewriter.create<LLVM::ReturnOp>(
        op->getLoc(), llvm::makeArrayRef(packed), llvm::ArrayRef<Block *>(),
        llvm::ArrayRef<llvm::ArrayRef<Value *>>(), op->getAttrs());
    return {};
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

/// A pass converting MLIR Standard and Builtin operations into the LLVM IR
/// dialect.
class LLVMLowering : public DialectConversion {
public:
  LLVMLowering() : DialectConversion(&passID) {}

  const static char passID = '\0';

protected:
  // Create a set of converters that live in the pass object by passing them a
  // reference to the LLVM IR dialect.  Store the module associated with the
  // dialect for further type conversion.
  llvm::DenseSet<DialectOpConversion *>
  initConverters(MLIRContext *mlirContext) override {
    converterStorage.Reset();
    auto *llvmDialect = static_cast<LLVM::LLVMDialect *>(
        mlirContext->getRegisteredDialect("llvm"));
    if (!llvmDialect) {
      mlirContext->emitError(UnknownLoc::get(mlirContext),
                             "LLVM IR dialect is not registered");
      return {};
    }

    module = &llvmDialect->getLLVMModule();

    // FIXME: this should be tablegen'ed
    return ConversionListBuilder<
        AddFOpLowering, AddIOpLowering, AllocOpLowering, BranchOpLowering,
        Call0OpLowering, CallIndirect0OpLowering, CallIndirectOpLowering,
        CallOpLowering, CmpIOpLowering, CondBranchOpLowering,
        ConstLLVMOpLowering, DeallocOpLowering, DimOpLowering, DivISOpLowering,
        DivIUOpLowering, DivFOpLowering, LoadOpLowering, MemRefCastOpLowering,
        MulFOpLowering, MulIOpLowering, RemISOpLowering, RemIUOpLowering,
        RemFOpLowering, ReturnOpLowering, SelectOpLowering, StoreOpLowering,
        SubFOpLowering, SubIOpLowering>::build(&converterStorage, *llvmDialect);
  }

  // Convert types using the stored LLVM IR module.
  Type convertType(Type t) override {
    return TypeConverter::convert(t, *module);
  }

  // Convert function signatures using the stored LLVM IR module.
  FunctionType convertFunctionSignatureType(FunctionType t) override {
    return TypeConverter::convertFunctionSignature(t, *module);
  }

private:
  // Storage for the conversion patterns.
  llvm::BumpPtrAllocator converterStorage;
  // LLVM IR module used to parse/create types.
  llvm::Module *module;
};

const char LLVMLowering::passID;

ModulePass *mlir::createConvertToLLVMIRPass() { return new LLVMLowering; }

static PassRegistration<LLVMLowering>
    pass("convert-to-llvmir", "Convert all functions to the LLVM IR dialect");
