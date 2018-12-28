//===- ConvertToLLVMIR.cpp - MLIR to LLVM IR conversion ---------*- C++ -*-===//
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
// This file implements a pass that converts CFG function to LLVM IR.  No ML
// functions must be presented in MLIR.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/Statements.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/SuperVectorOps/SuperVectorOps.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Functional.h"
#include "mlir/Target/LLVMIR.h"
#include "mlir/Translation.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

using namespace mlir;

namespace {
class ModuleLowerer {
public:
  explicit ModuleLowerer(llvm::LLVMContext &llvmContext)
      : llvmContext(llvmContext), builder(llvmContext) {}

  bool runOnModule(Module &m, llvm::Module &llvmModule);

private:
  bool convertBasicBlock(const BasicBlock &bb, bool ignoreArguments = false);
  bool convertCFGFunction(const CFGFunction &cfgFunc, llvm::Function &llvmFunc);
  bool convertFunctions(const Module &mlirModule, llvm::Module &llvmModule);
  bool convertInstruction(const OperationInst &inst);

  void connectPHINodes(const CFGFunction &cfgFunc);

  /// Type conversion functions.  If any conversion fails, report errors to the
  /// context of the MLIR type and return nullptr.
  /// \{
  llvm::FunctionType *convertFunctionType(FunctionType type);
  llvm::IntegerType *convertIndexType(IndexType type);
  llvm::IntegerType *convertIntegerType(IntegerType type);
  llvm::Type *convertFloatType(FloatType type);
  llvm::Type *convertType(Type type);
  /// Convert a MemRefType `type` into an LLVM aggregate structure type.  Each
  /// structure type starts with a pointer to the elemental type of the MemRef
  /// and continues with as many lowered to LLVM index types as MemRef has
  /// dynamic dimensions.  An instance of this type is called a MemRef decriptor
  /// and replaces the MemRef everywhere it is used so that any instruction has
  /// access to its dynamic sizes.
  /// For example, given that `index` is converted to `i64`, `memref<?x?xf32>`
  /// is converted to `{float*, i64, i64}` (two dynamic sizes, in order);
  /// `memref<42x?x42xi32>` is converted to `{i32*, i64}` (only one size is
  /// dynamic); `memref<2x3x4xf64>` is converted to `{double*}`.
  llvm::StructType *convertMemRefType(MemRefType type);

  /// Convert a 1D vector type to an LLVM vector type.
  llvm::VectorType *convertVectorType(VectorType type);
  /// \}

  /// Convert a list of types to an LLVM type suitable for being returned from a
  /// function.  If the list is empty, return VoidTy.  If it
  /// contains one element, return the converted element. Otherwise, create an
  /// LLVM StructType containing all the given types in order.
  llvm::Type *getPackedResultType(ArrayRef<Type> types);

  /// Get an a constant value of `indexType`.
  inline llvm::Constant *getIndexConstant(int64_t value);

  /// Given subscript indices and array sizes in row-major order,
  ///   i_n, i_{n-1}, ..., i_1
  ///   s_n, s_{n-1}, ..., s_1
  /// obtain a value that corresponds to the linearized subscript
  ///   i_n * s_{n-1} * s_{n-2} * ... * s_1 +
  ///   + i_{n-1} * s_{n-2} * s_{n_3} * ... * s_1 +
  ///   + ... +
  ///   + i_2 * s_1 +
  ///   + i_1.
  llvm::Value *linearizeSubscripts(ArrayRef<llvm::Value *> indices,
                                   ArrayRef<llvm::Value *> allocSizes);

  /// Emit LLVM IR instructions necessary to obtain a pointer to the element of
  /// `memRef` accessed by `op` with indices `opIndices`. In particular, extract
  /// any dynamic allocation sizes from the MemRef descriptor, linearize the
  /// access subscript given the sizes, extract the data pointer from the MemRef
  /// descriptor and get the pointer to the element indexed by the linearized
  /// subscript.  Return nullptr on errors.
  llvm::Value *emitMemRefElementAccess(
      const Value *memRef, const OperationInst &op,
      llvm::iterator_range<OperationInst::const_operand_iterator> opIndices);

  /// Emit LLVM IR corresponding to the given Alloc `op`.  In particular, create
  /// a Value for the MemRef descriptor, store any dynamic sizes passed to
  /// the alloc operation in the descriptor, allocate the buffer for the data
  /// using `allocFunc` and also store it in the descriptor.  Return the MemRef
  /// descriptor.  This function returns `nullptr` in case of errors.
  llvm::Value *emitMemRefAlloc(ConstOpPointer<AllocOp> allocOp);

  /// Emit LLVM IR corresponding to the given Dealloc `op`.  In particular,
  /// use `freeFunc` to free the memory allocated for the MemRef's buffer.  The
  /// MemRef descriptor allocated on stack will cease to exist when the current
  /// function returns without any extra action.  Returns an LLVM Value (call
  /// instruction) on success and nullptr on error.
  llvm::Value *emitMemRefDealloc(ConstOpPointer<DeallocOp> deallocOp);

  /// Emit a constant splat operation, i.e. an operation that broadcasts a
  /// single value to a vector.  The `op` must have an attribute `value` of
  /// SplatElementsAttr type.  Return an LLVM SSA value of the constant vector;
  /// return `nullptr` in case of errors.
  llvm::Value *emitConstantSplat(const ConstantOp &op);

  /// Create a single LLVM value of struct type that includes the list of
  /// given MLIR values.  The `values` list must contain at least 2 elements.
  llvm::Value *packValues(ArrayRef<const Value *> values);
  /// Extract a list of `num` LLVM values from a `value` of struct type.
  SmallVector<llvm::Value *, 4> unpackValues(llvm::Value *value, unsigned num);

  llvm::DenseMap<const Function *, llvm::Function *> functionMapping;
  llvm::DenseMap<const Value *, llvm::Value *> valueMapping;
  llvm::DenseMap<const BasicBlock *, llvm::BasicBlock *> blockMapping;
  llvm::LLVMContext &llvmContext;
  llvm::IRBuilder<llvm::ConstantFolder, llvm::IRBuilderDefaultInserter> builder;
  llvm::IntegerType *indexType;

  /// Allocation function : (index) -> i8*, declaration only.
  llvm::Constant *allocFunc;
  /// Deallocation function : (i8*) -> void, declaration only.
  llvm::Constant *freeFunc;
};

llvm::IntegerType *ModuleLowerer::convertIndexType(IndexType type) {
  return indexType;
}

llvm::IntegerType *ModuleLowerer::convertIntegerType(IntegerType type) {
  return builder.getIntNTy(type.getWidth());
}

llvm::Type *ModuleLowerer::convertFloatType(FloatType type) {
  MLIRContext *context = type.getContext();
  switch (type.getKind()) {
  case Type::Kind::F32:
    return builder.getFloatTy();
  case Type::Kind::F64:
    return builder.getDoubleTy();
  case Type::Kind::F16:
    return builder.getHalfTy();
  case Type::Kind::BF16:
    return context->emitError(UnknownLoc::get(context),
                              "unsupported type: BF16"),
           nullptr;
  default:
    llvm_unreachable("non-float type in convertFloatType");
  }
}

// Helper function for lambdas below.
static bool isTypeNull(llvm::Type *type) { return type == nullptr; }

// If `types` has more than one type, pack them into an LLVM StructType,
// otherwise just convert the type.
llvm::Type *ModuleLowerer::getPackedResultType(ArrayRef<Type> types) {
  // Convert result types one by one and check for errors.
  auto resultTypes =
      functional::map([this](Type t) { return convertType(t); }, types);
  if (llvm::any_of(resultTypes, isTypeNull))
    return nullptr;

  // LLVM does not support tuple returns.  If there are more than 2 results,
  // pack them into an LLVM struct type.
  if (resultTypes.empty())
    return llvm::Type::getVoidTy(llvmContext);
  if (resultTypes.size() == 1)
    return resultTypes.front();
  return llvm::StructType::get(llvmContext, resultTypes);
}

// Function types are converted to LLVM Function types by recursively converting
// argument and result types.  If MLIR Function has zero results, the LLVM
// Function has one VoidType result.  If MLIR Function has more than one result,
// they are into an LLVM StructType in their order of appearance.
llvm::FunctionType *ModuleLowerer::convertFunctionType(FunctionType type) {
  llvm::Type *resultType = getPackedResultType(type.getResults());
  if (!resultType)
    return nullptr;

  // Convert argument types one by one and check for errors.
  auto argTypes = functional::map([this](Type t) { return convertType(t); },
                                  type.getInputs());
  if (llvm::any_of(argTypes, isTypeNull))
    return nullptr;

  return llvm::FunctionType::get(resultType, argTypes, /*isVarArg=*/false);
}

// MemRefs are converted into LLVM structure types to accomodate dynamic sizes.
// The first element of a structure is a pointer to the elemental type of the
// MemRef.  The following N elements are values of the Index type, one for each
// of N dynamic dimensions of the MemRef.
llvm::StructType *ModuleLowerer::convertMemRefType(MemRefType type) {
  llvm::Type *elementType = convertType(type.getElementType());
  if (!elementType)
    return nullptr;
  elementType = elementType->getPointerTo();

  // Extra value for the memory space.
  unsigned numDynamicSizes = type.getNumDynamicDims();
  SmallVector<llvm::Type *, 8> types(numDynamicSizes + 1, indexType);
  types.front() = elementType;

  return llvm::StructType::get(llvmContext, types);
}

// Convert a 1D vector type to an LLVM vector type.
llvm::VectorType *ModuleLowerer::convertVectorType(VectorType type) {
  if (type.getRank() != 1) {
    MLIRContext *context = type.getContext();
    context->emitError(UnknownLoc::get(context),
                       "only 1D vectors are supported");
    return nullptr;
  }

  llvm::Type *elementType = convertType(type.getElementType());
  if (!elementType) {
    return nullptr;
  }

  return llvm::VectorType::get(elementType, type.getShape().front());
}

llvm::Type *ModuleLowerer::convertType(Type type) {
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
  return nullptr;
}

llvm::Constant *ModuleLowerer::getIndexConstant(int64_t value) {
  return llvm::Constant::getIntegerValue(
      indexType, llvm::APInt(indexType->getBitWidth(), value));
}

// Given subscript indices and array sizes in row-major order,
//   i_n, i_{n-1}, ..., i_1
//   s_n, s_{n-1}, ..., s_1
// obtain a value that corresponds to the linearized subscript
//   \sum_k i_k * \prod_{j=1}^{k-1} s_j
// by accumulating the running linearized value.
llvm::Value *
ModuleLowerer::linearizeSubscripts(ArrayRef<llvm::Value *> indices,
                                   ArrayRef<llvm::Value *> allocSizes) {
  assert(indices.size() == allocSizes.size() &&
         "mismatching number of indices and allocation sizes");
  assert(!indices.empty() && "cannot linearize a 0-dimensional access");

  llvm::Value *linearized = indices.front();
  for (unsigned i = 1, nSizes = allocSizes.size(); i < nSizes; ++i) {
    linearized = builder.CreateMul(linearized, allocSizes[i]);
    linearized = builder.CreateAdd(linearized, indices[i]);
  }

  return linearized;
}

// Check if the MemRefType `type` is supported by the lowering.  Emit errors at
// the location of `op` and return true.  Return false if the type is supported.
// TODO(zinenko): this function should disappear when the conversion fully
// supports MemRefs.
static bool checkSupportedMemRefType(MemRefType type, const OperationInst &op) {
  if (!type.getAffineMaps().empty())
    return op.emitError("NYI: memrefs with affine maps");
  if (type.getMemorySpace() != 0)
    return op.emitError("NYI: non-default memory space");
  return false;
}

llvm::Value *ModuleLowerer::emitMemRefElementAccess(
    const Value *memRef, const OperationInst &op,
    llvm::iterator_range<OperationInst::const_operand_iterator> opIndices) {
  auto type = memRef->getType().dyn_cast<MemRefType>();
  assert(type && "expected memRef value to have a MemRef type");
  if (checkSupportedMemRefType(type, op))
    return nullptr;

  // A MemRef-typed value is remapped to its descriptor.
  llvm::Value *memRefDescriptor = valueMapping.lookup(memRef);

  // Get the list of MemRef sizes.  Static sizes are defined as values.  Dynamic
  // sizes are extracted from the MemRef descriptor.
  llvm::SmallVector<llvm::Value *, 4> sizes;
  unsigned dynanmicSizeIdx = 0;
  for (int64_t s : type.getShape()) {
    llvm::Value *size = (s == -1) ? builder.CreateExtractValue(
                                        memRefDescriptor, 1 + dynanmicSizeIdx++)
                                  : getIndexConstant(s);
    sizes.push_back(size);
  }

  // Obtain the list of access subscripts as values and linearize it given the
  // list of sizes.
  auto indices = functional::map(
      [this](const Value *value) { return valueMapping.lookup(value); },
      opIndices);
  auto subscript = linearizeSubscripts(indices, sizes);

  // Extract the pointer to the data buffer and use LLVM's getelementptr to
  // repoint it to the element indexed by the subscript.
  llvm::Value *data = builder.CreateExtractValue(memRefDescriptor, 0);
  return builder.CreateGEP(data, subscript);
}

llvm::Value *ModuleLowerer::emitMemRefAlloc(ConstOpPointer<AllocOp> allocOp) {
  MemRefType type = allocOp->getType();
  if (checkSupportedMemRefType(type, *allocOp->getInstruction()))
    return nullptr;

  // Get actual sizes of the memref as values: static sizes are constant
  // values and dynamic sizes are passed to 'alloc' as operands.
  SmallVector<llvm::Value *, 4> sizes;
  sizes.reserve(allocOp->getNumOperands());
  unsigned i = 0;
  for (int s : type.getShape()) {
    llvm::Value *value = (s == -1)
                             ? valueMapping.lookup(allocOp->getOperand(i++))
                             : getIndexConstant(s);
    sizes.push_back(value);
  }
  assert(!sizes.empty() && "zero-dimensional allocation");

  // Compute the total numer of memref elements as Value.
  llvm::Value *cumulativeSize = sizes.front();
  for (unsigned i = 1, e = sizes.size(); i < e; ++i) {
    cumulativeSize = builder.CreateMul(cumulativeSize, sizes[i]);
  }

  // Allocate the MemRef descriptor on stack and load it.
  llvm::StructType *structType = convertMemRefType(type);
  llvm::Type *elementType = convertType(type.getElementType());
  if (!structType || !elementType)
    return nullptr;
  llvm::Value *memRefDescriptor = llvm::UndefValue::get(structType);

  // Take into account the size of the elemental type before allocation.
  // Elemental types can be scalars or vectors only.
  unsigned byteWidth = elementType->getScalarSizeInBits() / 8;
  assert(byteWidth > 0 && "could not determine size of a MemRef element");
  if (elementType->isVectorTy()) {
    byteWidth *= elementType->getVectorNumElements();
  }
  llvm::Value *byteWidthValue = getIndexConstant(byteWidth);
  cumulativeSize = builder.CreateMul(cumulativeSize, byteWidthValue);

  // Allocate the buffer for theMemRef and store a pointer to it in the MemRef
  // descriptor.
  llvm::Value *allocated = builder.CreateCall(allocFunc, cumulativeSize);
  allocated = builder.CreateBitCast(allocated, elementType->getPointerTo());
  memRefDescriptor = builder.CreateInsertValue(memRefDescriptor, allocated, 0);

  // Store dynamically allocated sizes in the descriptor.
  i = 0;
  for (auto indexedSize : llvm::enumerate(sizes)) {
    if (type.getShape()[indexedSize.index()] != -1)
      continue;
    memRefDescriptor = builder.CreateInsertValue(memRefDescriptor,
                                                 indexedSize.value(), 1 + i++);
  }

  // Return the final value of the descriptor (each insert returns a new,
  // updated value, the old is still accessible but has old data).
  return memRefDescriptor;
}

llvm::Value *
ModuleLowerer::emitMemRefDealloc(ConstOpPointer<DeallocOp> deallocOp) {
  // Extract the pointer to the MemRef buffer from its descriptor and call
  // `freeFunc` on it.
  llvm::Value *memRefDescriptor = valueMapping.lookup(deallocOp->getMemRef());
  llvm::Value *data = builder.CreateExtractValue(memRefDescriptor, 0);
  data = builder.CreateBitCast(data, builder.getInt8PtrTy());
  return builder.CreateCall(freeFunc, data);
}

// Return an LLVM constant of the `float` type for the given APvalue.
// This forcibly recreates the APFloat with IEEESingle semantics to make sure
// LLVM constructs a `float` constant.
static llvm::ConstantFP *getFloatConstant(APFloat APvalue,
                                          const OperationInst &inst,
                                          llvm::LLVMContext *context) {
  bool unused;
  APFloat::opStatus status = APvalue.convert(
      llvm::APFloat::IEEEsingle(), llvm::APFloat::rmTowardZero, &unused);
  if (status == APFloat::opInexact) {
    inst.emitWarning("lossy conversion of a float constant to the float type");
    // No return intended.
  }
  if (status != APFloat::opOK)
    return inst.emitError("failed to convert a floating point constant"),
           nullptr;
  auto value = APvalue.convertToFloat();
  return llvm::ConstantFP::get(*context, APFloat(value));
}

llvm::Value *ModuleLowerer::emitConstantSplat(const ConstantOp &op) {
  auto splatAttr = op.getValue().dyn_cast<SplatElementsAttr>();
  assert(splatAttr && "expected a splat constant");

  auto floatAttr = splatAttr.getValue().dyn_cast<FloatAttr>();
  if (!floatAttr)
    return op.emitError("NYI: only float splats are currently supported"),
           nullptr;

  llvm::Constant *cst = getFloatConstant(floatAttr.getValue(),
                                         *op.getInstruction(), &llvmContext);
  if (!cst)
    return nullptr;

  auto nElements = op.getType().cast<VectorType>().getShape()[0];
  return llvm::ConstantVector::getSplat(nElements, cst);
}

// Create an undef struct value and insert individual values into it.
llvm::Value *ModuleLowerer::packValues(ArrayRef<const Value *> values) {
  assert(values.size() > 1 && "cannot pack less than 2 values");

  auto types =
      functional::map([](const Value *v) { return v->getType(); }, values);
  llvm::Type *packedType = getPackedResultType(types);

  llvm::Value *packed = llvm::UndefValue::get(packedType);
  for (auto indexedValue : llvm::enumerate(values)) {
    packed = builder.CreateInsertValue(
        packed, valueMapping.lookup(indexedValue.value()),
        indexedValue.index());
  }
  return packed;
}

// Emit extract value instructions to unpack the struct.
SmallVector<llvm::Value *, 4> ModuleLowerer::unpackValues(llvm::Value *value,
                                                          unsigned num) {
  SmallVector<llvm::Value *, 4> unpacked;
  unpacked.reserve(num);
  for (unsigned i = 0; i < num; ++i)
    unpacked.push_back(builder.CreateExtractValue(value, i));
  return unpacked;
}

static llvm::CmpInst::Predicate getLLVMCmpPredicate(CmpIPredicate p) {
  switch (p) {
  case CmpIPredicate::EQ:
    return llvm::CmpInst::Predicate::ICMP_EQ;
  case CmpIPredicate::NE:
    return llvm::CmpInst::Predicate::ICMP_NE;
  case CmpIPredicate::SLT:
    return llvm::CmpInst::Predicate::ICMP_SLT;
  case CmpIPredicate::SLE:
    return llvm::CmpInst::Predicate::ICMP_SLE;
  case CmpIPredicate::SGT:
    return llvm::CmpInst::Predicate::ICMP_SGT;
  case CmpIPredicate::SGE:
    return llvm::CmpInst::Predicate::ICMP_SGE;
  case CmpIPredicate::ULT:
    return llvm::CmpInst::Predicate::ICMP_ULT;
  case CmpIPredicate::ULE:
    return llvm::CmpInst::Predicate::ICMP_ULE;
  case CmpIPredicate::UGT:
    return llvm::CmpInst::Predicate::ICMP_UGT;
  case CmpIPredicate::UGE:
    return llvm::CmpInst::Predicate::ICMP_UGE;
  default:
    llvm_unreachable("incorrect comparison predicate");
  }
}

// Convert specific operation instruction types LLVM instructions.
// FIXME(zinenko): this should eventually become a separate MLIR pass that
// converts MLIR standard operations into LLVM IR dialect; the translation in
// that case would become a simple 1:1 instruction and value remapping.
bool ModuleLowerer::convertInstruction(const OperationInst &inst) {
  if (auto op = inst.dyn_cast<AddIOp>())
    return valueMapping[op->getResult()] =
               builder.CreateAdd(valueMapping[op->getOperand(0)],
                                 valueMapping[op->getOperand(1)]),
           false;
  if (auto op = inst.dyn_cast<SubIOp>())
    return valueMapping[op->getResult()] =
               builder.CreateSub(valueMapping[op->getOperand(0)],
                                 valueMapping[op->getOperand(1)]),
           false;
  if (auto op = inst.dyn_cast<MulIOp>())
    return valueMapping[op->getResult()] =
               builder.CreateMul(valueMapping[op->getOperand(0)],
                                 valueMapping[op->getOperand(1)]),
           false;
  if (auto op = inst.dyn_cast<CmpIOp>())
    return valueMapping[op->getResult()] =
               builder.CreateICmp(getLLVMCmpPredicate(op->getPredicate()),
                                  valueMapping[op->getOperand(0)],
                                  valueMapping[op->getOperand(1)]),
           false;

  if (auto op = inst.dyn_cast<AddFOp>())
    return valueMapping[op->getResult()] =
               builder.CreateFAdd(valueMapping.lookup(op->getOperand(0)),
                                  valueMapping.lookup(op->getOperand(1))),
           false;
  if (auto op = inst.dyn_cast<SubFOp>())
    return valueMapping[op->getResult()] =
               builder.CreateFSub(valueMapping.lookup(op->getOperand(0)),
                                  valueMapping.lookup(op->getOperand(1))),
           false;
  if (auto op = inst.dyn_cast<MulFOp>())
    return valueMapping[op->getResult()] =
               builder.CreateFMul(valueMapping.lookup(op->getOperand(0)),
                                  valueMapping.lookup(op->getOperand(1))),
           false;

  if (auto constantOp = inst.dyn_cast<ConstantIndexOp>()) {
    auto attr = constantOp->getValue();
    valueMapping[constantOp->getResult()] = getIndexConstant(attr);
    return false;
  }
  if (auto constantOp = inst.dyn_cast<ConstantFloatOp>()) {
    llvm::Type *type = convertType(constantOp->getType());
    if (!type)
      return true;
    // TODO(somebody): float attributes have "double" semantics whatever the
    // type of the constant.  This should be fixed at the parser level.
    if (!type->isFloatTy())
      return inst.emitError("NYI: only floats are currently supported");

    auto APvalue = constantOp->getValue();
    auto llvmValue = getFloatConstant(APvalue, inst, &type->getContext());
    if (!llvmValue)
      return true;

    valueMapping[constantOp->getResult()] = llvmValue;
    return false;
  }
  if (auto constantOp = inst.dyn_cast<ConstantIntOp>()) {
    llvm::Type *type = convertType(constantOp->getType());
    if (!type)
      return true;

    // Create a new APInt even if we can extract one from the attribute, because
    // attributes are currently hardcoded to be 64-bit APInts and LLVM will
    // create an i64 constant from those.
    auto value = constantOp->getValue();
    valueMapping[constantOp->getResult()] = llvm::Constant::getIntegerValue(
        type, APInt(type->getIntegerBitWidth(), value));
    return false;
  }
  if (auto constantOp = inst.dyn_cast<ConstantOp>()) {
    llvm::Type *type = convertType(constantOp->getType());
    if (!type)
      return true;
    if (!isa<llvm::VectorType>(type))
      return inst.emitError("unsupported constant type");

    auto constantValue = constantOp->getValue();
    if (!constantValue.isa<SplatElementsAttr>())
      return inst.emitError("NYI: non-splat vector constants");

    llvm::Value *llvmValue = emitConstantSplat(*constantOp);
    if (!llvmValue)
      return true;
    valueMapping[constantOp->getResult()] = llvmValue;
    return false;
  }

  if (auto allocOp = inst.dyn_cast<AllocOp>()) {
    llvm::Value *memRefDescriptor = emitMemRefAlloc(allocOp);
    if (!memRefDescriptor)
      return true;

    valueMapping[allocOp->getResult()] = memRefDescriptor;
    return false;
  }
  if (auto deallocOp = inst.dyn_cast<DeallocOp>()) {
    return !emitMemRefDealloc(deallocOp);
  }

  if (auto loadOp = inst.dyn_cast<LoadOp>()) {
    llvm::Value *element = emitMemRefElementAccess(
        loadOp->getMemRef(), *loadOp->getInstruction(), loadOp->getIndices());
    if (!element)
      return true;

    valueMapping[loadOp->getResult()] = builder.CreateLoad(element);
    return false;
  }
  if (auto storeOp = inst.dyn_cast<StoreOp>()) {
    llvm::Value *element = emitMemRefElementAccess(storeOp->getMemRef(),
                                                   *storeOp->getInstruction(),
                                                   storeOp->getIndices());
    if (!element)
      return true;

    builder.CreateStore(valueMapping.lookup(storeOp->getValueToStore()),
                        element);
    return false;
  }
  if (auto dimOp = inst.dyn_cast<DimOp>()) {
    const Value *container = dimOp->getOperand();
    MemRefType type = container->getType().dyn_cast<MemRefType>();
    if (!type)
      return dimOp->emitError("only memref types are supported");

    auto shape = type.getShape();
    auto index = dimOp->getIndex();
    assert(index < shape.size() && "out-of-bounds 'dim' operation");

    // If the size is a constant, just define that constant.
    if (shape[index] != -1) {
      valueMapping[dimOp->getResult()] = getIndexConstant(shape[index]);
      return false;
    }

    // Otherwise, compute the position of the requested index in the list of
    // dynamic sizes stored in the MemRef descriptor and extract it from there.
    unsigned numLeadingDynamicSizes = 0;
    for (unsigned i = 0; i < index; ++i) {
      if (shape[i] == -1)
        ++numLeadingDynamicSizes;
    }
    llvm::Value *memRefDescriptor = valueMapping.lookup(container);
    llvm::Value *dynamicSize = builder.CreateExtractValue(
        memRefDescriptor, 1 + numLeadingDynamicSizes);
    valueMapping[dimOp->getResult()] = dynamicSize;
    return false;
  }

  if (auto callOp = inst.dyn_cast<CallOp>()) {
    auto operands = functional::map(
        [this](const Value *value) { return valueMapping.lookup(value); },
        callOp->getOperands());
    auto numResults = callOp->getNumResults();
    llvm::Value *result =
        builder.CreateCall(functionMapping[callOp->getCallee()], operands);
    if (numResults == 1) {
      valueMapping[callOp->getResult(0)] = result;
    } else if (numResults > 1) {
      auto unpacked = unpackValues(result, numResults);
      for (auto indexedValue : llvm::enumerate(unpacked)) {
        valueMapping[callOp->getResult(indexedValue.index())] =
            indexedValue.value();
      }
    }
    return false;
  }

  // TODO(zinenko): LLVM IR lowering should not be aware of all the other
  // dialects.  Instead, we should have separate definitions for conversions in
  // a global lowering framework.  However, this requires LLVM dialect to be
  // implemented, which is currently blocked by the absence of user-defined
  // types.
  if (auto vectorTypeCastOp = inst.dyn_cast<VectorTypeCastOp>()) {
    auto targetMemRefType = vectorTypeCastOp->getType().dyn_cast<MemRefType>();

    llvm::Value *oldDescriptor =
        valueMapping.lookup(vectorTypeCastOp->getOperand());
    llvm::StructType *llvmTargetMemrefStructType =
        convertMemRefType(targetMemRefType);
    llvm::Value *newDescriptor =
        llvm::UndefValue::get(llvmTargetMemrefStructType);
    llvm::Value *dataPtr = builder.CreateExtractValue(oldDescriptor, 0);
    dataPtr = builder.CreateBitCast(
        dataPtr, llvmTargetMemrefStructType->getElementType(0));
    newDescriptor = builder.CreateInsertValue(newDescriptor, dataPtr, 0);
    valueMapping[vectorTypeCastOp->getResult()] = newDescriptor;
    return false;
  }

  // Terminators.
  if (auto returnInst = inst.dyn_cast<ReturnOp>()) {
    unsigned numOperands = returnInst->getNumOperands();
    if (numOperands == 0) {
      builder.CreateRetVoid();
    } else if (numOperands == 1) {
      builder.CreateRet(valueMapping[returnInst->getOperand(0)]);
    } else {
      llvm::Value *packed =
          packValues(llvm::to_vector<4>(returnInst->getOperands()));
      if (!packed)
        return true;
      builder.CreateRet(packed);
    }

    return false;
  }
  if (auto branchInst = inst.dyn_cast<BranchOp>()) {
    builder.CreateBr(blockMapping[branchInst->getDest()]);
    return false;
  }
  if (auto condBranchInst = inst.dyn_cast<CondBranchOp>()) {
    builder.CreateCondBr(valueMapping[condBranchInst->getCondition()],
                         blockMapping[condBranchInst->getTrueDest()],
                         blockMapping[condBranchInst->getFalseDest()]);
    return false;
  }
  return inst.emitError("unsupported operation");
}

bool ModuleLowerer::convertBasicBlock(const BasicBlock &bb,
                                      bool ignoreArguments) {
  builder.SetInsertPoint(blockMapping[&bb]);

  // Before traversing instructions, make block arguments available through
  // value remapping and PHI nodes, but do not add incoming edges for the PHI
  // nodes just yet: those values may be defined by this or following blocks.
  // This step is omitted if "ignoreArguments" is set.  The arguments of the
  // first basic block have been already made available through the remapping of
  // LLVM function arguments.
  if (!ignoreArguments) {
    auto predecessors = bb.getPredecessors();
    unsigned numPredecessors =
        std::distance(predecessors.begin(), predecessors.end());
    for (const auto *arg : bb.getArguments()) {
      llvm::Type *type = convertType(arg->getType());
      if (!type)
        return true;
      llvm::PHINode *phi = builder.CreatePHI(type, numPredecessors);
      valueMapping[arg] = phi;
    }
  }

  // Traverse instructions.
  for (const auto &inst : bb) {
    auto *op = dyn_cast<OperationInst>(&inst);
    if (!op)
      return inst.emitError("unsupported operation");

    if (convertInstruction(*op))
      return true;
  }

  return false;
}

// Get the SSA value passed to the current block from the terminator instruction
// of its predecessor.
static const Value *getPHISourceValue(const BasicBlock *current,
                                      const BasicBlock *pred,
                                      unsigned numArguments, unsigned index) {
  auto &terminator = *pred->getTerminator();
  if (terminator.isa<BranchOp>()) {
    return terminator.getOperand(index);
  }

  // For conditional branches, we need to check if the current block is reached
  // through the "true" or the "false" branch and take the relevant operands.
  auto condBranchOp = terminator.dyn_cast<CondBranchOp>();
  assert(condBranchOp &&
         "only branch instructions can be terminators of a basic block that "
         "has successors");

  condBranchOp->emitError("NYI: conditional branches with arguments");
  return nullptr;
}

void ModuleLowerer::connectPHINodes(const CFGFunction &cfgFunc) {
  // Skip the first block, it cannot be branched to and its arguments correspond
  // to the arguments of the LLVM function.
  for (auto it = std::next(cfgFunc.begin()), eit = cfgFunc.end(); it != eit;
       ++it) {
    const BasicBlock *bb = &*it;
    llvm::BasicBlock *llvmBB = blockMapping[bb];
    auto phis = llvmBB->phis();
    auto numArguments = bb->getNumArguments();
    assert(numArguments == std::distance(phis.begin(), phis.end()));
    for (auto &numberedPhiNode : llvm::enumerate(phis)) {
      auto &phiNode = numberedPhiNode.value();
      unsigned index = numberedPhiNode.index();
      for (const auto *pred : bb->getPredecessors()) {
        phiNode.addIncoming(
            valueMapping[getPHISourceValue(bb, pred, numArguments, index)],
            blockMapping[pred]);
      }
    }
  }
}

bool ModuleLowerer::convertCFGFunction(const CFGFunction &cfgFunc,
                                       llvm::Function &llvmFunc) {
  // Clear the block mapping.  Blocks belong to a function, no need to keep
  // blocks from the previous functions around.  Furthermore, we use this
  // mapping to connect PHI nodes inside the function later.
  blockMapping.clear();
  // First, create all blocks so we can jump to them.
  for (const auto &bb : cfgFunc) {
    auto *llvmBB = llvm::BasicBlock::Create(llvmContext);
    llvmBB->insertInto(&llvmFunc);
    blockMapping[&bb] = llvmBB;
  }

  // Then, convert blocks one by one.
  for (auto indexedBB : llvm::enumerate(cfgFunc)) {
    const auto &bb = indexedBB.value();
    if (convertBasicBlock(bb, /*ignoreArguments=*/indexedBB.index() == 0))
      return true;
  }

  // Finally, after all blocks have been traversed and values mapped, connect
  // the PHI nodes to the results of preceding blocks.
  connectPHINodes(cfgFunc);
  return false;
}

bool ModuleLowerer::convertFunctions(const Module &mlirModule,
                                     llvm::Module &llvmModule) {
  // Declare all functions first because there may be function calls that form a
  // call graph with cycles.  We don't expect MLFunctions here.
  for (const Function &function : mlirModule) {
    const Function *functionPtr = &function;
    if (functionPtr->isML())
      continue;
    llvm::Constant *llvmFuncCst = llvmModule.getOrInsertFunction(
        function.getName(), convertFunctionType(function.getType()));
    assert(isa<llvm::Function>(llvmFuncCst));
    functionMapping[functionPtr] = cast<llvm::Function>(llvmFuncCst);
  }

  // Convert CFG functions.
  for (const Function &function : mlirModule) {
    const Function *functionPtr = &function;
    if (!functionPtr->isCFG())
      continue;
    llvm::Function *llvmFunc = functionMapping[functionPtr];

    // Add function arguments to the value remapping table.  In CFGFunction,
    // arguments of the first block are those of the function.
    assert(!functionPtr->getBlocks().empty() &&
           "expected at least one basic block in a CFGFunction");
    const BasicBlock &firstBlock = *functionPtr->begin();
    for (auto arg : llvm::enumerate(llvmFunc->args())) {
      valueMapping[firstBlock.getArgument(arg.index())] = &arg.value();
    }

    if (convertCFGFunction(*functionPtr, *functionMapping[functionPtr]))
      return true;
  }
  return false;
}

bool ModuleLowerer::runOnModule(Module &m, llvm::Module &llvmModule) {
  // Create index type once for the entire module, it needs module info that is
  // not available in the convert*Type calls.
  indexType =
      builder.getIntNTy(llvmModule.getDataLayout().getPointerSizeInBits());

  // Declare or obtain (de)allocation functions.
  allocFunc = llvmModule.getOrInsertFunction("__mlir_alloc",
                                             builder.getInt8PtrTy(), indexType);
  freeFunc = llvmModule.getOrInsertFunction("__mlir_free", builder.getVoidTy(),
                                            builder.getInt8PtrTy());

  return convertFunctions(m, llvmModule);
}
} // namespace

// Entry point for the lowering procedure.
std::unique_ptr<llvm::Module>
mlir::convertModuleToLLVMIR(Module &module, llvm::LLVMContext &llvmContext) {
  auto llvmModule = llvm::make_unique<llvm::Module>("FIXME_name", llvmContext);
  if (ModuleLowerer(llvmContext).runOnModule(module, *llvmModule))
    return nullptr;
  return llvmModule;
}

// MLIR to LLVM IR translation registration.
static TranslateFromMLIRRegistration MLIRToLLVMIRTranslate(
    "mlir-to-llvmir", [](Module *module, llvm::StringRef outputFilename) {
      if (!module)
        return true;

      llvm::LLVMContext llvmContext;
      auto llvmModule = convertModuleToLLVMIR(*module, llvmContext);
      if (!llvmModule)
        return true;

      auto file = openOutputFile(outputFilename);
      if (!file)
        return true;

      llvmModule->print(file->os(), nullptr);
      file->keep();
      return false;
    });
