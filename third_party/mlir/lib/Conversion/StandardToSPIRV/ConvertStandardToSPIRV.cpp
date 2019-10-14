//===- ConvertStandardToSPIRV.cpp - Standard to SPIR-V dialect conversion--===//
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
// into the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

static Type convertIndexType(MLIRContext *context) {
  // Convert to 32-bit integers for now. Might need a way to control this in
  // future.
  // TODO(ravishankarm): It is porbably better to make it 64-bit integers. To
  // this some support is needed in SPIR-V dialect for Conversion
  // instructions. The Vulkan spec requires the builtins like
  // GlobalInvocationID, etc. to be 32-bit (unsigned) integers which should be
  // SExtended to 64-bit for index computations.
  return IntegerType::get(32, context);
}

static Type convertIndexType(IndexType t) {
  return convertIndexType(t.getContext());
}

static Type basicTypeConversion(Type t) {
  // Check if the type is SPIR-V supported. If so return the type.
  if (spirv::SPIRVDialect::isValidType(t)) {
    return t;
  }

  if (auto indexType = t.dyn_cast<IndexType>()) {
    return convertIndexType(indexType);
  }

  if (auto memRefType = t.dyn_cast<MemRefType>()) {
    auto elementType = memRefType.getElementType();
    if (memRefType.hasStaticShape()) {
      // Convert to a multi-dimensional spv.array if size is known.
      for (auto size : reverse(memRefType.getShape())) {
        elementType = spirv::ArrayType::get(elementType, size);
      }
      return spirv::PointerType::get(elementType,
                                     spirv::StorageClass::StorageBuffer);
    } else {
      // Vulkan SPIR-V validation rules require runtime array type to be the
      // last member of a struct.
      return spirv::PointerType::get(spirv::RuntimeArrayType::get(elementType),
                                     spirv::StorageClass::StorageBuffer);
    }
  }
  return Type();
}

Type SPIRVBasicTypeConverter::convertType(Type t) {
  return basicTypeConversion(t);
}

//===----------------------------------------------------------------------===//
// Entry Function signature Conversion
//===----------------------------------------------------------------------===//

/// Generates the type of variable given the type of object.
static Type getGlobalVarTypeForEntryFnArg(Type t) {
  auto convertedType = basicTypeConversion(t);
  if (auto ptrType = convertedType.dyn_cast<spirv::PointerType>()) {
    if (!ptrType.getPointeeType().isa<spirv::StructType>()) {
      return spirv::PointerType::get(
          spirv::StructType::get(ptrType.getPointeeType()),
          ptrType.getStorageClass());
    }
  } else {
    return spirv::PointerType::get(spirv::StructType::get(convertedType),
                                   spirv::StorageClass::StorageBuffer);
  }
  return convertedType;
}

Type SPIRVTypeConverter::convertType(Type t) {
  return getGlobalVarTypeForEntryFnArg(t);
}

/// Computes the replacement value for an argument of an entry function. It
/// allocates a global variable for this argument and adds statements in the
/// entry block to get a replacement value within function scope.
static Value *createAndLoadGlobalVarForEntryFnArg(PatternRewriter &rewriter,
                                                  size_t origArgNum,
                                                  Value *origArg) {
  // Create a global variable for this argument.
  auto insertionOp = rewriter.getInsertionBlock()->getParent();
  auto module = insertionOp->getParentOfType<spirv::ModuleOp>();
  if (!module) {
    return nullptr;
  }
  auto funcOp = insertionOp->getParentOfType<FuncOp>();
  spirv::GlobalVariableOp var;
  {
    OpBuilder::InsertionGuard moduleInsertionGuard(rewriter);
    rewriter.setInsertionPointToStart(&module.getBlock());
    std::string varName =
        funcOp.getName().str() + "_arg_" + std::to_string(origArgNum);
    var = rewriter.create<spirv::GlobalVariableOp>(
        funcOp.getLoc(),
        rewriter.getTypeAttr(getGlobalVarTypeForEntryFnArg(origArg->getType())),
        rewriter.getStringAttr(varName), nullptr);
    var.setAttr(
        spirv::SPIRVDialect::getAttributeName(spirv::Decoration::DescriptorSet),
        rewriter.getI32IntegerAttr(0));
    var.setAttr(
        spirv::SPIRVDialect::getAttributeName(spirv::Decoration::Binding),
        rewriter.getI32IntegerAttr(origArgNum));
  }
  // Insert the addressOf and load instructions, to get back the converted value
  // type.
  auto addressOf = rewriter.create<spirv::AddressOfOp>(funcOp.getLoc(), var);
  auto indexType = convertIndexType(funcOp.getContext());
  auto zero = rewriter.create<spirv::ConstantOp>(
      funcOp.getLoc(), indexType, rewriter.getIntegerAttr(indexType, 0));
  auto accessChain = rewriter.create<spirv::AccessChainOp>(
      funcOp.getLoc(), addressOf.pointer(), zero.constant());
  // If the original argument is a tensor/memref type, the value is not
  // loaded. Instead the pointer value is returned to allow its use in access
  // chain ops.
  auto origArgType = origArg->getType();
  if (origArgType.isa<MemRefType>()) {
    return accessChain;
  }
  return rewriter.create<spirv::LoadOp>(
      funcOp.getLoc(), accessChain.component_ptr(), /*memory_access=*/nullptr,
      /*alignment=*/nullptr);
}

static FuncOp applySignatureConversion(
    FuncOp funcOp, ConversionPatternRewriter &rewriter,
    TypeConverter::SignatureConversion &signatureConverter) {
  // Create a new function with an updated signature.
  auto newFuncOp = rewriter.cloneWithoutRegions(funcOp);
  rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                              newFuncOp.end());
  newFuncOp.setType(FunctionType::get(signatureConverter.getConvertedTypes(),
                                      llvm::None, funcOp.getContext()));

  // Tell the rewriter to convert the region signature.
  rewriter.applySignatureConversion(&newFuncOp.getBody(), signatureConverter);
  rewriter.replaceOp(funcOp.getOperation(), llvm::None);
  return newFuncOp;
}

/// Gets the global variables that need to be specified as interface variable
/// with an spv.EntryPointOp. Traverses the body of a entry function to do so.
LogicalResult getInterfaceVariables(FuncOp funcOp,
                                    SmallVectorImpl<Attribute> &interfaceVars) {
  auto module = funcOp.getParentOfType<spirv::ModuleOp>();
  if (!module) {
    return failure();
  }
  llvm::SetVector<Operation *> interfaceVarSet;
  for (auto &block : funcOp) {
    // TODO(ravishankarm) : This should in reality traverse the entry function
    // call graph and collect all the interfaces. For now, just traverse the
    // instructions in this function.
    auto callOps = block.getOps<CallOp>();
    if (std::distance(callOps.begin(), callOps.end())) {
      return funcOp.emitError("Collecting interface variables through function "
                              "calls unimplemented");
    }
    for (auto op : block.getOps<spirv::AddressOfOp>()) {
      auto var = module.lookupSymbol<spirv::GlobalVariableOp>(op.variable());
      if (var.type().cast<spirv::PointerType>().getStorageClass() ==
          spirv::StorageClass::StorageBuffer) {
        continue;
      }
      interfaceVarSet.insert(var.getOperation());
    }
  }
  for (auto &var : interfaceVarSet) {
    interfaceVars.push_back(SymbolRefAttr::get(
        cast<spirv::GlobalVariableOp>(var).sym_name(), funcOp.getContext()));
  }
  return success();
}

namespace mlir {
LogicalResult lowerFunction(FuncOp funcOp, SPIRVTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter,
                            FuncOp &newFuncOp) {
  auto fnType = funcOp.getType();
  if (fnType.getNumResults()) {
    return funcOp.emitError("SPIR-V lowering only supports functions with no "
                            "return values right now");
  }
  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  auto basicTypeConverter = typeConverter->getBasicTypeConverter();
  for (auto origArgType : enumerate(fnType.getInputs())) {
    auto convertedType = basicTypeConverter->convertType(origArgType.value());
    if (!convertedType) {
      return funcOp.emitError("unable to convert argument of type '")
             << convertedType << "'";
    }
    signatureConverter.addInputs(origArgType.index(), convertedType);
  }
  newFuncOp = applySignatureConversion(funcOp, rewriter, signatureConverter);
  return success();
}

LogicalResult lowerAsEntryFunction(FuncOp funcOp,
                                   SPIRVTypeConverter *typeConverter,
                                   ConversionPatternRewriter &rewriter,
                                   FuncOp &newFuncOp) {
  auto fnType = funcOp.getType();
  if (fnType.getNumResults()) {
    return funcOp.emitError("SPIR-V lowering only supports functions with no "
                            "return values right now");
  }
  // For entry functions need to make the signature void(void). Compute the
  // replacement value for all arguments and replace all uses.
  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  {
    OpBuilder::InsertionGuard moduleInsertionGuard(rewriter);
    rewriter.setInsertionPointToStart(&funcOp.front());
    for (auto origArg : enumerate(funcOp.getArguments())) {
      auto replacement = createAndLoadGlobalVarForEntryFnArg(
          rewriter, origArg.index(), origArg.value());
      rewriter.replaceUsesOfBlockArgument(origArg.value(), replacement);
    }
  }
  newFuncOp = applySignatureConversion(funcOp, rewriter, signatureConverter);
  return success();
}

LogicalResult finalizeEntryFunction(FuncOp newFuncOp, OpBuilder &builder) {
  // Add the spv.EntryPointOp after collecting all the interface variables
  // needed.
  SmallVector<Attribute, 1> interfaceVars;
  if (failed(getInterfaceVariables(newFuncOp, interfaceVars))) {
    return failure();
  }
  builder.create<spirv::EntryPointOp>(newFuncOp.getLoc(),
                                      spirv::ExecutionModel::GLCompute,
                                      newFuncOp, interfaceVars);
  // Specify the spv.ExecutionModeOp.

  /// TODO(ravishankarm): Vulkan environment for SPIR-V requires "either a
  /// LocalSize execution mode or an object decorated with the WorkgroupSize
  /// decoration must be specified." Better approach is to use the
  /// WorkgroupSize GlobalVariable with initializer being a specialization
  /// constant. But current support for specialization constant does not allow
  /// for this. So for now use the execution mode. Hard-wiring this to {1, 1,
  /// 1} for now. To be fixed ASAP.
  builder.create<spirv::ExecutionModeOp>(newFuncOp.getLoc(), newFuncOp,
                                         spirv::ExecutionMode::LocalSize,
                                         ArrayRef<int32_t>{1, 1, 1});
  return success();
}
} // namespace mlir

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

namespace {

/// Convert constant operation with IndexType return to SPIR-V constant
/// operation. Since IndexType is not used within SPIR-V dialect, this needs
/// special handling to make sure the result type and the type of the value
/// attribute are consistent.
class ConstantIndexOpConversion final : public ConversionPattern {
public:
  ConstantIndexOpConversion(MLIRContext *context)
      : ConversionPattern(ConstantOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto constIndexOp = cast<ConstantOp>(op);
    if (!constIndexOp.getResult()->getType().isa<IndexType>()) {
      return matchFailure();
    }
    // The attribute has index type. Get the integer value and create a new
    // IntegerAttr.
    auto constAttr = constIndexOp.value().dyn_cast<IntegerAttr>();
    if (!constAttr) {
      return matchFailure();
    }

    // Use the bitwidth set in the value attribute to decide the result type of
    // the SPIR-V constant operation since SPIR-V does not support index types.
    auto constVal = constAttr.getValue();
    auto constValType = constAttr.getType().dyn_cast<IndexType>();
    if (!constValType) {
      return matchFailure();
    }
    auto spirvConstType = convertIndexType(constValType);
    auto spirvConstVal =
        rewriter.getIntegerAttr(spirvConstType, constAttr.getInt());
    auto spirvConstantOp = rewriter.create<spirv::ConstantOp>(
        op->getLoc(), spirvConstType, spirvConstVal);
    rewriter.replaceOp(op, spirvConstantOp.constant(), {});
    return matchSuccess();
  }
};

/// Convert integer binary operations to SPIR-V operations. Cannot use tablegen
/// for this. If the integer operation is on variables of IndexType, the type of
/// the return value of the replacement operation differs from that of the
/// replaced operation. This is not handled in tablegen-based pattern
/// specification.
template <typename StdOp, typename SPIRVOp>
class IntegerOpConversion final : public ConversionPattern {
public:
  IntegerOpConversion(MLIRContext *context)
      : ConversionPattern(StdOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.template replaceOpWithNewOp<SPIRVOp>(
        op, operands[0]->getType(), operands, ArrayRef<NamedAttribute>());
    return this->matchSuccess();
  }
};

/// Convert load -> spv.LoadOp. The operands of the replaced operation are of
/// IndexType while that of the replacement operation are of type i32. This is
/// not suppored in tablegen based pattern specification.
// TODO(ravishankarm) : These could potentially be templated on the operation
// being converted, since the same logic should work for linalg.load.
class LoadOpConversion final : public ConversionPattern {
public:
  LoadOpConversion(MLIRContext *context)
      : ConversionPattern(LoadOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    LoadOpOperandAdaptor loadOperands(operands);
    auto basePtr = loadOperands.memref();
    auto ptrType = basePtr->getType().dyn_cast<spirv::PointerType>();
    if (!ptrType) {
      return matchFailure();
    }
    auto loadPtr = rewriter.create<spirv::AccessChainOp>(
        op->getLoc(), basePtr, loadOperands.indices());
    auto loadPtrType = loadPtr.getType().cast<spirv::PointerType>();
    rewriter.replaceOpWithNewOp<spirv::LoadOp>(
        op, loadPtrType.getPointeeType(), loadPtr, /*memory_access =*/nullptr,
        /*alignment =*/nullptr);
    return matchSuccess();
  }
};

/// Convert return -> spv.Return.
class ReturnToSPIRVConversion : public ConversionPattern {
public:
  ReturnToSPIRVConversion(MLIRContext *context)
      : ConversionPattern(ReturnOp::getOperationName(), 1, context) {}
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands()) {
      return matchFailure();
    }
    rewriter.replaceOpWithNewOp<spirv::ReturnOp>(op);
    return matchSuccess();
  }
};

/// Convert store -> spv.StoreOp. The operands of the replaced operation are of
/// IndexType while that of the replacement operation are of type i32. This is
/// not suppored in tablegen based pattern specification.
// TODO(ravishankarm) : These could potentially be templated on the operation
// being converted, since the same logic should work for linalg.store.
class StoreOpConversion final : public ConversionPattern {
public:
  StoreOpConversion(MLIRContext *context)
      : ConversionPattern(StoreOp::getOperationName(), 1, context) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    StoreOpOperandAdaptor storeOperands(operands);
    auto value = storeOperands.value();
    auto basePtr = storeOperands.memref();
    auto ptrType = basePtr->getType().dyn_cast<spirv::PointerType>();
    if (!ptrType) {
      return matchFailure();
    }
    auto storePtr = rewriter.create<spirv::AccessChainOp>(
        op->getLoc(), basePtr, storeOperands.indices());
    rewriter.replaceOpWithNewOp<spirv::StoreOp>(op, storePtr, value,
                                                /*memory_access =*/nullptr,
                                                /*alignment =*/nullptr);
    return matchSuccess();
  }
};

} // namespace

namespace {
/// Import the Standard Ops to SPIR-V Patterns.
#include "StandardToSPIRV.cpp.inc"
} // namespace

namespace mlir {
void populateStandardToSPIRVPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns) {
  populateWithGenerated(context, &patterns);
  // Add the return op conversion.
  patterns.insert<ConstantIndexOpConversion,
                  IntegerOpConversion<AddIOp, spirv::IAddOp>,
                  IntegerOpConversion<MulIOp, spirv::IMulOp>, LoadOpConversion,
                  ReturnToSPIRVConversion, StoreOpConversion>(context);
}
} // namespace mlir
