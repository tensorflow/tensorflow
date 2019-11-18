//===- SPIRVLowering.cpp - Standard to SPIR-V dialect conversion--===//
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
// This file implements utilities used to lower to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"

#include "mlir/Dialect/SPIRV/LayoutUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

namespace {
Type convertIndexType(MLIRContext *context) {
  // Convert to 32-bit integers for now. Might need a way to control this in
  // future.
  // TODO(ravishankarm): It is porbably better to make it 64-bit integers. To
  // this some support is needed in SPIR-V dialect for Conversion
  // instructions. The Vulkan spec requires the builtins like
  // GlobalInvocationID, etc. to be 32-bit (unsigned) integers which should be
  // SExtended to 64-bit for index computations.
  return IntegerType::get(32, context);
}

Type convertIndexType(IndexType t) { return convertIndexType(t.getContext()); }

Type basicTypeConversion(Type t) {
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

Type getLayoutDecoratedType(spirv::StructType type) {
  VulkanLayoutUtils::Size size = 0, alignment = 0;
  return VulkanLayoutUtils::decorateType(type, size, alignment);
}

/// Generates the type of variable given the type of object.
static Type getGlobalVarTypeForEntryFnArg(Type t) {
  auto convertedType = basicTypeConversion(t);
  if (auto ptrType = convertedType.dyn_cast<spirv::PointerType>()) {
    if (!ptrType.getPointeeType().isa<spirv::StructType>()) {
      return spirv::PointerType::get(
          getLayoutDecoratedType(
              spirv::StructType::get(ptrType.getPointeeType())),
          ptrType.getStorageClass());
    }
  } else {
    return spirv::PointerType::get(
        getLayoutDecoratedType(spirv::StructType::get(convertedType)),
        spirv::StorageClass::StorageBuffer);
  }
  return convertedType;
}
} // namespace

Type SPIRVBasicTypeConverter::convertType(Type t) {
  return basicTypeConversion(t);
}

Type SPIRVTypeConverter::convertType(Type t) {
  return getGlobalVarTypeForEntryFnArg(t);
}

//===----------------------------------------------------------------------===//
// Builtin Variables
//===----------------------------------------------------------------------===//

namespace {
/// Look through all global variables in `moduleOp` and check if there is a
/// spv.globalVariable that has the same `builtin` attribute.
spirv::GlobalVariableOp getBuiltinVariable(spirv::ModuleOp &moduleOp,
                                           spirv::BuiltIn builtin) {
  for (auto varOp : moduleOp.getBlock().getOps<spirv::GlobalVariableOp>()) {
    if (auto builtinAttr = varOp.getAttrOfType<StringAttr>(convertToSnakeCase(
            stringifyDecoration(spirv::Decoration::BuiltIn)))) {
      auto varBuiltIn = spirv::symbolizeBuiltIn(builtinAttr.getValue());
      if (varBuiltIn && varBuiltIn.getValue() == builtin) {
        return varOp;
      }
    }
  }
  return nullptr;
}

/// Gets name of global variable for a buitlin.
std::string getBuiltinVarName(spirv::BuiltIn builtin) {
  return std::string("__builtin_var_") + stringifyBuiltIn(builtin).str() + "__";
}

/// Gets or inserts a global variable for a builtin within a module.
spirv::GlobalVariableOp getOrInsertBuiltinVariable(spirv::ModuleOp &moduleOp,
                                                   Location loc,
                                                   spirv::BuiltIn builtin,
                                                   OpBuilder &builder) {
  if (auto varOp = getBuiltinVariable(moduleOp, builtin)) {
    return varOp;
  }
  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&moduleOp.getBlock());
  auto name = getBuiltinVarName(builtin);
  spirv::GlobalVariableOp newVarOp;
  switch (builtin) {
  case spirv::BuiltIn::NumWorkgroups:
  case spirv::BuiltIn::WorkgroupSize:
  case spirv::BuiltIn::WorkgroupId:
  case spirv::BuiltIn::LocalInvocationId:
  case spirv::BuiltIn::GlobalInvocationId: {
    auto ptrType = spirv::PointerType::get(
        VectorType::get({3}, builder.getIntegerType(32)),
        spirv::StorageClass::Input);
    newVarOp = builder.create<spirv::GlobalVariableOp>(
        loc, TypeAttr::get(ptrType), builder.getStringAttr(name), nullptr);
    newVarOp.setAttr(
        convertToSnakeCase(stringifyDecoration(spirv::Decoration::BuiltIn)),
        builder.getStringAttr(stringifyBuiltIn(builtin)));
    break;
  }
  default:
    emitError(loc, "unimplemented builtin variable generation for ")
        << stringifyBuiltIn(builtin);
  }
  builder.restoreInsertionPoint(ip);
  return newVarOp;
}
} // namespace

/// Gets the global variable associated with a builtin and add
/// it if it doesnt exist.
Value *mlir::spirv::getBuiltinVariableValue(Operation *op,
                                            spirv::BuiltIn builtin,
                                            OpBuilder &builder) {
  auto moduleOp = op->getParentOfType<spirv::ModuleOp>();
  if (!moduleOp) {
    op->emitError("expected operation to be within a SPIR-V module");
    return nullptr;
  }
  auto varOp =
      getOrInsertBuiltinVariable(moduleOp, op->getLoc(), builtin, builder);
  auto ptr = builder
                 .create<spirv::AddressOfOp>(op->getLoc(), varOp.type(),
                                             builder.getSymbolRefAttr(varOp))
                 .pointer();
  return builder.create<spirv::LoadOp>(
      op->getLoc(),
      ptr->getType().template cast<spirv::PointerType>().getPointeeType(), ptr,
      /*memory_access =*/nullptr, /*alignment =*/nullptr);
}

//===----------------------------------------------------------------------===//
// Entry Function signature Conversion
//===----------------------------------------------------------------------===//

namespace {
/// Computes the replacement value for an argument of an entry function. It
/// allocates a global variable for this argument and adds statements in the
/// entry block to get a replacement value within function scope.
Value *createAndLoadGlobalVarForEntryFnArg(PatternRewriter &rewriter,
                                           size_t origArgNum, Value *origArg) {
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
    rewriter.setInsertionPoint(funcOp.getOperation());
    std::string varName =
        funcOp.getName().str() + "_arg_" + std::to_string(origArgNum);
    var = rewriter.create<spirv::GlobalVariableOp>(
        funcOp.getLoc(),
        TypeAttr::get(getGlobalVarTypeForEntryFnArg(origArg->getType())),
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

FuncOp applySignatureConversion(
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
} // namespace

LogicalResult mlir::spirv::lowerAsEntryFunction(
    FuncOp funcOp, SPIRVTypeConverter *typeConverter,
    ConversionPatternRewriter &rewriter, FuncOp &newFuncOp) {
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
      signatureConverter.remapInput(origArg.index(), replacement);
    }
  }
  newFuncOp = applySignatureConversion(funcOp, rewriter, signatureConverter);
  return success();
}

LogicalResult mlir::spirv::finalizeEntryFunction(FuncOp newFuncOp,
                                                 OpBuilder &builder) {
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
