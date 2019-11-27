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
#include "llvm/ADT/Sequence.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Attributes for ABI
//===----------------------------------------------------------------------===//

// Pull in the attributes needed for lowering.
namespace mlir {
#include "mlir/Dialect/SPIRV/SPIRVLowering.cpp.inc"
}

StringRef mlir::spirv::getInterfaceVarABIAttrName() {
  return "spirv.interface_var_abi";
}

mlir::spirv::InterfaceVarABIAttr
mlir::spirv::getInterfaceVarABIAttr(unsigned descriptorSet, unsigned binding,
                                    spirv::StorageClass storageClass,
                                    MLIRContext *context) {
  Type i32Type = IntegerType::get(32, context);
  return mlir::spirv::InterfaceVarABIAttr::get(
      IntegerAttr::get(i32Type, descriptorSet),
      IntegerAttr::get(i32Type, binding),
      IntegerAttr::get(i32Type, static_cast<int64_t>(storageClass)), context);
}

StringRef mlir::spirv::getEntryPointABIAttrName() {
  return "spirv.entry_point_abi";
}

mlir::spirv::EntryPointABIAttr
mlir::spirv::getEntryPointABIAttr(ArrayRef<int32_t> localSize,
                                  MLIRContext *context) {
  assert(localSize.size() == 3);
  return mlir::spirv::EntryPointABIAttr::get(
      DenseElementsAttr::get<int32_t>(
          VectorType::get(3, IntegerType::get(32, context)), localSize)
          .cast<DenseIntElementsAttr>(),
      context);
}

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

// TODO(ravishankarm): This is a utility function that should probably be
// exposed by the SPIR-V dialect. Keeping it local till the use case arises.
Optional<int64_t> getTypeNumBytes(Type t) {
  if (auto integerType = t.dyn_cast<IntegerType>()) {
    return integerType.getWidth() / 8;
  } else if (auto floatType = t.dyn_cast<FloatType>()) {
    return floatType.getWidth() / 8;
  }
  // TODO: Add size computation for other types.
  return llvm::None;
}

Type typeConversionImpl(Type t) {
  // Check if the type is SPIR-V supported. If so return the type.
  if (spirv::SPIRVDialect::isValidType(t)) {
    return t;
  }

  if (auto indexType = t.dyn_cast<IndexType>()) {
    return convertIndexType(t.getContext());
  }

  if (auto memRefType = t.dyn_cast<MemRefType>()) {
    // TODO(ravishankarm): For now only support default memory space. The memory
    // space description is not set is stone within MLIR, i.e. it depends on the
    // context it is being used. To map this to SPIR-V storage classes, we
    // should rely on the ABI attributes, and not on the memory space. This is
    // still evolving, and needs to be revisited when there is more clarity.
    if (memRefType.getMemorySpace()) {
      return Type();
    }
    auto elementType = typeConversionImpl(memRefType.getElementType());
    if (!elementType) {
      return Type();
    }
    auto elementSize = getTypeNumBytes(elementType);
    if (!elementSize) {
      return Type();
    }
    // TODO(ravishankarm) : Handle dynamic shapes.
    if (memRefType.hasStaticShape()) {
      // Get the strides and offset
      int64_t offset;
      SmallVector<int64_t, 4> strides;
      if (failed(getStridesAndOffset(memRefType, strides, offset)) ||
          offset == MemRefType::getDynamicStrideOrOffset() ||
          llvm::is_contained(strides, MemRefType::getDynamicStrideOrOffset())) {
        // TODO(ravishankarm) : Handle dynamic strides and offsets.
        return Type();
      }
      // Convert to a multi-dimensional spv.array if size is known.
      auto shape = memRefType.getShape();
      assert(shape.size() == strides.size());
      for (int i = shape.size(); i > 0; --i) {
        elementType = spirv::ArrayType::get(
            elementType, shape[i - 1], strides[i - 1] * elementSize.getValue());
      }
      // For the offset, need to wrap the array in a struct.
      auto structType =
          spirv::StructType::get(elementType, offset * elementSize.getValue());
      // For now initialize the storage class to StorageBuffer. This will be
      // updated later based on whats passed in w.r.t to the ABI attributes.
      return spirv::PointerType::get(structType,
                                     spirv::StorageClass::StorageBuffer);
    }
  }
  return Type();
}
} // namespace

Type SPIRVTypeConverter::convertType(Type t) { return typeConversionImpl(t); }

Type SPIRVTypeConverter::getIndexType(MLIRContext *context) {
  return convertType(IndexType::get(context));
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

FuncOp mlir::spirv::lowerAsEntryFunction(
    FuncOp funcOp, SPIRVTypeConverter &typeConverter,
    ConversionPatternRewriter &rewriter,
    ArrayRef<spirv::InterfaceVarABIAttr> argABIInfo,
    spirv::EntryPointABIAttr entryPointInfo) {
  auto fnType = funcOp.getType();
  if (fnType.getNumResults()) {
    funcOp.emitError("SPIR-V lowering only supports entry functions"
                     "with no return values right now");
    return nullptr;
  }
  if (fnType.getNumInputs() != argABIInfo.size()) {
    funcOp.emitError(
        "lowering as entry functions requires ABI info for all arguments");
    return nullptr;
  }
  // For entry functions need to make the signature void(void). Compute the
  // replacement value for all arguments and replace all uses.
  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  {
    for (auto argType : enumerate(funcOp.getType().getInputs())) {
      auto convertedType = typeConverter.convertType(argType.value());
      signatureConverter.addInputs(argType.index(), convertedType);
    }
  }
  auto newFuncOp = rewriter.cloneWithoutRegions(funcOp);
  rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                              newFuncOp.end());
  newFuncOp.setType(rewriter.getFunctionType(
      signatureConverter.getConvertedTypes(), llvm::None));
  rewriter.applySignatureConversion(&newFuncOp.getBody(), signatureConverter);
  rewriter.replaceOp(funcOp.getOperation(), llvm::None);

  // Set the attributes for argument and the function.
  StringRef argABIAttrName = spirv::getInterfaceVarABIAttrName();
  for (auto argIndex : llvm::seq<unsigned>(0, newFuncOp.getNumArguments())) {
    newFuncOp.setArgAttr(argIndex, argABIAttrName, argABIInfo[argIndex]);
  }
  newFuncOp.setAttr(spirv::getEntryPointABIAttrName(), entryPointInfo);
  return newFuncOp;
}
