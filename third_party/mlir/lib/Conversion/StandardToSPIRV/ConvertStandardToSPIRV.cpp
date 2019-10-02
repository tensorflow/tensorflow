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

using namespace mlir;

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

Type SPIRVBasicTypeConverter::convertType(Type t) {
  // Check if the type is SPIR-V supported. If so return the type.
  if (spirv::SPIRVDialect::isValidType(t)) {
    return t;
  }

  if (auto indexType = t.dyn_cast<IndexType>()) {
    // Return I32 for index types.
    return IntegerType::get(32, t.getContext());
  }

  if (auto memRefType = t.dyn_cast<MemRefType>()) {
    if (memRefType.hasStaticShape()) {
      // Convert MemrefType to a multi-dimensional spv.array if size is known.
      auto elementType = memRefType.getElementType();
      for (auto size : reverse(memRefType.getShape())) {
        elementType = spirv::ArrayType::get(elementType, size);
      }
      // TODO(ravishankarm) : For now hard-coding this to be StorageBuffer. Need
      // to support other Storage Classes.
      return spirv::PointerType::get(elementType,
                                     spirv::StorageClass::StorageBuffer);
    }
  }
  return Type();
}

//===----------------------------------------------------------------------===//
// Entry Function signature Conversion
//===----------------------------------------------------------------------===//

LogicalResult
SPIRVTypeConverter::convertSignatureArg(unsigned inputNo, Type type,
                                        SignatureConversion &result) {
  // Try to convert the given input type.
  auto convertedType = basicTypeConverter->convertType(type);
  // TODO(ravishankarm) : Vulkan spec requires these to be a
  // spirv::StructType. This is not a SPIR-V requirement, so just making this a
  // pointer type for now.
  if (!convertedType)
    return failure();
  // For arguments to entry functions, convert the type into a pointer type if
  // it is already not one, unless the original type was an index type.
  // TODO(ravishankarm): For arguments that are of index type, keep the
  // arguments as the scalar converted type, i.e. i32. These are still not
  // handled effectively. These are potentially best handled as specialization
  // constants.
  if (!convertedType.isa<spirv::PointerType>() && !type.isa<IndexType>()) {
    // TODO(ravishankarm) : For now hard-coding this to be StorageBuffer. Need
    // to support other Storage classes.
    convertedType = spirv::PointerType::get(convertedType,
                                            spirv::StorageClass::StorageBuffer);
  }

  // Add the new inputs.
  result.addInputs(inputNo, convertedType);
  return success();
}

static LogicalResult lowerFunctionImpl(
    FuncOp funcOp, ArrayRef<Value *> operands,
    ConversionPatternRewriter &rewriter, TypeConverter *typeConverter,
    TypeConverter::SignatureConversion &signatureConverter, FuncOp &newFuncOp) {
  auto fnType = funcOp.getType();

  if (fnType.getNumResults()) {
    return funcOp.emitError("SPIR-V dialect only supports functions with no "
                            "return values right now");
  }

  for (auto &argType : enumerate(fnType.getInputs())) {
    // Get the type of the argument
    if (failed(typeConverter->convertSignatureArg(
            argType.index(), argType.value(), signatureConverter))) {
      return funcOp.emitError("unable to convert argument type ")
             << argType.value() << " to SPIR-V type";
    }
  }

  // Create a new function with an updated signature.
  newFuncOp = rewriter.cloneWithoutRegions(funcOp);
  rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                              newFuncOp.end());
  newFuncOp.setType(FunctionType::get(signatureConverter.getConvertedTypes(),
                                      llvm::None, funcOp.getContext()));

  // Tell the rewriter to convert the region signature.
  rewriter.applySignatureConversion(&newFuncOp.getBody(), signatureConverter);
  rewriter.replaceOp(funcOp.getOperation(), llvm::None);
  return success();
}

namespace mlir {
LogicalResult lowerFunction(FuncOp funcOp, ArrayRef<Value *> operands,
                            SPIRVTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter,
                            FuncOp &newFuncOp) {
  auto fnType = funcOp.getType();
  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  return lowerFunctionImpl(funcOp, operands, rewriter,
                           typeConverter->getBasicTypeConverter(),
                           signatureConverter, newFuncOp);
}

LogicalResult lowerAsEntryFunction(FuncOp funcOp, ArrayRef<Value *> operands,
                                   SPIRVTypeConverter *typeConverter,
                                   ConversionPatternRewriter &rewriter,
                                   FuncOp &newFuncOp) {
  auto fnType = funcOp.getType();
  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  if (failed(lowerFunctionImpl(funcOp, operands, rewriter, typeConverter,
                               signatureConverter, newFuncOp))) {
    return failure();
  }
  // Create spv.globalVariable ops for each of the arguments. These need to be
  // bound by the runtime. For now use descriptor_set 0, and arg number as the
  // binding number.
  auto module = funcOp.getParentOfType<spirv::ModuleOp>();
  if (!module) {
    return funcOp.emitError("expected op to be within a spv.module");
  }
  auto ip = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointToStart(&module.getBlock());
  SmallVector<Attribute, 4> interface;
  for (auto &convertedArgType :
       llvm::enumerate(signatureConverter.getConvertedTypes())) {
    // TODO(ravishankarm) : The arguments to the converted function are either
    // spirv::PointerType or i32 type, the latter due to conversion of index
    // type to i32. Eventually entry function should be of signature
    // void(void). Arguments converted to spirv::PointerType, will be made
    // variables and those converted to i32 will be made specialization
    // constants. Latter is not implemented.
    if (!convertedArgType.value().isa<spirv::PointerType>()) {
      continue;
    }
    std::string varName = funcOp.getName().str() + "_arg_" +
                          std::to_string(convertedArgType.index());
    auto variableOp = rewriter.create<spirv::GlobalVariableOp>(
        funcOp.getLoc(), rewriter.getTypeAttr(convertedArgType.value()),
        rewriter.getStringAttr(varName), nullptr);
    variableOp.setAttr("descriptor_set", rewriter.getI32IntegerAttr(0));
    variableOp.setAttr("binding",
                       rewriter.getI32IntegerAttr(convertedArgType.index()));
    interface.push_back(rewriter.getSymbolRefAttr(variableOp.sym_name()));
  }
  // Create an entry point instruction for this function.
  // TODO(ravishankarm) : Add execution mode for the entry function
  rewriter.setInsertionPoint(&(module.getBlock().back()));
  rewriter.create<spirv::EntryPointOp>(
      funcOp.getLoc(),
      rewriter.getI32IntegerAttr(
          static_cast<int32_t>(spirv::ExecutionModel::GLCompute)),
      rewriter.getSymbolRefAttr(newFuncOp.getName()),
      rewriter.getArrayAttr(interface));
  rewriter.restoreInsertionPoint(ip);
  return success();
}
} // namespace mlir

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

namespace {

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
  patterns.insert<IntegerOpConversion<AddIOp, spirv::IAddOp>,
                  IntegerOpConversion<MulIOp, spirv::IMulOp>, LoadOpConversion,
                  ReturnToSPIRVConversion, StoreOpConversion>(context);
}
} // namespace mlir
