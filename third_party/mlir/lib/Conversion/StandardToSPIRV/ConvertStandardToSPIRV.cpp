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
#include "mlir/StandardOps/Ops.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

SPIRVTypeConverter::SPIRVTypeConverter(MLIRContext *context)
    : spirvDialect(context->getRegisteredDialect<spirv::SPIRVDialect>()) {}

Type SPIRVTypeConverter::convertType(Type t) {
  // Check if the type is SPIR-V supported. If so return the type.
  if (spirvDialect->isValidSPIRVType(t)) {
    return t;
  }

  if (auto memRefType = t.dyn_cast<MemRefType>()) {
    if (memRefType.hasStaticShape()) {
      // Convert MemrefType to spv.array if size is known.
      // TODO(ravishankarm) : For now hard-coding this to be StorageBuffer. Need
      // to support other Storage Classes.
      return spirv::PointerType::get(
          spirv::ArrayType::get(memRefType.getElementType(),
                                memRefType.getNumElements()),
          spirv::StorageClass::StorageBuffer);
    }
  }
  return Type();
}

//===----------------------------------------------------------------------===//
// Entry Function signature Conversion
//===----------------------------------------------------------------------===//

LogicalResult
SPIRVEntryFnTypeConverter::convertSignatureArg(unsigned inputNo, Type type,
                                               SignatureConversion &result) {
  // Try to convert the given input type.
  auto convertedType = convertType(type);
  // TODO(ravishankarm) : Vulkan spec requires these to be a
  // spirv::StructType. This is not a SPIR-V requirement, so just making this a
  // pointer type for now.
  if (!convertedType)
    return failure();
  // For arguments to entry functions, convert the type into a pointer type if
  // it is already not one.
  if (!convertedType.isa<spirv::PointerType>()) {
    // TODO(ravishankarm) : For now hard-coding this to be StorageBuffer. Need
    // to support other Storage classes.
    convertedType = spirv::PointerType::get(convertedType,
                                            spirv::StorageClass::StorageBuffer);
  }

  // Add the new inputs.
  result.addInputs(inputNo, convertedType);
  return success();
}

template <typename Converter>
static LogicalResult
lowerFunctionImpl(FuncOp funcOp, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter, Converter &typeConverter,
                  TypeConverter::SignatureConversion &signatureConverter,
                  FuncOp &newFuncOp) {
  auto fnType = funcOp.getType();

  if (fnType.getNumResults()) {
    return funcOp.emitError("SPIR-V dialect only supports functions with no "
                            "return values right now");
  }

  for (auto &argType : enumerate(fnType.getInputs())) {
    // Get the type of the argument
    if (failed(typeConverter.convertSignatureArg(
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

LogicalResult
SPIRVFnLowering::lowerFunction(FuncOp funcOp, ArrayRef<Value *> operands,
                               ConversionPatternRewriter &rewriter,
                               FuncOp &newFuncOp) const {
  auto fnType = funcOp.getType();
  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  return lowerFunctionImpl(funcOp, operands, rewriter, typeConverter,
                           signatureConverter, newFuncOp);
}

LogicalResult
SPIRVFnLowering::lowerAsEntryFunction(FuncOp funcOp, ArrayRef<Value *> operands,
                                      ConversionPatternRewriter &rewriter,
                                      FuncOp &newFuncOp) const {
  auto fnType = funcOp.getType();
  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  if (failed(lowerFunctionImpl(funcOp, operands, rewriter, entryFnConverter,
                               signatureConverter, newFuncOp))) {
    return failure();
  }
  // Create spv.Variable ops for each of the arguments. These need to be bound
  // by the runtime. For now use descriptor_set 0, and arg number as the binding
  // number.
  auto module = funcOp.getParentOfType<spirv::ModuleOp>();
  if (!module) {
    return funcOp.emitError("expected op to be within a spv.module");
  }
  OpBuilder builder(module.getOperation()->getRegion(0));
  SmallVector<Value *, 4> interface;
  for (auto &convertedArgType :
       llvm::enumerate(signatureConverter.getConvertedTypes())) {
    auto variableOp = builder.create<spirv::VariableOp>(
        funcOp.getLoc(), convertedArgType.value(),
        builder.getI32IntegerAttr(
            static_cast<int32_t>(spirv::StorageClass::StorageBuffer)),
        llvm::None);
    variableOp.setAttr("descriptor_set", builder.getI32IntegerAttr(0));
    variableOp.setAttr("binding",
                       builder.getI32IntegerAttr(convertedArgType.index()));
    interface.push_back(variableOp.getResult());
  }
  // Create an entry point instruction for this function.
  // TODO(ravishankarm) : Add execution mode for the entry function
  builder.setInsertionPoint(&(module.getBlock().back()));
  builder.create<spirv::EntryPointOp>(
      funcOp.getLoc(),
      builder.getI32IntegerAttr(
          static_cast<int32_t>(spirv::ExecutionModel::GLCompute)),
      builder.getSymbolRefAttr(newFuncOp.getName()), interface);
  return success();
}

//===----------------------------------------------------------------------===//
// Operation conversion
//===----------------------------------------------------------------------===//

namespace {
/// Convert return -> spv.Return.
class ReturnToSPIRVConversion : public ConversionPattern {
public:
  ReturnToSPIRVConversion(MLIRContext *context)
      : ConversionPattern(ReturnOp::getOperationName(), 1, context) {}
  virtual PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands()) {
      return matchFailure();
    }
    rewriter.replaceOpWithNewOp<spirv::ReturnOp>(op);
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
  patterns.insert<ReturnToSPIRVConversion>(context);
}
} // namespace mlir
