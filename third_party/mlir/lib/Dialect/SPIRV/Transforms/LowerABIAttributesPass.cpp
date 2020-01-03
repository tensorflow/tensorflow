//===- LowerABIAttributesPass.cpp - Decorate composite type ---------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to lower attributes that specify the shader ABI
// for the functions in the generated SPIR-V module.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/LayoutUtils.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

/// Checks if the `type` is a scalar or vector type. It is assumed that they are
/// valid for SPIR-V dialect already.
static bool isScalarOrVectorType(Type type) {
  return spirv::SPIRVDialect::isValidScalarType(type) || type.isa<VectorType>();
}

/// Creates a global variable for an argument based on the ABI info.
static spirv::GlobalVariableOp
createGlobalVariableForArg(FuncOp funcOp, OpBuilder &builder, unsigned argNum,
                           spirv::InterfaceVarABIAttr abiInfo) {
  auto spirvModule = funcOp.getParentOfType<spirv::ModuleOp>();
  if (!spirvModule) {
    return nullptr;
  }
  OpBuilder::InsertionGuard moduleInsertionGuard(builder);
  builder.setInsertionPoint(funcOp.getOperation());
  std::string varName =
      funcOp.getName().str() + "_arg_" + std::to_string(argNum);

  // Get the type of variable. If this is a scalar/vector type and has an ABI
  // info create a variable of type !spv.ptr<!spv.struct<elementTYpe>>. If not
  // it must already be a !spv.ptr<!spv.struct<...>>.
  auto varType = funcOp.getType().getInput(argNum);
  auto storageClass =
      static_cast<spirv::StorageClass>(abiInfo.storage_class().getInt());
  if (isScalarOrVectorType(varType)) {
    varType =
        spirv::PointerType::get(spirv::StructType::get(varType), storageClass);
  }
  auto varPtrType = varType.cast<spirv::PointerType>();
  auto varPointeeType = varPtrType.getPointeeType().cast<spirv::StructType>();

  // Set the offset information.
  VulkanLayoutUtils::Size size = 0, alignment = 0;
  varPointeeType =
      VulkanLayoutUtils::decorateType(varPointeeType, size, alignment)
          .cast<spirv::StructType>();
  varType =
      spirv::PointerType::get(varPointeeType, varPtrType.getStorageClass());

  return builder.create<spirv::GlobalVariableOp>(
      funcOp.getLoc(), varType, varName, abiInfo.descriptor_set().getInt(),
      abiInfo.binding().getInt());
}

/// Gets the global variables that need to be specified as interface variable
/// with an spv.EntryPointOp. Traverses the body of a entry function to do so.
static LogicalResult
getInterfaceVariables(FuncOp funcOp,
                      SmallVectorImpl<Attribute> &interfaceVars) {
  auto module = funcOp.getParentOfType<spirv::ModuleOp>();
  if (!module) {
    return failure();
  }
  llvm::SetVector<Operation *> interfaceVarSet;

  // TODO(ravishankarm) : This should in reality traverse the entry function
  // call graph and collect all the interfaces. For now, just traverse the
  // instructions in this function.
  funcOp.walk([&](spirv::AddressOfOp addressOfOp) {
    auto var =
        module.lookupSymbol<spirv::GlobalVariableOp>(addressOfOp.variable());
    if (var.type().cast<spirv::PointerType>().getStorageClass() !=
        spirv::StorageClass::StorageBuffer) {
      interfaceVarSet.insert(var.getOperation());
    }
  });
  for (auto &var : interfaceVarSet) {
    interfaceVars.push_back(SymbolRefAttr::get(
        cast<spirv::GlobalVariableOp>(var).sym_name(), funcOp.getContext()));
  }
  return success();
}

/// Lowers the entry point attribute.
static LogicalResult lowerEntryPointABIAttr(FuncOp funcOp, OpBuilder &builder) {
  auto entryPointAttrName = spirv::getEntryPointABIAttrName();
  auto entryPointAttr =
      funcOp.getAttrOfType<spirv::EntryPointABIAttr>(entryPointAttrName);
  if (!entryPointAttr) {
    return failure();
  }

  OpBuilder::InsertionGuard moduleInsertionGuard(builder);
  auto spirvModule = funcOp.getParentOfType<spirv::ModuleOp>();
  builder.setInsertionPoint(spirvModule.body().front().getTerminator());

  // Adds the spv.EntryPointOp after collecting all the interface variables
  // needed.
  SmallVector<Attribute, 1> interfaceVars;
  if (failed(getInterfaceVariables(funcOp, interfaceVars))) {
    return failure();
  }
  builder.create<spirv::EntryPointOp>(
      funcOp.getLoc(), spirv::ExecutionModel::GLCompute, funcOp, interfaceVars);
  // Specifies the spv.ExecutionModeOp.
  auto localSizeAttr = entryPointAttr.local_size();
  SmallVector<int32_t, 3> localSize(localSizeAttr.getValues<int32_t>());
  builder.create<spirv::ExecutionModeOp>(
      funcOp.getLoc(), funcOp, spirv::ExecutionMode::LocalSize, localSize);
  funcOp.removeAttr(entryPointAttrName);
  return success();
}

namespace {
/// Pattern rewriter for changing function signature to match the ABI specified
/// in attributes.
class FuncOpLowering final : public SPIRVOpLowering<FuncOp> {
public:
  using SPIRVOpLowering<FuncOp>::SPIRVOpLowering;
  PatternMatchResult
  matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pass to implement the ABI information specified as attributes.
class LowerABIAttributesPass final
    : public OperationPass<LowerABIAttributesPass, spirv::ModuleOp> {
private:
  void runOnOperation() override;
};
} // namespace

PatternMatchResult
FuncOpLowering::matchAndRewrite(FuncOp funcOp, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const {
  if (!funcOp.getAttrOfType<spirv::EntryPointABIAttr>(
          spirv::getEntryPointABIAttrName())) {
    // TODO(ravishankarm) : Non-entry point functions are not handled.
    return matchFailure();
  }
  TypeConverter::SignatureConversion signatureConverter(
      funcOp.getType().getNumInputs());

  auto attrName = spirv::getInterfaceVarABIAttrName();
  for (auto argType : llvm::enumerate(funcOp.getType().getInputs())) {
    auto abiInfo = funcOp.getArgAttrOfType<spirv::InterfaceVarABIAttr>(
        argType.index(), attrName);
    if (!abiInfo) {
      // TODO(ravishankarm) : For non-entry point functions, it should be legal
      // to pass around scalar/vector values and return a scalar/vector. For now
      // non-entry point functions are not handled in this ABI lowering and will
      // produce an error.
      return matchFailure();
    }
    auto var =
        createGlobalVariableForArg(funcOp, rewriter, argType.index(), abiInfo);
    if (!var) {
      return matchFailure();
    }

    OpBuilder::InsertionGuard funcInsertionGuard(rewriter);
    rewriter.setInsertionPointToStart(&funcOp.front());
    // Insert spirv::AddressOf and spirv::AccessChain operations.
    Value replacement =
        rewriter.create<spirv::AddressOfOp>(funcOp.getLoc(), var);
    // Check if the arg is a scalar or vector type. In that case, the value
    // needs to be loaded into registers.
    // TODO(ravishankarm) : This is loading value of the scalar into registers
    // at the start of the function. It is probably better to do the load just
    // before the use. There might be multiple loads and currently there is no
    // easy way to replace all uses with a sequence of operations.
    if (isScalarOrVectorType(argType.value())) {
      auto indexType =
          typeConverter.convertType(IndexType::get(funcOp.getContext()));
      auto zero =
          spirv::ConstantOp::getZero(indexType, funcOp.getLoc(), &rewriter);
      auto loadPtr = rewriter.create<spirv::AccessChainOp>(
          funcOp.getLoc(), replacement, zero.constant());
      replacement = rewriter.create<spirv::LoadOp>(funcOp.getLoc(), loadPtr,
                                                   /*memory_access=*/nullptr,
                                                   /*alignment=*/nullptr);
    }
    signatureConverter.remapInput(argType.index(), replacement);
  }

  // Creates a new function with the update signature.
  rewriter.updateRootInPlace(funcOp, [&] {
    funcOp.setType(rewriter.getFunctionType(
        signatureConverter.getConvertedTypes(), llvm::None));
    rewriter.applySignatureConversion(&funcOp.getBody(), signatureConverter);
  });
  return matchSuccess();
}

void LowerABIAttributesPass::runOnOperation() {
  // Uses the signature conversion methodology of the dialect conversion
  // framework to implement the conversion.
  spirv::ModuleOp module = getOperation();
  MLIRContext *context = &getContext();

  SPIRVTypeConverter typeConverter;
  OwningRewritePatternList patterns;
  patterns.insert<FuncOpLowering>(context, typeConverter);

  ConversionTarget target(*context);
  target.addLegalDialect<spirv::SPIRVDialect>();
  auto entryPointAttrName = spirv::getEntryPointABIAttrName();
  target.addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
    return op.getAttrOfType<spirv::EntryPointABIAttr>(entryPointAttrName) &&
           op.getNumResults() == 0 && op.getNumArguments() == 0;
  });
  target.addLegalOp<ReturnOp>();
  if (failed(
          applyPartialConversion(module, target, patterns, &typeConverter))) {
    return signalPassFailure();
  }

  // Walks over all the FuncOps in spirv::ModuleOp to lower the entry point
  // attributes.
  OpBuilder builder(context);
  SmallVector<FuncOp, 1> entryPointFns;
  module.walk([&](FuncOp funcOp) {
    if (funcOp.getAttrOfType<spirv::EntryPointABIAttr>(entryPointAttrName)) {
      entryPointFns.push_back(funcOp);
    }
  });
  for (auto fn : entryPointFns) {
    if (failed(lowerEntryPointABIAttr(fn, builder))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OpPassBase<spirv::ModuleOp>>
mlir::spirv::createLowerABIAttributesPass() {
  return std::make_unique<LowerABIAttributesPass>();
}

static PassRegistration<LowerABIAttributesPass>
    pass("spirv-lower-abi-attrs", "Lower SPIR-V ABI Attributes");
