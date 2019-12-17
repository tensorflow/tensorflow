//===- ConvertGPUToSPIRV.cpp - Convert GPU ops to SPIR-V dialect ----------===//
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
// This file implements the conversion patterns from GPU ops to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/IR/Module.h"

using namespace mlir;

namespace {

/// Pattern to convert a loop::ForOp within kernel functions into spirv::LoopOp.
class ForOpConversion final : public SPIRVOpLowering<loop::ForOp> {
public:
  using SPIRVOpLowering<loop::ForOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(loop::ForOp forOp, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern lowering GPU block/thread size/id to loading SPIR-V invocation
/// builin variables.
template <typename SourceOp, spirv::BuiltIn builtin>
class LaunchConfigConversion : public SPIRVOpLowering<SourceOp> {
public:
  using SPIRVOpLowering<SourceOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(SourceOp op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert a kernel function in GPU dialect within a spv.module.
class KernelFnConversion final : public SPIRVOpLowering<gpu::GPUFuncOp> {
public:
  KernelFnConversion(MLIRContext *context, SPIRVTypeConverter &converter,
                     ArrayRef<int64_t> workGroupSize,
                     PatternBenefit benefit = 1)
      : SPIRVOpLowering<gpu::GPUFuncOp>(context, converter, benefit) {
    auto config = workGroupSize.take_front(3);
    workGroupSizeAsInt32.assign(config.begin(), config.end());
    workGroupSizeAsInt32.resize(3, 1);
  }

  PatternMatchResult
  matchAndRewrite(gpu::GPUFuncOp funcOp, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override;

private:
  SmallVector<int32_t, 3> workGroupSizeAsInt32;
};

/// Pattern to convert a module with gpu.kernel_module attribute to a
/// spv.module.
class KernelModuleConversion final : public SPIRVOpLowering<ModuleOp> {
public:
  using SPIRVOpLowering<ModuleOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(ModuleOp moduleOp, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert a module terminator op to a terminator of spv.module op.
// TODO: Move this into DRR, but that requires ModuleTerminatorOp to be defined
// in ODS.
class KernelModuleTerminatorConversion final
    : public SPIRVOpLowering<ModuleTerminatorOp> {
public:
  using SPIRVOpLowering<ModuleTerminatorOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(ModuleTerminatorOp terminatorOp, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

/// Pattern to convert a gpu.return into a SPIR-V return.
// TODO: This can go to DRR when GPU return has operands.
class GPUReturnOpConversion final : public SPIRVOpLowering<gpu::ReturnOp> {
public:
  using SPIRVOpLowering<gpu::ReturnOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(gpu::ReturnOp returnOp, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

//===----------------------------------------------------------------------===//
// loop::ForOp.
//===----------------------------------------------------------------------===//

PatternMatchResult
ForOpConversion::matchAndRewrite(loop::ForOp forOp, ArrayRef<Value *> operands,
                                 ConversionPatternRewriter &rewriter) const {
  // loop::ForOp can be lowered to the structured control flow represented by
  // spirv::LoopOp by making the continue block of the spirv::LoopOp the loop
  // latch and the merge block the exit block. The resulting spirv::LoopOp has a
  // single back edge from the continue to header block, and a single exit from
  // header to merge.
  loop::ForOpOperandAdaptor forOperands(operands);
  auto loc = forOp.getLoc();
  auto loopControl = rewriter.getI32IntegerAttr(
      static_cast<uint32_t>(spirv::LoopControl::None));
  auto loopOp = rewriter.create<spirv::LoopOp>(loc, loopControl);
  loopOp.addEntryAndMergeBlock();

  OpBuilder::InsertionGuard guard(rewriter);
  // Create the block for the header.
  auto header = new Block();
  // Insert the header.
  loopOp.body().getBlocks().insert(std::next(loopOp.body().begin(), 1), header);

  // Create the new induction variable to use.
  BlockArgument *newIndVar =
      header->addArgument(forOperands.lowerBound()->getType());
  Block *body = forOp.getBody();

  // Apply signature conversion to the body of the forOp. It has a single block,
  // with argument which is the induction variable. That has to be replaced with
  // the new induction variable.
  TypeConverter::SignatureConversion signatureConverter(
      body->getNumArguments());
  signatureConverter.remapInput(0, newIndVar);
  body = rewriter.applySignatureConversion(&forOp.getLoopBody(),
                                           signatureConverter);

  // Delete the loop terminator.
  rewriter.eraseOp(body->getTerminator());

  // Move the blocks from the forOp into the loopOp. This is the body of the
  // loopOp.
  rewriter.inlineRegionBefore(forOp.getOperation()->getRegion(0), loopOp.body(),
                              std::next(loopOp.body().begin(), 2));

  // Branch into it from the entry.
  rewriter.setInsertionPointToEnd(&(loopOp.body().front()));
  rewriter.create<spirv::BranchOp>(loc, header, forOperands.lowerBound());

  // Generate the rest of the loop header.
  rewriter.setInsertionPointToEnd(header);
  auto mergeBlock = loopOp.getMergeBlock();
  auto cmpOp = rewriter.create<spirv::SLessThanOp>(
      loc, rewriter.getI1Type(), newIndVar, forOperands.upperBound());
  rewriter.create<spirv::BranchConditionalOp>(
      loc, cmpOp, body, ArrayRef<Value *>(), mergeBlock, ArrayRef<Value *>());

  // Generate instructions to increment the step of the induction variable and
  // branch to the header.
  Block *continueBlock = loopOp.getContinueBlock();
  rewriter.setInsertionPointToEnd(continueBlock);

  // Add the step to the induction variable and branch to the header.
  Value *updatedIndVar = rewriter.create<spirv::IAddOp>(
      loc, newIndVar->getType(), newIndVar, forOperands.step());
  rewriter.create<spirv::BranchOp>(loc, header, updatedIndVar);

  rewriter.eraseOp(forOp);
  return matchSuccess();
}

//===----------------------------------------------------------------------===//
// Builtins.
//===----------------------------------------------------------------------===//

template <typename SourceOp, spirv::BuiltIn builtin>
PatternMatchResult LaunchConfigConversion<SourceOp, builtin>::matchAndRewrite(
    SourceOp op, ArrayRef<Value *> operands,
    ConversionPatternRewriter &rewriter) const {
  auto dimAttr =
      op.getOperation()->template getAttrOfType<StringAttr>("dimension");
  if (!dimAttr) {
    return this->matchFailure();
  }
  int32_t index = 0;
  if (dimAttr.getValue() == "x") {
    index = 0;
  } else if (dimAttr.getValue() == "y") {
    index = 1;
  } else if (dimAttr.getValue() == "z") {
    index = 2;
  } else {
    return this->matchFailure();
  }

  // SPIR-V invocation builtin variables are a vector of type <3xi32>
  auto spirvBuiltin = spirv::getBuiltinVariableValue(op, builtin, rewriter);
  rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
      op, rewriter.getIntegerType(32), spirvBuiltin,
      rewriter.getI32ArrayAttr({index}));
  return this->matchSuccess();
}

//===----------------------------------------------------------------------===//
// GPUFuncOp
//===----------------------------------------------------------------------===//

// Legalizes a GPU function as an entry SPIR-V function.
static FuncOp
lowerAsEntryFunction(gpu::GPUFuncOp funcOp, SPIRVTypeConverter &typeConverter,
                     ConversionPatternRewriter &rewriter,
                     spirv::EntryPointABIAttr entryPointInfo,
                     ArrayRef<spirv::InterfaceVarABIAttr> argABIInfo) {
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
  // Update the signature to valid SPIR-V types and add the ABI
  // attributes. These will be "materialized" by using the
  // LowerABIAttributesPass.
  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  {
    for (auto argType : enumerate(funcOp.getType().getInputs())) {
      auto convertedType = typeConverter.convertType(argType.value());
      signatureConverter.addInputs(argType.index(), convertedType);
    }
  }
  auto newFuncOp = rewriter.create<FuncOp>(
      funcOp.getLoc(), funcOp.getName(),
      rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                               llvm::None),
      ArrayRef<NamedAttribute>());
  for (const auto &namedAttr : funcOp.getAttrs()) {
    if (namedAttr.first.is(impl::getTypeAttrName()) ||
        namedAttr.first.is(SymbolTable::getSymbolAttrName()))
      continue;
    newFuncOp.setAttr(namedAttr.first, namedAttr.second);
  }
  rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                              newFuncOp.end());
  rewriter.applySignatureConversion(&newFuncOp.getBody(), signatureConverter);
  rewriter.eraseOp(funcOp);

  spirv::setABIAttrs(newFuncOp, entryPointInfo, argABIInfo);
  return newFuncOp;
}

PatternMatchResult
KernelFnConversion::matchAndRewrite(gpu::GPUFuncOp funcOp,
                                    ArrayRef<Value *> operands,
                                    ConversionPatternRewriter &rewriter) const {
  if (!gpu::GPUDialect::isKernel(funcOp)) {
    return matchFailure();
  }

  SmallVector<spirv::InterfaceVarABIAttr, 4> argABI;
  for (auto argNum : llvm::seq<unsigned>(0, funcOp.getNumArguments())) {
    argABI.push_back(spirv::getInterfaceVarABIAttr(
        0, argNum, spirv::StorageClass::StorageBuffer, rewriter.getContext()));
  }

  auto context = rewriter.getContext();
  auto entryPointAttr =
      spirv::getEntryPointABIAttr(workGroupSizeAsInt32, context);
  FuncOp newFuncOp = lowerAsEntryFunction(funcOp, typeConverter, rewriter,
                                          entryPointAttr, argABI);
  if (!newFuncOp) {
    return matchFailure();
  }
  newFuncOp.removeAttr(Identifier::get(gpu::GPUDialect::getKernelFuncAttrName(),
                                       rewriter.getContext()));
  return matchSuccess();
}

//===----------------------------------------------------------------------===//
// ModuleOp with gpu.kernel_module.
//===----------------------------------------------------------------------===//

PatternMatchResult KernelModuleConversion::matchAndRewrite(
    ModuleOp moduleOp, ArrayRef<Value *> operands,
    ConversionPatternRewriter &rewriter) const {
  if (!moduleOp.getAttrOfType<UnitAttr>(
          gpu::GPUDialect::getKernelModuleAttrName())) {
    return matchFailure();
  }
  // TODO : Generalize this to account for different extensions,
  // capabilities, extended_instruction_sets, other addressing models
  // and memory models.
  auto spvModule = rewriter.create<spirv::ModuleOp>(
      moduleOp.getLoc(), spirv::AddressingModel::Logical,
      spirv::MemoryModel::GLSL450, spirv::Capability::Shader,
      spirv::Extension::SPV_KHR_storage_buffer_storage_class);
  // Move the region from the module op into the SPIR-V module.
  Region &spvModuleRegion = spvModule.getOperation()->getRegion(0);
  rewriter.inlineRegionBefore(moduleOp.getBodyRegion(), spvModuleRegion,
                              spvModuleRegion.begin());
  // The spv.module build method adds a block with a terminator. Remove that
  // block. The terminator of the module op in the remaining block will be
  // legalized later.
  spvModuleRegion.back().erase();
  rewriter.eraseOp(moduleOp);
  return matchSuccess();
}

//===----------------------------------------------------------------------===//
// ModuleTerminatorOp for gpu.kernel_module.
//===----------------------------------------------------------------------===//

PatternMatchResult KernelModuleTerminatorConversion::matchAndRewrite(
    ModuleTerminatorOp terminatorOp, ArrayRef<Value *> operands,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<spirv::ModuleEndOp>(terminatorOp);
  return matchSuccess();
}

//===----------------------------------------------------------------------===//
// GPU return inside kernel functions to SPIR-V return.
//===----------------------------------------------------------------------===//

PatternMatchResult GPUReturnOpConversion::matchAndRewrite(
    gpu::ReturnOp returnOp, ArrayRef<Value *> operands,
    ConversionPatternRewriter &rewriter) const {
  if (!operands.empty())
    return matchFailure();

  rewriter.replaceOpWithNewOp<spirv::ReturnOp>(returnOp);
  return matchSuccess();
}

//===----------------------------------------------------------------------===//
// GPU To SPIRV Patterns.
//===----------------------------------------------------------------------===//

namespace mlir {
void populateGPUToSPIRVPatterns(MLIRContext *context,
                                SPIRVTypeConverter &typeConverter,
                                OwningRewritePatternList &patterns,
                                ArrayRef<int64_t> workGroupSize) {
  patterns.insert<KernelFnConversion>(context, typeConverter, workGroupSize);
  patterns.insert<
      GPUReturnOpConversion, ForOpConversion, KernelModuleConversion,
      KernelModuleTerminatorConversion,
      LaunchConfigConversion<gpu::BlockDimOp, spirv::BuiltIn::WorkgroupSize>,
      LaunchConfigConversion<gpu::BlockIdOp, spirv::BuiltIn::WorkgroupId>,
      LaunchConfigConversion<gpu::GridDimOp, spirv::BuiltIn::NumWorkgroups>,
      LaunchConfigConversion<gpu::ThreadIdOp,
                             spirv::BuiltIn::LocalInvocationId>>(context,
                                                                 typeConverter);
}
} // namespace mlir
