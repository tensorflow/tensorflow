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

/// Pattern to convert a kernel function in GPU dialect (a FuncOp with the
/// attribute gpu.kernel) within a spv.module.
class KernelFnConversion final : public SPIRVOpLowering<FuncOp> {
public:
  using SPIRVOpLowering<FuncOp>::SPIRVOpLowering;

  PatternMatchResult
  matchAndRewrite(FuncOp funcOp, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override;
};

} // namespace

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

PatternMatchResult
KernelFnConversion::matchAndRewrite(FuncOp funcOp, ArrayRef<Value *> operands,
                                    ConversionPatternRewriter &rewriter) const {
  FuncOp newFuncOp;
  if (!gpu::GPUDialect::isKernel(funcOp)) {
    return matchFailure();
  }

  if (failed(spirv::lowerAsEntryFunction(funcOp, &typeConverter, rewriter,
                                         newFuncOp))) {
    return matchFailure();
  }
  return matchSuccess();
}

namespace mlir {
void populateGPUToSPIRVPatterns(MLIRContext *context,
                                SPIRVTypeConverter &typeConverter,
                                OwningRewritePatternList &patterns) {
  patterns.insert<
      ForOpConversion, KernelFnConversion,
      LaunchConfigConversion<gpu::BlockDimOp, spirv::BuiltIn::WorkgroupSize>,
      LaunchConfigConversion<gpu::BlockIdOp, spirv::BuiltIn::WorkgroupId>,
      LaunchConfigConversion<gpu::GridDimOp, spirv::BuiltIn::NumWorkgroups>,
      LaunchConfigConversion<gpu::ThreadIdOp,
                             spirv::BuiltIn::LocalInvocationId>>(context,
                                                                 typeConverter);
}
} // namespace mlir
