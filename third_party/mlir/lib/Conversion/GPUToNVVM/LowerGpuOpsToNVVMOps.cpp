//===- LowerGpuOpsToNVVMOps.cpp - MLIR GPU to NVVM lowering passes --------===//
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
// This file implements a pass to generate NVVMIR operations for higher-level
// GPU operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "../GPUCommon/IndexIntrinsicsOpLowering.h"

using namespace mlir;

namespace {

// Converts all_reduce op to LLVM/NVVM ops.
struct GPUAllReduceOpLowering : public LLVMOpLowering {
  explicit GPUAllReduceOpLowering(LLVMTypeConverter &lowering_)
      : LLVMOpLowering(gpu::AllReduce::getOperationName(),
                       lowering_.getDialect()->getContext(), lowering_),
        int32Type(LLVM::LLVMType::getInt32Ty(lowering_.getDialect())) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Value *result = createBlockReduce(op->getLoc(), operands.front(), rewriter);
    rewriter.replaceOp(op, {result});
    return matchSuccess();
  }

private:
  // Creates an all_reduce across the block.
  //
  // First reduce the elements within a warp. The first thread of each warp
  // writes the intermediate result to shared memory. After synchronizing the
  // block, the first warp reduces the values from shared memory. The result
  // is broadcasted to all threads through shared memory.
  //
  //     %warp_reduce = `createWarpReduce(%operand)`
  //     %shared_mem_ptr = llvm.mlir.addressof @reduce_buffer
  //     %zero = llvm.mlir.constant(0 : i32) : !llvm.i32
  //     %lane_id = nvvm.read.ptx.sreg.laneid  : !llvm.i32
  //     %is_first_lane = llvm.icmp "eq" %lane_id, %zero : !llvm.i1
  //     %thread_idx = `getLinearThreadIndex()` : !llvm.i32
  //     llvm.cond_br %is_first_lane, ^then1, ^continue1
  //   ^then1:
  //     %warp_id = `getWarpId()`
  //     %store_dst = llvm.getelementptr %shared_mem_ptr[%zero, %warp_id]
  //     llvm.store %store_dst, %warp_reduce
  //     llvm.br ^continue1
  //   ^continue1:
  //     nvvm.barrier0
  //     %num_warps = `getNumWarps()` : !llvm.i32
  //     %is_valid_warp = llvm.icmp "slt" %thread_idx, %num_warps
  //     %result_ptr = llvm.getelementptr %shared_mem_ptr[%zero, %zero]
  //     llvm.cond_br %is_first_lane, ^then2, ^continue2
  //   ^then2:
  //     %load_src = llvm.getelementptr %shared_mem_ptr[%zero, %thread_idx]
  //     %value = llvm.load %load_src
  //     %result = `createWarpReduce(%value)`
  //     llvm.store %result_ptr, %result
  //     llvm.br ^continue2
  //   ^continue2:
  //     nvvm.barrier0
  //     %result = llvm.load %result_ptr
  //     return %result
  //
  Value *createBlockReduce(Location loc, Value *operand,
                           ConversionPatternRewriter &rewriter) const {
    auto type = operand->getType().cast<LLVM::LLVMType>();

    // Reduce elements within each warp to produce the intermediate results.
    Value *warpReduce = createWarpReduce(loc, operand, rewriter);

    // Create shared memory array to store the warp reduction.
    auto module = warpReduce->getDefiningOp()->getParentOfType<ModuleOp>();
    assert(module && "op must belong to a module");
    Value *sharedMemPtr =
        createSharedMemoryArray(loc, module, type, kWarpSize, rewriter);

    Value *zero = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(0u));
    Value *laneId = rewriter.create<NVVM::LaneIdOp>(loc, int32Type);
    Value *isFirstLane = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, laneId, zero);
    Value *threadIdx = getLinearThreadIndex(loc, rewriter);

    // Write the intermediate results to shared memory, using the first lane of
    // each warp.
    createPredicatedBlock(
        loc, isFirstLane,
        [&] {
          Value *warpId = getDivideByWarpSize(threadIdx, rewriter);
          Value *storeDst = rewriter.create<LLVM::GEPOp>(
              loc, type, sharedMemPtr, ArrayRef<Value *>({zero, warpId}));
          rewriter.create<LLVM::StoreOp>(loc, warpReduce, storeDst);
        },
        rewriter);

    rewriter.create<NVVM::Barrier0Op>(loc);
    Value *numWarps = getNumWarps(loc, rewriter);
    Value *isValidWarp = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::slt, threadIdx, numWarps);
    Value *resultPtr = rewriter.create<LLVM::GEPOp>(
        loc, type, sharedMemPtr, ArrayRef<Value *>({zero, zero}));

    // Use the first numWarps threads to reduce the intermediate results from
    // shared memory. The final result is written to shared memory again.
    createPredicatedBlock(
        loc, isValidWarp,
        [&] {
          Value *loadSrc = rewriter.create<LLVM::GEPOp>(
              loc, type, sharedMemPtr, ArrayRef<Value *>({zero, threadIdx}));
          Value *value = rewriter.create<LLVM::LoadOp>(loc, type, loadSrc);
          Value *result = createWarpReduce(loc, value, rewriter);
          rewriter.create<LLVM::StoreOp>(loc, result, resultPtr);
        },
        rewriter);

    rewriter.create<NVVM::Barrier0Op>(loc);
    Value *result = rewriter.create<LLVM::LoadOp>(loc, type, resultPtr);

    return result;
  }

  // Creates an if-block skeleton to perform conditional execution of the
  // instructions generated by predicatedOpsFactory.
  //
  //     llvm.cond_br %condition, ^then, ^continue
  //   ^then:
  //     ... code created in `predicatedOpsFactory()`
  //     llvm.br ^continue
  //   ^continue:
  //
  template <typename Func>
  void createPredicatedBlock(Location loc, Value *condition,
                             Func &&predicatedOpsFactory,
                             ConversionPatternRewriter &rewriter) const {
    Block *currentBlock = rewriter.getInsertionBlock();
    auto currentPoint = rewriter.getInsertionPoint();

    Block *thenBlock = rewriter.splitBlock(currentBlock, currentPoint);
    Block *continueBlock = rewriter.splitBlock(thenBlock, currentPoint);

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(
        loc, llvm::makeArrayRef(condition),
        ArrayRef<Block *>{thenBlock, continueBlock});

    rewriter.setInsertionPointToEnd(thenBlock);
    predicatedOpsFactory();
    rewriter.create<LLVM::BrOp>(loc, ArrayRef<Value *>(),
                                llvm::makeArrayRef(continueBlock));

    rewriter.setInsertionPointToStart(continueBlock);
  }

  // Creates an all_reduce across the warp. Creates a preamble
  //
  //     %active_mask = llvm.mlir.constant(-1 : i32) : !llvm.i32
  //     %mask_and_clamp = llvm.mlir.constant(31 : i32) : !llvm.i32
  //
  // plus the accumulation for i = 1, 2, 4, 8, 16:
  //
  //     %offset = llvm.mlir.constant(i : i32) : !llvm.i32
  //     %value = nvvm.shfl.sync.bfly
  //        %active_mask, %operand, %offset, %mask_and_clamp
  //     %operand = llvm.fadd %operand, %value
  //
  // Each thread returns the same result.
  //
  // Note: this currently only supports reducing exactly 32 values.
  Value *createWarpReduce(Location loc, Value *operand,
                          ConversionPatternRewriter &rewriter) const {
    // TODO(csigg): Generalize to partial warps and other types of accumulation.
    static_assert(kWarpSize == 32, "Only warp size of 32 is supported.");
    auto activeMask = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(~0u));
    auto maskAndClamp = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(kWarpSize - 1));

    auto resultType = operand->getType();
    for (int i = 1; i < kWarpSize; i <<= 1) {
      auto offset = rewriter.create<LLVM::ConstantOp>(
          loc, int32Type, rewriter.getI32IntegerAttr(i));
      auto value = rewriter.create<NVVM::ShflBflyOp>(
          loc, resultType, activeMask, operand, offset, maskAndClamp);
      operand = rewriter.create<LLVM::FAddOp>(loc, resultType, operand, value);
    }
    return operand;
  }

  // Creates a global array stored in shared memory.
  //
  //     llvm.mlir.global @reduce_buffer()
  //         {addr_space = 3 : i32} : !llvm<"[32 x float]">
  //
  Value *createSharedMemoryArray(Location loc, ModuleOp module,
                                 LLVM::LLVMType elementType, int numElements,
                                 ConversionPatternRewriter &rewriter) const {
    OpBuilder builder(module.getBodyRegion());

    auto arrayType = LLVM::LLVMType::getArrayTy(elementType, numElements);
    StringRef name = "reduce_buffer";
    auto addrSpace =
        builder.getNamedAttr("addr_space", builder.getI32IntegerAttr(3));
    auto globalOp = builder.create<LLVM::GlobalOp>(
        loc, arrayType.cast<LLVM::LLVMType>(),
        /*isConstant=*/false, name, /*value=*/Attribute(),
        llvm::makeArrayRef(addrSpace));

    return rewriter.create<LLVM::AddressOfOp>(loc, globalOp);
  }

  // Returns the index of the thread within the block.
  Value *getLinearThreadIndex(Location loc,
                              ConversionPatternRewriter &rewriter) const {
    Value *dimX = rewriter.create<NVVM::BlockDimXOp>(loc, int32Type);
    Value *dimY = rewriter.create<NVVM::BlockDimYOp>(loc, int32Type);
    Value *idX = rewriter.create<NVVM::ThreadIdXOp>(loc, int32Type);
    Value *idY = rewriter.create<NVVM::ThreadIdYOp>(loc, int32Type);
    Value *idZ = rewriter.create<NVVM::ThreadIdZOp>(loc, int32Type);
    Value *tmp1 = rewriter.create<LLVM::MulOp>(loc, int32Type, idZ, dimY);
    Value *tmp2 = rewriter.create<LLVM::AddOp>(loc, int32Type, tmp1, idY);
    Value *tmp3 = rewriter.create<LLVM::MulOp>(loc, int32Type, tmp2, dimX);
    return rewriter.create<LLVM::AddOp>(loc, int32Type, tmp3, idX);
  }

  // Returns the number of warps in the block.
  Value *getNumWarps(Location loc, ConversionPatternRewriter &rewriter) const {
    auto blockSize = getBlockSize(loc, rewriter);
    auto warpSizeMinusOne = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(kWarpSize - 1));
    auto biasedBlockSize = rewriter.create<LLVM::AddOp>(
        loc, int32Type, blockSize, warpSizeMinusOne);
    return getDivideByWarpSize(biasedBlockSize, rewriter);
  }

  // Returns the number of threads in the block.
  Value *getBlockSize(Location loc, ConversionPatternRewriter &rewriter) const {
    Value *dimX = rewriter.create<NVVM::BlockDimXOp>(loc, int32Type);
    Value *dimY = rewriter.create<NVVM::BlockDimYOp>(loc, int32Type);
    Value *dimZ = rewriter.create<NVVM::BlockDimZOp>(loc, int32Type);
    Value *dimXY = rewriter.create<LLVM::MulOp>(loc, int32Type, dimX, dimY);
    return rewriter.create<LLVM::MulOp>(loc, int32Type, dimXY, dimZ);
  }

  // Returns value divided by the warp size (i.e. 32).
  Value *getDivideByWarpSize(Value *value,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = value->getLoc();
    auto warpSize = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(kWarpSize));
    return rewriter.create<LLVM::SDivOp>(loc, int32Type, value, warpSize);
  }

  LLVM::LLVMType int32Type;

  // TODO(csigg): Support other warp sizes.
  static constexpr int kWarpSize = 32;
};

// A pass that replaces all occurrences of GPU device operations with their
// corresponding NVVM equivalent.
//
// This pass only handles device code and is not meant to be run on GPU host
// code.
class LowerGpuOpsToNVVMOpsPass : public ModulePass<LowerGpuOpsToNVVMOpsPass> {
public:
  void runOnModule() override {
    ModuleOp m = getModule();
    if (!m.getAttrOfType<UnitAttr>(gpu::GPUDialect::getKernelModuleAttrName()))
      return;

    OwningRewritePatternList patterns;
    LLVMTypeConverter converter(m.getContext());
    populateStdToLLVMConversionPatterns(converter, patterns);
    patterns.insert<
        GPUIndexIntrinsicOpLowering<gpu::ThreadId, NVVM::ThreadIdXOp,
                                    NVVM::ThreadIdYOp, NVVM::ThreadIdZOp>,
        GPUIndexIntrinsicOpLowering<gpu::BlockDim, NVVM::BlockDimXOp,
                                    NVVM::BlockDimYOp, NVVM::BlockDimZOp>,
        GPUIndexIntrinsicOpLowering<gpu::BlockId, NVVM::BlockIdXOp,
                                    NVVM::BlockIdYOp, NVVM::BlockIdZOp>,
        GPUIndexIntrinsicOpLowering<gpu::GridDim, NVVM::GridDimXOp,
                                    NVVM::GridDimYOp, NVVM::GridDimZOp>,
        GPUAllReduceOpLowering>(converter);

    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<NVVM::NVVMDialect>();
    target.addDynamicallyLegalOp<FuncOp>(
        [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
    if (failed(applyPartialConversion(m, target, patterns, &converter)))
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createLowerGpuOpsToNVVMOpsPass() {
  return std::make_unique<LowerGpuOpsToNVVMOpsPass>();
}

static PassRegistration<LowerGpuOpsToNVVMOpsPass>
    pass("lower-gpu-ops-to-nvvm-ops",
         "Generate NVVM operations for gpu operations");
