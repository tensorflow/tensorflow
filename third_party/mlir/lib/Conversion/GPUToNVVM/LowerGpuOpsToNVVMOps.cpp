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

#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace {

// Rewriting that replaces Op with XOp, YOp, or ZOp depending on the dimension
// that Op operates on.  Op is assumed to return an `std.index` value and
// XOp, YOp and ZOp are assumed to return an `llvm.i32` value.  Depending on
// `indexBitwidth`, sign-extend or truncate the resulting value to match the
// bitwidth expected by the consumers of the value.
template <typename Op, typename XOp, typename YOp, typename ZOp>
struct GPUIndexIntrinsicOpLowering : public LLVMOpLowering {
private:
  enum dimension { X = 0, Y = 1, Z = 2, invalid };
  unsigned indexBitwidth;

  static dimension dimensionToIndex(Op op) {
    return llvm::StringSwitch<dimension>(op.dimension())
        .Case("x", X)
        .Case("y", Y)
        .Case("z", Z)
        .Default(invalid);
  }

  static unsigned getIndexBitWidth(LLVMTypeConverter &lowering) {
    auto dialect = lowering.getDialect();
    return dialect->getLLVMModule().getDataLayout().getPointerSizeInBits();
  }

public:
  explicit GPUIndexIntrinsicOpLowering(LLVMTypeConverter &lowering_)
      : LLVMOpLowering(Op::getOperationName(),
                       lowering_.getDialect()->getContext(), lowering_),
        indexBitwidth(getIndexBitWidth(lowering_)) {}

  // Convert the kernel arguments to an LLVM type, preserve the rest.
  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto dialect = lowering.getDialect();
    Value *newOp;
    switch (dimensionToIndex(cast<Op>(op))) {
    case X:
      newOp = rewriter.create<XOp>(loc, LLVM::LLVMType::getInt32Ty(dialect));
      break;
    case Y:
      newOp = rewriter.create<YOp>(loc, LLVM::LLVMType::getInt32Ty(dialect));
      break;
    case Z:
      newOp = rewriter.create<ZOp>(loc, LLVM::LLVMType::getInt32Ty(dialect));
      break;
    default:
      return matchFailure();
    }

    if (indexBitwidth > 32) {
      newOp = rewriter.create<LLVM::SExtOp>(
          loc, LLVM::LLVMType::getIntNTy(dialect, indexBitwidth), newOp);
    } else if (indexBitwidth < 32) {
      newOp = rewriter.create<LLVM::TruncOp>(
          loc, LLVM::LLVMType::getIntNTy(dialect, indexBitwidth), newOp);
    }

    rewriter.replaceOp(op, {newOp});
    return matchSuccess();
  }
};

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
  // block, each warp reduces all values from shared memory.
  //
  //     %warp_reduce = ... (see createWarpReduce)
  //     %buffer = llvm.mlir.addressof @reduce_buffer : !llvm<"[32 x float]*">
  //     %zero = llvm.mlir.constant(0 : i32) : !llvm.i32
  //     %lane_id = nvvm.read.ptx.sreg.laneid  : !llvm.i32
  //     %is_first_lane = llvm.icmp "eq" %lane_id, %zero : !llvm.i32
  //     llvm.cond_br %is_first_lane, ^then, ^continue
  //   ^then:
  //     %warp_id = ... (see getWarpId)
  //     %store_dst = llvm.getelementptr %buffer[%zero, %warp_id]
  //     llvm.store %store_dst, %warp_reduce : !llvm.float
  //     llvm.br ^continue
  //   ^continue:
  //     nvvm.barrier0
  //     %load_src = llvm.getelementptr %buffer[%zero, %lane_id]
  //     %value = llvm.load %load_src : !llvm.float
  //     %result = ... (see createWarpReduce)
  Value *createBlockReduce(Location loc, Value *operand,
                           ConversionPatternRewriter &rewriter) const {
    auto type = operand->getType().cast<LLVM::LLVMType>();

    Value *warpReduce = createWarpReduce(loc, operand, rewriter);

    auto module = warpReduce->getDefiningOp()->getParentOfType<ModuleOp>();
    assert(module && "op must belong to a module");
    Value *sharedMemPtr =
        createSharedMemoryArray(loc, module, type, kWarpSize, rewriter);

    Value *zero = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(0u));
    Value *laneId = rewriter.create<NVVM::LaneIdOp>(loc, int32Type);
    Value *isFirstLane = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, laneId, zero);

    Block *currentBlock = rewriter.getInsertionBlock();
    auto currentPoint = rewriter.getInsertionPoint();

    Block *thenBlock = rewriter.splitBlock(currentBlock, currentPoint);
    Block *continueBlock = rewriter.splitBlock(thenBlock, currentPoint);

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(
        loc, llvm::makeArrayRef(isFirstLane),
        ArrayRef<Block *>{thenBlock, continueBlock});

    rewriter.setInsertionPointToEnd(thenBlock);
    Value *warpId = getWarpId(loc, rewriter);
    Value *storeDst = rewriter.create<LLVM::GEPOp>(
        loc, type, sharedMemPtr, ArrayRef<Value *>({zero, warpId}));
    rewriter.create<LLVM::StoreOp>(loc, warpReduce, storeDst);
    rewriter.create<LLVM::BrOp>(loc, ArrayRef<Value *>(),
                                llvm::makeArrayRef(continueBlock));

    rewriter.setInsertionPointToStart(continueBlock);
    rewriter.create<NVVM::Barrier0Op>(loc);
    Value *loadSrc = rewriter.create<LLVM::GEPOp>(
        loc, type, sharedMemPtr, ArrayRef<Value *>({zero, laneId}));
    Value *value = rewriter.create<LLVM::LoadOp>(loc, type, loadSrc);
    Value *result = createWarpReduce(loc, value, rewriter);

    return result;
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
  //        %active_mask, %operand, %offset, %mask_and_clamp : !llvm.float
  //     %operand = llvm.fadd %operand, %value : !llvm.float
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

  // Returns the index of the warp within the block.
  //
  //     %warp_size = llvm.mlir.constant(32 : i32) : !llvm.i32
  //     %thread_idx = nvvm.read.ptx.sreg.tid.x  : !llvm.i32
  //     %warp_idx = llvm.sdiv %thread_idx, %warp_size : !llvm.i32
  //
  Value *getWarpId(Location loc, ConversionPatternRewriter &rewriter) const {
    auto warpSize = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(kWarpSize));
    auto threadIdx = getLinearThreadIndex(loc, rewriter);
    return rewriter.create<LLVM::SDivOp>(loc, int32Type, threadIdx, warpSize);
  }

  Value *getLinearThreadIndex(Location loc,
                              ConversionPatternRewriter &rewriter) const {
    // TODO(csigg): support 2- and 3-dimensional blocks.
    return rewriter.create<NVVM::ThreadIdXOp>(loc, int32Type);
  }

  LLVM::LLVMType int32Type;

  // TODO(csigg): Support other warp sizes.
  static constexpr int kWarpSize = 32;
};

// A pass that replaces all occurences of GPU device operations with their
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
