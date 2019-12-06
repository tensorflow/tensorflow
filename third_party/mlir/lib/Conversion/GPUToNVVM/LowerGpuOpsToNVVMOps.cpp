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
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/FormatVariadic.h"

#include "../GPUCommon/IndexIntrinsicsOpLowering.h"
#include "../GPUCommon/OpToFuncCallLowering.h"

using namespace mlir;

namespace {

/// Converts all_reduce op to LLVM/NVVM ops.
struct GPUAllReduceOpLowering : public LLVMOpLowering {
  using AccumulatorFactory = std::function<Value *(
      Location, Value *, Value *, ConversionPatternRewriter &)>;

  explicit GPUAllReduceOpLowering(LLVMTypeConverter &lowering_)
      : LLVMOpLowering(gpu::AllReduceOp::getOperationName(),
                       lowering_.getDialect()->getContext(), lowering_),
        int32Type(LLVM::LLVMType::getInt32Ty(lowering_.getDialect())) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value *operand = operands.front();

    // TODO(csigg): Generalize to other types of accumulation.
    assert(op->getOperand(0)->getType().isIntOrFloat());

    // Create the reduction using an accumulator factory.
    AccumulatorFactory factory =
        getFactory(cast<gpu::AllReduceOp>(op), operand);
    assert(factory && "failed to create accumulator factory");
    Value *result = createBlockReduce(loc, operand, factory, rewriter);

    rewriter.replaceOp(op, {result});
    return matchSuccess();
  }

private:
  /// Returns an accumulator factory using either the op attribute or the body
  /// region.
  AccumulatorFactory getFactory(gpu::AllReduceOp allReduce,
                                Value *operand) const {
    if (!allReduce.body().empty()) {
      return getFactory(allReduce.body());
    }
    if (allReduce.op()) {
      auto type = operand->getType().cast<LLVM::LLVMType>();
      return getFactory(*allReduce.op(), type.getUnderlyingType());
    }
    return AccumulatorFactory();
  }

  /// Returns an accumulator factory that clones the body. The body's entry
  /// block is expected to have 2 arguments. The gpu.yield return the
  /// accumulated value of the same type.
  AccumulatorFactory getFactory(Region &body) const {
    return AccumulatorFactory([&](Location loc, Value *lhs, Value *rhs,
                                  ConversionPatternRewriter &rewriter) {
      Block *block = rewriter.getInsertionBlock();
      Block *split = rewriter.splitBlock(block, rewriter.getInsertionPoint());

      // Insert accumulator body between split block.
      BlockAndValueMapping mapping;
      mapping.map(body.front().getArgument(0), lhs);
      mapping.map(body.front().getArgument(1), rhs);
      rewriter.cloneRegionBefore(body, *split->getParent(),
                                 split->getIterator(), mapping);

      // Add branch before inserted body, into body.
      block = block->getNextNode();
      rewriter.create<LLVM::BrOp>(loc, ArrayRef<Value *>{},
                                  llvm::makeArrayRef(block),
                                  llvm::ArrayRef<Value *>());

      // Replace all gpu.yield ops with branch out of body.
      for (; block != split; block = block->getNextNode()) {
        Operation *terminator = block->getTerminator();
        if (!llvm::isa<gpu::YieldOp>(terminator))
          continue;
        rewriter.setInsertionPointToEnd(block);
        rewriter.replaceOpWithNewOp<LLVM::BrOp>(
            terminator, ArrayRef<Value *>{}, llvm::makeArrayRef(split),
            llvm::makeArrayRef(terminator->getOperand(0)));
      }

      // Return accumulator result.
      rewriter.setInsertionPointToStart(split);
      return split->addArgument(lhs->getType());
    });
  }

  /// Returns an accumulator factory that creates an op specified by opName.
  AccumulatorFactory getFactory(StringRef opName, llvm::Type *type) const {
    if (type->isVectorTy() || type->isArrayTy())
      return getFactory(opName, type->getSequentialElementType());

    bool isFloatingPoint = type->isFloatingPointTy();

    if (opName == "add") {
      return isFloatingPoint ? getFactory<LLVM::FAddOp>()
                             : getFactory<LLVM::AddOp>();
    }
    if (opName == "mul") {
      return isFloatingPoint ? getFactory<LLVM::FMulOp>()
                             : getFactory<LLVM::MulOp>();
    }

    return AccumulatorFactory();
  }

  /// Returns an accumulator factory that creates an op of type T.
  template <typename T> AccumulatorFactory getFactory() const {
    return [](Location loc, Value *lhs, Value *rhs,
              ConversionPatternRewriter &rewriter) {
      return rewriter.create<T>(loc, lhs->getType(), lhs, rhs);
    };
  }

  /// Creates an all_reduce across the block.
  ///
  /// First reduce the elements within a warp. The first thread of each warp
  /// writes the intermediate result to shared memory. After synchronizing the
  /// block, the first warp reduces the values from shared memory. The result
  /// is broadcasted to all threads through shared memory.
  ///
  ///     %warp_reduce = `createWarpReduce(%operand)`
  ///     %shared_mem_ptr = llvm.mlir.addressof @reduce_buffer
  ///     %zero = llvm.mlir.constant(0 : i32) : !llvm.i32
  ///     %lane_id = nvvm.read.ptx.sreg.laneid  : !llvm.i32
  ///     %is_first_lane = llvm.icmp "eq" %lane_id, %zero : !llvm.i1
  ///     %thread_idx = `getLinearThreadIndex()` : !llvm.i32
  ///     llvm.cond_br %is_first_lane, ^then1, ^continue1
  ///   ^then1:
  ///     %warp_id = `getWarpId()`
  ///     %store_dst = llvm.getelementptr %shared_mem_ptr[%zero, %warp_id]
  ///     llvm.store %store_dst, %warp_reduce
  ///     llvm.br ^continue1
  ///   ^continue1:
  ///     nvvm.barrier0
  ///     %num_warps = `getNumWarps()` : !llvm.i32
  ///     %is_valid_warp = llvm.icmp "slt" %thread_idx, %num_warps
  ///     %result_ptr = llvm.getelementptr %shared_mem_ptr[%zero, %zero]
  ///     llvm.cond_br %is_first_lane, ^then2, ^continue2
  ///   ^then2:
  ///     %load_src = llvm.getelementptr %shared_mem_ptr[%zero, %thread_idx]
  ///     %value = llvm.load %load_src
  ///     %result = `createWarpReduce(%value)`
  ///     llvm.store %result_ptr, %result
  ///     llvm.br ^continue2
  ///   ^continue2:
  ///     nvvm.barrier0
  ///     %result = llvm.load %result_ptr
  ///     return %result
  ///
  Value *createBlockReduce(Location loc, Value *operand,
                           AccumulatorFactory &accumFactory,
                           ConversionPatternRewriter &rewriter) const {
    auto type = operand->getType().cast<LLVM::LLVMType>();

    // Create shared memory array to store the warp reduction.
    auto module = operand->getDefiningOp()->getParentOfType<ModuleOp>();
    assert(module && "op must belong to a module");
    Value *sharedMemPtr =
        createSharedMemoryArray(loc, module, type, kWarpSize, rewriter);

    Value *zero = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(0u));
    Value *laneId = rewriter.create<NVVM::LaneIdOp>(loc, int32Type);
    Value *isFirstLane = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, laneId, zero);
    Value *threadIdx = getLinearThreadIndex(loc, rewriter);
    Value *blockSize = getBlockSize(loc, rewriter);
    Value *activeWidth = getActiveWidth(loc, threadIdx, blockSize, rewriter);

    // Reduce elements within each warp to produce the intermediate results.
    Value *warpReduce = createWarpReduce(loc, activeWidth, laneId, operand,
                                         accumFactory, rewriter);

    // Write the intermediate results to shared memory, using the first lane of
    // each warp.
    createPredicatedBlock(loc, rewriter, isFirstLane, [&] {
      Value *warpId = getDivideByWarpSize(threadIdx, rewriter);
      Value *storeDst = rewriter.create<LLVM::GEPOp>(
          loc, type, sharedMemPtr, ArrayRef<Value *>({zero, warpId}));
      rewriter.create<LLVM::StoreOp>(loc, warpReduce, storeDst);
    });
    rewriter.create<NVVM::Barrier0Op>(loc);

    Value *numWarps = getNumWarps(loc, blockSize, rewriter);
    Value *isValidWarp = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::slt, threadIdx, numWarps);
    Value *resultPtr = rewriter.create<LLVM::GEPOp>(
        loc, type, sharedMemPtr, ArrayRef<Value *>({zero, zero}));

    // Use the first numWarps threads to reduce the intermediate results from
    // shared memory. The final result is written to shared memory again.
    createPredicatedBlock(loc, rewriter, isValidWarp, [&] {
      Value *loadSrc = rewriter.create<LLVM::GEPOp>(
          loc, type, sharedMemPtr, ArrayRef<Value *>({zero, threadIdx}));
      Value *value = rewriter.create<LLVM::LoadOp>(loc, type, loadSrc);
      Value *result = createWarpReduce(loc, numWarps, laneId, value,
                                       accumFactory, rewriter);
      rewriter.create<LLVM::StoreOp>(loc, result, resultPtr);
    });
    rewriter.create<NVVM::Barrier0Op>(loc);

    // Load and return result from shared memory.
    Value *result = rewriter.create<LLVM::LoadOp>(loc, type, resultPtr);
    return result;
  }

  /// Creates an if-block skeleton and calls the two factories to generate the
  /// ops in the `then` and `else` block..
  ///
  ///     llvm.cond_br %condition, ^then, ^continue
  ///   ^then:
  ///     %then_operands = `thenOpsFactory()`
  ///     llvm.br ^continue(%then_operands)
  ///   ^else:
  ///     %else_operands = `elseOpsFactory()`
  ///     llvm.br ^continue(%else_operands)
  ///   ^continue(%block_operands):
  ///
  template <typename ThenOpsFactory, typename ElseOpsFactory>
  void createIf(Location loc, ConversionPatternRewriter &rewriter,
                Value *condition, ThenOpsFactory &&thenOpsFactory,
                ElseOpsFactory &&elseOpsFactory) const {
    Block *currentBlock = rewriter.getInsertionBlock();
    auto currentPoint = rewriter.getInsertionPoint();

    Block *thenBlock = rewriter.splitBlock(currentBlock, currentPoint);
    Block *elseBlock = rewriter.splitBlock(thenBlock, thenBlock->begin());
    Block *continueBlock = rewriter.splitBlock(elseBlock, elseBlock->begin());

    rewriter.setInsertionPointToEnd(currentBlock);
    rewriter.create<LLVM::CondBrOp>(loc, llvm::makeArrayRef(condition),
                                    ArrayRef<Block *>{thenBlock, elseBlock});

    auto addBranch = [&](ArrayRef<Value *> operands) {
      rewriter.create<LLVM::BrOp>(loc, ArrayRef<Value *>{},
                                  llvm::makeArrayRef(continueBlock),
                                  llvm::makeArrayRef(operands));
    };

    rewriter.setInsertionPointToStart(thenBlock);
    auto thenOperands = thenOpsFactory();
    addBranch(thenOperands);

    rewriter.setInsertionPointToStart(elseBlock);
    auto elseOperands = elseOpsFactory();
    addBranch(elseOperands);

    assert(thenOperands.size() == elseOperands.size());
    rewriter.setInsertionPointToStart(continueBlock);
    for (auto *operand : thenOperands)
      continueBlock->addArgument(operand->getType());
  }

  /// Shortcut for createIf with empty else block and no block operands.
  template <typename Factory>
  void createPredicatedBlock(Location loc, ConversionPatternRewriter &rewriter,
                             Value *condition,
                             Factory &&predicatedOpsFactory) const {
    createIf(
        loc, rewriter, condition,
        [&] {
          predicatedOpsFactory();
          return ArrayRef<Value *>();
        },
        [&] { return ArrayRef<Value *>(); });
  }

  /// Creates a reduction across the first activeWidth lanes of a warp.
  /// The first lane returns the result, all others return values are undefined.
  Value *createWarpReduce(Location loc, Value *activeWidth, Value *laneId,
                          Value *operand, AccumulatorFactory accumFactory,
                          ConversionPatternRewriter &rewriter) const {
    Value *warpSize = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(kWarpSize));
    Value *maskAndClamp = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(kWarpSize - 1));
    Value *isPartialWarp = rewriter.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::slt, activeWidth, warpSize);
    auto type = operand->getType().cast<LLVM::LLVMType>();

    createIf(
        loc, rewriter, isPartialWarp,
        // Generate reduction over a (potentially) partial warp.
        [&] {
          Value *value = operand;
          Value *one = rewriter.create<LLVM::ConstantOp>(
              loc, int32Type, rewriter.getI32IntegerAttr(1));
          // Bit mask of active lanes: `(1 << activeWidth) - 1`.
          Value *activeMask = rewriter.create<LLVM::SubOp>(
              loc, int32Type,
              rewriter.create<LLVM::ShlOp>(loc, int32Type, one, activeWidth),
              one);
          auto dialect = lowering.getDialect();
          auto predTy = LLVM::LLVMType::getInt1Ty(dialect);
          auto shflTy = LLVM::LLVMType::getStructTy(dialect, {type, predTy});
          auto returnValueAndIsValidAttr = rewriter.getUnitAttr();

          // Repeatedly shuffle value from 'laneId ^ i' and accumulate if source
          // lane is within the active range. All lanes contain the final
          // result, but only the first lane's result is used.
          for (int i = 1; i < kWarpSize; i <<= 1) {
            Value *offset = rewriter.create<LLVM::ConstantOp>(
                loc, int32Type, rewriter.getI32IntegerAttr(i));
            Value *shfl = rewriter.create<NVVM::ShflBflyOp>(
                loc, shflTy, activeMask, value, offset, maskAndClamp,
                returnValueAndIsValidAttr);
            Value *isActiveSrcLane = rewriter.create<LLVM::ExtractValueOp>(
                loc, predTy, shfl, rewriter.getIndexArrayAttr(1));
            // Skip the accumulation if the shuffle op read from a lane outside
            // of the active range.
            createIf(
                loc, rewriter, isActiveSrcLane,
                [&] {
                  Value *shflValue = rewriter.create<LLVM::ExtractValueOp>(
                      loc, type, shfl, rewriter.getIndexArrayAttr(0));
                  return llvm::SmallVector<Value *, 1>{
                      accumFactory(loc, value, shflValue, rewriter)};
                },
                [&] { return llvm::makeArrayRef(value); });
            value = rewriter.getInsertionBlock()->getArgument(0);
          }
          return llvm::SmallVector<Value *, 1>{value};
        },
        // Generate a reduction over the entire warp. This is a specialization
        // of the above reduction with unconditional accumulation.
        [&] {
          Value *value = operand;
          Value *activeMask = rewriter.create<LLVM::ConstantOp>(
              loc, int32Type, rewriter.getI32IntegerAttr(~0u));
          for (int i = 1; i < kWarpSize; i <<= 1) {
            Value *offset = rewriter.create<LLVM::ConstantOp>(
                loc, int32Type, rewriter.getI32IntegerAttr(i));
            Value *shflValue = rewriter.create<NVVM::ShflBflyOp>(
                loc, type, activeMask, value, offset, maskAndClamp,
                /*return_value_and_is_valid=*/UnitAttr());
            value = accumFactory(loc, value, shflValue, rewriter);
          }
          return llvm::SmallVector<Value *, 1>{value};
        });
    return rewriter.getInsertionBlock()->getArgument(0);
  }

  /// Creates a global array stored in shared memory.
  Value *createSharedMemoryArray(Location loc, ModuleOp module,
                                 LLVM::LLVMType elementType, int numElements,
                                 ConversionPatternRewriter &rewriter) const {
    OpBuilder builder(module.getBodyRegion());

    auto arrayType = LLVM::LLVMType::getArrayTy(elementType, numElements);
    StringRef name = "reduce_buffer";
    auto globalOp = builder.create<LLVM::GlobalOp>(
        loc, arrayType.cast<LLVM::LLVMType>(),
        /*isConstant=*/false, LLVM::Linkage::Internal, name,
        /*value=*/Attribute(), gpu::GPUDialect::getWorkgroupAddressSpace());

    return rewriter.create<LLVM::AddressOfOp>(loc, globalOp);
  }

  /// Returns the index of the thread within the block.
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

  /// Returns the number of threads in the block.
  Value *getBlockSize(Location loc, ConversionPatternRewriter &rewriter) const {
    Value *dimX = rewriter.create<NVVM::BlockDimXOp>(loc, int32Type);
    Value *dimY = rewriter.create<NVVM::BlockDimYOp>(loc, int32Type);
    Value *dimZ = rewriter.create<NVVM::BlockDimZOp>(loc, int32Type);
    Value *dimXY = rewriter.create<LLVM::MulOp>(loc, int32Type, dimX, dimY);
    return rewriter.create<LLVM::MulOp>(loc, int32Type, dimXY, dimZ);
  }

  /// Returns the number of warps in the block.
  Value *getNumWarps(Location loc, Value *blockSize,
                     ConversionPatternRewriter &rewriter) const {
    auto warpSizeMinusOne = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(kWarpSize - 1));
    auto biasedBlockSize = rewriter.create<LLVM::AddOp>(
        loc, int32Type, blockSize, warpSizeMinusOne);
    return getDivideByWarpSize(biasedBlockSize, rewriter);
  }

  /// Returns the number of active threads in the warp, not clamped to 32.
  Value *getActiveWidth(Location loc, Value *threadIdx, Value *blockSize,
                        ConversionPatternRewriter &rewriter) const {
    Value *threadIdxMask = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(~(kWarpSize - 1)));
    Value *numThreadsWithSmallerWarpId =
        rewriter.create<LLVM::AndOp>(loc, threadIdx, threadIdxMask);
    return rewriter.create<LLVM::SubOp>(loc, blockSize,
                                        numThreadsWithSmallerWarpId);
  }

  /// Returns value divided by the warp size (i.e. 32).
  Value *getDivideByWarpSize(Value *value,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = value->getLoc();
    auto warpSize = rewriter.create<LLVM::ConstantOp>(
        loc, int32Type, rewriter.getI32IntegerAttr(kWarpSize));
    return rewriter.create<LLVM::SDivOp>(loc, int32Type, value, warpSize);
  }

  LLVM::LLVMType int32Type;

  static constexpr int kWarpSize = 32;
};

namespace {

struct FuncOpLowering : LLVMOpLowering {
  explicit FuncOpLowering(LLVMTypeConverter &typeConverter)
      : LLVMOpLowering(gpu::GPUFuncOp::getOperationName(),
                       typeConverter.getDialect()->getContext(),
                       typeConverter) {}

  PatternMatchResult
  matchAndRewrite(Operation *op, ArrayRef<Value *> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.empty() && "func op is not expected to have operands");
    auto gpuFuncOp = cast<gpu::GPUFuncOp>(op);
    Location loc = gpuFuncOp.getLoc();

    SmallVector<LLVM::GlobalOp, 3> workgroupBuffers;
    workgroupBuffers.reserve(gpuFuncOp.getNumWorkgroupAttributions());
    for (auto en : llvm::enumerate(gpuFuncOp.getWorkgroupAttributions())) {
      Value *attribution = en.value();

      auto type = attribution->getType().dyn_cast<MemRefType>();
      assert(type && type.hasStaticShape() && "unexpected type in attribution");

      uint64_t numElements = type.getNumElements();

      auto elementType =
          lowering.convertType(type.getElementType()).cast<LLVM::LLVMType>();
      auto arrayType = LLVM::LLVMType::getArrayTy(elementType, numElements);
      std::string name =
          llvm::formatv("__wg_{0}_{1}", gpuFuncOp.getName(), en.index());
      auto globalOp = rewriter.create<LLVM::GlobalOp>(
          gpuFuncOp.getLoc(), arrayType, /*isConstant=*/false,
          LLVM::Linkage::Internal, name, /*value=*/Attribute(),
          gpu::GPUDialect::getWorkgroupAddressSpace());
      workgroupBuffers.push_back(globalOp);
    }

    // Rewrite the original GPU function to an LLVM function.
    // TODO(zinenko): there is a hack in the std->llvm lowering that promotes
    // structs to pointers that probably needs to be replicated here.
    auto funcType = lowering.convertType(gpuFuncOp.getType())
                        .cast<LLVM::LLVMType>()
                        .getPointerElementTy();

    // Remap proper input types.
    TypeConverter::SignatureConversion signatureConversion(
        gpuFuncOp.front().getNumArguments());
    for (unsigned i = 0, e = funcType.getFunctionNumParams(); i < e; ++i)
      signatureConversion.addInputs(i, funcType.getFunctionParamType(i));

    // Create the new function operation. Only copy those attributes that are
    // not specific to function modeling.
    SmallVector<NamedAttribute, 4> attributes;
    for (const auto &attr : gpuFuncOp.getAttrs()) {
      if (attr.first.is(SymbolTable::getSymbolAttrName()) ||
          attr.first.is(impl::getTypeAttrName()) ||
          attr.first.is(gpu::GPUFuncOp::getNumWorkgroupAttributionsAttrName()))
        continue;
      attributes.push_back(attr);
    }
    auto llvmFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        gpuFuncOp.getLoc(), gpuFuncOp.getName(), funcType,
        LLVM::Linkage::External, attributes);

    {
      // Insert operations that correspond to converted workgroup and private
      // memory attributions to the body of the function. This must operate on
      // the original function, before the body region is inlined in the new
      // function to maintain the relation between block arguments and the
      // parent operation that assigns their semantics.
      OpBuilder::InsertionGuard guard(rewriter);

      // Rewrite workgroup memory attributions to addresses of global buffers.
      rewriter.setInsertionPointToStart(&gpuFuncOp.front());
      unsigned numProperArguments = gpuFuncOp.getNumArguments();
      auto i32Type = LLVM::LLVMType::getInt32Ty(lowering.getDialect());

      Value *zero = nullptr;
      if (!workgroupBuffers.empty())
        zero = rewriter.create<LLVM::ConstantOp>(loc, i32Type,
                                                 rewriter.getI32IntegerAttr(0));
      for (auto en : llvm::enumerate(workgroupBuffers)) {
        LLVM::GlobalOp global = en.value();
        Value *address = rewriter.create<LLVM::AddressOfOp>(loc, global);
        auto elementType = global.getType().getArrayElementType();
        Value *memory = rewriter.create<LLVM::GEPOp>(
            loc, elementType.getPointerTo(global.addr_space().getZExtValue()),
            address, ArrayRef<Value *>{zero, zero});

        // Build a memref descriptor pointing to the buffer to plug with the
        // existing memref infrastructure. This may use more registers than
        // otherwise necessary given that memref sizes are fixed, but we can try
        // and canonicalize that away later.
        Value *attribution = gpuFuncOp.getWorkgroupAttributions()[en.index()];
        auto type = attribution->getType().cast<MemRefType>();
        auto descr = MemRefDescriptor::fromStaticShape(rewriter, loc, lowering,
                                                       type, memory);
        signatureConversion.remapInput(numProperArguments + en.index(), descr);
      }

      // Rewrite private memory attributions to alloca'ed buffers.
      unsigned numWorkgroupAttributions =
          gpuFuncOp.getNumWorkgroupAttributions();
      auto int64Ty = LLVM::LLVMType::getInt64Ty(lowering.getDialect());
      for (auto en : llvm::enumerate(gpuFuncOp.getPrivateAttributions())) {
        Value *attribution = en.value();
        auto type = attribution->getType().cast<MemRefType>();
        assert(type && type.hasStaticShape() &&
               "unexpected type in attribution");

        auto ptrType = lowering.convertType(type.getElementType())
                           .cast<LLVM::LLVMType>()
                           .getPointerTo(type.getMemorySpace());
        Value *numElements = rewriter.create<LLVM::ConstantOp>(
            gpuFuncOp.getLoc(), int64Ty,
            rewriter.getI64IntegerAttr(type.getNumElements()));
        Value *allocated = rewriter.create<LLVM::AllocaOp>(
            gpuFuncOp.getLoc(), ptrType, numElements, /*alignment=*/0);
        auto descr = MemRefDescriptor::fromStaticShape(rewriter, loc, lowering,
                                                       type, allocated);
        signatureConversion.remapInput(
            numProperArguments + numWorkgroupAttributions + en.index(), descr);
      }
    }

    rewriter.inlineRegionBefore(gpuFuncOp.getBody(), llvmFuncOp.getBody(),
                                llvmFuncOp.end());
    rewriter.applySignatureConversion(&llvmFuncOp.getBody(),
                                      signatureConversion);

    rewriter.eraseOp(gpuFuncOp);
    return matchSuccess();
  }
};

} // end namespace

/// Import the GPU Ops to NVVM Patterns.
#include "GPUToNVVM.cpp.inc"

/// A pass that replaces all occurrences of GPU device operations with their
/// corresponding NVVM equivalent.
///
/// This pass only handles device code and is not meant to be run on GPU host
/// code.
class LowerGpuOpsToNVVMOpsPass : public ModulePass<LowerGpuOpsToNVVMOpsPass> {
public:
  void runOnModule() override {
    ModuleOp m = getModule();
    if (!m.getAttrOfType<UnitAttr>(gpu::GPUDialect::getKernelModuleAttrName()))
      return;

    OwningRewritePatternList patterns;
    LLVMTypeConverter converter(m.getContext());
    populateStdToLLVMConversionPatterns(converter, patterns);
    populateWithGenerated(&getContext(), &patterns);
    patterns.insert<
        GPUIndexIntrinsicOpLowering<gpu::ThreadIdOp, NVVM::ThreadIdXOp,
                                    NVVM::ThreadIdYOp, NVVM::ThreadIdZOp>,
        GPUIndexIntrinsicOpLowering<gpu::BlockDimOp, NVVM::BlockDimXOp,
                                    NVVM::BlockDimYOp, NVVM::BlockDimZOp>,
        GPUIndexIntrinsicOpLowering<gpu::BlockIdOp, NVVM::BlockIdXOp,
                                    NVVM::BlockIdYOp, NVVM::BlockIdZOp>,
        GPUIndexIntrinsicOpLowering<gpu::GridDimOp, NVVM::GridDimXOp,
                                    NVVM::GridDimYOp, NVVM::GridDimZOp>,
        GPUAllReduceOpLowering, FuncOpLowering>(converter);
    patterns.insert<OpToFuncCallLowering<ExpOp>>(converter, "__nv_expf",
                                                 "__nv_exp");
    ConversionTarget target(getContext());
    target.addIllegalDialect<gpu::GPUDialect>();
    target.addIllegalOp<LLVM::ExpOp>();
    target.addIllegalOp<FuncOp>();
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<NVVM::NVVMDialect>();
    // TODO(csigg): Remove once we support replacing non-root ops.
    target.addLegalOp<gpu::YieldOp>();
    if (failed(applyPartialConversion(m, target, patterns, &converter)))
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<OpPassBase<ModuleOp>> mlir::createLowerGpuOpsToNVVMOpsPass() {
  return std::make_unique<LowerGpuOpsToNVVMOpsPass>();
}

static PassRegistration<LowerGpuOpsToNVVMOpsPass>
    pass("convert-gpu-to-nvvm", "Generate NVVM operations for gpu operations");
