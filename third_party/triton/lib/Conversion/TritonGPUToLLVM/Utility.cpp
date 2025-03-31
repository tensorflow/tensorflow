#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"

#if defined(_MSC_VER) && !defined(__clang__)
// from https://gist.github.com/pps83/3210a2f980fd02bb2ba2e5a1fc4a2ef0
#include <intrin.h>

static int __builtin_clz(unsigned x) {
  unsigned long r;
  _BitScanReverse(&r, x);
  return static_cast<int>(r ^ 31);
}

static int __builtin_ctz(unsigned x) {
  unsigned long r;
  _BitScanForward(&r, x);
  return static_cast<int>(r);
}

#endif

// This reverts #5645, because it introduced increased register pressure in AMD
// backend.
// TODO: remove when new implementation performance reaches target level
namespace {

LinearLayout getRegToSharedLayout(MLIRContext *ctx, ArrayRef<int64_t> shape,
                                  LinearLayout regLayout,
                                  triton::gpu::SharedEncodingTrait dstEnc,
                                  int elemBitWidth) {
  StringAttr kBlock = StringAttr::get(ctx, ("block"));
  int rank = shape.size();

  LinearLayout sharedLayout = triton::gpu::toLinearLayout(shape, dstEnc);
  auto sharedOrder = triton::gpu::getOrder(dstEnc, shape);

  // sharedLayout's in-dims are currently (offset, block).  Reshape to
  // (offsetX1, offsetX2, ..., block) so that we can apply the N-dimensional
  // shmem strides.  (The offsetX's appear in minor-to-major order.)
  auto sharedLegacy = cast<triton::gpu::SwizzledSharedEncodingAttr>(dstEnc);
  SmallVector<std::pair<StringAttr, int32_t>> multiDimSharedSize;
  for (int i = 0; i < rank; i++) {
    int dim = sharedOrder[i];
    int64_t size = std::max(
        int64_t{1},
        shape[dim] / sharedLegacy.getCTALayout().getCTASplitNum()[dim]);
    multiDimSharedSize.push_back(
        {StringAttr::get(ctx, ("offset" + std::to_string(dim))), size});
  }
  multiDimSharedSize.push_back({kBlock, sharedLayout.getInDimSize(kBlock)});
  sharedLayout = sharedLayout.reshapeIns(multiDimSharedSize);

  // regToSharedLayout maps from (register, lane, warp, block) to (offsetX1,
  // ..., offsetXN, block), where the offsetX's are in minor-to-major order.
  return regLayout.invertAndCompose(sharedLayout);
}

} // namespace

namespace mlir {

namespace triton::gpu {
Type getFunctionType(Type resultType, ValueRange operands) {
  SmallVector<Type> operandTypes(operands.getTypes());
  return LLVM::LLVMFunctionType::get(resultType, operandTypes);
}

LLVM::LLVMFuncOp appendOrGetExternFuncOp(RewriterBase &rewriter, Operation *op,
                                         StringRef funcName, Type funcType,
                                         StringRef libname /*= ""*/,
                                         StringRef libpath /*= ""*/) {
  using LLVM::LLVMFuncOp;

  auto funcAttr = StringAttr::get(op->getContext(), funcName);
  Operation *funcOp = SymbolTable::lookupNearestSymbolFrom(op, funcAttr);
  if (funcOp)
    return cast<LLVMFuncOp>(*funcOp);

  Operation *parent = op;
  if (!isa<LLVM::LLVMFuncOp>(op))
    parent = op->getParentOfType<LLVM::LLVMFuncOp>();
  OpBuilder b(parent);
  auto ret = b.create<LLVMFuncOp>(op->getLoc(), funcName, funcType);
  ret.getOperation()->setAttr("libname",
                              StringAttr::get(op->getContext(), libname));
  ret.getOperation()->setAttr("libpath",
                              StringAttr::get(op->getContext(), libpath));
  return ret;
}
} // namespace triton::gpu

SmallVector<std::pair<StringAttr, Value>>
applyLinearLayout(Location loc, RewriterBase &rewriter,
                  const LinearLayout &layout,
                  ArrayRef<std::pair<StringAttr, Value>> indices) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  assert(layout.getNumInDims() == indices.size());
  for (auto [inDimName, idx] : indices) {
    assert(layout.hasInDim(inDimName) && "Invalid inDimName");
  }

  // This function can emit a lot of MLIR code, which ultimately makes
  // compilation slow.  (We think this shouldn't be the case -- it's not *that*
  // much code -- but we're not clear on how to fix the slowness, which happens
  // in the bowels of MLIR.)
  //
  // As a result we go through some contortions to avoid emitting code where
  // possible.

  // Manually constant-fold the layout where possible.
  SmallVector<std::pair<StringAttr, int32_t>> constantIns;
  for (auto [inDimName, idx] : indices) {
    if (auto constant = idx.getDefiningOp<LLVM::ConstantOp>()) {
      constantIns.push_back(
          {inDimName, cast<IntegerAttr>(constant.getValue()).getInt()});
    } else {
      constantIns.push_back({inDimName, 0});
    }
  }
  SmallVector<int32_t> constantComponent =
      llvm::to_vector(llvm::make_second_range(layout.apply(constantIns)));

  Value zero = b.i32_val(0);
  SmallVector<std::pair<StringAttr, Value>> outIndices;
  for (auto [i, outDimName] : llvm::enumerate(layout.getOutDimNames())) {
    if (constantComponent[i] == 0)
      outIndices.push_back({outDimName, zero});
    else
      outIndices.push_back({outDimName, b.i32_val(constantComponent[i])});
  }

  for (auto [inDimName, idx] : indices) {
    if (idx.getDefiningOp<LLVM::ConstantOp>()) {
      continue;
    }

    int nBits = layout.getInDimSizeLog2(inDimName);
    for (int i = 0; i < nBits; i++) {
      Value bit = b.and_(idx, b.i32_val(1 << i));
      Value bit_is_zero = b.icmp_eq(bit, zero);
      for (auto &[outDimName, outIdx] : outIndices) {
        int32_t basis = layout.getBasis(inDimName, i, outDimName);
        if (basis == 0)
          continue;
        outIdx = b.xor_(outIdx, b.select(bit_is_zero, zero, b.i32_val(basis)));
      }
    }
  }

  return outIndices;
}

std::optional<int> getWarpGroupStartThreadId(Block *block) {
  using namespace triton::gpu;

  // Look for an enclosing `ttg.warp_specialize` op.
  while (block && block->getParentOp() &&
         !isa<WarpSpecializePartitionsOp>(block->getParentOp()))
    block = block->getParentOp()->getBlock();
  if (!block || !block->getParentOp())
    return {};

  auto partitions = cast<WarpSpecializePartitionsOp>(block->getParentOp());
  unsigned idx = block->getParent()->getRegionNumber();
  WarpSpecializeOp ws = partitions.getParentOp();
  std::optional<ArrayRef<int32_t>> startIds = ws.getWarpGroupStartIds();
  assert(startIds && "cannot get warp group ID before warp group allocation");
  int32_t warpStartId = (*startIds)[idx];
  int threadsPerWarp =
      TritonGPUDialect::getThreadsPerWarp(ws->getParentOfType<ModuleOp>());
  return warpStartId * threadsPerWarp;
}

Value getThreadId(OpBuilder &rewriter, Location loc) {
  Value tid =
      rewriter.create<::mlir::gpu::ThreadIdOp>(loc, ::mlir::gpu::Dimension::x);
  tid = rewriter.create<arith::IndexCastOp>(loc, i32_ty, tid);

  // If this is being created inside a warp specialize op, compute the relative
  // thread ID within the warp group.
  if (std::optional<int> startId =
          getWarpGroupStartThreadId(rewriter.getInsertionBlock())) {
    TritonLLVMOpBuilder b(loc, rewriter);
    tid = rewriter.create<arith::SubIOp>(loc, tid, b.i32_val(*startId));
  }

  return tid;
}

Value getLaneId(OpBuilder &rewriter, Location loc) {
  TritonLLVMOpBuilder b(loc, rewriter);
  Value tid = getThreadId(rewriter, loc);
  int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
  return b.urem(tid, b.i32_val(threadsPerWarp));
}

std::pair<Value, Value> getLaneAndWarpId(OpBuilder &rewriter, Location loc) {
  TritonLLVMOpBuilder b(loc, rewriter);
  Value tid = getThreadId(rewriter, loc);
  int threadsPerWarp = triton::gpu::lookupThreadsPerWarp(rewriter);
  Value warpSizeVal = b.i32_val(threadsPerWarp);

  Value laneId = b.urem(tid, warpSizeVal);
  Value warpId = b.udiv(tid, warpSizeVal);
  return {laneId, warpId};
}

SmallVector<SmallVector<Value>>
emitIndices(Location loc, RewriterBase &rewriter, const TargetInfoBase &target,
            Attribute layout, RankedTensorType type, bool withCTAOffset) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = rewriter.getContext();
  auto shape = type.getShape();

  LinearLayout ll = triton::gpu::toLinearLayout(shape, layout);

  // TODO(jlebar): We could add strong typing if we wanted; for now this is
  // "stringly typed".
  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  StringAttr kWarp = str_attr("warp");
  StringAttr kBlock = str_attr("block");

  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId =
      withCTAOffset ? target.getClusterCTAId(rewriter, loc) : b.i32_val(0);
  unsigned rank = shape.size();
  SmallVector<SmallVector<Value>> ret;
  // Linear layout function is split in two parts below:
  // L(r, t, w, b) = L(0, t, w, b) xor L(r, 0, 0, 0)
  //     idxs      =    idxsBase   xor    idxsReg
  //
  // L(0, t, w, b) part is the same for all registers,
  // so we hoist it out of the main register loop in the below.
  //
  // This approach produces code with lower register pressure and
  // less computations, compared to fused L(r,t,w,b) method.
  auto idxsBase = applyLinearLayout(loc, rewriter, ll,
                                    {{kRegister, b.i32_val(0)},
                                     {kLane, laneId},
                                     {kWarp, warpId},
                                     {kBlock, blockId}});
  for (unsigned reg = 0; reg < ll.getInDimSize(str_attr("register")); reg++) {
    auto idxsReg =
        ll.apply({{kRegister, reg}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    SmallVector<std::pair<StringAttr, Value>> idxs;
    for (auto [idxBase, idxReg] : llvm::zip(idxsBase, idxsReg)) {
      auto dimName = idxBase.first;
      assert(dimName == idxReg.first &&
             "dim names of block+warp+thread and register idx should be equal");
      auto idx = b.xor_(idxBase.second, b.i32_val(idxReg.second));
      idxs.emplace_back(dimName, idx);
    }
    assert(idxs.size() == rank);
    for (unsigned k = 0; k < rank; ++k) {
      assert(idxs[k].first == str_attr("dim" + std::to_string(k)));
    }
    ret.push_back(llvm::to_vector(llvm::make_second_range(idxs)));
  }

  return ret;
}

namespace {

Value getSmemVecAddr(const LinearLayout &regLayout,
                     const LinearLayout &regToSharedLayout,
                     const LinearLayout &invertAllocSharedLayout,
                     const SharedMemoryObject &smemObj,
                     triton::gpu::MemDescType sharedTy, Type elemLlvmTy,
                     Value regId, Value laneId, Value warpId, Value blockId,
                     Location loc, RewriterBase &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  MLIRContext *ctx = rewriter.getContext();
  StringAttr kBlock = str_attr("block");
  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  StringAttr kWarp = str_attr("warp");
  auto shape = sharedTy.getShape();
  auto allocShape = sharedTy.getAllocShape();
  auto rank = shape.size();
  auto sharedEnc =
      cast<triton::gpu::SharedEncodingTrait>(sharedTy.getEncoding());

  auto smemBase = smemObj.getBase();
  auto smemOffsets = smemObj.getOffsets();
  auto smemStrides = smemObj.getStrides(sharedTy, loc, rewriter);
  Value smemOffset;
  // When loading or storing to shared memory, we consider two cases for
  // performance reasons:
  //
  //   1. Non-swizzled shared memory.
  //   2. Swizzled shared memory.
  //
  // Consider lowering `ttg.local_load %a`. In the first case, we can
  // directly construct a linear layout using `%a`'s shape and shared memory
  // encoding, irrespective of `%a`'s rank or whether it represents a slice of a
  // larger tensor.
  //
  // The method does not apply for swizzled shared memory in some scenarios.
  // Key properties of swizzling in Triton are:
  //
  //   - Swizzling applies only to tensors with rank ≥ 2.
  //   - It is restricted to the last two dimensions of the tensor.
  //   - These last two dimensions are always treated as the most "minor."
  //
  // An important edge case arises when `%a` results from `%a = ttg.subview %b`,
  // where `%b` is swizzled (and so is `%a`). In this case, constructing a
  // layout and determining shared memory offsets using `%a`'s shape is
  // incorrect. This is because swizzling depends on the original shape of `%b`,
  // which differs from `%a`'s shape. As a result, some locations may fall
  // outside `%a`'s contiguous view of memory. Specifically, an element `[i
  // (row_idx), j (col_idx)]` in `%a` might map to `[i, j']` after swizzling,
  // where `j'` lies outside `%a`'s shape but still within `%b`'s shape.
  //
  // We propose case 2 (see comments below), which provides a more general
  // solution for all swizzled shared memory scenarios, including the edge case
  // mentioned above.
  if (isSimpleSharedMemoryAccess(shape, allocShape, sharedEnc)) { // Case 1
    smemOffset = applyLinearLayout(loc, rewriter, regToSharedLayout,
                                   {{kRegister, regId},
                                    {kLane, laneId},
                                    {kWarp, warpId},
                                    {kBlock, blockId}})[0]
                     .second;
    // This reverts #5645, because it introduced increased register pressure in
    // AMD backend.
    // TODO: remove when new implementation performance reaches target level
    if (auto swizzledSharedEnc =
            mlir::dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(
                sharedEnc)) {
      auto regToSharedLayout =
          getRegToSharedLayout(ctx, shape, regLayout, swizzledSharedEnc,
                               elemLlvmTy.getIntOrFloatBitWidth());
      auto smemOrder = swizzledSharedEnc.getOrder();
      smemOffsets = llvm::to_vector(llvm::drop_end(llvm::make_second_range(
          applyLinearLayout(loc, rewriter, regToSharedLayout,
                            {{kRegister, regId},
                             {kLane, laneId},
                             {kWarp, warpId},
                             {kBlock, b.i32_val(0)}}))));
      // Reorder strides according to `order`.  This way they match the
      // multi-dimensional offsets in regToSharedLayout.
      smemOffset = dot(rewriter, loc, smemOffsets,
                       applyPermutation(smemStrides, smemOrder));
    }
  } else { // Case 2 -> rank-reduced swizzling
    assert(rank >= 2 && "Swizzling only applies to tensors with rank >= 2");
    assert((isa<triton::gpu::SwizzledSharedEncodingAttr,
                triton::gpu::AMDRotatingSharedEncodingAttr>(sharedEnc)) &&
           "NVMMA layout not supported for sliced tensors");
    // We define both tensor offsets and shared memory offsets:
    //
    //   - Tensor offsets: Relative offsets within a given tensor.
    //   - Shared memory offsets: Absolute offsets within the shared memory.
    //
    // In Triton, the shared memory layout provides an invertible, one-to-one
    // mapping between tensor offsets and shared memory offsets. The `base`
    // field of any shared memory object represents both the shared memory
    // offset and the tensor offset relative to the original tensor at
    // allocation, prior to any subview operations.
    //
    // To determine the shared memory offsets for a specific register when
    // dealing with swizzled and sliced tensors, the process involves:
    //
    //   1. Retrieving the original tensor's `invertAllocSharedLayout`, which
    //   maps the allocated tensor's offsets back to shared memory offsets.
    //   2. Reconstructing the register's offsets in the allocated tensor by
    //   summing:
    //      - The shared memory offsets of the current view's base, and
    //      - The relative tensor offsets of the register.
    //
    // This approach ensures that "absolute" tensor offsets can be
    // mapped to the correct shared memory addresses using
    // `invertAllocSharedLayout`.
    auto multiDimTensorOffsets =
        llvm::to_vector(applyLinearLayout(loc, rewriter, regLayout,
                                          {{kRegister, regId},
                                           {kLane, laneId},
                                           {kWarp, warpId},
                                           {kBlock, blockId}}));
    for (auto i = 0; i < rank; i++) {
      multiDimTensorOffsets[i].second =
          b.add(multiDimTensorOffsets[i].second, smemOffsets[i]);
    }
    smemOffset = applyLinearLayout(loc, rewriter, invertAllocSharedLayout,
                                   multiDimTensorOffsets)[0]
                     .second;
    Value baseToAllocBaseDist = dot(rewriter, loc, smemOffsets, smemStrides);
    smemOffset = b.sub(smemOffset, baseToAllocBaseDist);
  }
  auto ptrTy = smemBase.getType();
  auto vecAddr = b.gep(ptrTy, elemLlvmTy, smemBase, smemOffset);
  vecAddr.setInbounds(true);
  return vecAddr;
}

} // namespace

bool emitTransferBetweenRegistersAndShared(
    LinearLayout &regLayout, triton::gpu::MemDescType sharedTy, Type elemLlvmTy,
    std::optional<int32_t> maxVecElems, const SharedMemoryObject &smemObj,
    Location loc, RewriterBase &rewriter, const TargetInfoBase &target,
    std::function<void(VectorType, Value /*shmemAddr*/)> perVectorCallback) {
  MLIRContext *ctx = rewriter.getContext();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  StringAttr kBlock = str_attr("block");
  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  StringAttr kWarp = str_attr("warp");

  auto shape = sharedTy.getShape();
  LinearLayout sharedLayout =
      triton::gpu::toLinearLayout(shape, sharedTy.getEncoding());
  LinearLayout regToSharedLayout = regLayout.invertAndCompose(sharedLayout);

  // TODO(jlebar): We don't currently support loading from shared memory in a
  // different CTA.  We'd need to emit `mapa.shared::cluster` instructions.
  for (int inBlock = 1; inBlock < regToSharedLayout.getInDimSize(kBlock);
       inBlock *= 2) {
    auto idx = regToSharedLayout.apply(
        {{kRegister, 0}, {kLane, 0}, {kWarp, 0}, {kBlock, inBlock}});
    // Intra-block offset must be 0
    int32_t offset = idx[0].second;
    if (offset != 0) {
      return false;
    }
    // Check if there's any cross CTA load.
    int32_t outBlock = idx[1].second;
    if (outBlock != inBlock) {
      return false;
    }
  }

  // Determine how many consecutive registers map to consecutive shmem elements
  // in out-dimension offsetN.  This is our load instruction's vector width.
  //
  // It's OK if the vector width we choose here is wider than the hardware
  // supports; LLVM will legalize it.
  const int vecElems =
      std::min(regToSharedLayout.getNumConsecutiveInOut(),
               maxVecElems.value_or(std::numeric_limits<int>::max()));

  auto withCTAOffset = triton::gpu::getNumCTAs(sharedTy.getEncoding()) > 1;
  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value blockId =
      withCTAOffset ? target.getClusterCTAId(rewriter, loc) : b.i32_val(0);

  // For kernels with a single CTA, `allocSharedLayout.sublayout(S("block"),
  // outDims) == 0`. We need to take out the "block" dimension in order to use
  // `invert`.
  // For kernels with multiple CTAs per CGA,
  // `allocSharedLayout.sublayout(S("block"), outDims) != 0`. We do not need to
  // take out the "block" dimension.
  // Thus we use `pseudoinvert` instead of `invert` here for simplicity.
  auto allocShape = sharedTy.getAllocShape();
  LinearLayout invertAllocSharedLayout =
      triton::gpu::toLinearLayout(allocShape.take_back(sharedTy.getRank()),
                                  sharedTy.getEncoding())
          .pseudoinvert();

  int numElems = regToSharedLayout.getInDimSize(kRegister);
  auto vecTy = vec_ty(elemLlvmTy, vecElems);
  SmallVector<Value> ret;
  for (int i = 0; i < numElems / vecElems; i++) {
    auto regId = b.i32_val(i * vecElems);
    auto vecAddr = getSmemVecAddr(
        regLayout, regToSharedLayout, invertAllocSharedLayout, smemObj,
        sharedTy, elemLlvmTy, regId, laneId, warpId, blockId, loc, rewriter);
    perVectorCallback(vecTy, vecAddr);
  }
  return true;
}

bool emitTransferBetweenRegistersAndShared(
    RankedTensorType registerTy, triton::gpu::MemDescType sharedTy,
    Type elemLlvmTy, std::optional<int32_t> maxVecElems,
    const SharedMemoryObject &smemObj, Location loc, RewriterBase &rewriter,
    const TargetInfoBase &target,
    std::function<void(VectorType, Value /*shmemAddr*/)> perVectorCallback) {
  auto regLayout = triton::gpu::toLinearLayout(registerTy.getShape(),
                                               registerTy.getEncoding());
  return emitTransferBetweenRegistersAndShared(
      regLayout, sharedTy, elemLlvmTy, maxVecElems, smemObj, loc, rewriter,
      target, perVectorCallback);
}

SmallVector<Value> loadSharedToDistributed(RankedTensorType dstTy,
                                           triton::gpu::MemDescType srcTy,
                                           Type elemLlvmTy,
                                           const SharedMemoryObject &smemObj,
                                           Location loc, RewriterBase &rewriter,
                                           const TargetInfoBase &target) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  SmallVector<Value> ret;
  bool success = emitTransferBetweenRegistersAndShared(
      dstTy, srcTy, elemLlvmTy, /*maxVecElems=*/std::nullopt, smemObj, loc,
      rewriter, target, [&](VectorType vecTy, Value vecAddr) {
        auto vecVal = b.load(vecTy, vecAddr);
        vecVal.setAlignment(vecTy.getNumElements() *
                            elemLlvmTy.getIntOrFloatBitWidth() / 8);

        for (int v = 0; v < vecTy.getNumElements(); v++) {
          ret.push_back(b.extract_element(elemLlvmTy, vecVal, b.i32_val(v)));
        }
      });
  if (!success)
    llvm::report_fatal_error("Failed to emit transfer from shared to register");

  return ret;
}

void storeDistributedToShared(triton::gpu::MemDescType dstTy,
                              RankedTensorType srcTy, Type elemLlvmTy,
                              ArrayRef<Value> srcVals,
                              const SharedMemoryObject &smemObj, Location loc,
                              RewriterBase &rewriter,
                              const TargetInfoBase &target,
                              std::pair<size_t, Type> *const llvmOpCount) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  bool success = emitTransferBetweenRegistersAndShared(
      srcTy, dstTy, elemLlvmTy, /*maxVecElems=*/std::nullopt, smemObj, loc,
      rewriter, target, [&](VectorType vecTy, Value vecAddr) {
        ArrayRef<Value> vals = srcVals.take_front(vecTy.getNumElements());
        srcVals = srcVals.drop_front(vecTy.getNumElements());

        Value vec = b.undef(vecTy);
        for (int i = 0; i < vals.size(); i++) {
          vec = b.insert_element(vec, vals[i], b.i32_val(i));
        }
        b.store(vec, vecAddr)
            .setAlignment(vecTy.getNumElements() *
                          elemLlvmTy.getIntOrFloatBitWidth() / 8);
        if (llvmOpCount) {
          ++(llvmOpCount->first);
          llvmOpCount->second = vecTy;
        }
      });

  if (!success)
    llvm::report_fatal_error("Failed to emit transfer from register to shared");
}

SmallVector<SmallVector<unsigned>> emitOffsetForLayout(Attribute layout,
                                                       RankedTensorType type) {
  MLIRContext *ctx = layout.getContext();
  auto shape = type.getShape();
  unsigned rank = shape.size();

  auto ll = triton::gpu::toLinearLayout(shape, layout);

  StringAttr kRegister = str_attr("register");
  StringAttr kLane = str_attr("lane");
  StringAttr kWarp = str_attr("warp");
  StringAttr kBlock = str_attr("block");

  SmallVector<SmallVector<unsigned>> offsets;
  for (int i = 0; i < ll.getInDimSize(str_attr("register")); i++) {
    auto idxs = ll.apply({{kRegister, i}, {kLane, 0}, {kWarp, 0}, {kBlock, 0}});
    assert(idxs.size() == rank);
    for (unsigned k = 0; k < rank; ++k) {
      assert(idxs[k].first == str_attr("dim" + std::to_string(k)));
    }
    offsets.push_back(
        llvm::to_vector_of<unsigned>(llvm::make_second_range(idxs)));
  }
  return offsets;
}

namespace LLVM {
using namespace mlir::triton;
using mlir::triton::gpu::getOrder;

Value createConstantI1(Location loc, OpBuilder &rewriter, bool v) {
  auto i1ty = rewriter.getIntegerType(1);
  return rewriter.create<LLVM::ConstantOp>(loc, i1ty,
                                           IntegerAttr::get(i1ty, v));
}

Value createConstantI32(Location loc, OpBuilder &rewriter, int32_t v) {
  auto i32ty = rewriter.getIntegerType(32);
  return rewriter.create<LLVM::ConstantOp>(loc, i32ty,
                                           IntegerAttr::get(i32ty, v));
}

Value createConstantI64(Location loc, OpBuilder &rewriter, int64_t v) {
  auto i64ty = rewriter.getIntegerType(64);
  return rewriter.create<LLVM::ConstantOp>(loc, i64ty,
                                           IntegerAttr::get(i64ty, v));
}

Value createConstantF16(Location loc, OpBuilder &rewriter, float v) {
  auto type = type::f16Ty(rewriter.getContext());
  return rewriter.create<LLVM::ConstantOp>(loc, type,
                                           rewriter.getF16FloatAttr(v));
}

Value createConstantBF16(Location loc, OpBuilder &rewriter, float v) {
  APFloat apf(v);
  bool ignored;
  apf.convert(APFloat::BFloat(), APFloat::rmNearestTiesToEven, &ignored);
  auto type = type::bf16Ty(rewriter.getContext());
  auto attr = FloatAttr::get(type, apf);
  return rewriter.create<LLVM::ConstantOp>(loc, type, attr);
}

Value createConstantF32(Location loc, OpBuilder &rewriter, float v) {
  auto type = type::f32Ty(rewriter.getContext());
  return rewriter.create<LLVM::ConstantOp>(loc, type,
                                           rewriter.getF32FloatAttr(v));
}

Value createConstantF64(Location loc, OpBuilder &rewriter, double v) {
  auto type = type::f64Ty(rewriter.getContext());
  return rewriter.create<LLVM::ConstantOp>(loc, type,
                                           rewriter.getF64FloatAttr(v));
}

Value createNaNConstant(Location loc, OpBuilder &rewriter, Type type) {
  if (!isa<FloatType>(type)) {
    llvm::report_fatal_error("Creating NaN constant for non-float type!");
  }
  return rewriter.create<LLVM::ConstantOp>(
      loc, type, APFloat::getNaN(cast<FloatType>(type).getFloatSemantics()));
}

// Create an index type constant.
Value createIndexConstant(OpBuilder &builder, Location loc,
                          const TypeConverter *converter, int64_t value) {
  Type ty = converter->convertType(builder.getIndexType());
  return builder.create<LLVM::ConstantOp>(loc, ty,
                                          builder.getIntegerAttr(ty, value));
}

// Create an integer constant of \param width bits.
Value createLLVMIntegerConstant(OpBuilder &builder, Location loc, short width,
                                int64_t value) {
  Type ty = builder.getIntegerType(width);
  return builder.create<LLVM::ConstantOp>(loc, ty,
                                          builder.getIntegerAttr(ty, value));
}

LLVM::CallOp createLLVMCallOp(OpBuilder &builder, Location loc,
                              LLVMFuncOp funcOp, ValueRange args) {
  auto op = builder.create<LLVM::CallOp>(loc, funcOp, args);
  op.getProperties().setOpBundleSizes(builder.getDenseI32ArrayAttr({}));
  op.getProperties().setOperandSegmentSizes({static_cast<int>(args.size()), 0});
  return op;
}

LLVM::CallIntrinsicOp
createLLVMIntrinsicCallOp(OpBuilder &builder, Location loc, StringRef intrinsic,
                          TypeRange types, ValueRange args) {
  auto op = builder.create<LLVM::CallIntrinsicOp>(loc, types, args);
  op.getProperties().setIntrin(builder.getStringAttr(intrinsic));
  op.getProperties().setOpBundleSizes(builder.getDenseI32ArrayAttr({}));
  op.getProperties().setOperandSegmentSizes({static_cast<int>(args.size()), 0});
  return op;
}

bool isConstantZero(Value v) {
  if (auto constantOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
      return attr.getValue().isZero();
    }
    if (auto attr = dyn_cast<FloatAttr>(constantOp.getValue())) {
      return attr.getValue().isZero();
    }
  }
  return false;
}

Value getStructFromSharedMemoryObject(Location loc,
                                      const SharedMemoryObject &smemObj,
                                      RewriterBase &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto elems = smemObj.getElems();
  auto types = smemObj.getTypes();
  auto structTy =
      LLVM::LLVMStructType::getLiteral(rewriter.getContext(), types);
  // pack into struct
  Value llvmStruct = rewriter.create<LLVM::UndefOp>(loc, structTy);
  for (const auto &v : llvm::enumerate(elems)) {
    assert(v.value() && "can not insert null values");
    llvmStruct = b.insert_val(structTy, llvmStruct, v.value(), v.index());
  }
  return llvmStruct;
}

SharedMemoryObject getSharedMemoryObjectFromStruct(Location loc,
                                                   Value llvmStruct,
                                                   Type elemTy,
                                                   RewriterBase &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  ArrayRef<Type> types =
      cast<LLVM::LLVMStructType>(llvmStruct.getType()).getBody();
  SmallVector<Value> elems(types.size());
  for (unsigned i = 0; i < types.size(); ++i) {
    Type type = types[i];
    elems[i] = b.extract_val(type, llvmStruct, i);
  }
  return {/*base=*/elems[0],
          /*baseElemType=*/elemTy,
          /*offsets=*/{elems.begin() + 1, elems.end()}};
}

// Extract the bits of `a` that are set in `mask`
Value pext_i32(RewriterBase &rewriter, Location loc, Value a, uint32_t mask) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  assert(a.getType() == i32_ty && "a must be i32");
  // Handle width = 32 to avoid doing 1 << 32
  if (mask == 0xFFFFFFFF)
    return a;

  // Implements the blocked algorithm from
  // https://forums.developer.nvidia.com/t/pdep-and-pext-functionality-for-cuda/270973
  uint32_t mskConst = mask;
  uint32_t extcnt = 0;
  Value result = b.i32_val(0);
  while (mskConst) {
    uint32_t oldmsk = mskConst;
    uint32_t bitgrplsb = mskConst & (-mskConst);
    mskConst &= bitgrplsb + mskConst;
    uint32_t bitgrp = mskConst ^ oldmsk;
    uint32_t lsbpos = 31 - __builtin_clz(bitgrplsb);
    // like popcount for a number 0..01..1..0 but portable
    uint32_t grplen = __builtin_ctz(~(bitgrp >> lsbpos));
    uint32_t shift = lsbpos - extcnt;
    extcnt += grplen;
    result =
        b.or_(result, b.lshr(b.and_(b.i32_val(bitgrp), a), b.i32_val(shift)));
  }
  return result;
}

std::tuple<SmallVector<Value>, Value>
delinearize(RewriterBase &rewriter, Location loc,
            triton::gpu::DistributedEncodingTrait layout,
            ArrayRef<int64_t> shape, StringAttr dimName, Value linear) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto ll = triton::gpu::toLinearLayout(shape, layout);
  auto linearLayout =
      triton::gpu::LinearEncodingAttr::get(rewriter.getContext(), ll);
  assert(ll.hasInDim(dimName));
  int32_t freeVarMask = ll.getFreeVariableMasks()[dimName];
  auto isRepresentative = b.true_val();
  if (freeVarMask != 0) {
    isRepresentative =
        b.icmp_eq(b.and_(b.i32_val(freeVarMask), linear), b.i32_val(0));
    // We remove the bits of linear that are set to one in freeVarMask
    int32_t nonFreeVarMask = ~freeVarMask & (ll.getInDimSize(dimName) - 1);
    linear = pext_i32(rewriter, loc, linear, nonFreeVarMask);
  }

  auto orderDim = linearLayout.orderPerDim(dimName, linearLayout.getOrder());
  auto shapeDim = linearLayout.basesPerDim(dimName);
  auto multiDim = delinearize(rewriter, loc, linear, shapeDim, orderDim);

  return std::make_tuple(std::move(multiDim), isRepresentative);
}

// Convert an \param index to a multi-dim coordinate given \param shape and
// \param order.
SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               Value linear, ArrayRef<unsigned> shape,
                               ArrayRef<unsigned> order) {
  unsigned rank = shape.size();
  assert(rank == order.size());
  auto reordered = applyPermutation(shape, order);
  SmallVector<Value> reorderedMultiDim(rank);
  if (auto constantOp = linear.getDefiningOp<arith::ConstantOp>()) {
    unsigned intVal = mlir::cast<IntegerAttr>(constantOp.getValue())
                          .getValue()
                          .getSExtValue();
    reorderedMultiDim = delinearize(rewriter, loc, intVal, reordered);
  } else {
    reorderedMultiDim = delinearize(rewriter, loc, linear, reordered);
  }
  SmallVector<Value> multiDim(rank);
  for (unsigned i = 0; i < rank; ++i) {
    multiDim[order[i]] = reorderedMultiDim[i];
  }
  return multiDim;
}

SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               unsigned linear, ArrayRef<unsigned> shape) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned rank = shape.size();
  assert(rank > 0);
  SmallVector<Value> multiDim(rank);
  unsigned remained = linear;
  for (auto &&en : llvm::enumerate(shape)) {
    unsigned dimSize = en.value();
    multiDim[en.index()] = b.i32_val(remained % dimSize);
    remained = remained / dimSize;
  }
  return multiDim;
}

SmallVector<Value> delinearize(RewriterBase &rewriter, Location loc,
                               Value linear, ArrayRef<unsigned> shape) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  unsigned rank = shape.size();
  assert(rank > 0);
  SmallVector<Value> multiDim(rank);
  Value remained = linear;
  for (auto &&en : llvm::enumerate(shape)) {
    Value dimSize = b.i32_val(en.value());
    multiDim[en.index()] = b.urem(remained, dimSize);
    remained = b.udiv(remained, dimSize);
  }
  return multiDim;
}

SmallVector<unsigned> delinearize(unsigned linear, ArrayRef<unsigned> shape,
                                  ArrayRef<unsigned> order) {
  auto rank = shape.size();
  assert(order.size() == rank);
  SmallVector<unsigned> multiDim(rank);
  for (auto dim : order) {
    multiDim[dim] = linear % shape[dim];
    linear /= shape[dim];
  }
  assert(linear == 0);
  return multiDim;
}

Value linearize(RewriterBase &rewriter, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape, ArrayRef<unsigned> order) {
  return linearize(rewriter, loc, applyPermutation(multiDim, order),
                   applyPermutation(shape, order));
}

Value linearize(RewriterBase &rewriter, Location loc, ArrayRef<Value> multiDim,
                ArrayRef<unsigned> shape) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto rank = multiDim.size();
  Value linear = b.i32_val(0);
  if (rank > 0) {
    linear = multiDim.back();
    for (auto [dim, dimShape] :
         llvm::reverse(llvm::zip(multiDim.drop_back(), shape.drop_back()))) {
      Value dimSize = b.i32_val(dimShape);
      linear = b.add(b.mul(linear, dimSize), dim);
    }
  }
  return linear;
}

size_t linearize(ArrayRef<unsigned> multiDim, ArrayRef<unsigned> shape,
                 ArrayRef<unsigned> order) {
  size_t linear = 0;
  for (unsigned dim : llvm::reverse(order))
    linear = linear * shape[dim] + multiDim[dim];
  return linear;
}

Value addStringToModule(Location loc, RewriterBase &rewriter, StringRef key,
                        StringRef content) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  auto ctx = moduleOp.getContext();
  unsigned stringNumber = 0;
  SmallString<16> stringConstName;
  do {
    stringConstName.clear();
    (key + Twine(stringNumber++)).toStringRef(stringConstName);
  } while (moduleOp.lookupSymbol(stringConstName));

  llvm::SmallString<64> contentStr(content);
  size_t contentSize = contentStr.size_in_bytes();
  auto globalType = LLVM::LLVMArrayType::get(i8_ty, contentSize);

  LLVM::GlobalOp global;
  {
    RewriterBase::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    global = rewriter.create<LLVM::GlobalOp>(
        UnknownLoc::get(ctx), globalType,
        /*isConstant=*/true, LLVM::Linkage::Internal, stringConstName,
        rewriter.getStringAttr(contentStr));
  }

  Value zero = b.i32_val(0);
  Type globalPtrType = LLVM::LLVMPointerType::get(ctx, global.getAddrSpace());
  Value globalPtr = rewriter.create<LLVM::AddressOfOp>(
      UnknownLoc::get(ctx), globalPtrType, global.getSymName());
  Value stringStart =
      b.gep(ptr_ty(ctx), i8_ty, globalPtr, SmallVector<Value>({zero}));
  return stringStart;
}

} // namespace LLVM

SharedMemoryObject
getExpandedSharedMemoryObject(ConversionPatternRewriter &rewriter, Location loc,
                              SharedMemoryObject smemObj,
                              ArrayRef<int64_t> shape) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  assert(shape.size() == 2 || shape.size() == 3);
  auto offsets = smemObj.getOffsets();
  auto rank = offsets.size();
  assert(rank == shape.size());
  if (rank == 3)
    return smemObj;
  offsets.insert(offsets.begin(), b.i32_val(0));
  auto expandedSmemObj =
      SharedMemoryObject(smemObj.getBase(), smemObj.getBaseElemType(), offsets);
  return expandedSmemObj;
}

} // namespace mlir
