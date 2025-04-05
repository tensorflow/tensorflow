#include "Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "AtomicRMWOpsEmitter.h"

using namespace triton::AMD;

namespace {

Value generateI32DppMove(RewriterBase &rewriter, Value val, int dppCtrl,
                         int rowMask = 0b1111,  // enable all rows
                         int bankMask = 0b1111, // enable all banks
                         bool boundCtrl = false) {
  assert(val.getType().isInteger(32));
  auto loc = val.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value old = b.i32_val(0);
  auto dppMovOp = rewriter.create<ROCDL::DPPUpdateOp>(
      loc, i32_ty, old, val, dppCtrl, rowMask, bankMask, boundCtrl);
  return dppMovOp.getResult();
}

Value shiftLeftI32ByDpp(RewriterBase &rewriter, Value val) {
  return generateI32DppMove(rewriter, val, 0x101); // shift left
}

Value shiftRightI32ByDpp(RewriterBase &rewriter, Value val) {
  return generateI32DppMove(rewriter, val, 0x111); // shift right 1 lane
}

Value generatePopcount64(RewriterBase &rewriter, Value val) {
  auto loc = val.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value m1 = b.i64_val(0x5555555555555555); // binary: 0101 0101..
  Value m2 = b.i64_val(0x3333333333333333); // binary: 0011 0011..
  Value m4 = b.i64_val(0x0f0f0f0f0f0f0f0f); // binary: 0000 1111..
  // binary: 0000 0001 0000 0001..
  Value h01 = b.i64_val(0x0101010101010101);
  // put count of each 2 bits into those 2 bits
  val = b.sub(val, b.and_(m1, b.lshr(val, b.i64_val(1))));
  // put count of each 4 bits into those 4 bits
  val = b.add(b.and_(val, m2), b.and_(b.lshr(val, b.i64_val(2)), m2));
  // put count of each 8 bits into those 8 bits
  val = b.and_(b.add(val, b.lshr(val, b.i64_val(4))), m4);
  // left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
  return b.lshr(b.mul(val, h01), b.i64_val(56));
}

Value genReadFirstLane(RewriterBase &rewriter, Value v) {
  auto loc = v.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  std::string intrinsic = "llvm.amdgcn.readfirstlane";
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, i32_ty, v)
      ->getResult(0);
}

Value genPermute(RewriterBase &rewriter, Value v, Value dst) {
  auto loc = v.getLoc();
  std::string intrinsic = "llvm.amdgcn.ds.permute";
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, i32_ty,
                                         ValueRange{dst, v})
      ->getResult(0);
}

Value genBPermute(RewriterBase &rewriter, Value v, Value dst) {
  auto loc = v.getLoc();
  std::string intrinsic = "llvm.amdgcn.ds.bpermute";
  return LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, i32_ty,
                                         ValueRange{dst, v})
      ->getResult(0);
}

template <typename Generator, typename... Values>
Value genI32TiledOp(RewriterBase &rewriter, Generator genCall, Value argToSplit,
                    Values... args) {
  auto loc = argToSplit.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type ty = argToSplit.getType();
  size_t tySize = ty.getIntOrFloatBitWidth();
  size_t i32Size = i32_ty.getIntOrFloatBitWidth();
  size_t count = tySize / i32Size;
  assert(tySize % i32Size == 0 && count > 0 &&
         "Unalligned types are not supported yet.");
  Type i32VecValTy = vec_ty(i32_ty, count);
  Value vec = b.undef(i32VecValTy);
  Value valCasted = b.bitcast(argToSplit, i32VecValTy);
  for (int i = 0; i < count; i++) {
    Value subVal = b.extract_element(i32_ty, valCasted, b.i32_val(i));
    Value result = genCall(rewriter, subVal, args...);
    vec = b.insert_element(i32VecValTy, vec, result, b.i32_val(i));
  }
  return b.bitcast(vec, ty);
}

Value genPrefixSum(RewriterBase &rewriter, Value v0) {
  auto loc = v0.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  auto v0Ty = v0.getType();
  assert(v0Ty.getIntOrFloatBitWidth() == i32_ty.getIntOrFloatBitWidth());

  Value v1 = v0;
  // v_add_f32 v1, v0, v0 row_shr:1 bound_ctrl:0
  Value tmp = generateI32DppMove(rewriter, v0, 0x111);
  v1 = b.add(v1, tmp);

  // v_add_f32 v1, v0, v1 row_shr:2 bound_ctrl:0
  tmp = generateI32DppMove(rewriter, v0, 0x112);
  v1 = b.add(v1, tmp);

  // v_add_f32 v1, v0, v1 row_shr:3 bound_ctrl:0
  tmp = generateI32DppMove(rewriter, v0, 0x113);
  v1 = b.add(v1, tmp);

  // v_add_f32 v1, v1, v1 row_shr:4 bank_mask:0xe
  tmp = generateI32DppMove(rewriter, v1, 0x114, 0xF, 0xE, true);
  v1 = b.add(v1, tmp);

  // v_add_f32 v1, v1, v1 row_shr:8 bank_mask:0xc
  tmp = generateI32DppMove(rewriter, v1, 0x118, 0xF, 0xC, true);
  v1 = b.add(v1, tmp);

  // v_add_f32 v1, v1, v1 row_bcast:15 row_mask:0xa
  tmp = generateI32DppMove(rewriter, v1, 0x142, 0xA, 0xF, true);
  v1 = b.add(v1, tmp);

  // v_add_f32 v1, v1, v1 row_bcast:31 row_mask:0xc
  tmp = generateI32DppMove(rewriter, v1, 0x143, 0xC, 0xF, true);
  v1 = b.add(v1, tmp);

  return v1;
}
} // namespace

namespace mlir::LLVM::AMD {

Value AtomicRMWEmitter::emitAtomicRMW(RewriterBase &rewriter, Value rmwPtr,
                                      Value valElem, Value rmwMask,
                                      std::optional<Value> sharedMemBase,
                                      bool enableIntraWaveReduce) const {
  auto loc = rmwPtr.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type retType = valElem.getType();
  Value undefVal = b.undef(retType);
  // Build blocks to bypass the atomic instruction for ~rmwMask.
  auto *curBlock = rewriter.getInsertionBlock();
  auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
  auto *atomicBlock = rewriter.createBlock(
      curBlock->getParent(), std::next(Region::iterator(curBlock)));
  endBlock->addArgument({retType}, {loc});

  rewriter.setInsertionPointToEnd(curBlock);

  // intraWave reduce optimization for atomic ops needs all active threads
  // at the beginning of a wave. This is achieved as:
  // 1. Compute the prefix sum of the mask, then each active lane gets a
  //    different value (offset) from its previous lane.
  // 2. Multiply the mask and the offset, so only active lanes have a
  //    non-zero offset, and the offset is different in each active lane
  // 3. Sub 1 from offset to get the idx each active lane is moved to
  // 4. Call ds_permute to move active lanes to the beginning of a wave
  // 5. Update mask of each lane
  if (enableIntraWaveReduce) {
    Value maskI32 = b.zext(i32_ty, rmwMask);
    Value offset = genPrefixSum(rewriter, maskI32);
    offset = b.mul(offset, maskI32);
    Value waveSize =
        b.i32_val(mlir::triton::gpu::lookupThreadsPerWarp(rewriter));
    offset = b.select(b.icmp_eq(offset, b.i32_val(0)), waveSize, offset);
    Value idx = b.sub(offset, b.i32_val(1));
    idx = b.mul(idx, b.i32_val(4));
    valElem = genI32TiledOp(rewriter, genPermute, valElem, idx);
    Value castedAddr = b.ptrtoint(i64_ty, rmwPtr);
    castedAddr = genI32TiledOp(rewriter, genPermute, castedAddr, idx);
    rmwPtr = b.inttoptr(rmwPtr.getType(), castedAddr);

    // update mask
    Value maskFlag = targetInfo.ballot(rewriter, loc, i64_ty, rmwMask);
    Value numActiveLanes =
        b.trunc(i32_ty, generatePopcount64(rewriter, maskFlag));

    Value laneID = b.urem(getThreadId(rewriter, loc), waveSize);
    rmwMask = b.icmp_ult(laneID, numActiveLanes);
  }

  rewriter.create<LLVM::CondBrOp>(loc, rmwMask, atomicBlock, endBlock,
                                  undefVal);

  rewriter.setInsertionPointToEnd(atomicBlock);
  Value atom = enableIntraWaveReduce
                   ? atomicIntraWaveReduce(rewriter, rmwPtr, valElem, binOp,
                                           memOrder, scopeStr.c_str())
                   : rewriter
                         .create<LLVM::AtomicRMWOp>(loc, binOp, rmwPtr, valElem,
                                                    memOrder, scopeStr.c_str())
                         .getResult();

  if (sharedMemBase.has_value()) {
    Value atomPtr = *sharedMemBase;
    b.store(atom, atomPtr);
  }
  rewriter.create<LLVM::BrOp>(loc, atom, endBlock);
  rewriter.setInsertionPointToStart(endBlock);

  return endBlock->getArgument(0);
}

Value AtomicRMWEmitter::emitPairedAtomicForEvenTID(RewriterBase &rewriter,
                                                   Value rmwPtr, Value valElem,
                                                   Value rmwMask,
                                                   bool checkPairs) const {
  auto loc = rmwPtr.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Value i64Ones = b.i64_val(~uint64_t(0));
  Value isOddI32 = b.urem(getThreadId(rewriter, loc), b.i32_val(2));
  // First check if odd threads hold adjacent ptrs to even ones.
  Value castedAddr = b.ptrtoint(i64_ty, rmwPtr);
  // Set casted addr to all ones if the thread is disabled.
  castedAddr = b.select(rmwMask, castedAddr, i64Ones);

  Type valueElemTy = valElem.getType();
  Type packF16Ty = vec_ty(valueElemTy, 2);

  // Move %val to left neighbour to proceed packed atomic further.
  Value packedVal = b.null(packF16Ty);
  packedVal = b.insert_element(packF16Ty, packedVal, valElem, isOddI32);
  // Pack to i32 type to simplify transaction.
  packedVal = b.bitcast(packedVal, i32_ty);
  // Zero operands for disabled threads to make addition no op.
  packedVal = b.select(rmwMask, packedVal, b.i32_val(0));
  Value dppMoveRes = shiftLeftI32ByDpp(rewriter, packedVal);
  Value operand = b.bitcast(b.or_(packedVal, dppMoveRes), packF16Ty);

  // If a runtime check is unnecessary (`checkPairs` is `false`),
  // `rightNeighbourPtr` is irrelevant.
  // Set the conditional value `enablePackedOpt` to `true` to enable DCE on the
  // runtime check branch.
  Value rightNeighbourPtr = rmwPtr;
  Value enablePackedOpt = b.true_val();
  if (checkPairs) {
    Value rightNeighbourAddr =
        genI32TiledOp(rewriter, shiftLeftI32ByDpp, castedAddr);

    // Packing optimization only supported if following conditions are true:
    // 1. address is aligned by 4 bytes
    // 2. right neighbour has adjacent address
    // 3. both threads are active
    Value isAligned = b.icmp_eq(b.urem(castedAddr, b.i64_val(4)), b.i64_val(0));
    Value neighbourAddrAdjacent = b.icmp_eq(
        rightNeighbourAddr,
        b.add(castedAddr, b.i64_val(valueElemTy.getIntOrFloatBitWidth() / 8)));
    Value neighbourEnabled = b.icmp_ne(i64Ones, rightNeighbourAddr);
    Value bothEnabled = b.and_(neighbourEnabled, rmwMask);
    enablePackedOpt =
        b.and_(b.and_(isAligned, bothEnabled), neighbourAddrAdjacent);

    // Enable only the even threads.
    Value anyEnabled = b.or_(neighbourEnabled, rmwMask);
    // If one of the threads is disabled, use the neighbour's addr.
    rightNeighbourAddr =
        b.select(neighbourEnabled, rightNeighbourAddr, castedAddr);
    castedAddr = b.select(rmwMask, castedAddr, rightNeighbourAddr);

    rmwMask = b.and_(anyEnabled, b.icmp_eq(isOddI32, b.i32_val(0)));

    // Unpack results back
    rightNeighbourPtr = b.inttoptr(rmwPtr.getType(), rightNeighbourAddr);
    rmwPtr = b.inttoptr(rmwPtr.getType(), castedAddr);
  } else {
    rmwMask = b.and_(rmwMask, b.icmp_eq(isOddI32, b.i32_val(0)));
  }

  Value undefVal = b.undef(packF16Ty);
  // Build blocks to bypass the atomic instruction for ~rmwMask.
  auto *curBlock = rewriter.getInsertionBlock();
  auto *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
  auto *atomicBlock = rewriter.createBlock(
      curBlock->getParent(), std::next(Region::iterator(curBlock)));
  endBlock->addArgument({packF16Ty}, {loc});

  rewriter.setInsertionPointToEnd(curBlock);
  rewriter.create<LLVM::CondBrOp>(loc, rmwMask, atomicBlock, endBlock,
                                  undefVal);

  rewriter.setInsertionPointToEnd(atomicBlock);

  // Determine on the runtime what atomic intrinsic to execute:
  // packed or regular.
  auto *packedBlock = atomicBlock->splitBlock(rewriter.getInsertionPoint());
  auto *regularBlock = rewriter.createBlock(
      atomicBlock->getParent(), std::next(Region::iterator(atomicBlock)));
  rewriter.setInsertionPointToEnd(atomicBlock);

  // If `checkPairs` was set to `false`, `packedBlock` must be removed by DCE
  rewriter.create<LLVM::CondBrOp>(loc, enablePackedOpt, packedBlock,
                                  regularBlock);

  // Fill out the regular block, where we issue two atomic ops.
  rewriter.setInsertionPointToEnd(regularBlock);
  Value pairedOperand0 = b.extract_element(valueElemTy, operand, b.i32_val(0));
  Value pairedOperand1 = b.extract_element(valueElemTy, operand, b.i32_val(1));
  Value atomNonVec0 = rewriter.create<LLVM::AtomicRMWOp>(
      loc, binOp, rmwPtr, pairedOperand0, memOrder, scopeStr.c_str());
  Value atomNonVec1 = rewriter.create<LLVM::AtomicRMWOp>(
      loc, binOp, rightNeighbourPtr, pairedOperand1, memOrder,
      scopeStr.c_str());
  Value packedRes = b.undef(packF16Ty);
  packedRes = b.insert_element(packF16Ty, packedRes, atomNonVec0, b.i32_val(0));
  packedRes = b.insert_element(packF16Ty, packedRes, atomNonVec1, b.i32_val(1));
  rewriter.create<LLVM::BrOp>(loc, packedRes, endBlock);

  // Start to fill out the packed block.
  rewriter.setInsertionPointToEnd(packedBlock);

  Value atom = rewriter.create<LLVM::AtomicRMWOp>(loc, binOp, rmwPtr, operand,
                                                  memOrder, scopeStr.c_str());

  rewriter.create<LLVM::BrOp>(loc, atom, endBlock);

  rewriter.setInsertionPointToStart(endBlock);
  Value atomRes = endBlock->getArgument(0);
  // Return packed to i32 result after atomic operation back from
  // master lane.
  auto packedRet = b.bitcast(atomRes, i32_ty);
  Value dppMovRes = shiftRightI32ByDpp(rewriter, packedRet);
  // Unpack results back
  Value unpackedDppRes = b.bitcast(dppMovRes, packF16Ty);
  atomRes = b.insert_element(
      packF16Ty, atomRes,
      b.extract_element(valueElemTy, unpackedDppRes, b.i32_val(1)),
      b.i32_val(1));
  return b.extract_element(valueElemTy, atomRes,
                           b.urem(getThreadId(rewriter, loc), b.i32_val(2)));
}

Value AtomicRMWEmitter::atomicIntraWaveReduce(RewriterBase &rewriter,
                                              Value rmwPtr, Value operand,
                                              LLVM::AtomicBinOp opKind,
                                              LLVM::AtomicOrdering memOrdering,
                                              StringRef scope) const {
  // This approach minimizes intra-warp thread contention when accessing
  // global memory pointers. It is particularly advantageous for certain ISA
  // families, such as CDNA3. The algorithm follows these steps:
  // 1. Analyze thread groups and their relative positions:
  // 1.1. Consider groups of threads sharing identical pointers using
  //      `readfirstlane` and ballot `intrinsics`.
  // 1.2. Compute parameters to form contiguous groups and further optimize
  //      them.
  // 1.3. Disable threads that have already been processed.
  // 1.4. If thread was not considered, jump to `1.1.`.
  // 2. Form contiguous groups:
  //    Use `permute` instructions to organize threads within the wavefront
  //    into continuous groups.
  // 4. Reduce Groups to Leader threads:
  //    Apply `bpermute` and operation-specific arithmetic based on the
  //    opKind to consolidate group data into leader threads.
  // 5. Perform global atomic operations by leader threads.
  auto loc = operand.getLoc();
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  Type operandElemType = operand.getType();
  Type origPtrType = rmwPtr.getType();

  rmwPtr = b.ptrtoint(i64_ty, rmwPtr);

  auto *curBlock = rewriter.getInsertionBlock();
  auto *atomicBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
  atomicBlock->addArgument(i64_ty, loc);
  atomicBlock->addArgument(operandElemType, loc);
  auto *initLoop = rewriter.createBlock(curBlock->getParent(),
                                        std::next(Region::iterator(curBlock)));

  rewriter.setInsertionPointToEnd(curBlock);

  // check how many adjacent address are in the wave
  Value rightNeighbourAddr = genI32TiledOp(rewriter, generateI32DppMove, rmwPtr,
                                           0x130, 0xF, 0xF, false);
  Value elemSize = b.i64_val(operandElemType.getIntOrFloatBitWidth() / 8);
  Value isNeighbour = b.icmp_eq(rightNeighbourAddr, b.add(rmwPtr, elemSize));
  Value neighbourFlag = targetInfo.ballot(rewriter, loc, i64_ty, isNeighbour);
  Value numNeighbours =
      b.trunc(i32_ty, generatePopcount64(rewriter, neighbourFlag));
  // Heuristic that atomic_add is optimizated only if the number of
  // neighbouring addresses in a wave is less than 32.
  // TODO: Calculate actual number of difference addresses in a wave.
  Value optAtomic = b.icmp_ult(numNeighbours, b.i32_val(32));

  rewriter.create<LLVM::CondBrOp>(loc, optAtomic, initLoop, atomicBlock,
                                  ValueRange({rmwPtr, operand}));
  rewriter.setInsertionPointToEnd(initLoop);

  auto *afterLoopBlock = initLoop->splitBlock(rewriter.getInsertionPoint());
  afterLoopBlock->addArgument(i32_ty, loc);    // idx
  afterLoopBlock->addArgument(i32_ty, loc);    // cnt
  afterLoopBlock->addArgument(int_ty(1), loc); // isLeader

  auto *loopBody = rewriter.createBlock(initLoop->getParent(),
                                        std::next(Region::iterator(initLoop)));
  loopBody->addArgument(i32_ty, loc);

  rewriter.setInsertionPointToEnd(initLoop);
  rewriter.create<LLVM::BrOp>(loc, b.i32_val(0), loopBody);

  // Greed search of same addr within wavefront. Also collect auxiliary
  // information about relative position:
  // - idx in a group + base laneId. This param is required to form
  // continuous
  //   groups further;
  // - cnt of remaining threads in a group after current thread;
  // - leadership status of the current thread.
  rewriter.setInsertionPointToEnd(loopBody);
  // `readfirstlane` considers only enabled threads
  Value chosen = genI32TiledOp(rewriter, genReadFirstLane, rmwPtr);
  // this flag is required to disable thread if we have already checked its
  // pointer
  Value done = b.icmp_eq(chosen, rmwPtr);
  Value mask = targetInfo.ballot(rewriter, loc, i64_ty, done);
  Value start = loopBody->getArgument(0);
  Value cnt = b.trunc(i32_ty, generatePopcount64(rewriter, mask));
  Value mbcntLoRes = rewriter
                         .create<ROCDL::MbcntLoOp>(
                             loc, i32_ty, b.trunc(i32_ty, mask), b.i32_val(0))
                         ->getResult(0);
  Value idx = rewriter.create<ROCDL::MbcntHiOp>(
      loc, i32_ty, b.trunc(i32_ty, b.lshr(mask, b.i64_val(32))), mbcntLoRes);
  Value base = b.add(start, cnt);
  Value leader = b.icmp_eq(idx, b.i32_val(0));
  cnt = b.sub(cnt, idx);
  idx = b.add(idx, start);
  rewriter.create<LLVM::CondBrOp>(loc, done, afterLoopBlock,
                                  ValueRange({idx, cnt, leader}), loopBody,
                                  ValueRange({base}));

  rewriter.setInsertionPointToEnd(afterLoopBlock);

  Value idxRes = afterLoopBlock->getArgument(0);
  Value cntRes = afterLoopBlock->getArgument(1);
  Value leaderRes = afterLoopBlock->getArgument(2);
  Value idxScaledForPermute = b.mul(idxRes, b.i32_val(4));

  // Make groups continuous
  rmwPtr = genI32TiledOp(rewriter, genPermute, rmwPtr, idxScaledForPermute);
  operand = genI32TiledOp(rewriter, genPermute, operand, idxScaledForPermute);
  // Actualize auxiliary info as well
  Value packedRoleInfo = genI32TiledOp(
      rewriter, genPermute,
      b.or_(b.zext(i32_ty, leaderRes),
            b.or_(idxScaledForPermute, b.shl(cntRes, b.i32_val(8)))),
      idxScaledForPermute);
  idxScaledForPermute = packedRoleInfo;
  cntRes = b.and_(b.lshr(packedRoleInfo, b.i32_val(8)), b.i32_val(0xff));
  leaderRes = b.icmp_ne(b.and_(packedRoleInfo, b.i32_val(1)), b.i32_val(0));

  auto *afterRedBlock =
      afterLoopBlock->splitBlock(rewriter.getInsertionPoint());
  afterRedBlock->addArgument(operandElemType, loc);
  auto *partialReductionBlock = rewriter.createBlock(
      afterLoopBlock->getParent(), std::next(Region::iterator(afterLoopBlock)));
  rewriter.setInsertionPointToEnd(afterLoopBlock);
  Value reductionCond = b.icmp_ne(
      targetInfo.ballot(rewriter, loc, i64_ty, b.icmp_ne(cntRes, b.i32_val(1))),
      b.i64_val(0));
  rewriter.create<LLVM::CondBrOp>(loc, reductionCond, partialReductionBlock,
                                  afterRedBlock, operand);
  rewriter.setInsertionPointToEnd(partialReductionBlock);

  auto performOp = [&](Value res, Value v) -> Value {
    switch (opKind) {
    case LLVM::AtomicBinOp::_and:
      return b.and_(res, v);
    case LLVM::AtomicBinOp::_or:
      return b.or_(res, v);
    case LLVM::AtomicBinOp::_xor:
      return b.xor_(res, v);
    case LLVM::AtomicBinOp::add:
      return b.add(res, v);
    case LLVM::AtomicBinOp::fadd:
      return b.fadd(res, v);
    case LLVM::AtomicBinOp::max:
    case LLVM::AtomicBinOp::umax:
      return b.umax(v, res);
    case LLVM::AtomicBinOp::min:
    case LLVM::AtomicBinOp::umin:
      return b.umin(v, res);
    case LLVM::AtomicBinOp::xchg:
      return v;
    default:
      llvm_unreachable("Unsupported atomic binary operation.");
    }
  };
  Value acc = operand;
  // Reduce to leader thread
  for (int i = 32; i != 0; i /= 2) {
    Value tmp = genI32TiledOp(rewriter, genBPermute, acc,
                              b.add(idxScaledForPermute, b.i32_val(i * 4)));
    acc = b.select(b.icmp_ult(b.i32_val(i), cntRes), performOp(acc, tmp), acc);
  }

  rewriter.create<LLVM::BrOp>(loc, acc, afterRedBlock);
  rewriter.setInsertionPointToEnd(afterRedBlock);

  auto *endBlock = afterRedBlock->splitBlock(rewriter.getInsertionPoint());
  endBlock->addArgument(operandElemType, loc);
  rewriter.setInsertionPointToEnd(afterRedBlock);
  Value leaderCond = leaderRes;
  Value defaultRes = b.undef(operandElemType);
  rewriter.create<LLVM::CondBrOp>(
      loc, leaderCond, atomicBlock,
      ValueRange({rmwPtr, afterRedBlock->getArgument(0)}), endBlock,
      ValueRange({defaultRes}));
  rewriter.setInsertionPointToEnd(atomicBlock);
  // Utilize global atomic only by leader threads
  Value addr = atomicBlock->getArgument(0);
  Value atomAddr = b.inttoptr(origPtrType, addr);
  Value atom = rewriter.create<LLVM::AtomicRMWOp>(
      loc, opKind, atomAddr, atomicBlock->getArgument(1), memOrdering, scope);
  rewriter.create<LLVM::BrOp>(loc, atom, endBlock);
  rewriter.setInsertionPointToStart(endBlock);

  return endBlock->getArgument(0);
}

} // namespace mlir::LLVM::AMD
