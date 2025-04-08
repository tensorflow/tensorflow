#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/Debug.h"

// InThreadTranspose pass optimizes inefficient
// tt.load->ttg.local_store->ttg.local_load chains.
//
// For details please look pass description in
// TritonAMDGPUTransforms/Passes.td

#define DEBUG_TYPE "tritonamdgpu-in-thread-transpose"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttag = mlir::triton::amdgpu;

namespace {

static Type replaceEncoding(Type type, Attribute encoding) {
  RankedTensorType tensorType = cast<RankedTensorType>(type);
  return RankedTensorType::get(tensorType.getShape(),
                               tensorType.getElementType(), encoding);
}

/// Replace load encoding with given one.
///
/// This functions converts load inputs to given one
/// and replaces old load with new:
///
///   %load_val = tt.load %addr : #blocked
///
/// converts to:
///
///   %addr_new = ttg.convert_layout %addr : #blocked -> #new_blocked
///   %load_val_new = tt.load %addr_new : #new_blocked
///   %load_val = ttg.convert_layout %load_val_new : #new_blocked -> #blocked
///
/// \param rewriter
/// \param encoding new encoding
/// \param load tt.load operation to replace
void refineGlobalLoadLayout(PatternRewriter &rewriter, Attribute encoding,
                            tt::LoadOp load) {
  auto loc = load->getLoc();
  rewriter.setInsertionPoint(load);
  // Convert operands
  SmallVector<Value, 4> newArgs;
  for (auto operand : load->getOperands()) {
    auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
    if (tensorType) {
      Type newType = replaceEncoding(tensorType, encoding);
      newArgs.push_back(
          rewriter.create<ttg::ConvertLayoutOp>(loc, newType, operand));
    } else {
      newArgs.push_back(operand);
    }
  }

  // Construct new load with the new encoding
  auto attrs = load->getAttrs();
  auto newLoad = rewriter.create<tt::LoadOp>(loc, newArgs, attrs);

  // Cast the results back to the original layout
  auto loadType = load.getType();
  Value newResult = newLoad.getResult();
  rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(load, loadType, newResult);
}

void transposeInRegsitersBeforeStoreInLocalMemory(
    PatternRewriter &rewriter, Operation *memStoreOp,
    ArrayRef<int64_t> loadShape, ttg::BlockedEncodingAttr newLoadEncoding) {
  assert((mlir::isa<ttg::LocalAllocOp, ttg::LocalStoreOp>(memStoreOp)));
  // skip local_alloc with zero arguments
  if (memStoreOp->getNumOperands() == 0)
    return;
  auto data = memStoreOp->getOperand(0);
  rewriter.setInsertionPoint(memStoreOp);

  auto transposedLayout =
      ttag::InThreadTransposeOp::deduceOutputLayout(loadShape, newLoadEncoding);
  auto transposedEncoding =
      ttg::LinearEncodingAttr::get(memStoreOp->getContext(), transposedLayout);

  auto loc = memStoreOp->getLoc();
  auto newLoadType = replaceEncoding(data.getType(), newLoadEncoding);
  auto nonTransposed =
      rewriter.create<ttg::ConvertLayoutOp>(loc, newLoadType, data);

  auto transposedType = replaceEncoding(data.getType(), transposedEncoding);
  auto inThreadTransposed = rewriter.create<ttag::InThreadTransposeOp>(
      loc, transposedType, nonTransposed);
  rewriter.startOpModification(memStoreOp);
  memStoreOp->setOperand(0, inThreadTransposed);
  rewriter.finalizeOpModification(memStoreOp);
}

Attribute createNewSharedEncoding(RankedTensorType operandType) {
  auto ctx = operandType.getContext();
  auto dotOperandEnc =
      cast<ttg::DotOperandEncodingAttr>(operandType.getEncoding());
  auto ctaLayout = ttg::getCTALayout(dotOperandEnc);
  auto bitWidth = operandType.getElementTypeBitWidth();
  SmallVector<unsigned> order{1, 0};
  if (dotOperandEnc.getOpIdx() == 1)
    std::swap(order[0], order[1]);

  auto tempAttr = ttg::SwizzledSharedEncodingAttr::get(
      ctx, dotOperandEnc, operandType.getShape(), order, ctaLayout, bitWidth,
      /*needTrans=*/false);

  auto sharedVec = tempAttr.getVec();
  auto perPhase = tempAttr.getPerPhase();
  auto maxPhase = tempAttr.getMaxPhase();

  auto newSharedEnc = ttg::AMDRotatingSharedEncodingAttr::get(
      ctx, sharedVec, perPhase, maxPhase, order, ctaLayout);

  return newSharedEnc;
}

void changeSharedEncoding(PatternRewriter &rewriter, Value memVal,
                          Attribute newEncoding) {
  auto originalType = cast<ttg::MemDescType>(memVal.getType());
  auto sharedEnc =
      dyn_cast<ttg::SwizzledSharedEncodingAttr>(originalType.getEncoding());
  // Already transformed this value
  if (!sharedEnc)
    return;

  auto newType = ttg::MemDescType::get(
      originalType.getShape(), originalType.getElementType(), newEncoding,
      originalType.getMemorySpace(), originalType.getMutableMemory());
  auto parentOp = memVal.getParentBlock()->getParentOp();
  rewriter.startOpModification(parentOp);
  memVal.setType(newType);
  rewriter.finalizeOpModification(parentOp);
}

/// Structure describes operations involved in tt.load -> ttg.local_store op
/// chain
struct GlobalToSharedMemoryOpChain {
  SmallVector<tt::LoadOp> globalLoads;
  // list of localAllocOp and localStoreOp operations
  SmallVector<Operation *> localAllocStores;
  // list of MemDescSubviewOp, control flow results and block operands
  SmallVector<Value> sharedMemVals;
};

FailureOr<SmallVector<Value>> traverseCFForValueDefs(Value val);

FailureOr<SmallVector<Value>> traverseForOpForDefs(scf::ForOp forOp,
                                                   int argIdx) {
  int iterArgIdx = argIdx - 1; // Skip induction variable
  if (iterArgIdx >= 0) {
    Value yieldVal = forOp.getBody()->getTerminator()->getOperand(iterArgIdx);
    // look inside of a loop
    auto inLoop = traverseCFForValueDefs(yieldVal);
    // look outside of a loop
    int forOpArgIdx = iterArgIdx + forOp.getNumControlOperands();
    auto outLoop = traverseCFForValueDefs(forOp.getOperand(forOpArgIdx));
    if (failed(inLoop) || failed(outLoop))
      return failure();

    SmallVector<Value> foundDefs = inLoop.value();
    foundDefs.append(outLoop.value());
    return foundDefs;
  } else {
    // Induction variable
    auto search = traverseCFForValueDefs(forOp.getOperand(0));
    if (failed(search))
      return failure();
    return search.value();
  }
}

FailureOr<SmallVector<Value>> traverseIfOpForDefs(scf::IfOp ifOp, int argIdx) {
  auto thenYield = ifOp.thenYield();
  auto elseYield = ifOp.elseYield();

  SmallVector<Value> foundDefs;
  // Track all possible yielded values from then/else blocks
  if (thenYield) {
    auto ops = traverseCFForValueDefs(thenYield->getOperand(argIdx));
    if (failed(ops))
      return failure();
    foundDefs.append(ops.value());
  }
  if (elseYield) {
    auto ops = traverseCFForValueDefs(elseYield->getOperand(argIdx));
    if (failed(ops))
      return failure();
    foundDefs.append(ops.value());
  }
  return foundDefs;
}

FailureOr<SmallVector<Value>> traverseWhileOpForDefs(scf::WhileOp whileOp,
                                                     int argIdx) {
  auto terminator = whileOp.getBefore().front().getTerminator();
  auto search = traverseCFForValueDefs(terminator->getOperand(argIdx));
  if (failed(search))
    return failure();
  SmallVector<Value> foundDefs = search.value();
  search = traverseCFForValueDefs(whileOp.getInits()[argIdx]);
  if (failed(search))
    return failure();
  foundDefs.append(search.value());
  return foundDefs;
}

FailureOr<SmallVector<Value>>
traverseRegionBranchOpForDefs(RegionBranchOpInterface regionBranch,
                              int argIdx) {
  // Deal with the case that convert_layout intakes from scf.if, etc.
  llvm::SmallVector<scf::YieldOp> yieldOps;
  regionBranch->walk([&](Operation *op) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      yieldOps.push_back(yieldOp);
    }
  });

  SmallVector<Value> foundDefs;
  for (auto yieldOp : yieldOps) {
    auto ops = traverseCFForValueDefs(yieldOp->getOperand(argIdx));
    if (failed(ops))
      return failure();
    foundDefs.append(ops.value());
  }
  return foundDefs;
}

/// For a given value, traverse the control flow graph yield structure to find
/// all initial source operations.
///
/// If val is a result of operation, return definingOp.
/// If val is a result of some control flow operation or block argument,
/// traverse control flow instructions.
FailureOr<SmallVector<Value>> traverseCFForValueDefs(Value val) {
  LDBG("    traverseCFForValueDefs processing " << val);
  auto attachValue =
      [val](
          FailureOr<SmallVector<Value>> res) -> FailureOr<SmallVector<Value>> {
    if (failed(res))
      return failure();
    SmallVector<Value> result(std::move(res.value()));
    result.push_back(val);
    return result;
  };

  // traverse inside CFG operation
  if (auto regionBranch = val.getDefiningOp<RegionBranchOpInterface>()) {
    auto resId = cast<OpResult>(val).getResultNumber();
    return attachValue(traverseRegionBranchOpForDefs(regionBranch, resId));
  }

  // if val is not a CFG op and not a block argument, it is a "normal" operation
  if (!isa<BlockArgument>(val)) {
    return SmallVector<Value>{val};
  }
  auto blockArg = dyn_cast<BlockArgument>(val);
  Block *block = blockArg.getOwner();

  // Get parent operation (e.g., scf.for, scf.if, scf.while)
  Operation *parentOp = block->getParentOp();
  if (!parentOp) {
    LDBG("    block without parent op, can not analyze further");
    return failure();
  }

  // If block belongs to a function, stop tracking (function arguments)
  if (isa<tt::FuncOp>(parentOp)) {
    LDBG("    can not traverse def-use chains, found function argument");
    return failure();
  }

  // Traverse outside CFG operations
  int argIdx = blockArg.getArgNumber();
  if (auto forOp = dyn_cast<scf::ForOp>(parentOp))
    return attachValue(traverseForOpForDefs(forOp, argIdx));
  if (auto ifOp = dyn_cast<scf::IfOp>(parentOp))
    return attachValue(traverseIfOpForDefs(ifOp, argIdx));
  if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp))
    return attachValue(traverseWhileOpForDefs(whileOp, argIdx));

  LDBG("    can not traverse def-use chains, unsupported control flow "
       "operation");
  return failure();
}

struct ForwardSearchAnalysis {
  SmallVector<Operation *> ops;
  SmallVector<Value> transitiveCF;
};

/// For a given value return all operations that uses it.
///
/// Traverses control flow instructions forward.
FailureOr<ForwardSearchAnalysis> traverseCFForValueUses(Value val) {
  LDBG("    traverseCFForValueUses processing " << val);
  ForwardSearchAnalysis result;
  for (auto &use : val.getUses()) {
    auto user = use.getOwner();
    LDBG("      processing user " << *user);
    if (isa<tt::ReturnOp>(user)) {
      LDBG("    Reached return from function");
      return failure();
    }
    if (isa<scf::YieldOp>(user)) {
      auto opIdx = use.getOperandNumber();
      auto parent = user->getParentOp();
      // traverse outside data flow
      auto parentResult = parent->getResult(opIdx);
      auto cfSearch = traverseCFForValueUses(parentResult);
      if (failed(cfSearch))
        return failure();
      result.ops.append(cfSearch.value().ops);
      result.transitiveCF.push_back(parentResult);
      result.transitiveCF.append(cfSearch.value().transitiveCF);

      // traverse loop internal data flow
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        int forBodyOperandIdx = opIdx + forOp.getNumInductionVars();
        auto blockArg = forOp.getBody()->getArgument(forBodyOperandIdx);
        auto cfSearch = traverseCFForValueUses(blockArg);
        if (failed(cfSearch))
          return failure();
        result.ops.append(cfSearch.value().ops);
        result.transitiveCF.push_back(blockArg);
        result.transitiveCF.append(cfSearch.value().transitiveCF);
      }
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      LDBG("      for op num operands: " << forOp.getNumOperands());
      LDBG("      for op body num operands: "
           << forOp.getBody()->getNumArguments());
      assert(use.getOperandNumber() >= forOp.getNumControlOperands());
      int blockArgIdx = use.getOperandNumber() - forOp.getNumControlOperands() +
                        forOp.getNumInductionVars();
      auto blockArg = forOp.getBody()->getArgument(blockArgIdx);
      auto cfSearch = traverseCFForValueUses(blockArg);
      if (failed(cfSearch))
        return failure();
      result.ops.append(cfSearch.value().ops);
      result.transitiveCF.push_back(blockArg);
      result.transitiveCF.append(cfSearch.value().transitiveCF);
      continue;
    }
    if (isa<scf::SCFDialect>(user->getDialect())) {
      LDBG("    can not traverse def-use chains, unsupported control flow "
           "operation");
      return failure();
    }
    result.ops.push_back(user);
  }
  return result;
}

/// Look for defining operation, hopping over control flow.
///
/// Gather all operations of type T within one def-use hop from val,
/// control flow constructions are not considered as an operations.
/// \returns true on success, false if analysis failed
template <typename Op>
FailureOr<SmallVector<Op>> findAllDefiningOps(Value val) {
  auto candidates = traverseCFForValueDefs(val);
  if (failed(candidates))
    return failure();
  SmallVector<Op> result;
  for (auto candidateValue : candidates.value()) {
    auto op = candidateValue.getDefiningOp();
    if (!op)
      continue;
    if (auto typedOp = dyn_cast<Op>(op))
      result.push_back(typedOp);
  }
  return result;
}

/// Find all shared mem related operations reachable from given ttg.local_load
/// along shared memory data flow.
///
/// Traversal bypasses control flow operations.
///
/// Example of found operation network:
///
/// ttg.local_alloc -----x-------------------------> ttg.local_dealloc
///                      V
/// tt.load -> ttg.local_store -> ttg.mem_subview -> ttg.local_load
///
/// \returns partially filled GlobalToSharedMemoryOpChain structure of failure.
FailureOr<GlobalToSharedMemoryOpChain>
findReachableSMemOps(ttg::LocalLoadOp root,
                     SetVector<mlir::Operation *> &visited) {
  // breadth-first search for reachable opeations
  GlobalToSharedMemoryOpChain foundNetwork;
  SmallVector<Operation *> traversalStep{root};
  while (!traversalStep.empty()) {
    LDBG("begin new step in smem op analysis");
    SmallVector<Operation *> nextTraversalStep;
    for (auto candidate : traversalStep) {
      if (visited.contains(candidate))
        continue;
      visited.insert(candidate);
      LDBG("  processing in smem op analysis: " << *candidate);

      // Each smem operation could have at most 1 result and at most 1 memory
      // operand smemOperand is a smem operand of "candidate" operation
      // smemOutput is smem output of "candidate" operation
      Value smemOperand;
      Value smemOutput;
      if (isa<ttg::LocalAllocOp>(candidate)) {
        foundNetwork.localAllocStores.push_back(candidate);
        smemOutput = candidate->getResult(0);
      } else if (isa<ttg::LocalStoreOp>(candidate)) {
        foundNetwork.localAllocStores.push_back(candidate);
        smemOperand = candidate->getOperand(1);
      } else if (isa<ttg::MemDescSubviewOp>(candidate)) {
        smemOutput = candidate->getResult(0);
        smemOperand = candidate->getOperand(0);
      } else if (isa<ttg::LocalLoadOp, ttg::LocalDeallocOp>(candidate)) {
        smemOperand = candidate->getOperand(0);
      } else {
        // this operation is not part of shared memory def-use network,
        // algorithm should not reach this point
        LDBG("  catched operation unrelated to shared memory" << *candidate);
        // this is critical error, assert in debug mode.
        assert(false && "  catched operation unrelated to shared memory");
        return failure();
      }

      // Look backward
      if (smemOperand) {
        auto backwardSearch = traverseCFForValueDefs(smemOperand);
        if (failed(backwardSearch))
          return failure();
        for (auto def : backwardSearch.value()) {
          foundNetwork.sharedMemVals.push_back(def);
          if (Operation *op = def.getDefiningOp()) {
            // additional check, to ignore control flow operations
            if (isa<ttg::MemDescSubviewOp, ttg::LocalAllocOp>(op))
              nextTraversalStep.push_back(op);
          }
        }
      }

      // Look forward
      if (smemOutput) {
        auto forwardSearch = traverseCFForValueUses(smemOutput);
        if (failed(forwardSearch))
          return failure();
        foundNetwork.sharedMemVals.append(forwardSearch.value().transitiveCF);
        nextTraversalStep.append(forwardSearch.value().ops);
      }
    }
    traversalStep = std::move(nextTraversalStep);
  }
  return foundNetwork;
}

unsigned getMaxSizePerThread(RankedTensorType type, int dimIdx) {
  auto loadEnc = type.getEncoding();
  auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(loadEnc);
  if (!blockedEnc)
    return 0;
  auto lanes = blockedEnc.getThreadsPerWarp();
  auto warps = blockedEnc.getWarpsPerCTA();
  auto shape = type.getShape();
  int maxSize = shape[dimIdx] / (lanes[dimIdx] * warps[dimIdx]);
  return std::max(1, maxSize);
}

// Looking for def-use network of following kind:
// ttg.local_alloc ---x
//                    |
//                    V
// tt.load --> ttg.local_store --> ttg.memdesc_subview --> ttg.local_load
//
// Actual network could vary, because of different control flow,
// optional ttg.memdesc_subview and ttg.local_store operations.
//
// If data flow pattern match, check applicability
// of inThreadTrasnpose optimization and return found pattern.
llvm::FailureOr<GlobalToSharedMemoryOpChain>
matchInThreadTransposePattern(ttg::LocalLoadOp lLoad) {
  auto opTensorTy = cast<RankedTensorType>(lLoad.getType());
  auto opEnc = opTensorTy.getEncoding();
  auto opDotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(opEnc);
  if (!opDotOpEnc)
    return failure();

  int kDimNum = opDotOpEnc.getOpIdx() == 0 ? 1 : 0;
  // TODO: support wmma
  if (!isa<ttg::AMDMfmaEncodingAttr, ttg::AMDWmmaEncodingAttr>(
          opDotOpEnc.getParent())) {
    LDBG("Operand's parent encoding is not MFMA");
    return failure();
  }

  SetVector<Operation *> visited;
  // find local_alloc, local_store, local_load and ttg.memdesc_subview
  // operations
  auto sharedMemSearch = findReachableSMemOps(lLoad, visited);
  if (failed(sharedMemSearch)) {
    LDBG("Failed to traverse shared memmory operation network");
    return failure();
  }
  auto pattern = sharedMemSearch.value();

  if (pattern.localAllocStores.empty()) {
    LDBG("Did not find local alloc or store operations");
    return failure();
  }

  for (auto localMemStore : pattern.localAllocStores) {
    LDBG("processing local mem store operation: " << *localMemStore);
    // check if it is a local alloc with no predecessor
    if (localMemStore->getNumOperands() == 0)
      continue;
    Value loadCandidate = localMemStore->getOperand(0);
    auto loadedEnc =
        cast<RankedTensorType>(loadCandidate.getType()).getEncoding();
    auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(loadedEnc);
    if (!blockedEnc)
      return failure();
    auto order = blockedEnc.getOrder();
    if (order[0] == kDimNum) {
      return failure();
    }
    auto globalLoadSearch = findAllDefiningOps<tt::LoadOp>(loadCandidate);
    if (failed(globalLoadSearch)) {
      LDBG("Failed to traverse path to global loads");
      return failure();
    }
    pattern.globalLoads.append(globalLoadSearch.value());
  }
  if (pattern.globalLoads.empty()) {
    LDBG("Did not find global load operation");
    return failure();
  }
  // check that all global loads have same type(i.e. shape and layout),
  // otherwise can not guarantee transformation overhead is cheap
  auto expectedLoadType =
      cast<RankedTensorType>(pattern.globalLoads.front().getResult().getType());
  // TODO support non 2d tensors:
  // in_thread_transpose operation and getTransposableBlockedEnc function
  // are limited to 2d tensors
  if (expectedLoadType.getRank() != 2)
    return failure();

  auto kDimMaxSizePerThread = getMaxSizePerThread(expectedLoadType, kDimNum);
  // kDimRepeats == 0 means loadType has unexpected layout
  // kDimRepeats == 1 means there are no room in k dimension in layout to
  // transpose in registers
  if (kDimMaxSizePerThread < 2) {
    LDBG("Can not extend load layout");
    return failure();
  }

  for (auto load : pattern.globalLoads) {
    if (load->getResult(0).getType() != expectedLoadType) {
      LDBG("Mismatch between global loads result types");
      return failure();
    }
  }

  // TODO implement general heuristic,
  // analyzing local load/store vectorization and estimating bank conflicts?

  return pattern;
}

/// Extends global load layout sizePerThread across k dimension, so it could be
/// transposed in registers.
///
/// Consider 2d dot operand idx = 1(i.e. kDim idx = 0), and global load layout
/// is n-continous:
///   #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA
///   = [1, 1], order = [1, 0]}>
/// Possible output is:
///   #ttg.blocked<{sizePerThread = [4, 8], threadsPerWarp = [8, 8], warpsPerCTA
///   = [1, 1], order = [1, 0]}>
///
/// Consider 2d dot operand idx = 0(i.e. kDim idx = 1), global load layout is
/// m-continous:
///   #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA
///   = [1, 1], order = [0, 1]}>
/// Possible output is:
///   #ttg.blocked<{sizePerThread = [8, 8], threadsPerWarp = [8, 8], warpsPerCTA
///   = [1, 1], order = [0, 1]}>
///
/// Number of elements added across K dimension is limited by tensor dtype bit
/// width and shape across K
ttg::BlockedEncodingAttr getTransposableBlockedEnc(int dotOperandIdx,
                                                   RankedTensorType loadType) {
  // get the K dim according to dotOp operand's index
  auto shape = loadType.getShape();
  int kDimNum = dotOperandIdx == 0 ? 1 : 0;
  // get the current blocked encoding
  auto loadEnc = loadType.getEncoding();
  auto blockedEnc = cast<ttg::BlockedEncodingAttr>(loadEnc);

  auto maxkDimSizePerThread = getMaxSizePerThread(loadType, kDimNum);
  auto elemBitwidth = loadType.getElementType().getIntOrFloatBitWidth();
  // Current the widest is set to ds_write_b64
  // In some cases b64 works best, in others 128
  // TODO introduce a heuristic
  const unsigned dsBitWidth = 64;
  auto newKDimSize = std::min(maxkDimSizePerThread, dsBitWidth / elemBitwidth);
  LDBG("Choose the minimum of numIters: " << newKDimSize << " and numElements: "
                                          << dsBitWidth / elemBitwidth);
  SmallVector<unsigned> newSizePerThread{blockedEnc.getSizePerThread()};
  newSizePerThread[kDimNum] = newKDimSize;

  // return the new blocked encoding
  auto order = blockedEnc.getOrder();
  auto ctx = blockedEnc.getContext();
  auto numWarps = product(blockedEnc.getWarpsPerCTA());
  auto threadsPerWarp = product(blockedEnc.getThreadsPerWarp());
  auto numCTAs = product(blockedEnc.getCTALayout().getCTAsPerCGA());
  return ttg::BlockedEncodingAttr::get(ctx, shape, newSizePerThread, order,
                                       numWarps, threadsPerWarp, numCTAs);
}

class InThreadTransposePattern : public OpRewritePattern<ttg::LocalLoadOp> {
public:
  InThreadTransposePattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(ttg::LocalLoadOp localLoad,
                                PatternRewriter &rewriter) const override {
    LDBG("Consider " << localLoad);
    auto matchResult = matchInThreadTransposePattern(localLoad);
    if (!llvm::succeeded(matchResult)) {
      LDBG("Failed to match InThreadTranspose pattern and nothing to be "
           "done");
      return failure();
    }
    auto pattern = matchResult.value();

    auto dotOpEnc =
        cast<ttg::DotOperandEncodingAttr>(localLoad.getType().getEncoding());

    LDBG("Adjusting global loads");
    RankedTensorType loadResultType =
        cast<RankedTensorType>(pattern.globalLoads[0].getResult().getType());
    auto newBlockedEnc =
        getTransposableBlockedEnc(dotOpEnc.getOpIdx(), loadResultType);
    auto loadShape = loadResultType.getShape();

    for (auto gLoad : pattern.globalLoads) {
      LDBG("operand newBlockedEnc = " << newBlockedEnc);
      refineGlobalLoadLayout(rewriter, newBlockedEnc, gLoad);
    }

    LDBG("Inserting transpose in registers before store in LDS");
    for (auto memOp : pattern.localAllocStores)
      transposeInRegsitersBeforeStoreInLocalMemory(rewriter, memOp, loadShape,
                                                   newBlockedEnc);

    LDBG("Adjust shared encoding");
    auto newSharedEncoding =
        createNewSharedEncoding(cast<RankedTensorType>(localLoad.getType()));
    for (auto memVal : pattern.sharedMemVals)
      changeSharedEncoding(rewriter, memVal, newSharedEncoding);
    return success();
  }
};

} // namespace

class TritonAMDGPUInThreadTransposePass
    : public TritonAMDGPUInThreadTransposeBase<
          TritonAMDGPUInThreadTransposePass> {

public:
  TritonAMDGPUInThreadTransposePass() = default;

  void runOnOperation() override {
    tt::FuncOp f = getOperation();

    auto ctx = f.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<::InThreadTransposePattern>(ctx, /*benefit=*/1);
    walkAndApplyPatterns(f, std::move(patterns));
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUInThreadTransposePass() {
  return std::make_unique<TritonAMDGPUInThreadTransposePass>();
}
