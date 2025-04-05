#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
//
// This pass works after all other passes, inserting fences to ensure that
// memory operations are properly ordered across generic and async proxy.
//
//===----------------------------------------------------------------------===//

using namespace mlir;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

struct FenceInsertionPass
    : public TritonGPUFenceInsertionBase<FenceInsertionPass> {

public:
  FenceInsertionPass() = default;
  FenceInsertionPass(int computeCapability) {
    this->computeCapability = computeCapability;
  }
  // TODO: support more general patterns to insert fences. eg. any op(generic)
  // to shared in use-def chain which refers by async proxy. We have generic(
  // convertlayout with sts/stmatix) + fence + async(wgmma) up to now
  void runOnOperation() override {
    // Only insert fences for compute capability 9.0
    if (computeCapability < 90)
      return;
    if (::triton::tools::getBoolEnv("DISABLE_MMA_V3"))
      return;
    ModuleOp mod = getOperation();
    DenseSet<std::pair<Operation *, unsigned>> trace;
    mod.walk([&](Operation *op) {
      if (!isa<ttng::WarpGroupDotOp>(op))
        return WalkResult::advance();
      OpBuilder builder(op);
      auto a = op->getOperand(0);
      auto b = op->getOperand(1);
      auto mmaEncoding = dyn_cast<ttg::NvidiaMmaEncodingAttr>(
          cast<RankedTensorType>(op->getResult(0).getType()).getEncoding());
      if (!mmaEncoding || !mmaEncoding.isHopper())
        return WalkResult::advance();
      bool aDependsOnShared = dependOnSharedEncOperand(a, trace);
      bool bDependsOnShared = dependOnSharedEncOperand(b, trace);
      if (!aDependsOnShared && !bDependsOnShared)
        return WalkResult::advance();
      Operation *fence = builder.create<ttng::FenceAsyncSharedOp>(
          op->getLoc(), /*bCluster=*/false);
      // If there is all the dependencies are outside of the loop try to hoist
      // the fence.
      while (auto loopOp = fence->getParentOfType<LoopLikeOpInterface>()) {
        if (aDependsOnShared &&
            loopOp->isAncestor(a.getParentBlock()->getParentOp()))
          break;
        if (bDependsOnShared &&
            loopOp->isAncestor(b.getParentBlock()->getParentOp()))
          break;
        loopOp.moveOutOfLoop(fence);
      }
      return WalkResult::advance();
    });
  }

private:
  bool dependOnSharedEncOperand(Value operand, DenseSet<std::pair<Operation *, unsigned>> &trace) {
    auto op = operand.getDefiningOp();
    // avoid redundant insertion
    if (op && isa<mlir::triton::DotOpInterface>(op))
      return false;
    // reach convertlayout
    if (op && isa<ttg::LocalAllocOp>(op) &&
        cast<ttg::LocalAllocOp>(op).getSrc())
      return true;
    // root and not BlockArgument
    if (!op && !isa<BlockArgument>(operand))
      return false;
    // op and not BlockArgument
    if (op && !isa<BlockArgument>(operand)) {
      for (auto v : op->getOperands()) {
        if (dependOnSharedEncOperand(v, trace))
          return true;
      }
    }
    // reach BlockArgument
    // TODO: support other scf ops, IfOp, WhileOp, etc.
    if (BlockArgument arg = dyn_cast<BlockArgument>(operand)) {
      unsigned argNum = arg.getArgNumber();
      Operation *argOwner = arg.getOwner()->getParentOp();
      // support ForOp only
      if (auto forOp = dyn_cast<scf::ForOp>(argOwner)) {
        // prologue
        auto iterOperands = forOp.getInitArgs();
        if (argNum == 0)
          return false;
        if (dependOnSharedEncOperand(iterOperands[argNum - 1], trace))
          return true;
        // yield
        auto yieldOp = forOp.getBody()->getTerminator();
        Value v = yieldOp->getOperand(argNum - 1);
        auto entry = std::make_pair<Operation *, unsigned>(std::move(yieldOp),
                                                           std::move(argNum));
        // avoid cyclic
        if (trace.contains(entry))
          return false;
        else
          trace.insert(entry);

        if (dependOnSharedEncOperand(v, trace))
          return true;
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(argOwner)) {
        assert(false && "FenceInsertionPass does not supported WhileOp");
      } else if (auto ifOp = dyn_cast<scf::IfOp>(argOwner)) {
        assert(false && "FenceInsertionPass does not supported IfOp");
      }
    }
    return false;
  }
};
} // namespace

std::unique_ptr<Pass>
mlir::createTritonNvidiaGPUFenceInsertionPass(int computeCapability) {
  return std::make_unique<FenceInsertionPass>(computeCapability);
}
