#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUREORDERINSTRUCTIONS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

static bool willIncreaseRegisterPressure(Operation *op) {
  if (isa<triton::gpu::LocalLoadOp>(op))
    return true;
  auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op);
  if (!cvt)
    return false;
  if (mlir::isa<triton::gpu::DotOperandEncodingAttr>(
          cvt.getType().getEncoding()))
    return true;
  return false;
}

class TritonGPUReorderInstructionsPass
    : public impl::TritonGPUReorderInstructionsBase<
          TritonGPUReorderInstructionsPass> {
public:
  TritonGPUReorderInstructionsPass() = default;

  Operation *getFirstUse(Operation *op) {
    std::vector<Operation *> users;
    for (auto user : op->getUsers()) {
      if (Operation *ancestor = op->getBlock()->findAncestorOpInBlock(*user))
        users.push_back(ancestor);
    }
    auto minOpIt =
        llvm::min_element(users, [](mlir::Operation *a, mlir::Operation *b) {
          return a->isBeforeInBlock(b);
        });
    return minOpIt != users.end() ? *minOpIt : nullptr;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    mlir::DominanceInfo dom(m);
    // sink conversion after the last dealloc
    // before the first use ancestor in its block
    m.walk([&](triton::gpu::ConvertLayoutOp op) {
      auto curr = mlir::Block::iterator(op);
      auto end = op->getBlock()->end();
      for (; curr != end && &*curr != getFirstUse(op); curr++)
        if (isa<triton::gpu::LocalDeallocOp>(&*curr))
          op->moveAfter(&*curr);
    });
    // Sink conversions into loops when they will increase
    // register pressure
    DenseMap<Operation *, Operation *> opToMove;
    auto moveAfter = [](Operation *lhs, Operation *rhs) {
      lhs->moveAfter(rhs);
    };
    m.walk([&](Operation *op) {
      if (!willIncreaseRegisterPressure(op))
        return;
      auto user_begin = op->user_begin();
      auto user_end = op->user_end();
      if (std::distance(user_begin, user_end) != 1)
        return;
      if (user_begin->getParentOfType<scf::ForOp>() ==
          op->getParentOfType<scf::ForOp>())
        return;
      opToMove.insert({op, *user_begin});
    });
    for (auto &kv : opToMove)
      kv.first->moveBefore(kv.second);
    // Move alloc(load) immediately after dependent load
    m.walk([&](triton::gpu::LocalAllocOp op) {
      if (!op.getSrc())
        return;
      Operation *argOp = op.getSrc().getDefiningOp();
      if (!argOp)
        return;
      moveAfter(op, argOp);
    });
    // Move transpositions just after their definition
    opToMove.clear();
    m.walk([&](triton::TransposeOpInterface op) {
      Operation *argOp = op.getSrc().getDefiningOp();
      if (!argOp)
        return;
      moveAfter(op, argOp);
    });
    // Move `dot` operand so that conversions to opIdx=1 happens after
    // conversions to opIdx=0
    m.walk([&](triton::gpu::LocalLoadOp op) {
      auto dstEncoding = mlir::dyn_cast<triton::gpu::DotOperandEncodingAttr>(
          op.getType().getEncoding());
      if (!dstEncoding)
        return;
      int opIdx = dstEncoding.getOpIdx();
      if (opIdx != 1)
        return;
      if (!op->hasOneUse())
        return;
      auto dotUser = dyn_cast<triton::DotOp>(*op->user_begin());
      if (!dotUser)
        return;
      auto AOp =
          dotUser.getOperand(0).getDefiningOp<triton::gpu::LocalLoadOp>();
      if (!AOp)
        return;
      // Check that the conversion to OpIdx=1 happens before and can be moved
      // after the conversion to OpIdx=0.
      if (!dom.dominates(op.getOperation(), AOp.getOperation()))
        return;
      moveAfter(op, AOp);
    });
    return;
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
