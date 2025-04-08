#include "mlir/IR/Dominance.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include <memory>

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUCOMBINETENSORSELECTANDIF
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

/// The user of select maybe inside either the ThenRegion or ElseRegion of
/// the scf.if. So, canonicalize user of select in scf.if first.
static void canonicalizeSelectUsersInSCFIf(ModuleOp input) {
  llvm::MapVector<std::pair<Value, Value>, SmallVector<Operation *>>
      usersNeedreplaced;
  input.walk([&](arith::SelectOp selectOp) {
    auto *parentBlock = selectOp->getBlock();
    Value condition = selectOp.getOperand(0);
    Value trueVal = selectOp.getOperand(1);
    Value falseVal = selectOp.getOperand(2);
    Value resVal = selectOp.getResult();
    for (auto *condUser : condition.getUsers()) {
      if (!llvm::isa<scf::IfOp>(condUser))
        continue;
      scf::IfOp ifOp = llvm::cast<scf::IfOp>(condUser);
      for (auto *resUser : resVal.getUsers()) {
        if (ifOp->isProperAncestor(resUser)) {
          if (ifOp.getThenRegion().findAncestorOpInRegion(*resUser) !=
              nullptr) {
            // The user is inside the ThenRegion of the scf.if.
            usersNeedreplaced[std::make_pair(resVal, trueVal)].push_back(
                resUser);
          } else {
            // The user is inside the ElseRegion of the scf.if.
            usersNeedreplaced[std::make_pair(resVal, falseVal)].push_back(
                resUser);
          }
        }
      }
    }
  });

  // Replace the operand of user.
  for (auto [replacedSrcAndDst, users] :
       llvm::make_early_inc_range(usersNeedreplaced)) {
    Value srcVal = replacedSrcAndDst.first;
    Value dstVal = replacedSrcAndDst.second;
    for (Operation *user : llvm::make_early_inc_range(users)) {
      srcVal.replaceUsesWithIf(
          dstVal, [&](OpOperand &use) { return use.getOwner() == user; });
    }
  }
}

/// Return true if the select could be merged into the If without breaking SSA
/// rules.
static bool canMergeIntoIf(arith::SelectOp selectOp, scf::IfOp ifOp,
                           DominanceInfo &dom) {
  // If needs to be dominated by the select.
  if (!dom.dominates(selectOp.getOperation(), ifOp.getOperation())) {
    return false;
  }
  // If needs to dominate all the select's users.
  for (auto user : selectOp.getResult().getUsers()) {
    if (!dom.dominates(ifOp, user)) {
      return false;
    }
  }
  return true;
}

class CombineTensorSelectAndIfPass
    : public impl::TritonGPUCombineTensorSelectAndIfBase<
          CombineTensorSelectAndIfPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    canonicalizeSelectUsersInSCFIf(m);

    // Go over the arith.select ops, look if there is an if
    // with the same condition.
    DominanceInfo dom(m);
    llvm::MapVector<scf::IfOp, SmallVector<arith::SelectOp>> selectToIf;
    m.walk([&](arith::SelectOp selectOp) {
      // Apply only to selects with a tensor result. Scalars are cheap enough to
      // predicate.
      if (!isa<RankedTensorType>(selectOp.getResult().getType()))
        return;
      // Look if there is an if in the same block, with the same condition.
      auto *parentBlock = selectOp->getBlock();
      Value condition = selectOp.getOperand(0);
      SetVector<Operation *> conditionUsers(condition.getUsers().begin(),
                                            condition.getUsers().end());
      // sort the users in topological order.
      conditionUsers = multiRootTopologicalSort(conditionUsers);
      // Get condition's users
      for (Operation *user : conditionUsers) {
        auto ifOp = dyn_cast<scf::IfOp>(user);
        if (!ifOp || ifOp->getBlock() != parentBlock)
          continue;
        if (canMergeIntoIf(selectOp, ifOp, dom)) {
          selectToIf[ifOp].push_back(selectOp);
          break;
        }
      }
    });

    for (auto [ifOp, selectOps] : selectToIf) {
      // Add new return value to the if (and create else block if necessary),
      // then yield the select value in the then block and the else block.
      OpBuilder builder(ifOp);
      auto loc = ifOp.getLoc();
      // Create an scf::IfOp with extra return value.
      SmallVector<Type> newResultTypes = {ifOp.getResultTypes().begin(),
                                          ifOp.getResultTypes().end()};
      for (arith::SelectOp selectOp : selectOps) {
        newResultTypes.push_back(selectOp.getResult().getType());
      }
      auto newIfOp = builder.create<scf::IfOp>(
          loc, newResultTypes, ifOp.getCondition(), /*hasElse*/ true);
      // Move the existing blocks to the new if.
      newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());

      if (ifOp.elseBlock()) {
        newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
      } else {
        // Create an empty yield
        auto yieldOp = newIfOp.getElseBodyBuilder().create<scf::YieldOp>(loc);
      }

      SmallVector<Value> ifYieldOperands = newIfOp.thenYield().getOperands();
      SmallVector<Value> elseYieldOperands = newIfOp.elseYield().getOperands();
      for (arith::SelectOp selectOp : selectOps) {
        Value thenValue = selectOp.getTrueValue();
        Value elseValue = selectOp.getFalseValue();
        ifYieldOperands.push_back(thenValue);
        elseYieldOperands.push_back(elseValue);
      }
      // Update yields
      auto updateYield = [&](scf::YieldOp yield, SmallVector<Value> &operands) {
        builder.setInsertionPoint(yield);
        builder.create<scf::YieldOp>(loc, operands);
        yield.erase();
      };
      updateYield(newIfOp.thenYield(), ifYieldOperands);
      updateYield(newIfOp.elseYield(), elseYieldOperands);

      int resultIdx = 0;
      // Replace old if with the new one.
      for (auto result : ifOp.getResults()) {
        result.replaceAllUsesWith(newIfOp->getResult(resultIdx++));
      }
      // Replace the select with the new return value.
      for (arith::SelectOp selectOp : selectOps) {
        selectOp.replaceAllUsesWith(newIfOp->getResult(resultIdx++));
        selectOp.erase();
      }

      ifOp.erase();
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
