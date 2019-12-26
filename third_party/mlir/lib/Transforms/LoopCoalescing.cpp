//===- LoopCoalescing.cpp - Pass transforming loop nests into single loops-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/Debug.h"

#define PASS_NAME "loop-coalescing"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;

namespace {
class LoopCoalescingPass : public FunctionPass<LoopCoalescingPass> {
public:
  void runOnFunction() override {
    FuncOp func = getFunction();

    func.walk([](loop::ForOp op) {
      // Ignore nested loops.
      if (op.getParentOfType<loop::ForOp>())
        return;

      SmallVector<loop::ForOp, 4> loops;
      getPerfectlyNestedLoops(loops, op);
      LLVM_DEBUG(llvm::dbgs()
                 << "found a perfect nest of depth " << loops.size() << '\n');

      // Look for a band of loops that can be coalesced, i.e. perfectly nested
      // loops with bounds defined above some loop.
      // 1. For each loop, find above which parent loop its operands are
      // defined.
      SmallVector<unsigned, 4> operandsDefinedAbove(loops.size());
      for (unsigned i = 0, e = loops.size(); i < e; ++i) {
        operandsDefinedAbove[i] = i;
        for (unsigned j = 0; j < i; ++j) {
          if (areValuesDefinedAbove(loops[i].getOperands(),
                                    loops[j].region())) {
            operandsDefinedAbove[i] = j;
            break;
          }
        }
        LLVM_DEBUG(llvm::dbgs()
                   << "  bounds of loop " << i << " are known above depth "
                   << operandsDefinedAbove[i] << '\n');
      }

      // 2. Identify bands of loops such that the operands of all of them are
      // defined above the first loop in the band.  Traverse the nest bottom-up
      // so that modifications don't invalidate the inner loops.
      for (unsigned end = loops.size(); end > 0; --end) {
        unsigned start = 0;
        for (; start < end - 1; ++start) {
          auto maxPos =
              *std::max_element(std::next(operandsDefinedAbove.begin(), start),
                                std::next(operandsDefinedAbove.begin(), end));
          if (maxPos > start)
            continue;

          assert(maxPos == start &&
                 "expected loop bounds to be known at the start of the band");
          LLVM_DEBUG(llvm::dbgs() << "  found coalesceable band from " << start
                                  << " to " << end << '\n');

          auto band =
              llvm::makeMutableArrayRef(loops.data() + start, end - start);
          coalesceLoops(band);
          break;
        }
        // If a band was found and transformed, keep looking at the loops above
        // the outermost transformed loop.
        if (start != end - 1)
          end = start + 1;
      }
    });
  }
};

} // namespace

std::unique_ptr<OpPassBase<FuncOp>> mlir::createLoopCoalescingPass() {
  return std::make_unique<LoopCoalescingPass>();
}

static PassRegistration<LoopCoalescingPass>
    reg(PASS_NAME,
        "coalesce nested loops with independent bounds into a single loop");
