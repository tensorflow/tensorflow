#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/Support/Debug.h"
#include <queue>

namespace mlir {
namespace triton {
namespace gpu {

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_TRITONGPUFUSENESTEDLOOPS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

// This attribute is set by the front-end to control whether fusion is on.
static constexpr llvm::StringLiteral kFlattenAttr = "tt.flatten";
// This attribute indicates the inner loop length has been speculated.
static constexpr llvm::StringLiteral kMustExecuteAttrName = "ttg.must-execute";
// This attribute is just used for testing the pass.
static constexpr llvm::StringLiteral kAlwaysFuseAttrName = "ttg.always-fuse";

namespace {
struct FuseNestedLoopsPass
    : public impl::TritonGPUFuseNestedLoopsBase<FuseNestedLoopsPass> {
  using TritonGPUFuseNestedLoopsBase::TritonGPUFuseNestedLoopsBase;

  void runOnOperation() override;
};

//===----------------------------------------------------------------------===//
// LoopNest
//===----------------------------------------------------------------------===//

// A node in the loop nest represents a single for loop with a list of
// immediately nested loops.
struct LoopNestNode {
  LoopNestNode(scf::ForOp loop) : loop(loop) {}

  // The for loop.
  scf::ForOp loop;
  // Loops nested immediately below this loop.
  SmallVector<LoopNestNode *, 1> children;
};

// A loop nest is a tree of loops.
struct LoopNest {
  LoopNest(scf::ForOp outermost);

  // Print the loop nest.
  void print(raw_ostream &os) const;
  // Dump the loop nest for debugging.
  LLVM_DUMP_METHOD void dump() const;

  // Owner of the memory of the nodes.
  SmallVector<std::unique_ptr<LoopNestNode>> nodes;

  // The outermost loop in the nest, which has no preconditions. Even if the
  // outermost loop is contained within an if, its preconditions relative to the
  // loop nest are empty.
  LoopNestNode *root;
};
} // namespace

LoopNest::LoopNest(scf::ForOp outermost)
    : root(
          nodes.emplace_back(std::make_unique<LoopNestNode>(outermost)).get()) {
}

void LoopNest::print(raw_ostream &os) const {
  // Print just the first line of the loop's textual IR.
  std::string buffer;
  auto printLoopFirstLine = [&](scf::ForOp loop) {
    buffer.clear();
    llvm::raw_string_ostream str(buffer);
    loop.print(str);
    os << buffer.substr(0, buffer.find('\n'));
  };

  os << "LoopNest:\n";
  SmallVector<std::pair<LoopNestNode *, unsigned>> stack;
  stack.emplace_back(root, 0);
  while (!stack.empty()) {
    auto [node, indent] = stack.pop_back_val();

    // Print the current loop.
    os << std::string(indent * 2, ' ');
    printLoopFirstLine(node->loop);
    os << "\n";

    // Push the children of the current loop.
    for (LoopNestNode *child : node->children)
      stack.emplace_back(child, indent + 1);
  }
  os << "\n";
}

void LoopNest::dump() const { print(llvm::dbgs()); }

//===----------------------------------------------------------------------===//
// findLoopNests
//===----------------------------------------------------------------------===//

// Forward declaration.
static void findLoopNests(Operation *container,
                          SmallVectorImpl<LoopNest> &nests);

// Recursively construct a loop nest.
static void constructLoopNest(LoopNestNode *parent, LoopNest &nest,
                              SmallVectorImpl<LoopNest> &nests) {
  parent->loop->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (op == parent->loop)
      return WalkResult::advance();

    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      auto &child =
          nest.nodes.emplace_back(std::make_unique<LoopNestNode>(forOp));
      parent->children.push_back(child.get());
      // Recurse with the current loop nest.
      constructLoopNest(child.get(), nest, nests);
      return WalkResult::skip();
    }

    // If the traversal encounters any other operation with regions, restart the
    // traversal and construct new loop nests. This means ops like `scf.while`
    // divide the analysis domain, but it also means loop fusion won't "see"
    // across `scf.if`, for example.
    // TODO: Handle loop nests with preconditions. The traversal can keep a
    // stack of `scf.if` preconditions while constructing the loop nest.
    if (op->getNumRegions()) {
      findLoopNests(op, nests);
      return WalkResult::skip();
    }

    return WalkResult::advance();
  });
}

// Find all the loop nests in the operation. The only region operation that
// allows CFG regions is `tt.func`. That means we can just walk starting from
// the function body and can build loop nests directly off the region trees
// contained in the function -- we don't have to worry about CFGs inside the
// nested region trees.
static void findLoopNests(Operation *container,
                          SmallVectorImpl<LoopNest> &nests) {
  container->walk<mlir::WalkOrder::PreOrder>([&](scf::ForOp loop) {
    LoopNest nest(loop);
    constructLoopNest(nest.root, nest, nests);
    nests.push_back(std::move(nest));
    return WalkResult::skip();
  });
}

//===----------------------------------------------------------------------===//
// Logue
//===----------------------------------------------------------------------===//

namespace {
// A prologue or epilogue.
struct Logue {
  // Move the ops in the logue before the iterator.
  void moveBefore(Block *block, Block::iterator it) {
    for (Operation *op : ops)
      op->moveBefore(block, it);
  }

  // Replace all uses of the logue results with the given values, where `logue`
  // comprises all the ops in `containingRegion`.
  void replaceAllUsesWith(ValueRange values, Region &containingRegion) {
    for (auto [newOut, output] : llvm::zip(values, outputs)) {
      // Replace uses of the prologue outputs that are not in the prologue, i.e.
      // inside the `then` region where it got spliced.
      output.replaceUsesWithIf(newOut, [&](OpOperand &use) {
        return !containingRegion.isAncestor(use.getOwner()->getParentRegion());
      });
    }
  }

  // Get the number of outputs.
  unsigned getNumOutputs() const { return outputs.size(); }
  // Get the outputs as a `ValueRange`.
  ValueRange getOutputs() const { return outputs; }
  // Get the types of the outputs.
  TypeRange getOutputTypes() const { return getOutputs().getTypes(); }

  // A contiguous range of ops representing the prologue or epilogue.
  SmallVector<Operation *> ops;
  // The outputs of the logue. These are the SSA value results of `ops` that are
  // used by ops outside of `ops`.
  SmallVector<Value> outputs;
};
} // namespace

// Given a range of ops, form it into a logue by finding the outputs.
static Logue createLogueFrom(llvm::iterator_range<Block::iterator> ops,
                             mlir::DominanceInfo &domInfo) {
  Logue logue;
  for (Operation &op : ops)
    logue.ops.push_back(&op);

  if (ops.empty())
    return logue;

  // An op result is an output of the logue if the last operation in the logue
  // dominates any of its users.
  Operation &lastOp = *std::prev(ops.end());
  auto isOutput = [&](OpResult result) {
    for (Operation *user : result.getUsers()) {
      if (domInfo.properlyDominates(&lastOp, user))
        return true;
    }
    return false;
  };

  // Find the outputs.
  for (Operation &op : ops) {
    for (OpResult result : op.getOpResults()) {
      if (isOutput(result))
        logue.outputs.push_back(result);
    }
  }

  return logue;
}

//===----------------------------------------------------------------------===//
// fuseOneLevel
//===----------------------------------------------------------------------===//

// Only hoist operations that are side-effect free and "cheap" (i.e. only scalar
// operands). Importantly, we need to be able to hoist code generated by fusing
// children loops into their parents so the algorithm can be applied
// recursively.
static bool canHoistLoopBoundComputation(Operation *op) {
  auto isScalar = [](Type type) { return type.isIntOrIndexOrFloat(); };
  return isMemoryEffectFree(op) &&
         llvm::all_of(op->getOperandTypes(), isScalar) &&
         llvm::all_of(op->getResultTypes(), isScalar);
}

// Determine if all of `values` are or can be made invariant to the outer loop
// by hoisting operations. `toHoist` is shared across all child loop bounds.
static bool isOuterLoopInvariant(mlir::DominanceInfo &domInfo, scf::ForOp outer,
                                 ArrayRef<Value> values,
                                 llvm::SetVector<Operation *> &toHoist) {
  // The set of operations within `outer` that are being checked if they can be
  // hoisted. This set prevents checking operations twice but also if the
  // computation can be hoisted, this becomes the set of operations to hoist.
  llvm::SetVector<Operation *> visited;

  // Climb the use-def chain breadth-first so that operations can be hoisted in
  // the reverse visitation order.
  std::queue<Value> queue;
  for (Value value : values)
    queue.push(value);

  while (!queue.empty()) {
    Value value = queue.front();
    queue.pop();

    // If the value properly dominates the outer loop, then it must be invariant
    // to it.
    if (domInfo.properlyDominates(value, outer))
      continue;
    // If the value is a block argument, it cannot be hoisted.
    if (auto arg = dyn_cast<BlockArgument>(value))
      return false;

    Operation *op = value.getDefiningOp();
    // Check if the op was already visited.
    if (visited.contains(op))
      continue;
    // If the defining op cannot be hoisted, then the value cannot be made loop
    // invariant.
    if (!canHoistLoopBoundComputation(op))
      return false;
    visited.insert(op);
    // Recurse on the operands of the op.
    for (Value operand : op->getOperands())
      queue.push(operand);
  }

  // The operations in `visited` must be hoisted. Note that operations are not
  // added to `toHoist` unless all of `values` can be hoisted. This is to avoid
  // hoisting operations for loops that don't end up getting fused if one of
  // their bounds operands cannot be hoisted.
  toHoist.insert(visited.begin(), visited.end());

  return true;
}

// Pessimistically assume the internal storage bitwidth for index types.
static unsigned getIntTypeWidth(Type type) {
  if (isa<IndexType>(type))
    return IndexType::kInternalStorageBitWidth;
  return cast<IntegerType>(type).getWidth();
}

// Generate IR to compute the number of iterations of a loop.
static Value computeNumIters(ImplicitLocOpBuilder &b, scf::ForOp loop) {
  // len(range(lb, ub, step)) = ceildiv(ub - lb, step)
  // This works even if step is negative.
  Value diff =
      b.create<arith::SubIOp>(loop.getUpperBound(), loop.getLowerBound());
  // Let someone else prove it can be unsigned.
  return b.create<arith::CeilDivSIOp>(diff, loop.getStep());
}

// Cast an integer or index value to an integer or index `type`, if necessary.
static Value castIntIfNecessary(ImplicitLocOpBuilder &b, Value value,
                                Type type) {
  if (value.getType() == type)
    return value;
  if (isa<IndexType>(value.getType()) || isa<IndexType>(type))
    return b.create<arith::IndexCastOp>(type, value);
  if (cast<IntegerType>(value.getType()).getWidth() >
      cast<IntegerType>(type).getWidth())
    return b.create<arith::TruncIOp>(type, value);
  return b.create<arith::ExtSIOp>(type, value);
}

// To model an "undef" value, i.e. a value that is known to never be read on
// live code paths, create a zero-valued constant where possible, otherwise use
// a poison value. PTXAS appears to generate better code with zeros compared to
// poison values.
static Value createPoisonOrZero(ImplicitLocOpBuilder &b, Type type) {
  Type elTy = getElementTypeOrSelf(type);
  if (!elTy.isIntOrIndexOrFloat() ||
      (!isa<RankedTensorType>(type) && type != elTy))
    return b.create<ub::PoisonOp>(type);

  TypedAttr attr = isa<FloatType>(elTy) ? TypedAttr(b.getFloatAttr(elTy, 0))
                                        : b.getIntegerAttr(elTy, 0);
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    attr = SplatElementsAttr::get(tensor, attr);
  return b.create<arith::ConstantOp>(attr);
}

static scf::YieldOp getYield(Region &body) {
  return cast<scf::YieldOp>(body.front().back());
}

static scf::IfOp eraseIfResults(ImplicitLocOpBuilder &b, scf::IfOp ifOp,
                                llvm::BitVector indices,
                                SmallVector<Value> replaceWith) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(ifOp);
  while (indices.size() < ifOp.getNumResults())
    indices.push_back(false);

  getYield(ifOp.getThenRegion())->eraseOperands(indices);
  getYield(ifOp.getElseRegion())->eraseOperands(indices);

  TypeRange newTypes = getYield(ifOp.getThenRegion()).getOperandTypes();
  auto newIf = b.create<scf::IfOp>(newTypes, ifOp.getCondition());
  newIf.getThenRegion().takeBody(ifOp.getThenRegion());
  newIf.getElseRegion().takeBody(ifOp.getElseRegion());

  SmallVector<Value> replacements;
  auto replIt = replaceWith.begin();
  auto resIt = newIf->result_begin();
  for (unsigned i : llvm::seq(ifOp.getNumResults()))
    replacements.push_back(indices[i] ? *replIt++ : *resIt++);
  assert(ValueRange(replacements).getTypes() == ifOp.getResultTypes());
  ifOp.replaceAllUsesWith(replacements);
  ifOp.erase();
  return newIf;
}

// Given a one level loop nest in the form
//
//   for i in range(lbi, ubi, stepi):
//     prologue0(i)
//     for j0 in range(lbj0, ubj0, stepj0):
//       body0(i, j0)
//     epilogue1(i)
//     for j1 in range(lbj1, ubj1, stepj1):
//       body1(i, j1)
//     epilogue2(i)
//     ...
//     for jN in range(lbjN, ubjN, stepjN):
//       bodyN(i, jN)
//     epilogue(i)
//
// Rewrite this into a single loop in the form:
//
//   len_i = len(range(lbi, ubi, stepi))
//   len_j0 = len(range(lbj0, ubj0, stepj0))
//   len_j1 = len(range(lbj1, ubj1, stepj1))
//   ...
//   len_jN = len(range(lbjN, ubjN, stepjN))
//   inner_len = max(1, len_j0) + max(1, len_j1) + ... + max(1, len_jN) - N
//   total_iters = len_i * inner_len
//
//   T = -1
//   i = lbi - stepi
//   for _ in range(total_iters):
//     T = 0 if T == (inner_len - 1) else T + 1
//
//     if T == 0:
//       i += stepi
//       prologue0(i)
//       j0 = lbj0
//     if T >= 0 and T < len_j0:
//       body0(i, j0)
//       j0 += stepj0
//
//     if T == max(1, len_j0) - 1:
//       prologue1(i)
//       j1 = lbj1
//     if T >= max(1, len_j0) - 1
//    and T <  max(1, len_j0) - 1 + len_j1:
//       body1(i, j1)
//       j1 += stepj1
//
//     if T == max(1, len_j0) + max(1, len_j1) - 2:
//       prologue2(i)
//       j2 = lbj2
//     if T >= max(1, len_j0) + max(1, len_j1) - 2
//    and T <  max(1, len_j0) + max(1, len_j1) - 2 + len_j2:
//       body2(i, j2)
//       j2 += stepj2
//
//     ...
//
//     if T == max(1, len_j0) + max(1, len_j1) + ... + max(1, len_jN-1) - N:
//       prologueN(i)
//       jN = lbjN
//     if T >= max(1, len_j0) + max(1, len_j1) + ... + max(1, len_jN-1) - N
//    and T <  max(1, len_j0) + max(1, len_j1) + ... + max(1, len_jN-1) - N +
//             len_jN:
//       bodyN(i, jN)
//       jN += stepjN
//
//     if T == max(1, len_j0) + max(1, len_j1) + ... + max(1, len_jN) - (N + 1):
//       epilogue(i)
//
// This routine can be applied recursively on a loop nest tree, leaf-to-root, to
// flatten the loop nest into a single loop. However, this routine only fuses
// child loops whose loop bounds are invariant to the parent loop. For child
// loops where this is not the case, the function will ignore them.
//
// We could fuse loops with parent-loop-variant or even data-dependent bounds,
// but this will require generating `scf.while` in a form that is not friendly
// to the pipeliner. In order to effectively fuse and pipeline these kinds of
// loop nests, loop nest fusion and the pipeliner need to share a higher-level
// representation (or perhaps be the same pass).
//
// Note that there are many potential forms of the fused loop. This routine will
// attempt to minimize the number of fused loop iterations by overlapping the
// iteration spaces of the child loops and the epilogues. E.g. the last
// iteration of bodyjK will execute on the same fused loop iteration as
// epilogueK and the first iteration of bodyj(K+1). Hence the `- N` term in the
// total number of iterations.
//
// What the above Python-psuedo-code glosses over is SSA dependency management.
// To interpret the pseudocode as SSA IR, just imagine everything is put back
// into allocas and SSA formation re-runs after fusion, which one should note
// will introduce undefs.
//
// Handling dependencies will require turning implicit captures into
// loop-carried dependencies. Consider:
//
//   scf.for %i = %lbi to %ubi step %stepi {
//     %a = tt.call @func(%i)
//     scf.for %j = %lbj to %ubj step %stepj {
//       %b = tt.call @use(%a, %j)
//     }
//   }
//
// This needs to be rewritten into:
//
//   %poison = ub.poison
//   %Tlast, %ilast, %jlast, %alast = scf.for %unused = ...
//       iter_args(%Tprev = %c-1_i32,
//                 %iprev = %lbi - %stepi,
//                 %jprev = %poison,
//                 %aprev = %poison) -> (i32, i32, i32, i32) {
//     %T = (%Tprev + 1) mod (...)
//     %a, %i, %j = scf.if %T == 0 {
//       %inext = %iprev + 1
//       %jnext = %lbj - %stepj
//
//       %anext = tt.call @func(%i)
//       yield %inext, %jnext, %anext
//     } else {
//       yield %iprev, %jprev, %aprev
//     }
//
//     scf.if %T >= 0 and %T < ... {
//       tt.call @use(%a, %j)
//     }
//
// Note: the induction variables will be initialized to their lower bound to
// avoid underflow in lbjk - stepjk, with the exception of the outer loop
// induction variable, which needs to be incremented inside the prologue to
// avoid a dependency on the epilogue. This helps the scheduler behave.
//
// Any inputs and outputs of the loop bodies would also need to be handled
// similarly: initialized as undef if appropriate and carried through the fused
// loop. This is why fusion will increase liveranges. To minimize the number of
// additional loop-carried values, the routine will analyze the subblock of IR
// inside each `prologueK` and determine its "outputs" as intermediate SSA
// values that are used later in the loop nest.
static void fuseOneLevel(LoopNestNode *parent, mlir::DominanceInfo &domInfo) {
  scf::ForOp outer = parent->loop;

  SmallVector<scf::ForOp> innerLoops;
  llvm::SetVector<Operation *> toHoist;
  for (LoopNestNode *child : parent->children) {
    scf::ForOp inner = child->loop;
    assert(child->children.empty() && "fuseOneLevel runs leaf-to-root");

    // Check if the inner loop bounds are or can be made invariant to the outer
    // loop. Check them all at once to avoid adding ops to `toHoist` if not
    // necessary.
    if (!isOuterLoopInvariant(
            domInfo, outer,
            {inner.getLowerBound(), inner.getUpperBound(), inner.getStep()},
            toHoist))
      continue;

    // Add this child to the list of loops to fuse.
    innerLoops.push_back(child->loop);
  }

  // From the perspective of the overall analysis, we can delete all the
  // children of the current loop node. Child loops that cannot be fused are now
  // treated opaquely by the rest of the analysis. This allows partial fusing of
  // the constructed loop nest.
  parent->children.clear();

  // If there are no child loops to fuse, then there is nothing to do.
  if (innerLoops.empty())
    return;

  // The transformation will definitely succeed on `childrenToFuse`. `toHoist`
  // only contains the operations that must be hoisted for `childrenToFuse` to
  // be fusible.
  toHoist = topologicalSort(toHoist);
  for (Operation *op : toHoist)
    op->moveBefore(outer);

  // Determine the integer type to use for the length computations. Use an
  // integer bitwidth twice the size of the largest integer, up to 64 bits, to
  // avoid overflow.
  unsigned intTyWidth = getIntTypeWidth(outer.getInductionVar().getType());

  // Generate the computations of the fused loop bounds.
  Location loc = outer.getLoc();
  ImplicitLocOpBuilder b(loc, outer);
  Value lenOuter = computeNumIters(b, outer);
  SmallVector<Value> lenInners;
  for (scf::ForOp loop : innerLoops) {
    // len_jk = len(range(lbjk, ubjk, stepjk))
    Value lenInner = computeNumIters(b, loop);
    intTyWidth = std::max(intTyWidth, getIntTypeWidth(lenInner.getType()));
    lenInners.push_back(lenInner);
  }
  auto intTy = b.getIntegerType(intTyWidth);

  auto intTyCst = [&](int64_t v) {
    return b.create<arith::ConstantOp>(IntegerAttr::get(intTy, v));
  };

  // inner_len = max(1, len_j0) + max(1, len_j1) + ... + max(1, len_jN) - N
  unsigned N = innerLoops.size() - 1;
  Value innerLen = intTyCst(0);
  // Keep all the partial sums because we need them later.
  SmallVector<Value> partialInnerSums;
  partialInnerSums.push_back(innerLen);
  for (Value lenInner : lenInners) {
    lenInner = castIntIfNecessary(b, lenInner, intTy);
    lenInner = b.create<arith::MaxSIOp>(intTyCst(1), lenInner);
    innerLen = b.create<arith::AddIOp>(innerLen, lenInner);
    partialInnerSums.push_back(innerLen);
  }
  innerLen = b.create<arith::SubIOp>(innerLen, intTyCst(N));

  // total_iters = len_i * inner_len
  Value totalIters =
      b.create<arith::MulIOp>(castIntIfNecessary(b, lenOuter, intTy), innerLen);

  // The outputs of the prologue, each epilogue, and all inner loop bodies need
  // to carried through the fused loop.
  SmallVector<Logue> logues;
  auto addLogue = [&](Block::iterator begin, Block::iterator end) {
    logues.push_back(createLogueFrom({begin, end}, domInfo));
  };
  // prologue0
  addLogue(outer.getBody()->begin(), innerLoops.front()->getIterator());
  // prologuek where 0 < k <= N
  for (auto i : llvm::seq<unsigned>(0, innerLoops.size() - 1)) {
    addLogue(std::next(innerLoops[i]->getIterator()),
             innerLoops[i + 1]->getIterator());
  }
  // epilogue
  addLogue(std::next(innerLoops.back()->getIterator()),
           // Don't include the outer loop yield.
           std::prev(outer.getBody()->end()));

  // We need iter args for:
  // - The fused loop induction var
  // - The outer loop induction var
  // - The outer loop iter args
  // - The induction vars for each inner loop
  // - The outputs of each child loop
  // - The outputs of each logue
  SmallVector<Value> fusedInits;

  // T = -1
  fusedInits.push_back(intTyCst(-1));
  // i = lbi - stepi
  fusedInits.push_back(
      b.create<arith::SubIOp>(outer.getLowerBound(), outer.getStep()));

  unsigned outerArgsStartIdx = fusedInits.size();
  llvm::append_range(fusedInits, outer.getInits());

  // Everything else is initialized to undef.
  unsigned ivarStartIdx = fusedInits.size();
  for (scf::ForOp loop : innerLoops) {
    fusedInits.push_back(
        createPoisonOrZero(b, loop.getInductionVar().getType()));
  }
  unsigned innerOutsStartIdx = fusedInits.size();
  for (scf::ForOp loop : innerLoops) {
    for (Type resultType : loop.getResultTypes())
      fusedInits.push_back(createPoisonOrZero(b, resultType));
  }
  unsigned logueOutsStartIdx = fusedInits.size();
  for (Logue &logue : llvm::drop_end(logues)) {
    for (Type outputType : logue.getOutputTypes())
      fusedInits.push_back(createPoisonOrZero(b, outputType));
  }

  // for _ in range(total_iters):
  auto fused =
      b.create<scf::ForOp>(intTyCst(0), totalIters, intTyCst(1), fusedInits);
  // Replace the outer loop args with the args in the fused loop args.
  for (auto [arg, fusedArg] :
       llvm::zip(outer.getRegionIterArgs(),
                 fused.getRegionIterArgs().slice(outerArgsStartIdx))) {
    arg.replaceAllUsesWith(fusedArg);
  }
  b.setInsertionPointToStart(fused.getBody());

  // T = 0 if T == (inner_len - 1) else T + 1
  Value T = fused.getRegionIterArg(0);
  Value nextT = b.create<arith::AddIOp>(T, intTyCst(1));
  Value rollover =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, T,
                              b.create<arith::SubIOp>(innerLen, intTyCst(1)));
  T = b.create<arith::SelectOp>(rollover, intTyCst(0), nextT);

  // `i` is computed inside the first prologue.
  Value curI = fused.getRegionIterArg(1);
  Value i;

  assert(partialInnerSums.size() == N + 2);
  ArrayRef<BlockArgument> ivars = fused.getRegionIterArgs().slice(ivarStartIdx);
  auto bodyOutsIt =
      ValueRange(fused.getRegionIterArgs()).begin() + innerOutsStartIdx;
  auto logueOutsIt =
      ValueRange(fused.getRegionIterArgs()).begin() + logueOutsStartIdx;
  SmallVector<scf::IfOp> prologueIfs, bodyIfs;
  for (unsigned k = 0; k <= N; ++k) {
    // if T == max(1, len_j0) + ... max(1, len_jk-1) - k
    //   [[if k == 0]] i += stepi
    //   prologuek(i)
    //   jk = lbjk
    Value innerStartT =
        b.create<arith::SubIOp>(partialInnerSums[k], intTyCst(k));
    Value prologueCond =
        b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, T, innerStartT);

    // The `scf.if` outputs will be `jk` and the outputs of prologuek. We also
    // have to initialize the inner loop iter args.
    scf::ForOp inner = innerLoops[k];
    Logue &prologue = logues[k];

    SmallVector<Type> prologueOutTypes{inner.getInductionVar().getType()};
    llvm::append_range(prologueOutTypes, prologue.getOutputTypes());
    llvm::append_range(prologueOutTypes, inner.getInits().getTypes());
    if (k == 0)
      prologueOutTypes.push_back(curI.getType());
    auto prologueIf = b.create<scf::IfOp>(prologueOutTypes, prologueCond);
    prologueIfs.push_back(prologueIf);

    // Splice prologuek into the `then` region.
    Block *thenBlock = b.createBlock(&prologueIf.getThenRegion());
    prologue.moveBefore(thenBlock, thenBlock->end());

    if (k == 0) {
      // Increment `i` and replace its uses inside the prologue.
      b.setInsertionPointToStart(thenBlock);
      i = b.create<arith::AddIOp>(curI, outer.getStep());
      mlir::replaceAllUsesInRegionWith(outer.getInductionVar(), i,
                                       prologueIf.getThenRegion());
    }

    // Yield the initialized jk, the prologue outputs, and the initial values of
    // the inner loop.
    b.setInsertionPointToEnd(thenBlock);
    SmallVector<Value> thenOuts{inner.getLowerBound()};
    llvm::append_range(thenOuts, prologue.getOutputs());
    llvm::append_range(thenOuts, inner.getInits());
    if (k == 0)
      thenOuts.push_back(i);
    b.create<scf::YieldOp>(thenOuts);

    // In the `else` region, just yield the last values of jk, the outputs, and
    // the iter args.
    b.createBlock(&prologueIf.getElseRegion());
    Value lastJk = ivars[k];
    unsigned numOuts = prologue.getNumOutputs();
    SmallVector<Value> elseOuts{lastJk};
    elseOuts.append(logueOutsIt, logueOutsIt + numOuts);
    elseOuts.append(bodyOutsIt, bodyOutsIt + inner.getNumResults());
    if (k == 0)
      elseOuts.push_back(curI);
    logueOutsIt += numOuts;
    b.create<scf::YieldOp>(elseOuts);

    // The results of the `scf.if` become the values of jk and the prologue
    // outputs for the rest of the fused loop.
    Value jk = prologueIf.getResult(0);
    ValueRange prologueOuts = prologueIf.getResults().slice(1, numOuts);
    ValueRange prologueInits =
        prologueIf.getResults().slice(1 + numOuts, inner.getNumResults());
    inner.getInductionVar().replaceAllUsesWith(jk);
    prologue.replaceAllUsesWith(prologueOuts, prologueIf.getThenRegion());
    for (auto [init, iterArg] :
         llvm::zip(prologueInits, inner.getRegionIterArgs()))
      iterArg.replaceAllUsesWith(init);
    // Replace uses of `i` elsewhere with the prologue result.
    if (k == 0) {
      i = prologueIf.getResults().back();
      outer.getInductionVar().replaceAllUsesWith(i);
    }

    // if  T >= max(1, len_j0) + max(1, len_j1) + ... + max(1, len_jk-1) - k
    // and T <  max(1, len_j0) + max(1, len_j1) + ... + max(1, len_jk-1) - k +
    //          len_jk
    //   bodyk(i, jk)
    //   jk += stepjk
    b.setInsertionPointAfter(prologueIf);
    Value innerEndT = b.create<arith::AddIOp>(
        innerStartT, castIntIfNecessary(b, lenInners[k], intTy));
    Value ge =
        b.create<arith::CmpIOp>(arith::CmpIPredicate::sge, T, innerStartT);
    Value lt = b.create<arith::CmpIOp>(arith::CmpIPredicate::slt, T, innerEndT);
    Value bodyCond = b.create<arith::AndIOp>(ge, lt);

    // The outputs will be the outputs of the inner loop body and the next jk.
    SmallVector<Type> bodyOutTypes{jk.getType()};
    llvm::append_range(bodyOutTypes, inner->getResultTypes());
    auto bodyIf = b.create<scf::IfOp>(bodyOutTypes, bodyCond);
    bodyIfs.push_back(bodyIf);

    // Splice bodyk into the `then` region.
    inner.getBody()->eraseArguments([](Value arg) { return true; });
    bodyIf.getThenRegion().takeBody(inner.getBodyRegion());
    auto yield = getYield(bodyIf.getThenRegion());
    b.setInsertionPoint(yield);
    Value nextJk = b.create<arith::AddIOp>(jk, inner.getStep());
    yield->insertOperands(0, nextJk);

    // The `else` region just forwards the values.
    b.createBlock(&bodyIf.getElseRegion());
    SmallVector<Value> bodyForwardedOuts{jk};
    bodyForwardedOuts.append(bodyOutsIt, bodyOutsIt + inner.getNumResults());
    bodyOutsIt += inner->getNumResults();
    b.create<scf::YieldOp>(bodyForwardedOuts);

    // Now we can replace the results of the inner loop with the outputs of the
    // body if.
    inner.replaceAllUsesWith(
        bodyIf.getResults().slice(1, inner.getNumResults()));

    // If the inner loop must execute, then its body does not have to be wrapped
    // in a conditional.
    if (inner->hasAttr(kMustExecuteAttrName)) {
      b.setInsertionPoint(bodyIf);
      bodyIf.getConditionMutable().assign(
          b.create<arith::ConstantOp>(b.getBoolAttr(true)));
    }

    // Move the insertion point for the next iteration.
    b.setInsertionPointAfter(bodyIf);
  }

  // if T == len_j0 + len_j1 + ... + len_jN - N - 1:
  //   epilogue(i)
  Logue &epilogue = logues.back();

  // The only possible use of an epilogue output is the yield.
  auto outerYield = cast<scf::YieldOp>(outer.getBody()->getTerminator());
  SmallVector<Value> usedIterArgs;
  for (Value output : epilogue.getOutputs()) {
    for (OpOperand &use : output.getUses()) {
      if (use.getOwner() == outerYield) {
        usedIterArgs.push_back(fused.getRegionIterArgs().drop_front(
            outerArgsStartIdx)[use.getOperandNumber()]);
      }
    }
  }

  auto epilogueCond =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, T,
                              b.create<arith::SubIOp>(innerLen, intTyCst(1)));
  auto epilogueIf =
      b.create<scf::IfOp>(epilogue.getOutputTypes(), epilogueCond);

  Block *thenBlock = b.createBlock(&epilogueIf.getThenRegion());
  epilogue.moveBefore(thenBlock, thenBlock->end());

  b.setInsertionPointToEnd(thenBlock);
  b.create<scf::YieldOp>(epilogue.getOutputs());
  b.createBlock(&epilogueIf.getElseRegion());
  b.create<scf::YieldOp>(usedIterArgs);
  epilogue.replaceAllUsesWith(epilogueIf.getResults(),
                              epilogueIf.getThenRegion());

  // Finally, create the yield of the fused loop.
  SmallVector<Value> outerOuts{T, i};
  llvm::append_range(outerOuts, outerYield.getOperands());
  for (scf::IfOp bodyIf : bodyIfs)
    outerOuts.push_back(/*jk=*/bodyIf.getResult(0));
  for (auto [bodyIf, loop] : llvm::zip(bodyIfs, innerLoops)) {
    llvm::append_range(outerOuts,
                       bodyIf.getResults().slice(1, loop.getNumResults()));
  }
  for (auto [logueIf, logue] : llvm::zip(prologueIfs, llvm::drop_end(logues))) {
    llvm::append_range(outerOuts,
                       logueIf.getResults().slice(1, logue.getNumOutputs()));
  }

  b.setInsertionPointToEnd(fused.getBody());
  b.create<scf::YieldOp>(outerOuts);
  outer.replaceAllUsesWith(
      fused.getResults().slice(outerArgsStartIdx, outer.getNumResults()));

  // Reduce dependencies across inner loops by hoisting the initialization of
  // inner loop iter args to the outer loop when possible, and then placing the
  // reset of these values in the epilogue.
  auto fusedInitsIt = fused.getInitsMutable().begin() + innerOutsStartIdx;
  auto fusedArgsIt = fused.getRegionIterArgs().begin() + innerOutsStartIdx;
  auto fusedYieldIt = getYield(fused.getBodyRegion())->getOpOperands().begin() +
                      innerOutsStartIdx;
  SmallVector<OpOperand *> yieldsToUpdate;
  SmallVector<Value> reset, forwarded;
  for (auto [loop, ifOp, bodyIf, prologue] :
       llvm::zip(innerLoops, prologueIfs, bodyIfs, logues)) {
    unsigned numResults = loop.getNumResults();
    unsigned prologueSkip = 1 + prologue.getNumOutputs();

    llvm::BitVector removeIndices(prologueSkip + numResults);
    SmallVector<Value> replaceWith;
    for (auto [i, init] : llvm::enumerate(loop.getInits())) {
      if (init.getParentRegion() == &fused.getBodyRegion())
        continue;
      // Initialize this in the outer loop.
      fusedInitsIt[i].assign(init);
      replaceWith.push_back(fusedArgsIt[i]);
      removeIndices.set(prologueSkip + i);
      yieldsToUpdate.push_back(&fusedYieldIt[i]);
      forwarded.push_back(bodyIf.getResult(1 + i));
      reset.push_back(init);
    }
    // Remove the initializers in the corresponding prologue.
    eraseIfResults(b, ifOp, removeIndices, replaceWith);

    fusedInitsIt += numResults;
    fusedArgsIt += numResults;
    fusedYieldIt += numResults;
  }
  if (!yieldsToUpdate.empty()) {
    MutableOperandRange(getYield(epilogueIf.getThenRegion())).append(reset);
    MutableOperandRange(getYield(epilogueIf.getElseRegion())).append(forwarded);
    b.setInsertionPoint(epilogueIf);
    TypeRange newTypes = getYield(epilogueIf.getThenRegion()).getOperandTypes();
    auto newIf = b.create<scf::IfOp>(newTypes, epilogueIf.getCondition());
    newIf.getThenRegion().takeBody(epilogueIf.getThenRegion());
    newIf.getElseRegion().takeBody(epilogueIf.getElseRegion());
    epilogueIf.replaceAllUsesWith(
        newIf.getResults().take_front(epilogueIf.getNumResults()));
    ResultRange newResults =
        newIf.getResults().drop_front(epilogueIf.getNumResults());
    for (auto [i, yieldOperand] : llvm::enumerate(yieldsToUpdate))
      yieldOperand->set(newResults[i]);
    epilogueIf.erase();
  }

  // Propagate the `tt.disallow_acc_multi_buffer` attribute to the parent loop.
  bool disallowAccMultiBuffer = getDisallowAccMultiBuffer(outer);
  for (scf::ForOp loop : innerLoops) {
    disallowAccMultiBuffer |= getDisallowAccMultiBuffer(loop);
  }
  if (disallowAccMultiBuffer)
    fused->setAttr(kDisallowAccMultiBufferAttrName, b.getUnitAttr());

  // Update the parent's loop to the fused loop. Set the new stage count to the
  // max stage count of the inner loops.
  int numStages = 1;
  if (auto stageAttr = outer->getAttrOfType<IntegerAttr>(kNumStagesAttrName))
    numStages = stageAttr.getInt();
  for (scf::ForOp loop : innerLoops) {
    if (auto stageAttr = loop->getAttrOfType<IntegerAttr>(kNumStagesAttrName))
      numStages = std::max<int>(numStages, stageAttr.getInt());
    loop.erase();
  }
  outer.erase();
  parent->loop = fused;
  if (numStages > 1)
    fused->setAttr(kNumStagesAttrName, b.getI32IntegerAttr(numStages));
}

//===----------------------------------------------------------------------===//
// flattenLoopNest
//===----------------------------------------------------------------------===//

// Completely flatten a loop nest by recursively fusing loops in a post-order
// traversal with `fuseOneLevel`.
static void flattenLoopNest(LoopNestNode *node, mlir::DominanceInfo &domInfo) {
  for (LoopNestNode *child : node->children)
    flattenLoopNest(child, domInfo);
  fuseOneLevel(node, domInfo);
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

// Fuse simple loop nests with a single outer and inner loop, and where the
// inner loop has a `tt.dot` operation.
static bool shouldFuse(const LoopNest &nest) {
  if (nest.root->loop->hasAttr(kAlwaysFuseAttrName))
    return true;

  // Only fuse simple loop nests.
  return nest.nodes.size() == 2 && nest.root->children.size() == 1 &&
         nest.root->loop->hasAttr(kFlattenAttr);
}

// This function identifies a subgraph of cheap ops that can be sunk between two
// regions in the loop nest and moves them, reducing their liveranges.
static void sinkOps(Region &limit, Block *sinkBlock, Block::iterator sinkBefore,
                    llvm::iterator_range<Block::iterator> prologue,
                    function_ref<bool(Operation *)> inSinkRegion) {
  llvm::SetVector<Operation *> sunkOps;
  auto canBeSunk = [&](Operation &op) -> std::pair<bool, bool> {
    if (!isPure(&op) || isa<DotOpInterface>(op))
      return {false, false};
    // An op can be sunk if all its users are inside the inner loop or are
    // marked for sinking.
    bool isRoot = true;
    for (Operation *user : op.getUsers()) {
      if (inSinkRegion(user))
        continue;
      isRoot = false;
      if (sunkOps.contains(user))
        continue;
      return {false, false};
    }
    return {true, isRoot};
  };

  // Find the subgraph of operations that can be sunk.
  SmallVector<Operation *> roots;
  for (Operation &op : llvm::reverse(prologue)) {
    auto [canSink, isRoot] = canBeSunk(op);
    if (canSink)
      sunkOps.insert(&op);
    if (isRoot)
      roots.push_back(&op);
  }
  if (sunkOps.empty())
    return;

  sunkOps = topologicalSort(sunkOps);
  for (Operation *op : sunkOps)
    op->moveBefore(sinkBlock, sinkBefore);
}

// Sink ops from the prologue into the epilogue when possible.
static void optimizeEpilogueDependencies(scf::ForOp outerLoop,
                                         scf::ForOp innerLoop,
                                         mlir::DominanceInfo &domInfo) {
  auto inEpilogue = [&](Operation *op) {
    return domInfo.properlyDominates(innerLoop, op, /*enclosingOpOk=*/false);
  };
  Region &limit = outerLoop.getBodyRegion();
  sinkOps(limit, outerLoop.getBody(), std::next(innerLoop->getIterator()),
          {outerLoop.getBody()->begin(), innerLoop->getIterator()}, inEpilogue);
}

// Speculate the length of the inner loop such that the loop is known to execute
// at least once. This way, the inner loop body does not have to be placed
// inside a conditional in the fused loop, which interacts better with the
// pipeliner.
static LogicalResult speculateInnerLoopLength(scf::ForOp outerLoop,
                                              scf::ForOp innerLoop,
                                              mlir::DominanceInfo &domInfo) {
  // The inner loop bounds must be outer-loop invariant to speculate from
  // outside the loop nest.
  Location loc = innerLoop.getLoc();
  llvm::SetVector<Operation *> toHoist;
  if (!isOuterLoopInvariant(domInfo, outerLoop,
                            {innerLoop.getLowerBound(),
                             innerLoop.getUpperBound(), innerLoop.getStep()},
                            toHoist))
    return failure();

  // Hoist the inner loop bounds computations if necessary.
  toHoist = topologicalSort(toHoist);
  for (Operation *op : toHoist)
    op->moveBefore(outerLoop);

  // Mark the inner loop.
  ImplicitLocOpBuilder b(loc, outerLoop);
  innerLoop->setAttr(kMustExecuteAttrName, b.getUnitAttr());

  // Speculate on whether the length of the inner loop is zero.
  Value lenInner = computeNumIters(b, innerLoop);
  auto zeroAttr = IntegerAttr::get(lenInner.getType(), 0);
  Value innerLoopEmpty =
      b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, lenInner,
                              b.create<arith::ConstantOp>(zeroAttr));
  auto ifOp = b.create<scf::IfOp>(outerLoop.getResultTypes(), innerLoopEmpty);

  // In the `then` branch, the inner loop does not execute. Clone the loop nest
  // into it and remove the inner loop.
  mlir::IRMapping map;
  b.createBlock(&ifOp.getThenRegion());
  auto newLoop = cast<scf::ForOp>(b.clone(*outerLoop, map));
  b.create<scf::YieldOp>(newLoop.getResults());
  auto newInnerLoop = cast<scf::ForOp>(map.lookup(innerLoop));
  newInnerLoop.replaceAllUsesWith(newInnerLoop.getInits());
  newInnerLoop.erase();

  // Move the loop nest into the `else` branch.
  outerLoop.replaceAllUsesWith(ifOp.getResults());
  Block *block = b.createBlock(&ifOp.getElseRegion());
  outerLoop->remove();
  b.insert(outerLoop);
  b.create<scf::YieldOp>(outerLoop.getResults());

  return success();
}

static LogicalResult preprocessLoopNest(const LoopNest &nest,
                                        mlir::DominanceInfo &domInfo) {
  assert(nest.nodes.size() == 2 && nest.root->children.size() == 1);

  scf::ForOp &outerLoop = nest.root->loop;
  scf::ForOp &innerLoop = nest.root->children.front()->loop;

  moveLoopInvariantCode(outerLoop);
  optimizeEpilogueDependencies(outerLoop, innerLoop, domInfo);
  return speculateInnerLoopLength(outerLoop, innerLoop, domInfo);
}

void FuseNestedLoopsPass::runOnOperation() {
  auto &domInfo = getAnalysis<DominanceInfo>();

  for (auto func : getOperation().getOps<FuncOp>()) {
    SmallVector<LoopNest> nests;
    findLoopNests(func, nests);
    for (LoopNest &nest : nests) {
      if (!shouldFuse(nest))
        continue;
      if (!nest.root->loop->hasAttr(kAlwaysFuseAttrName) &&
          failed(preprocessLoopNest(nest, domInfo)))
        continue;
      flattenLoopNest(nest.root, domInfo);
    }
  }
}

} // namespace gpu
} // namespace triton
} // namespace mlir
