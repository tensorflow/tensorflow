//===- SliceAnalysis.h - Analysis for Transitive UseDef chains --*- C++ -*-===//
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

#ifndef MLIR_ANALYSIS_SLICEANALYSIS_H_
#define MLIR_ANALYSIS_SLICEANALYSIS_H_

#include <functional>
#include <vector>

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SetVector.h"

namespace mlir {

class Instruction;

/// Type of the condition to limit the propagation of transitive use-defs.
/// This can be used in particular to limit the propagation to a given Scope or
/// to avoid passing through certain types of instruction in a configurable
/// manner.
using TransitiveFilter = std::function<bool(Instruction *)>;

/// Fills `forwardSlice` with the computed forward slice (i.e. all
/// the transitive uses of inst), **without** including that instruction.
///
/// This additionally takes a TransitiveFilter which acts as a frontier:
/// when looking at uses transitively, a instruction that does not pass the
/// filter is never propagated through. This allows in particular to carve out
/// the scope within a ForInst or the scope within an IfInst.
///
/// The implementation traverses the use chains in postorder traversal for
/// efficiency reasons: if a instruction is already in `forwardSlice`, no
/// need to traverse its uses again. Since use-def chains form a DAG, this
/// terminates.
///
/// Upon return to the root call, `forwardSlice` is filled with a
/// postorder list of uses (i.e. a reverse topological order). To get a proper
/// topological order, we just just reverse the order in `forwardSlice` at
/// the topLevel before returning.
///
/// Example starting from node 0
/// ============================
///
///              0
///    ___________|___________
///    1       2      3      4
///    |_______|      |______|
///    |   |             |
///    |   5             6
///    |___|_____________|
///      |               |
///      7               8
///      |_______________|
///              |
///              9
///
/// Assuming all local orders match the numbering order:
/// 1. after getting back to the root getForwardSlice, `forwardSlice` may
///    contain:
///      {9, 7, 8, 5, 1, 2, 6, 3, 4}
/// 2. reversing the result of 1. gives:
///      {4, 3, 6, 2, 1, 5, 8, 7, 9}
///
void getForwardSlice(
    Instruction *inst, llvm::SetVector<Instruction *> *forwardSlice,
    TransitiveFilter filter = /* pass-through*/
    [](Instruction *) { return true; },
    bool topLevel = true);

/// Fills `backwardSlice` with the computed backward slice (i.e.
/// all the transitive defs of inst), **without** including that instruction.
///
/// This additionally takes a TransitiveFilter which acts as a frontier:
/// when looking at defs transitively, a instruction that does not pass the
/// filter is never propagated through. This allows in particular to carve out
/// the scope within a ForInst or the scope within an IfInst.
///
/// The implementation traverses the def chains in postorder traversal for
/// efficiency reasons: if a instruction is already in `backwardSlice`, no
/// need to traverse its definitions again. Since useuse-def chains form a DAG,
/// this terminates.
///
/// Upon return to the root call, `backwardSlice` is filled with a
/// postorder list of defs. This happens to be a topological order, from the
/// point of view of the use-def chains.
///
/// Example starting from node 8
/// ============================
///
///    1       2      3      4
///    |_______|      |______|
///    |   |             |
///    |   5             6
///    |___|_____________|
///      |               |
///      7               8
///      |_______________|
///              |
///              9
///
/// Assuming all local orders match the numbering order:
///    {1, 2, 5, 7, 3, 4, 6, 8}
///
void getBackwardSlice(
    Instruction *inst, llvm::SetVector<Instruction *> *backwardSlice,
    TransitiveFilter filter = /* pass-through*/
    [](Instruction *) { return true; },
    bool topLevel = true);

/// Iteratively computes backward slices and forward slices until
/// a fixed point is reached. Returns an `llvm::SetVector<Instruction *>` which
/// **includes** the original instruction.
///
/// This allows building a slice (i.e. multi-root DAG where everything
/// that is reachable from an Value in forward and backward direction is
/// contained in the slice).
/// This is the abstraction we need to materialize all the instructions for
/// supervectorization without worrying about orderings and Value
/// replacements.
///
/// Example starting from any node
/// ==============================
///
///    1       2      3      4
///    |_______|      |______|
///    |   |             |   |
///    |   5             6___|
///    |___|_____________|   |
///      |               |   |
///      7               8   |
///      |_______________|   |
///              |           |
///              9          10
///
/// Return the whole DAG in some topological order.
///
/// The implementation works by just filling up a worklist with iterative
/// alternate calls to `getBackwardSlice` and `getForwardSlice`.
///
/// The following section describes some additional implementation
/// considerations for a potentially more efficient implementation but they are
/// just an intuition without proof, we still use a worklist for now.
///
/// Additional implementation considerations
/// ========================================
/// Consider the defs-inst-uses hourglass.
///    ____
///    \  /  defs (in some topological order)
///     \/
///    inst
///     /\
///    /  \  uses (in some topological order)
///   /____\
///
/// We want to iteratively apply `getSlice` to construct the whole
/// list of Instruction that are reachable by (use|def)+ from inst.
/// We want the resulting slice in topological order.
/// Ideally we would like the ordering to be maintained in-place to avoid
/// copying Instruction at each step. Keeping this ordering by construction
/// seems very unclear, so we list invariants in the hope of seeing whether
/// useful properties pop up.
///
/// In the following:
///   we use |= for set inclusion;
///   we use << for set topological ordering (i.e. each pair is ordered).
///
/// Assumption:
/// ===========
/// We wish to maintain the following property by a recursive argument:
///   """
///      defs << {inst} <<uses are in topological order.
///   """
/// The property clearly holds for 0 and 1-sized uses and defs;
///
/// Invariants:
///   2. defs and uses are in topological order internally, by construction;
///   3. for any {x} |= defs, defs(x) |= defs;    because all go through inst
///   4. for any {x} |= uses,    defs |= defs(x); because all go through inst
///   5. for any {x} |= defs,    uses |= uses(x); because all go through inst
///   6. for any {x} |= uses, uses(x) |= uses;    because all go through inst
///
/// Intuitively, we should be able to recurse like:
///   preorder(defs) - inst - postorder(uses)
/// and keep things ordered but this is still hand-wavy and not worth the
/// trouble for now: punt to a simple worklist-based solution.
///
llvm::SetVector<Instruction *> getSlice(
    Instruction *inst,
    TransitiveFilter backwardFilter = /* pass-through*/
    [](Instruction *) { return true; },
    TransitiveFilter forwardFilter = /* pass-through*/
    [](Instruction *) { return true; });

/// Multi-root DAG topological sort.
/// Performs a topological sort of the Instruction in the `toSort` SetVector.
/// Returns a topologically sorted SetVector.
llvm::SetVector<Instruction *>
topologicalSort(const llvm::SetVector<Instruction *> &toSort);

} // end namespace mlir

#endif // MLIR_ANALYSIS_SLICEANALYSIS_H_
