// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_PIPELINE_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_PIPELINE_H_

// This is a fork of upstream pipeline transformation. This will be merged back
// upstream once we have a stable solution.

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {

class RewriterBase;
class Operation;
class Value;

namespace scf {
class ForOp;
}

namespace triton {

/// Options to dictate how loops should be pipelined.
struct PipeliningOption {
  /// Lambda returning all the operations in the forOp, with their stage, in the
  /// order picked for the pipelined loop.
  using GetScheduleFnType = std::function<void(
      scf::ForOp, std::vector<std::pair<Operation *, unsigned>> &)>;
  GetScheduleFnType getScheduleFn = nullptr;
  enum class PipelinerPart {
    Prologue,
    Kernel,
    Epilogue,
  };
  /// Lambda called by the pipeliner to allow the user to annotate the IR while
  /// it is generated.
  /// The callback passes the operation created along with the part of the
  /// pipeline and the iteration index. The iteration index is always 0 for the
  /// kernel. For the prologue and epilogue, it corresponds to the iteration
  /// peeled out of the loop in the range [0, maxStage[.
  using AnnotationlFnType =
      std::function<void(Operation *, PipelinerPart, unsigned)>;
  AnnotationlFnType annotateFn = nullptr;

  /// Control whether the epilogue should be peeled out of the loop or
  /// operations should be predicated to skip the early stages in the last loop
  /// iterations. If the epilogue is predicated; the user needs to provide a
  /// lambda to generate the predicated version of operations.
  bool peelEpilogue = true;

  /// Control whether the transformation checks that the number of iterations is
  /// greater or equal to the number of stages and skip the transformation if
  /// this is not the case. If the loop is dynamic and this is set to true the
  /// pipeliner will have to predicate operations in the prologue/epilogue.
  bool supportDynamicLoops = false;

  // Callback to predicate operations when the prologue or epilogue are not
  // peeled. This takes the original operation, an i1 predicate value and the
  // pattern rewriter. It is expected to replace the given operation with
  // the predicated equivalent and return it, or return nullptr if the
  // predication is impossible. In the latter case, pipelining will fail and
  // may leave IR in a partially transformed state.
  using PredicateOpFnType =
      std::function<Operation *(RewriterBase &, Operation *, Value)>;
  PredicateOpFnType predicateFn = nullptr;

  // TODO: add option to decide if the prologue should be peeled.
};

/// Generate a pipelined version of the scf.for loop based on the schedule given
/// as option. This applies the mechanical transformation of changing the loop
/// and generating the prologue/epilogue for the pipelining and doesn't make any
/// decision regarding the schedule.
/// Based on the options the loop is split into several stages.
/// The transformation assumes that the scheduling given by user is valid.
/// For example if we break a loop into 3 stages named S0, S1, S2 we would
/// generate the following code with the number in parenthesis as the iteration
/// index:
///
///   S0(0)                        // Prologue
///   S0(1) S1(0)                  // Prologue
///   scf.for %I = %C0 to %N - 2 {
///     S0(I+2) S1(I+1) S2(I)       // Pipelined kernel
///   }
///   S1(N) S2(N-1)                // Epilogue
///   S2(N)                        // Epilogue
///
/// If `modifiedIR` is provided, it will be set to a value that indicates
/// whether pipelining modified the IR before failing, signaling to the caller
/// whether they can proceed with different transformations.
FailureOr<scf::ForOp> pipelineForLoop(RewriterBase &rewriter, scf::ForOp forOp,
                                      const PipeliningOption &options,
                                      bool *modifiedIR = nullptr);

} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_PIPELINE_H_
