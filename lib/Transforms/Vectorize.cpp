//===- Vectorize.cpp - Vectorize Pass Impl ----------------------*- C++ -*-===//
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
//
// This file implements vectorization of loops, operations and data types to
// a target-independent, n-D super-vector abstraction.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/MLFunctionMatcher.h"
#include "mlir/Analysis/VectorAnalysis.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLValue.h"
#include "mlir/IR/SSAValue.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass.h"
#include "mlir/StandardOps/StandardOps.h"
#include "mlir/SuperVectorOps/SuperVectorOps.h"
#include "mlir/Support/Functional.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

///
/// Implements a high-level vectorization strategy on an MLFunction.
/// The abstraction used is that of super-vectors, which provide a single,
/// compact, representation in the vector types, information that is expected
/// to reduce the impact of the phase ordering problem
///
/// Vector granularity:
/// ===================
/// This pass is designed to perform vectorization at a super-vector
/// granularity. A super-vector is loosely defined as a vector type that is a
/// multiple of a "good" vector size so the HW can efficiently implement a set
/// of high-level primitives. Multiple is understood along any dimension; e.g.
/// both vector<16xf32> and vector<2x8xf32> are valid super-vectors for a
/// vector<8xf32> HW vector. Note that a "good vector size so the HW can
/// efficiently implement a set of high-level primitives" is not necessarily an
/// integer multiple of actual hardware registers. We leave details of this
/// distinction unspecified for now.
///
/// Some may prefer the terminology a "tile of HW vectors". In this case, one
/// should note that super-vectors implement an "always full tile" abstraction.
/// They guarantee no partial-tile separation is necessary by relying on a
/// high-level copy-reshape abstraction that we call vector_transfer. This
/// copy-reshape operations is also responsible for performing layout
/// transposition if necessary. In the general case this will require a scoped
/// allocation in some notional local memory.
///
/// Whatever the mental model one prefers to use for this abstraction, the key
/// point is that we burn into a single, compact, representation in the vector
/// types, information that is expected to reduce the impact of the phase
/// ordering problem. Indeed, a vector type conveys information that:
///   1. the associated loops have dependency semantics that do not prevent
///      vectorization;
///   2. the associate loops have been sliced in chunks of static sizes that are
///      compatible with vector sizes (i.e. similar to unroll-and-jam);
///   3. the inner loops, in the unroll-and-jam analogy of 2, are captured by
///   the
///      vector type and no vectorization hampering transformations can be
///      applied to them anymore;
///   4. the underlying memrefs are accessed in some notional contiguous way
///      that allows loading into vectors with some amount of spatial locality;
/// In other words, super-vectorization provides a level of separation of
/// concern by way of opacity to subsequent passes. This has the effect of
/// encapsulating and propagating vectorization constraints down the list of
/// passes until we are ready to lower further.
///
/// For a particular target, a notion of minimal n-d vector size will be
/// specified and vectorization targets a multiple of those. In the following
/// paragraph, let "k ." represent "a multiple of", to be understood as a
/// multiple in the same dimension (e.g. vector<16 x k . 128> summarizes
/// vector<16 x 128>, vector<16 x 256>, vector<16 x 1024>, etc).
///
/// Some non-exhaustive notable super-vector sizes of interest include:
///   - CPU: vector<k . HW_vector_size>,
///          vector<k' . core_count x k . HW_vector_size>,
///          vector<socket_count x k' . core_count x k . HW_vector_size>;
///   - GPU: vector<k . warp_size>,
///          vector<k . warp_size x float2>,
///          vector<k . warp_size x float4>,
///          vector<k . warp_size x 4 x 4x 4> (for tensor_core sizes).
///
/// Loops and operations are emitted that operate on those super-vector shapes.
/// Subsequent lowering passes will materialize to actual HW vector sizes. These
/// passes are expected to be (gradually) more target-specific.
///
/// At a high level, a vectorized load in a loop will resemble:
/// ```mlir
///   for %i = ? to ? step ? {
///     %v_a = "vector_transfer_read" (A, %i) : (memref<?xf32>, index) ->
///                                              vector<128xf32>
///   }
/// ```
/// It is the reponsibility of the implementation of the vector_transfer_read
/// to materialize vector registers from the original scalar memrefs.
/// A later (more target-dependent) lowering pass will materialize to actual HW
/// vector sizes. This lowering may be occur at different times:
///   1. at the MLIR level into a combination of loops, unrolling, DmaStartOp +
///   DmaWaitOp + vectorized operations
///      for data transformations and shuffle; thus opening opportunities for
///      unrolling and pipelining. This is an instance of library call
///      "whiteboxing"; or
///   2. later in the a target-specific lowering pass or hand-written library
///      call; achieving full separation of concerns. This is an instance of
///      library call; or
///   3. a mix of both, e.g. based on a model.
/// In the future, these operations will expose a contract to constrain the
/// search on vectorization patterns and sizes.
///
/// Occurrence of super-vectorization in the compiler flow:
/// =======================================================
/// This is an active area of investigation. We start with 2 remarks to position
/// super-vectorization in the context of existing ongoing work: LLVM VPLAN
/// and LLVM SLP Vectorizer.
///
/// LLVM VPLAN:
/// -----------
/// The astute reader may have noticed that in the limit, super-vectorization
/// can be applied at a similar time and with similar objectives than VPLAN.
/// For instance, in the case of a traditional, polyhedral compilation-flow (for
/// instance, the PPCG project uses ISL to provide dependence analysis,
/// multi-level(scheduling + tiling), lifting footprint to fast memory,
/// communication synthesis, mapping, register optimizations) and before
/// unrolling. When vectorization is applied at this *late* level in a typical
/// polyhedral flow, and is instantiated with actual hardware vector sizes,
/// super-vectorization is expected to match (or subsume) the type of patterns
/// that LLVM's VPLAN aims at targeting. The main difference here is that MLIR
/// is higher level and our implementation should be significantly simpler. Also
/// note that in this mode, recursive patterns are probably a bit of an overkill
/// although it is reasonable to expect that mixing a bit of outer loop and
/// inner loop vectorization + unrolling will provide interesting choices to
/// MLIR.
///
/// LLVM SLP Vectorizer:
/// --------------------
/// Super-vectorization however is not meant to be usable in a similar fashion
/// to the SLP vectorizer. The main difference lies in the information that
/// both vectorizers use: super-vectorization examines contiguity of memory
/// references along fastest varying dimensions and loops with recursive nested
/// patterns capturing imperfectly-nested loop nests; the SLP vectorizer, on
/// the other hand, performs flat pattern matching inside a single unrolled loop
/// body and stitches together pieces of load and store instructions into full
/// 1-D vectors. We envision that the SLP vectorizer is a good way to capture
/// innermost loop, control-flow dependent patterns that super-vectorization may
/// not be able to capture easily. In other words, super-vectorization does not
/// aim at replacing the SLP vectorizer and the two solutions are complementary.
///
/// Ongoing investigations:
/// -----------------------
/// We discuss the following *early* places where super-vectorization is
/// applicable and touch on the expected benefits and risks . We list the
/// opportunities in the context of the traditional polyhedral compiler flow
/// described in PPCG. There are essentially 6 places in the MLIR pass pipeline
/// we expect to experiment with super-vectorization:
/// 1. Right after language lowering to MLIR: this is the earliest time where
///    super-vectorization is expected to be applied. At this level, all the
///    language/user/library-level annotations are available and can be fully
///    exploited. Examples include loop-type annotations (such as parallel,
///    reduction, scan, dependence distance vector, vectorizable) as well as
///    memory access annotations (such as non-aliasing writes guaranteed,
///    indirect accesses that are permutations by construction) accesses or
///    that a particular operation is prescribed atomic by the user. At this
///    level, anything that enriches what dependence analysis can do should be
///    aggressively exploited. At this level we are close to having explicit
///    vector types in the language, except we do not impose that burden on the
///    programmer/library: we derive information from scalar code + annotations.
/// 2. After dependence analysis and before polyhedral scheduling: the
///    information that supports vectorization does not need to be supplied by a
///    higher level of abstraction. Traditional dependence anaysis is available
///    in MLIR and will be used to drive vectorization and cost models.
///
/// Let's pause here and remark that applying super-vectorization as described
/// in 1. and 2. presents clear opportunities and risks:
///   - the opportunity is that vectorization is burned in the type system and
///   is protected from the adverse effect of loop scheduling, tiling, loop
///   interchange and all passes downstream. Provided that subsequent passes are
///   able to operate on vector types; the vector shapes, associated loop
///   iterator properties, alignment, and contiguity of fastest varying
///   dimensions are preserved until we lower the super-vector types. We expect
///   this to significantly rein in on the adverse effects of phase ordering.
///   - the risks are that a. all passes after super-vectorization have to work
///   on elemental vector types (not that this is always true, wherever
///   vectorization is applied) and b. that imposing vectorization constraints
///   too early may be overall detrimental to loop fusion, tiling and other
///   transformations because the dependence distances are coarsened when
///   operating on elemental vector types. For this reason, the pattern
///   profitability analysis should include a component that also captures the
///   maximal amount of fusion available under a particular pattern. This is
///   still at the stage of rought ideas but in this context, search is our
///   friend as the Tensor Comprehensions and auto-TVM contributions
///   demonstrated previously.
/// Bottom-line is we do not yet have good answers for the above but aim at
/// making it easy to answer such questions.
///
/// Back to our listing, the last places where early super-vectorization makes
/// sense are:
/// 3. right after polyhedral-style scheduling: PLUTO-style algorithms are known
///    to improve locality, parallelism and be configurable (e.g. max-fuse,
///    smart-fuse etc). They can also have adverse effects on contiguity
///    properties that are required for vectorization but the vector_transfer
///    copy-reshape-pad-transpose abstraction is expected to help recapture
///    these properties.
/// 4. right after polyhedral-style scheduling+tiling;
/// 5. right after scheduling+tiling+rescheduling: points 4 and 5 represent
///    probably the most promising places because applying tiling achieves a
///    separation of concerns that allows rescheduling to worry less about
///    locality and more about parallelism and distribution (e.g. min-fuse).
///
/// At these levels the risk-reward looks different: on one hand we probably
/// lost a good deal of language/user/library-level annotation; on the other
/// hand we gained parallelism and locality through scheduling and tiling.
/// However we probably want to ensure tiling is compatible with the
/// full-tile-only abstraction used in super-vectorization or suffer the
/// consequences. It is too early to place bets on what will win but we expect
/// super-vectorization to be the right abstraction to allow exploring at all
/// these levels. And again, search is our friend.
///
/// Lastly, we mention it again here:
/// 6. as a MLIR-based alternative to VPLAN.
///
/// Lowering, unrolling, pipelining:
/// ================================
/// TODO(ntv): point to the proper places.
///
/// Algorithm:
/// ==========
/// The algorithm proceeds in a few steps:
///  1. defining super-vectorization patterns and matching them on the tree of
///     ForStmt. A super-vectorization pattern is defined as a recursive data
///     structures that matches and captures nested, imperfectly-nested loops
///     that have a. comformable loop annotations attached (e.g. parallel,
///     reduction, vectoriable, ...) as well as b. all contiguous load/store
///     operations along a specified minor dimension (not necessarily the
///     fastest varying) ;
///  2. analyzing those patterns for profitability (TODO(ntv): and
///     interference);
///  3. Then, for each pattern in order:
///    a. applying iterative rewriting of the loop and the load operations in
///       DFS postorder. Rewriting is implemented by coarsening the loops and
///       turning load operations into opaque vector_transfer_read ops;
///    b. keeping track of the load operations encountered as "roots" and the
///       store operations as "terminators";
///    c. traversing the use-def chains starting from the roots and iteratively
///       propagating vectorized values. Scalar values that are encountered
///       during this process must come from outside the scope of the current
///       pattern (TODO(ntv): enforce this and generalize). Such a scalar value
///       is vectorized only if it is a constant (into a vector splat). The
///       non-constant case is not supported for now and results in the pattern
///       failing to vectorize;
///    d. performing a second traversal on the terminators (store ops) to
///       rewriting the scalar value they write to memory into vector form.
///       If the scalar value has been vectorized previously, we simply replace
///       it by its vector form. Otherwise, if the scalar value is a constant,
///       it is vectorized into a splat. In all other cases, vectorization for
///       the pattern currently fails.
///    e. if everything under the root ForStmt in the current pattern vectorizes
///       properly, we commit that loop to the IR. Otherwise we discard it and
///       restore a previously cloned version of the loop. Thanks to the
///       recursive scoping nature of matchers and captured patterns, this is
///       transparently achieved by a simple RAII implementation.
///    f. vectorization is applied on the next pattern in the list. Because
///       pattern interference avoidance is not yet implemented and that we do
///       not support further vectorizing an already vector load we need to
///       re-verify that the pattern is still vectorizable. This is expected to
///       make cost models more difficult to write and is subject to improvement
///       in the future.
///
/// Points c. and d. above are worth additional comment. In most passes that
/// do not change the type of operands, it is usually preferred to eagerly
/// `replaceAllUsesWith`. Unfortunately this does not work for vectorization
/// because during the use-def chain traversal, all the operands of an operation
/// must be available in vector form. Trying to propagate eagerly makes the IR
/// temporarily invalid and results in errors such as:
///   `vectorize.mlir:308:13: error: 'addf' op requires the same type for all
///   operands and results
///      %s5 = addf %a5, %b5 : f32`
///
/// Lastly, we show a minimal example for which use-def chains rooted in load /
/// vector_transfer_read are not enough. This is what motivated splitting
/// terminator processing out of the use-def chains starting from loads. In the
/// following snippet, there is simply no load::
/// ```mlir
/// mlfunc @fill(%A : memref<128xf32>) -> () {
///   %f1 = constant 1.0 : f32
///   for %i0 = 0 to 32 {
///     store %f1, %A[%i0] : memref<128xf32, 0>
///   }
///   return
/// }
/// ```
///
/// Choice of loop transformation to support the algorithm:
/// =======================================================
/// The choice of loop transformation to apply for coarsening vectorized loops
/// is still subject to exploratory tradeoffs. In particular, say we want to
/// vectorize by a factor 128, we want to transform the following input:
/// ```mlir
///   for %i = %M to %N {
///     %a = load A[%i] : memref<?xf32>
///   }
/// ```
///
/// Traditionally, one would vectorize late (after scheduling, tiling,
/// memory promotion etc) say after stripmining (and potentially unrolling in
/// the case of LLVM's SLP vectorizer):
/// ```mlir
///   for %i = floor(%M, 128) to ceil(%N, 128) {
///     for %ii = max(%M, 128 * %i) to min(%N, 128*%i + 127) {
///       %a = load A[%ii] : memref<?xf32>
///     }
///   }
/// ```
///
/// Instead, we seek to vectorize early and freeze vector types before
/// scheduling, so we want to generate a pattern that resembles:
/// ```mlir
///   for %i = ? to ? step ? {
///     %v_a = "vector_transfer_read" (A, %i) : (memref<?xf32>, index) ->
///                                              vector<128xf32>
///   }
/// ```
///
/// i. simply dividing the lower / upper bounds by 128 creates issues
///    when representing expressions such as ii + 1 because now we only
///    have access to original values that have been divided. Additional
///    information is needed to specify accesses at below-128 granularity;
/// ii. another alternative is to coarsen the loop step but this may have
///    consequences on dependence analysis and fusability of loops: fusable
///    loops probably need to have the same step (because we don't want to
///    stripmine/unroll to enable fusion).
/// As a consequence, we choose to represent the coarsening using the loop
/// step for now and reevaluate in the future. Note that we can renormalize
/// loop steps later if/when we have evidence that they are problematic.
///
/// For the simple strawman example above, vectorizing for a 1-D vector
/// abstraction of size 128 returns code similar to:
/// ```mlir
///   for %i = %M to %N step 128 {
///     %v_a = "vector_transfer_read" (A, %i) : (memref<?xf32>, index) ->
///                                              vector<128xf32>
///   }
/// ```
///
/// Unsupported cases, extensions, and work in progress (help welcome :-) ):
/// ========================================================================
///   1. lowering to concrete vector types for various HW;
///   2. reduction support;
///   3. non-effecting padding during vector_transfer_read and filter during
///      vector_transfer_write;
///   4. misalignment support vector_transfer_read / vector_transfer_write
///      (hopefully without read-modify-writes);
///   5. control-flow support;
///   6. cost-models, heuristics and search;
///   7. Op implementation, extensions and implication on memref views;
///   8. many TODOs left around.
///
/// Examples:
/// =========
/// Consider the following MLFunction:
/// ```mlir
/// mlfunc @vector_add_2d(%M : index, %N : index) -> f32 {
///   %A = alloc (%M, %N) : memref<?x?xf32, 0>
///   %B = alloc (%M, %N) : memref<?x?xf32, 0>
///   %C = alloc (%M, %N) : memref<?x?xf32, 0>
///   %f1 = constant 1.0 : f32
///   %f2 = constant 2.0 : f32
///   for %i0 = 0 to %M {
///     for %i1 = 0 to %N {
///       // non-scoped %f1
///       store %f1, %A[%i0, %i1] : memref<?x?xf32, 0>
///     }
///   }
///   for %i2 = 0 to %M {
///     for %i3 = 0 to %N {
///       // non-scoped %f2
///       store %f2, %B[%i2, %i3] : memref<?x?xf32, 0>
///     }
///   }
///   for %i4 = 0 to %M {
///     for %i5 = 0 to %N {
///       %a5 = load %A[%i4, %i5] : memref<?x?xf32, 0>
///       %b5 = load %B[%i4, %i5] : memref<?x?xf32, 0>
///       %s5 = addf %a5, %b5 : f32
///       // non-scoped %f1
///       %s6 = addf %s5, %f1 : f32
///       // non-scoped %f2
///       %s7 = addf %s5, %f2 : f32
///       // diamond dependency.
///       %s8 = addf %s7, %s6 : f32
///       store %s8, %C[%i4, %i5] : memref<?x?xf32, 0>
///     }
///   }
///   %c7 = constant 7 : index
///   %c42 = constant 42 : index
///   %res = load %C[%c7, %c42] : memref<?x?xf32, 0>
///   return %res : f32
/// }
/// ```
///
/// TODO(ntv): update post b/119731251.
/// The -vectorize pass with the following arguments:
/// ```
/// -vectorize -virtual-vector-size 256 --test-fastest-varying=0
/// ```
///
/// produces this standard innermost-loop vectorized code:
/// ```mlir
/// mlfunc @vector_add_2d(%arg0 : index, %arg1 : index) -> f32 {
///   %0 = alloc(%arg0, %arg1) : memref<?x?xf32>
///   %1 = alloc(%arg0, %arg1) : memref<?x?xf32>
///   %2 = alloc(%arg0, %arg1) : memref<?x?xf32>
///   %cst = constant 1.0 : f32
///   %cst_0 = constant 2.0 : f32
///   for %i0 = 0 to %arg0 {
///     for %i1 = 0 to %arg1 step 256 {
///       %cst_1 = constant splat<vector<256xf32>, 1.0> :
///                vector<256xf32>
///       "vector_transfer_write"(%cst_1, %0, %i0, %i1) :
///                (vector<256xf32>, memref<?x?xf32>, index, index) -> ()
///     }
///   }
///   for %i2 = 0 to %arg0 {
///     for %i3 = 0 to %arg1 step 256 {
///       %cst_2 = constant splat<vector<256xf32>, 2.0> :
///                vector<256xf32>
///       "vector_transfer_write"(%cst_2, %1, %i2, %i3) :
///                (vector<256xf32>, memref<?x?xf32>, index, index) -> ()
///     }
///   }
///   for %i4 = 0 to %arg0 {
///     for %i5 = 0 to %arg1 step 256 {
///       %3 = "vector_transfer_read"(%0, %i4, %i5) :
///                      (memref<?x?xf32>, index, index) -> vector<256xf32>
///       %4 = "vector_transfer_read"(%1, %i4, %i5) :
///                      (memref<?x?xf32>, index, index) -> vector<256xf32>
///       %5 = addf %3, %4 : vector<256xf32>
///       %cst_3 = constant splat<vector<256xf32>, 1.0> :
///                vector<256xf32>
///       %6 = addf %5, %cst_3 : vector<256xf32>
///       %cst_4 = constant splat<vector<256xf32>, 2.0> :
///                vector<256xf32>
///       %7 = addf %5, %cst_4 : vector<256xf32>
///       %8 = addf %7, %6 : vector<256xf32>
///       "vector_transfer_write"(%8, %2, %i4, %i5) :
///                (vector<256xf32>, memref<?x?xf32>, index, index) -> ()
///     }
///   }
///   %c7 = constant 7 : index
///   %c42 = constant 42 : index
///   %9 = load %2[%c7, %c42] : memref<?x?xf32>
///   return %9 : f32
/// }
/// ```
///
/// TODO(ntv): update post b/119731251.
/// The -vectorize pass with the following arguments:
/// ```
/// -vectorize -virtual-vector-size 32 -virtual-vector-size 256
/// --test-fastest-varying=1 --test-fastest-varying=0
/// ```
///
/// produces this more insteresting mixed outer-innermost-loop vectorized code:
/// ```mlir
/// mlfunc @vector_add_2d(%arg0 : index, %arg1 : index) -> f32 {
///   %0 = alloc(%arg0, %arg1) : memref<?x?xf32>
///   %1 = alloc(%arg0, %arg1) : memref<?x?xf32>
///   %2 = alloc(%arg0, %arg1) : memref<?x?xf32>
///   %cst = constant 1.0 : f32
///   %cst_0 = constant 2.0 : f32
///   for %i0 = 0 to %arg0 step 32 {
///     for %i1 = 0 to %arg1 step 256 {
///       %cst_1 = constant splat<vector<32x256xf32>, 1.0> :
///                vector<32x256xf32>
///       "vector_transfer_write"(%cst_1, %0, %i0, %i1) :
///                (vector<32x256xf32>, memref<?x?xf32>, index, index) -> ()
///     }
///   }
///   for %i2 = 0 to %arg0 step 32 {
///     for %i3 = 0 to %arg1 step 256 {
///       %cst_2 = constant splat<vector<32x256xf32>, 2.0> :
///                vector<32x256xf32>
///       "vector_transfer_write"(%cst_2, %1, %i2, %i3) :
///                (vector<32x256xf32>, memref<?x?xf32>, index, index) -> ()
///     }
///   }
///   for %i4 = 0 to %arg0 step 32 {
///     for %i5 = 0 to %arg1 step 256 {
///       %3 = "vector_transfer_read"(%0, %i4, %i5) :
///                (memref<?x?xf32>, index, index) -> vector<32x256xf32>
///       %4 = "vector_transfer_read"(%1, %i4, %i5) :
///                (memref<?x?xf32>, index, index) -> vector<32x256xf32>
///       %5 = addf %3, %4 : vector<32x256xf32>
///       %cst_3 = constant splat<vector<32x256xf32>, 1.0> :
///                vector<32x256xf32>
///       %6 = addf %5, %cst_3 : vector<32x256xf32>
///       %cst_4 = constant splat<vector<32x256xf32>, 2.0> :
///                vector<32x256xf32>
///       %7 = addf %5, %cst_4 : vector<32x256xf32>
///       %8 = addf %7, %6 : vector<32x256xf32>
///       "vector_transfer_write"(%8, %2, %i4, %i5) :
///                (vector<32x256xf32>, memref<?x?xf32>, index, index) -> ()
///     }
///   }
///   %c7 = constant 7 : index
///   %c42 = constant 42 : index
///   %9 = load %2[%c7, %c42] : memref<?x?xf32>
///   return %9 : f32
/// }
/// ```
///
/// Of course, much more intricate n-D imperfectly-nested patterns can be
/// vectorized too and specified in a fully declarative fashion.

#define DEBUG_TYPE "early-vect"

using functional::apply;
using functional::makePtrDynCaster;
using functional::map;
using functional::ScopeGuard;
using llvm::dbgs;
using llvm::DenseSet;
using llvm::SetVector;

static llvm::cl::list<int> clVirtualVectorSize(
    "virtual-vector-size",
    llvm::cl::desc("Specify n-D virtual vector size for early vectorization"),
    llvm::cl::ZeroOrMore);

static llvm::cl::list<int> clFastestVaryingPattern(
    "test-fastest-varying",
    llvm::cl::desc(
        "Specify a 1-D, 2-D or 3-D pattern of fastest varying memory"
        " dimensions to match. See defaultPatterns in Vectorize.cpp for a"
        " description and examples. This is used for testing purposes"),
    llvm::cl::ZeroOrMore);

/// Forward declaration.
static FilterFunctionType
isVectorizableLoopPtrFactory(unsigned fastestVaryingMemRefDimension);

// Build a bunch of predetermined patterns that will be traversed in order.
// Due to the recursive nature of MLFunctionMatchers, this captures
// arbitrarily nested pairs of loops at any position in the tree.
/// Note that this currently only matches 2 nested loops and will be extended.
// TODO(ntv): support 3-D loop patterns with a common reduction loop that can
// be matched to GEMMs.
static std::vector<MLFunctionMatcher> defaultPatterns() {
  using matcher::For;
  return std::vector<MLFunctionMatcher>{
      // 3-D patterns
      For(isVectorizableLoopPtrFactory(2),
          For(isVectorizableLoopPtrFactory(1),
              For(isVectorizableLoopPtrFactory(0)))),
      // for i { for j { A[??f(not i, not j), f(i, not j), f(not i, j)];}}
      // test independently with:
      //   --test-fastest-varying=1 --test-fastest-varying=0
      For(isVectorizableLoopPtrFactory(1),
          For(isVectorizableLoopPtrFactory(0))),
      // for i { for j { A[??f(not i, not j), f(i, not j), ?, f(not i, j)];}}
      // test independently with:
      //   --test-fastest-varying=2 --test-fastest-varying=0
      For(isVectorizableLoopPtrFactory(2),
          For(isVectorizableLoopPtrFactory(0))),
      // for i { for j { A[??f(not i, not j), f(i, not j), ?, ?, f(not i, j)];}}
      // test independently with:
      //   --test-fastest-varying=3 --test-fastest-varying=0
      For(isVectorizableLoopPtrFactory(3),
          For(isVectorizableLoopPtrFactory(0))),
      // for i { for j { A[??f(not i, not j), f(not i, j), f(i, not j)];}}
      // test independently with:
      //   --test-fastest-varying=0 --test-fastest-varying=1
      For(isVectorizableLoopPtrFactory(0),
          For(isVectorizableLoopPtrFactory(1))),
      // for i { for j { A[??f(not i, not j), f(not i, j), ?, f(i, not j)];}}
      // test independently with:
      //   --test-fastest-varying=0 --test-fastest-varying=2
      For(isVectorizableLoopPtrFactory(0),
          For(isVectorizableLoopPtrFactory(2))),
      // for i { for j { A[??f(not i, not j), f(not i, j), ?, ?, f(i, not j)];}}
      // test independently with:
      //   --test-fastest-varying=0 --test-fastest-varying=3
      For(isVectorizableLoopPtrFactory(0),
          For(isVectorizableLoopPtrFactory(3))),
      // for i { A[??f(not i) , f(i)];}
      // test independently with:  --test-fastest-varying=0
      For(isVectorizableLoopPtrFactory(0)),
      // for i { A[??f(not i) , f(i), ?];}
      // test independently with:  --test-fastest-varying=1
      For(isVectorizableLoopPtrFactory(1)),
      // for i { A[??f(not i) , f(i), ?, ?];}
      // test independently with:  --test-fastest-varying=2
      For(isVectorizableLoopPtrFactory(2)),
      // for i { A[??f(not i) , f(i), ?, ?, ?];}
      // test independently with:  --test-fastest-varying=3
      For(isVectorizableLoopPtrFactory(3))};
}

/// Creates a vectorization pattern from the command line arguments.
/// Up to 3-D patterns are supported.
/// If the command line argument requests a pattern of higher order, returns an
/// empty pattern list which will conservatively result in no vectorization.
static std::vector<MLFunctionMatcher> makePatterns() {
  using matcher::For;
  if (clFastestVaryingPattern.empty()) {
    return defaultPatterns();
  }
  switch (clFastestVaryingPattern.size()) {
  case 1:
    return {For(isVectorizableLoopPtrFactory(clFastestVaryingPattern[0]))};
  case 2:
    return {For(isVectorizableLoopPtrFactory(clFastestVaryingPattern[0]),
                For(isVectorizableLoopPtrFactory(clFastestVaryingPattern[1])))};
  case 3:
    return {For(
        isVectorizableLoopPtrFactory(clFastestVaryingPattern[0]),
        For(isVectorizableLoopPtrFactory(clFastestVaryingPattern[1]),
            For(isVectorizableLoopPtrFactory(clFastestVaryingPattern[2]))))};
  default:
    return std::vector<MLFunctionMatcher>();
  }
}

namespace {

struct Vectorize : public FunctionPass {
  Vectorize() : FunctionPass(&Vectorize::passID) {}

  PassResult runOnMLFunction(MLFunction *f) override;

  // Thread-safe RAII contexts local to pass, BumpPtrAllocator freed on exit.
  MLFunctionMatcherContext MLContext;

  static char passID;
};

} // end anonymous namespace

char Vectorize::passID = 0;

/////// TODO(ntv): Hoist to a VectorizationStrategy.cpp when appropriate. //////
namespace {

struct VectorizationStrategy {
  ArrayRef<int> vectorSizes;
  DenseMap<ForStmt *, unsigned> loopToVectorDim;
};

} // end anonymous namespace

static void vectorizeLoopIfProfitable(ForStmt *loop, unsigned depthInPattern,
                                      unsigned patternDepth,
                                      VectorizationStrategy *strategy) {
  assert(patternDepth > depthInPattern &&
         "patternDepth is greater than depthInPattern");
  if (patternDepth - depthInPattern > strategy->vectorSizes.size()) {
    // Don't vectorize this loop
    return;
  }
  strategy->loopToVectorDim[loop] =
      strategy->vectorSizes.size() - (patternDepth - depthInPattern);
}

/// Implements a simple strawman strategy for vectorization.
/// Given a matched pattern `matches` of depth `patternDepth`, this strategy
/// greedily assigns the fastest varying dimension ** of the vector ** to the
/// innermost loop in the pattern.
/// When coupled with a pattern that looks for the fastest varying dimension in
/// load/store MemRefs, this creates a generic vectorization strategy that works
/// for any loop in a hierarchy (outermost, innermost or intermediate).
///
/// TODO(ntv): In the future we should additionally increase the power of the
/// profitability analysis along 3 directions:
///   1. account for loop extents (both static and parametric + annotations);
///   2. account for data layout permutations;
///   3. account for impact of vectorization on maximal loop fusion.
/// Then we can quantify the above to build a cost model and search over
/// strategies.
static bool analyzeProfitability(MLFunctionMatches matches,
                                 unsigned depthInPattern, unsigned patternDepth,
                                 VectorizationStrategy *strategy) {
  for (auto m : matches) {
    auto *loop = cast<ForStmt>(m.first);
    bool fail = analyzeProfitability(m.second, depthInPattern + 1, patternDepth,
                                     strategy);
    if (fail) {
      return fail;
    }
    vectorizeLoopIfProfitable(loop, depthInPattern, patternDepth, strategy);
  }
  return false;
}

///// end TODO(ntv): Hoist to a VectorizationStrategy.cpp when appropriate /////

namespace {

struct VectorizationState {
  /// Adds an entry of pre/post vectorization statements in the state.
  void registerReplacement(OperationStmt *key, OperationStmt *value);
  /// When the current vectorization pattern is successful, this erases the
  /// instructions that were marked for erasure in the proper order and resets
  /// the internal state for the next pattern.
  void finishVectorizationPattern();

  // In-order tracking of original OperationStmt that have been vectorized.
  // Erase in reverse order.
  SmallVector<OperationStmt *, 16> toErase;
  // Set of OperationStmt that have been vectorized (the values in the
  // vectorizationMap for hashed access). The vectorizedSet is used in
  // particular to filter the statements that have already been vectorized by
  // this pattern, when iterating over nested loops in this pattern.
  DenseSet<OperationStmt *> vectorizedSet;
  // Map of old scalar OperationStmt to new vectorized OperationStmt.
  DenseMap<OperationStmt *, OperationStmt *> vectorizationMap;
  // Map of old scalar MLValue to new vectorized MLValue.
  DenseMap<const MLValue *, MLValue *> replacementMap;
  // The strategy drives which loop to vectorize by which amount.
  const VectorizationStrategy *strategy;
  // Use-def roots. These represent the starting points for the worklist in the
  // vectorizeOperations function. They consist of the subset of load operations
  // that have been vectorized. They can be retrieved from `vectorizationMap`
  // but it is convenient to keep track of them in a separate data structure.
  DenseSet<OperationStmt *> roots;
  // Terminator statements for the worklist in the vectorizeOperations function.
  // They consist of the subset of store operations that have been vectorized.
  // They can be retrieved from `vectorizationMap` but it is convenient to keep
  // track of them in a separate data structure. Since they do not necessarily
  // belong to use-def chains starting from loads (e.g storing a constant), we
  // need to handle them in a post-pass.
  DenseSet<OperationStmt *> terminators;
  // Checks that the type of `stmt` is StoreOp and adds it to the terminators
  // set.
  void registerTerminator(OperationStmt *stmt);

private:
  void registerReplacement(const SSAValue *key, SSAValue *value);
};

} // end namespace

void VectorizationState::registerReplacement(OperationStmt *key,
                                             OperationStmt *value) {
  LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ commit vectorized op: ");
  LLVM_DEBUG(key->print(dbgs()));
  LLVM_DEBUG(dbgs() << "  into  ");
  LLVM_DEBUG(value->print(dbgs()));
  assert(key->getNumResults() == 1 && "already registered");
  assert(value->getNumResults() == 1 && "already registered");
  assert(vectorizedSet.count(value) == 0 && "already registered");
  assert(vectorizationMap.count(key) == 0 && "already registered");
  toErase.push_back(key);
  vectorizedSet.insert(value);
  vectorizationMap.insert(std::make_pair(key, value));
  registerReplacement(key->getResult(0), value->getResult(0));
  if (key->isa<LoadOp>()) {
    assert(roots.count(key) == 0 && "root was already inserted previously");
    roots.insert(key);
  }
}

void VectorizationState::registerTerminator(OperationStmt *stmt) {
  assert(stmt->isa<StoreOp>() && "terminator must be a StoreOp");
  assert(terminators.count(stmt) == 0 &&
         "terminator was already inserted previously");
  terminators.insert(stmt);
}

void VectorizationState::finishVectorizationPattern() {
  while (!toErase.empty()) {
    auto *stmt = toErase.pop_back_val();
    LLVM_DEBUG(dbgs() << "\n[early-vect] finishVectorizationPattern erase: ");
    LLVM_DEBUG(stmt->print(dbgs()));
    stmt->erase();
  }
}

void VectorizationState::registerReplacement(const SSAValue *key,
                                             SSAValue *value) {
  assert(replacementMap.count(cast<MLValue>(key)) == 0 &&
         "replacement already registered");
  replacementMap.insert(
      std::make_pair(cast<MLValue>(key), cast<MLValue>(value)));
}

////// TODO(ntv): Hoist to a VectorizationMaterialize.cpp when appropriate. ////

/// Handles the vectorization of load and store MLIR operations.
///
/// LoadOp operations are the roots of the vectorizeOperations call. They are
/// vectorized immediately. The resulting vector_transfer_read is immediately
/// registered to replace all uses of the LoadOp in this pattern's scope.
///
/// StoreOp are the terminators of the vectorizeOperations call. They need
/// to be vectorized late once all the use-def chains have been traversed.
/// Additionally, they may have ssa-values operands which come from outside
/// the scope of the current pattern.
/// Such special cases force us to delay the vectorization of the stores
/// until the last step. Here we merely register the store operation.
template <typename LoadOrStoreOpPointer>
static bool vectorizeRootOrTerminal(MLValue *iv, LoadOrStoreOpPointer memoryOp,
                                    VectorizationState *state) {
  auto memRefType =
      memoryOp->getMemRef()->getType().template cast<MemRefType>();

  auto elementType = memRefType.getElementType();
  // TODO(ntv): ponder whether we want to further vectorize a vector value.
  assert(VectorType::isValidElementType(elementType) &&
         "Not a valid vector element type");
  auto vectorType = VectorType::get(state->strategy->vectorSizes, elementType);

  // Materialize a MemRef with 1 vector.
  auto *opStmt = cast<OperationStmt>(memoryOp->getOperation());
  // For now, vector_transfers must be aligned, operate only on indices with an
  // identity subset of AffineMap and do not change layout.
  // TODO(ntv): increase the expressiveness power of vector_transfer operations
  // as needed by various targets.
  if (opStmt->template isa<LoadOp>()) {
    auto permutationMap =
        makePermutationMap(opStmt, state->strategy->loopToVectorDim);
    LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ permutationMap: ");
    LLVM_DEBUG(permutationMap.print(dbgs()));
    MLFuncBuilder b(opStmt);
    auto transfer = b.create<VectorTransferReadOp>(
        opStmt->getLoc(), vectorType, memoryOp->getMemRef(),
        map(makePtrDynCaster<SSAValue>(), memoryOp->getIndices()),
        permutationMap);
    state->registerReplacement(opStmt,
                               cast<OperationStmt>(transfer->getOperation()));
  } else {
    state->registerTerminator(opStmt);
  }
  return false;
}
/// end TODO(ntv): Hoist to a VectorizationMaterialize.cpp when appropriate. ///

/// Coarsens the loops bounds and transforms all remaining load and store
/// operations into the appropriate vector_transfer.
static bool vectorizeForStmt(ForStmt *loop, int64_t step,
                             VectorizationState *state) {
  using namespace functional;
  loop->setStep(step);

  FilterFunctionType notVectorizedThisPattern = [state](const Statement &stmt) {
    if (!matcher::isLoadOrStore(stmt)) {
      return false;
    }
    auto *opStmt = cast<OperationStmt>(&stmt);
    return state->vectorizationMap.count(opStmt) == 0 &&
           state->vectorizedSet.count(opStmt) == 0 &&
           state->roots.count(opStmt) == 0 &&
           state->terminators.count(opStmt) == 0;
  };
  auto loadAndStores = matcher::Op(notVectorizedThisPattern);
  auto matches = loadAndStores.match(loop);
  for (auto ls : matches) {
    auto *opStmt = cast<OperationStmt>(ls.first);
    auto load = opStmt->dyn_cast<LoadOp>();
    auto store = opStmt->dyn_cast<StoreOp>();
    LLVM_DEBUG(opStmt->print(dbgs()));
    auto fail = load ? vectorizeRootOrTerminal(loop, load, state)
                     : vectorizeRootOrTerminal(loop, store, state);
    if (fail) {
      return fail;
    }
  }
  return false;
}

/// Returns a FilterFunctionType that can be used in MLFunctionMatcher to
/// match a loop whose underlying load/store accesses are all varying along the
/// `fastestVaryingMemRefDimension`.
/// TODO(ntv): In the future, allow more interesting mixed layout permutation
/// once we understand better the performance implications and we are confident
/// we can build a cost model and a search procedure.
static FilterFunctionType
isVectorizableLoopPtrFactory(unsigned fastestVaryingMemRefDimension) {
  return [fastestVaryingMemRefDimension](const Statement &forStmt) {
    const auto &loop = cast<ForStmt>(forStmt);
    return isVectorizableLoopAlongFastestVaryingMemRefDim(
        loop, fastestVaryingMemRefDimension);
  };
}

/// Forward-declaration.
static bool vectorizeNonRoot(MLFunctionMatches matches,
                             VectorizationState *state);

/// Apply vectorization of `loop` according to `state`. This is only triggered
/// if all vectorizations in `childrenMatches` have already succeeded
/// recursively in DFS post-order.
static bool doVectorize(MLFunctionMatches::EntryType oneMatch,
                        VectorizationState *state) {
  ForStmt *loop = cast<ForStmt>(oneMatch.first);
  MLFunctionMatches childrenMatches = oneMatch.second;

  // 1. DFS postorder recursion, if any of my children fails, I fail too.
  auto fail = vectorizeNonRoot(childrenMatches, state);
  if (fail) {
    // Early exit and trigger RAII cleanups at the root.
    return fail;
  }

  // 2. This loop may have been omitted from vectorization for various reasons
  // (e.g. due to the performance model or pattern depth > vector size).
  auto it = state->strategy->loopToVectorDim.find(loop);
  if (it == state->strategy->loopToVectorDim.end()) {
    return false;
  }

  // 3. Actual post-order transformation.
  auto vectorDim = it->second;
  assert(vectorDim < state->strategy->vectorSizes.size() &&
         "vector dim overflow");
  //   a. get actual vector size
  auto vectorSize = state->strategy->vectorSizes[vectorDim];
  //   b. loop transformation for early vectorization is still subject to
  //     exploratory tradeoffs (see top of the file). Apply coarsening, i.e.:
  //        | ub -> ub
  //        | step -> step * vectorSize
  LLVM_DEBUG(dbgs() << "\n[early-vect] vectorizeForStmt by " << vectorSize
                    << " : ");
  LLVM_DEBUG(loop->print(dbgs()));
  return vectorizeForStmt(loop, loop->getStep() * vectorSize, state);
}

/// Non-root pattern iterates over the matches at this level, calls doVectorize
/// and exits early if anything below fails.
static bool vectorizeNonRoot(MLFunctionMatches matches,
                             VectorizationState *state) {
  for (auto m : matches) {
    auto fail = doVectorize(m, state);
    if (fail) {
      // Early exit and trigger RAII cleanups at the root.
      return fail;
    }
  }
  return false;
}

/// Tries to transform a scalar constant into a vector splat of that constant.
/// Returns the vectorized splat operation if the constant is a valid vector
/// element type.
/// If `type` is not a valid vector type or if the scalar constant is not a
/// valid vector element type, returns nullptr.
static MLValue *vectorizeConstant(Statement *stmt, const ConstantOp &constant,
                                  Type type) {
  if (!type || !type.isa<VectorType>() ||
      !VectorType::isValidElementType(constant.getType())) {
    return nullptr;
  }
  MLFuncBuilder b(stmt);
  Location loc = stmt->getLoc();
  auto vectorType = type.cast<VectorType>();
  auto attr = SplatElementsAttr::get(vectorType, constant.getValue());
  auto *constantOpStmt = cast<OperationStmt>(constant.getOperation());
  auto *splat = cast<OperationStmt>(b.createOperation(
      loc, constantOpStmt->getName(), {}, {vectorType},
      {make_pair(Identifier::get("value", b.getContext()), attr)}));
  return cast<MLValue>(splat->getResult(0));
}

/// Returns a uniqu'ed VectorType.
/// In the case `v`'s defining statement is already part of the `state`'s
/// vectorizedSet, just returns the type of `v`.
/// Otherwise, constructs a new VectorType of shape defined by `state.strategy`
/// and of elemental type the type of `v`.
static Type getVectorType(SSAValue *v, const VectorizationState &state) {
  if (!VectorType::isValidElementType(v->getType())) {
    return Type();
  }
  auto *definingOpStmt = cast<OperationStmt>(v->getDefiningStmt());
  if (state.vectorizedSet.count(definingOpStmt) > 0) {
    return v->getType().cast<VectorType>();
  }
  return VectorType::get(state.strategy->vectorSizes, v->getType());
};

/// Tries to vectorize a given operand `op` of Statement `stmt` during def-chain
/// propagation or during terminator vectorization, by applying the following
/// logic:
/// 1. if the defining statement is part of the vectorizedSet (i.e. vectorized
///    useby -def propagation), `op` is already in the proper vector form;
/// 2. otherwise, the `op` may be in some other vector form that fails to
///    vectorize atm (i.e. broadcasting required), returns nullptr to indicate
///    failure;
/// 3. if the `op` is a constant, returns the vectorized form of the constant;
/// 4. non-constant scalars are currently non-vectorizable, in particular to
///    guard against vectorizing an index which may be loop-variant and needs
///    special handling.
///
/// In particular this logic captures some of the use cases where definitions
/// that are not scoped under the current pattern are needed to vectorize.
/// One such example is top level function constants that need to be splatted.
///
/// Returns an operand that has been vectorized to match `state`'s strategy if
/// vectorization is possible with the above logic. Returns nullptr otherwise.
///
/// TODO(ntv): handle more complex cases.
static MLValue *vectorizeOperand(SSAValue *operand, Statement *stmt,
                                 VectorizationState *state) {
  LLVM_DEBUG(dbgs() << "\n[early-vect]vectorize operand: ");
  LLVM_DEBUG(operand->print(dbgs()));
  auto *definingStatement = cast<OperationStmt>(operand->getDefiningStmt());
  // 1. If this value has already been vectorized this round, we are done.
  if (state->vectorizedSet.count(definingStatement) > 0) {
    LLVM_DEBUG(dbgs() << " -> already vector operand");
    return cast<MLValue>(operand);
  }
  // 1.b. Delayed on-demand replacement of a use.
  //    Note that we cannot just call replaceAllUsesWith because it may result
  //    in ops with mixed types, for ops whose operands have not all yet
  //    been vectorized. This would be invalid IR.
  auto it = state->replacementMap.find(cast<MLValue>(operand));
  if (it != state->replacementMap.end()) {
    auto *res = cast<MLValue>(it->second);
    LLVM_DEBUG(dbgs() << "-> delayed replacement by: ");
    LLVM_DEBUG(res->print(dbgs()));
    return res;
  }
  // 2. TODO(ntv): broadcast needed.
  if (operand->getType().isa<VectorType>()) {
    LLVM_DEBUG(dbgs() << "-> non-vectorizable");
    return nullptr;
  }
  // 3. vectorize constant.
  if (auto constant = operand->getDefiningStmt()->dyn_cast<ConstantOp>()) {
    return vectorizeConstant(stmt, *constant,
                             getVectorType(operand, *state).cast<VectorType>());
  }
  // 4. currently non-vectorizable.
  LLVM_DEBUG(dbgs() << "-> non-vectorizable");
  LLVM_DEBUG(operand->print(dbgs()));
  return nullptr;
};

/// Encodes OperationStmt-specific behavior for vectorization. In general we
/// assume that all operands of an op must be vectorized but this is not always
/// true. In the future, it would be nice to have a trait that describes how a
/// particular operation vectorizes. For now we implement the case distinction
/// here.
/// Returns a vectorized form of stmt or nullptr if vectorization fails.
/// TODO(ntv): consider adding a trait to Op to describe how it gets vectorized.
/// Maybe some Ops are not vectorizable or require some tricky logic, we cannot
/// do one-off logic here; ideally it would be TableGen'd.
static OperationStmt *vectorizeOneOperationStmt(MLFuncBuilder *b,
                                                OperationStmt *opStmt,
                                                VectorizationState *state) {
  // Sanity checks.
  assert(!opStmt->isa<LoadOp>() &&
         "all loads must have already been fully vectorized independently");
  assert(!opStmt->isa<VectorTransferReadOp>() &&
         "vector_transfer_read cannot be further vectorized");
  assert(!opStmt->isa<VectorTransferWriteOp>() &&
         "vector_transfer_write cannot be further vectorized");

  if (auto store = opStmt->dyn_cast<StoreOp>()) {
    auto *memRef = store->getMemRef();
    auto *value = store->getValueToStore();
    auto *vectorValue = vectorizeOperand(value, opStmt, state);
    auto indices = map(makePtrDynCaster<SSAValue>(), store->getIndices());
    MLFuncBuilder b(opStmt);
    auto permutationMap =
        makePermutationMap(opStmt, state->strategy->loopToVectorDim);
    LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ permutationMap: ");
    LLVM_DEBUG(permutationMap.print(dbgs()));
    auto transfer = b.create<VectorTransferWriteOp>(
        opStmt->getLoc(), vectorValue, memRef, indices, permutationMap);
    auto *res = cast<OperationStmt>(transfer->getOperation());
    LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ vectorized store: " << *res);
    // "Terminators" (i.e. StoreOps) are erased on the spot.
    opStmt->erase();
    return res;
  }

  auto types = map([state](SSAValue *v) { return getVectorType(v, *state); },
                   opStmt->getResults());
  auto vectorizeOneOperand = [opStmt, state](SSAValue *op) {
    return vectorizeOperand(op, opStmt, state);
  };
  auto operands = map(vectorizeOneOperand, opStmt->getOperands());
  // Check whether a single operand is null. If so, vectorization failed.
  bool success = llvm::any_of(operands, [](SSAValue *op) { return op; });
  if (!success) {
    LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ an operand failed vectorize");
    return nullptr;
  }

  // Create a clone of the op with the proper operands and return types.
  // TODO(ntv): The following assumes there is always an op with a fixed
  // name that works both in scalar mode and vector mode.
  // TODO(ntv): Is it worth considering an OperationStmt.clone operation
  // which changes the type so we can promote an OperationStmt with less
  // boilerplate?
  return cast<OperationStmt>(b->createOperation(opStmt->getLoc(),
                                                opStmt->getName(), operands,
                                                types, opStmt->getAttrs()));
}

/// Iterates over the OperationStmt in the loop and rewrites them using their
/// vectorized counterpart by:
///   1. iteratively building a worklist of uses of the OperationStmt vectorized
///   so far by this pattern;
///   2. for each OperationStmt in the worklist, create the vector form of this
///   operation and replace all its uses by the vectorized form. For this step,
///   the worklist must be traversed in order;
///   3. verify that all operands of the newly vectorized operation have been
///   vectorized by this pattern.
static bool vectorizeOperations(VectorizationState *state) {
  // 1. create initial worklist with the uses of the roots.
  SetVector<OperationStmt *> worklist;
  auto insertUsesOf = [&worklist, state](Operation *vectorized) {
    for (auto *r : cast<OperationStmt>(vectorized)->getResults())
      for (auto &u : r->getUses()) {
        auto *stmt = cast<OperationStmt>(u.getOwner());
        // Don't propagate to terminals, a separate pass is needed for those.
        // TODO(ntv)[b/119759136]: use isa<> once Op is implemented.
        if (state->terminators.count(stmt) > 0) {
          continue;
        }
        worklist.insert(stmt);
      }
  };
  apply(insertUsesOf, state->roots);

  // Note: Worklist size increases iteratively. At each round we evaluate the
  // size again. By construction, the order of elements in the worklist is
  // consistent across iterations.
  for (unsigned i = 0; i < worklist.size(); ++i) {
    auto *stmt = worklist[i];
    LLVM_DEBUG(dbgs() << "\n[early-vect] vectorize use: ");
    LLVM_DEBUG(stmt->print(dbgs()));

    // 2. Create vectorized form of the statement.
    // Insert it just before stmt, on success register stmt as replaced.
    MLFuncBuilder b(stmt);
    auto *vectorizedStmt = vectorizeOneOperationStmt(&b, stmt, state);
    if (!vectorizedStmt) {
      return true;
    }

    // 3. Register replacement for future uses in the scop.
    //    Note that we cannot just call replaceAllUsesWith because it may
    //    result in ops with mixed types, for ops whose operands have not all
    //    yet been vectorized. This would be invalid IR.
    state->registerReplacement(cast<OperationStmt>(stmt), vectorizedStmt);

    // 4. Augment the worklist with uses of the statement we just vectorized.
    // This preserves the proper order in the worklist.
    apply(insertUsesOf, ArrayRef<Operation *>{stmt});
  }
  return false;
}

/// Vectorization is a recursive procedure where anything below can fail.
/// The root match thus needs to maintain a clone for handling failure.
/// Each root may succeed independently but will otherwise clean after itself if
/// anything below it fails.
static bool vectorizeRootMatches(MLFunctionMatches matches,
                                 VectorizationStrategy *strategy) {
  for (auto m : matches) {
    auto *loop = cast<ForStmt>(m.first);
    VectorizationState state;
    state.strategy = strategy;

    // Since patterns are recursive, they can very well intersect.
    // Since we do not want a fully greedy strategy in general, we decouple
    // pattern matching, from profitability analysis, from application.
    // As a consequence we must check that each root pattern is still
    // vectorizable. If a pattern is not vectorizable anymore, we just skip it.
    // TODO(ntv): implement a non-greedy profitability analysis that keeps only
    // non-intersecting patterns.
    if (!isVectorizableLoop(*loop)) {
      LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ loop is not vectorizable");
      continue;
    }
    MLFuncBuilder builder(loop); // builder to insert in place of loop
    DenseMap<const MLValue *, MLValue *> nomap;
    ForStmt *clonedLoop = cast<ForStmt>(builder.clone(*loop, nomap));
    auto fail = doVectorize(m, &state);
    /// Sets up error handling for this root loop. This is how the root match
    /// maintains a clone for handling failure and restores the proper state via
    /// RAII.
    ScopeGuard sg2([&fail, loop, clonedLoop]() {
      if (fail) {
        loop->replaceAllUsesWith(clonedLoop);
        loop->erase();
      } else {
        clonedLoop->erase();
      }
    });
    if (fail) {
      LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ failed root doVectorize");
      continue;
    }

    // Form the root operationsthat have been set in the replacementMap.
    // For now, these roots are the loads for which vector_transfer_read
    // operations have been inserted.
    auto getDefiningOperation = [](const MLValue *val) {
      return const_cast<MLValue *>(val)->getDefiningOperation();
    };
    using ReferenceTy = decltype(*(state.replacementMap.begin()));
    auto getKey = [](ReferenceTy it) { return it.first; };
    auto roots = map(getDefiningOperation, map(getKey, state.replacementMap));

    // Vectorize the root operations and everything reached by use-def chains
    // except the terminators (store statements) that need to be post-processed
    // separately.
    fail = vectorizeOperations(&state);
    if (fail) {
      LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ failed vectorizeOperations");
      continue;
    }

    // Finally, vectorize the terminators. If anything fails to vectorize, skip.
    auto vectorizeOrFail = [&fail, &state](OperationStmt *stmt) {
      if (fail) {
        return;
      }
      MLFuncBuilder b(stmt);
      auto *res = vectorizeOneOperationStmt(&b, stmt, &state);
      if (res == nullptr) {
        fail = true;
      }
    };
    apply(vectorizeOrFail, state.terminators);
    if (fail) {
      LLVM_DEBUG(
          dbgs() << "\n[early-vect]+++++ failed to vectorize terminators");
      continue;
    } else {
      LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ success vectorizing pattern");
    }

    // Finish this vectorization pattern.
    state.finishVectorizationPattern();
  }
  return false;
}

/// Applies vectorization to the current MLFunction by searching over a bunch of
/// predetermined patterns.
PassResult Vectorize::runOnMLFunction(MLFunction *f) {
  for (auto pat : makePatterns()) {
    LLVM_DEBUG(dbgs() << "\n******************************************");
    LLVM_DEBUG(dbgs() << "\n******************************************");
    LLVM_DEBUG(dbgs() << "\n[early-vect] new pattern on MLFunction\n");
    LLVM_DEBUG(f->print(dbgs()));
    unsigned patternDepth = pat.getDepth();
    auto matches = pat.match(f);
    // Iterate over all the top-level matches and vectorize eagerly.
    // This automatically prunes intersecting matches.
    for (auto m : matches) {
      VectorizationStrategy strategy;
      // TODO(ntv): depending on profitability, elect to reduce the vector size.
      strategy.vectorSizes = clVirtualVectorSize;
      auto fail = analyzeProfitability(m.second, 1, patternDepth, &strategy);
      if (fail) {
        continue;
      }
      auto *loop = cast<ForStmt>(m.first);
      vectorizeLoopIfProfitable(loop, 0, patternDepth, &strategy);
      // TODO(ntv): if pattern does not apply, report it; alter the
      // cost/benefit.
      fail = vectorizeRootMatches(matches, &strategy);
      assert(!fail && "top-level failure should not happen");
      // TODO(ntv): some diagnostics.
    }
  }
  LLVM_DEBUG(dbgs() << "\n");
  return PassResult::Success;
}

FunctionPass *mlir::createVectorizePass() { return new Vectorize(); }

static PassRegistration<Vectorize>
    pass("vectorize",
         "Vectorize to a target independent n-D vector abstraction");
