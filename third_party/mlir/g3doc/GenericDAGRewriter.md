# MLIR Generic DAG Rewriter Infrastructure

## Introduction and Motivation

The goal of a compiler IR is to represent code - at various levels of
abstraction which pose different sets of tradeoffs in terms of representational
capabilities and ease of transformation. However, the ability to represent code
is not itself very useful - you also need to be able to implement those
transformations.

There are many different sorts of compiler transformations, but this document
focuses on a particularly important class of transformation that comes up
repeatedly at scale, and is important for the immediate goals of MLIR: that of
pattern matching on a set of operations and replacing with another set. This is
the key algorithm required to implement the "op fission" algorithm used by the
tf2xla bridge, pattern matching rewrites from TF ops to TF/Lite, peephole
optimizations like "eliminate identity nodes" or "replace x+0 with x", as well
as a useful abstraction to implement optimization algorithms for MLIR graphs at
all levels.

A particular strength of MLIR (and a major difference vs other compiler
infrastructures like LLVM, GCC, XLA, TensorFlow, etc) is that it uses a single
compiler IR to represent code at multiple levels of abstraction: an MLIR
operation can be a "TensorFlow operation", an "XLA HLO", a "TF Lite
FlatBufferModel op", a TPU LLO instruction, an LLVM IR instruction (transitively
including X86, Lanai, CUDA, and other target specific instructions), or anything
else that the MLIR type system can reasonably express. Because MLIR spans such a
wide range of different problems, a single infrastructure for performing
graph-to-graph rewrites can help solve many diverse domain challenges, including
TensorFlow graph level down to the machine code level.

[Static single assignment](https://en.wikipedia.org/wiki/Static_single_assignment_form)
(SSA) representations like MLIR make it easy to access the operands and "users"
of an operation. As such, a natural abstraction for these graph-to-graph
rewrites is that of DAG pattern matching: clients define DAG tile patterns, and
each pattern includes a result DAG to produce and the cost of the result (or,
inversely, the benefit of doing the replacement). A common infrastructure
efficiently finds and perform the rewrites.

While this concept is simple, the details are more nuanced. This proposal
defines and explores a set of abstractions that we feel can solve a wide range
of different problems, and can be applied to many different sorts of problems
that MLIR is - and is expected to - face over time. We do this by separating the
pattern definition and matching algorithm from the "driver" of the computation
loop, and make space for the patterns to be defined declaratively in the future.

## Related Work

There is a huge amount of related work to consider, given that pretty much every
compiler in existence has to solve this problem many times over. Here are a few
graph rewrite systems we have used, along with the pros and cons of this related
work. One unifying problem with all of these is that these systems are only
trying to solve one particular and usually narrow problem: our proposal would
like to solve many of these problems with a single infrastructure. Of these, the
most similar design to our proposal is the LLVM DAG-to-DAG instruction selection
algorithm at the end.

### Constant folding

A degenerate but pervasive case of DAG-to-DAG pattern matching is constant
folding: given an operation whose operands contain constants can often be folded
to a result constant value.

MLIR already has constant folding routines which provide a simpler API than a
general DAG-to-DAG pattern matcher, and we expect it to remain because the
simpler contract makes it applicable in some cases that a generic matcher would
not. For example, a DAG-rewrite can remove arbitrary nodes in the current
function, which could invalidate iterators. Constant folding as an API does not
remove any nodes, it just provides a (list of) constant values and allows the
clients to update their data structures as necessary.

### AST-Level Pattern Matchers

The literature is full of source-to-source translators which transform
identities in order to improve performance (e.g. transforming `X*0` into `0`).
One large example that I'm aware of is the GCC `fold` function, which performs
[many optimizations](https://github.com/gcc-mirror/gcc/blob/master/gcc/fold-const.c)
on ASTs. Clang has
[similar routines](http://releases.llvm.org/3.5.0/tools/clang/docs/InternalsManual.html#constant-folding-in-the-clang-ast)
for simple constant folding of expressions (as required by the C++ standard) but
doesn't perform general optimizations on its ASTs.

The primary downside of tree optimizers is that you can't see across operations
that have multiple uses. It is
[well known in literature](https://llvm.org/pubs/2008-06-LCTES-ISelUsingSSAGraphs.pdf)
that DAG pattern matching is more powerful than tree pattern matching, but OTOH,
DAG pattern matching can lead to duplication of computation which needs to be
checked for.

### "Combiners" and other peephole optimizers

Compilers end up with a lot of peephole optimizers for various things, e.g. the
GCC
["combine" routines](https://github.com/gcc-mirror/gcc/blob/master/gcc/combine.c)
(which try to merge two machine instructions into a single one), the LLVM
[Inst Combine](http://llvm.org/viewvc/llvm-project/llvm/trunk/lib/Transforms/InstCombine/)
[pass](https://llvm.org/docs/Passes.html#instcombine-combine-redundant-instructions),
LLVM's
[DAG Combiner](https://github.com/llvm-mirror/llvm/blob/master/lib/CodeGen/SelectionDAG/DAGCombiner.cpp),
the Swift compiler's
[SIL Combiner](https://github.com/apple/swift/tree/master/lib/SILOptimizer/SILCombiner),
etc. These generally match one or more operations and produce zero or more
operations as a result. The LLVM
[Legalization](http://llvm.org/viewvc/llvm-project/llvm/trunk/lib/CodeGen/SelectionDAG/)
infrastructure has a different outer loop but otherwise works the same way.

These passes have a lot of diversity, but also have a unifying structure: they
mostly have a worklist outer loop which visits operations. They then use the C++
visitor pattern (or equivalent) to switch over the class of operation and
dispatch to a method. That method contains a long list of hand-written C++ code
that pattern-matches various special cases. LLVM introduced a "match" function
that allows writing patterns in a somewhat more declarative style using template
metaprogramming (MLIR has similar facilities). Here's a simple example:

```c++
  // Y - (X + 1) --> ~X + Y
  if (match(Op1, m_OneUse(m_Add(m_Value(X), m_One()))))
    return BinaryOperator::CreateAdd(Builder.CreateNot(X), Op0);
```

Here is a somewhat more complicated one (this is not the biggest or most
complicated :)

```c++
  // C2 is ODD
  // LHS = XOR(Y,C1), Y = AND(Z,C2), C1==(C2+1) => LHS == NEG(OR(Z, ~C2))
  // ADD(LHS, RHS) == SUB(RHS, OR(Z, ~C2))
  if (match(LHS, m_Xor(m_Value(Y), m_APInt(C1))))
    if (C1->countTrailingZeros() == 0)
      if (match(Y, m_And(m_Value(Z), m_APInt(C2))) && *C1 == (*C2 + 1)) {
        Value NewOr = Builder.CreateOr(Z, ~(*C2));
        return Builder.CreateSub(RHS, NewOr, "sub");
      }
```

These systems are simple to set up, and pattern matching templates have some
advantages (they are extensible for new sorts of sub-patterns, look compact at
point of use). OTOH, they have lots of well known problems, for example:

*   These patterns are very error prone to write, and contain lots of
    redundancies.
*   The IR being matched often has identities (e.g. when matching commutative
    operators) and the C++ code has to handle it manually - take a look at
    [the full code](http://llvm.org/viewvc/llvm-project/llvm/trunk/lib/Transforms/InstCombine/InstCombineAddSub.cpp?view=markup#l775)
    for checkForNegativeOperand that defines the second pattern).
*   The matching code compiles slowly, both because it generates tons of code
    and because the templates instantiate slowly.
*   Adding new patterns (e.g. for count leading zeros in the example above) is
    awkward and doesn't often happen.
*   The cost model for these patterns is not really defined - it is emergent
    based on the order the patterns are matched in code.
*   They are non-extensible without rebuilding the compiler.
*   It isn't practical to apply theorem provers and other tools to these
    patterns - they cannot be reused for other purposes.

In addition to structured "combiners" like these, there are lots of ad-hoc
systems like the
[LLVM Machine code peephole optimizer](http://llvm.org/viewvc/llvm-project/llvm/trunk/lib/CodeGen/PeepholeOptimizer.cpp?view=markup)
which are related.

### LLVM's DAG-to-DAG Instruction Selection Infrastructure

The instruction selection subsystem in LLVM is the result of many years worth of
iteration and discovery, driven by the need for LLVM to support code generation
for lots of targets, the complexity of code generators for modern instruction
sets (e.g. X86), and the fanatical pursuit of reusing code across targets. Eli
wrote a
[nice short overview](https://eli.thegreenplace.net/2013/02/25/a-deeper-look-into-the-llvm-code-generator-part-1)
of how this works, and the
[LLVM documentation](https://llvm.org/docs/CodeGenerator.html#select-instructions-from-dag)
describes it in more depth including its advantages and limitations. It allows
writing patterns like this.

```
def : Pat<(or GR64:$src, (not (add GR64:$src, 1))),
          (BLCI64rr GR64:$src)>;
```

This example defines a matcher for the
["blci" instruction](https://en.wikipedia.org/wiki/Bit_Manipulation_Instruction_Sets#TBM_\(Trailing_Bit_Manipulation\))
in the
[X86 target description](http://llvm.org/viewvc/llvm-project/llvm/trunk/lib/Target/X86/X86InstrInfo.td?view=markup),
there are many others in that file (look for `Pat<>` patterns, since they aren't
entangled in details of the compiler like assembler/disassembler generation
logic).

For our purposes, there is much to like about this system, for example:

*   It is defined in a declarative format.
*   It is extensible to target-defined operations.
*   It automates matching across identities, like commutative patterns.
*   It allows custom abstractions and intense factoring of target-specific
    commonalities.
*   It generates compact code - it compiles into a state machine, which is
    interpreted.
*   It allows the instruction patterns to be defined and reused for multiple
    purposes.
*   The patterns are "type checked" at compile time, detecting lots of bugs
    early and eliminating redundancy from the pattern specifications.
*   It allows the use of general C++ code for weird/complex cases.

While there is a lot that is good here, there is also a lot of bad things:

*   All of this machinery is only applicable to instruction selection. Even
    directly adjacent problems like the DAGCombiner and Legalizer can't use it.
*   This isn't extensible at compiler runtime, you have to rebuild the compiler
    to extend it.
*   The error messages when failing to match a pattern
    [are not exactly optimal](https://www.google.com/search?q=llvm+cannot+select).
*   It has lots of implementation problems and limitations (e.g. can't write a
    pattern for a multi-result operation) as a result of working with the
    awkward SelectionDAG representation and being designed and implemented
    lazily.
*   This stuff all grew organically over time and has lots of sharp edges.

### Summary

MLIR will face a wide range of pattern matching and graph rewrite problems, and
one of the major advantages of having a common representation for code at
multiple levels that it allows us to invest in - and highly leverage - a single
infra for doing this sort of work.

## Goals

This proposal includes support for defining pattern matching and rewrite
algorithms on MLIR. We'd like these algorithms to encompass many problems in the
MLIR space, including 1-to-N expansions (e.g. as seen in the TF/XLA bridge when
lowering a "tf.AddN" to multiple "add" HLOs), M-to-1 patterns (as seen in
Grappler optimization passes, e.g. that convert multiple/add into a single
muladd op), as well as general M-to-N patterns (e.g. instruction selection for
target instructions). Patterns should have a cost associated with them, and the
common infrastructure should be responsible for sorting out the lowest cost
match for a given application.

We separate the task of picking a particular locally optimal pattern from a
given root node, the algorithm used to rewrite an entire graph given a
particular set of goals, and the definition of the patterns themselves. We do
this because DAG tile pattern matching is NP complete, which means that there
are no known polynomial time algorithms to optimally solve this problem.
Additionally, we would like to support iterative rewrite algorithms that
progressively transform the input program through multiple steps. Furthermore,
we would like to support many different sorts of clients across the MLIR stack,
and they may have different tolerances for compile time cost, different demands
for optimality, and other algorithmic goals or constraints.

We aim for MLIR transformations to be easy to implement and reduce the
likelihood for compiler bugs. We expect there to be a very very large number of
patterns that are defined over time, and we believe that these sorts of patterns
will have a very large number of legality/validity constraints - many of which
are difficult to reason about in a consistent way, may be target specific, and
whose implementation may be particularly bug-prone. As such, we aim to design the
API around pattern definition to be simple, resilient to programmer errors, and
allow separation of concerns between the legality of the nodes generated from
the idea of the pattern being defined.

Finally, error handling is a topmost concern: in addition to allowing patterns
to be defined in a target-independent way that may not apply for all hardware,
we also want failure for any pattern to match to be diagnosable in a reasonable
way. To be clear, this is not a solvable problem in general - the space of
malfunction is too great to be fully enumerated and handled optimally, but there
are better and worse ways to handle the situation. MLIR is already designed to
represent the provenance of an operation well. This project aims to propagate
that provenance information precisely, as well as diagnose pattern match
failures with the rationale for why a set of patterns do not apply.

### Non goals

This proposal doesn't aim to solve all compiler problems, it is simply a
DAG-to-DAG pattern matching system, starting with a greedy driver algorithm.
Compiler algorithms that require global dataflow analysis (e.g. common
subexpression elimination, conditional constant propagation, and many many
others) will not be directly solved by this infrastructure.

This proposal is limited to DAG patterns, which (by definition) prevent the
patterns from seeing across cycles in a graph. In an SSA-based IR like MLIR,
this means that these patterns don't see across PHI nodes / basic block
arguments. We consider this acceptable given the set of problems we are trying
to solve - we don't know of any other system that attempts to do so, and
consider the payoff of worrying about this to be low.

This design includes the ability for DAG patterns to have associated costs
(benefits), but those costs are defined in terms of magic numbers (typically
equal to the number of nodes being replaced). For any given application, the
units of magic numbers will have to be defined.

## Overall design

We decompose the problem into four major pieces:

1.  the code that is used to define patterns to match, cost, and their
    replacement actions
1.  the driver logic to pick the best match for a given root node
1.  the client that is implementing some transformation (e.g. a combiner)
1.  (future) the subsystem that allows patterns to be described with a
    declarative syntax, which sugars step #1.

We sketch the first three of these pieces, each in turn. This is not intended to
be a concrete API proposal, merely to describe the design

### Defining Patterns

Each pattern will be an instance of a mlir::Pattern class, whose subclasses
implement methods like this. Note that this API is meant for exposition, the
actual details are different for efficiency and coding standards reasons (e.g.
the memory management of `PatternState` is not specified below, etc):

```c++
class Pattern {
  /// Return the benefit (the inverse of "cost") of matching this pattern.  The
  /// benefit of a Pattern is always static - rewrites that may have dynamic
  /// benefit can be instantiated multiple times (different Pattern instances)
  /// for each benefit that they may return, and be guarded by different match
  /// condition predicates.
  PatternBenefit getBenefit() const { return benefit; }

  /// Return the root node that this pattern matches.  Patterns that can
  /// match multiple root types are instantiated once per root.
  OperationName getRootKind() const { return rootKind; }

  /// Attempt to match against code rooted at the specified operation,
  /// which is the same operation code as getRootKind().  On failure, this
  /// returns a None value.  On success it a (possibly null) pattern-specific
  /// state wrapped in a Some.  This state is passed back into its rewrite
  /// function if this match is selected.
  virtual Optional<PatternState*> match(Operation *op) const = 0;

  /// Rewrite the IR rooted at the specified operation with the result of
  /// this pattern, generating any new operations with the specified
  /// rewriter.  If an unexpected error is encountered (an internal
  /// compiler error), it is emitted through the normal MLIR diagnostic
  /// hooks and the IR is left in a valid state.
  virtual void rewrite(Operation *op, PatternState *state,
                       PatternRewriter &rewriter) const;
};
```

In practice, the first patterns we implement will directly subclass and
implement this stuff, but we will define some helpers to reduce boilerplate.
When we have a declarative way to describe patterns, this should be
automatically generated from the description.

Instances of `Pattern` have a benefit that is static upon construction of the
pattern instance, but may be computed dynamically at pattern initialization
time, e.g. allowing the benefit to be derived from domain specific information,
like the target architecture). This limitation allows us MLIR to (eventually)
perform pattern fusion and compile patterns into an efficient state machine, and
[Thier, Ertl, and Krall](https://dl.acm.org/citation.cfm?id=3179501) have shown
that match predicates eliminate the need for dynamically computed costs in
almost all cases: you can simply instantiate the same pattern one time for each
possible cost and use the predicate to guard the match.

The two-phase nature of this API (match separate from rewrite) is important for
two reasons: 1) some clients may want to explore different ways to tile the
graph, and only rewrite after committing to one tiling. 2) We want to support
runtime extensibility of the pattern sets, but want to be able to statically
compile the bulk of known patterns into a state machine at "compiler compile
time". Both of these reasons lead to us needing to match multiple patterns
before committing to an answer.

### Picking and performing a replacement

In the short term, this API can be very simple, something like this can work and
will be useful for many clients:

```c++
class PatternMatcher {
   // Create a pattern matcher with a bunch of patterns.  This constructor
   // looks across all of the specified patterns, and builds an internal
   // data structure that allows efficient matching.
   PatternMatcher(ArrayRef<Pattern*> patterns);

   // Given a specific operation, see if there is some rewrite that is
   // interesting.  If so, return success and return the list of new
   // operations that were created.  If not, return failure.
   bool matchAndRewrite(Operation *op,
                        SmallVectorImpl<Operation*> &newlyCreatedOps);
};
```

In practice the interesting part of this class is the acceleration structure it
builds internally. It buckets up the patterns by root operation, and sorts them
by their static benefit. When performing a match, it tests any dynamic patterns,
then tests statically known patterns from highest to lowest benefit.

### First Client: A Greedy Worklist Combiner

We expect that there will be lots of clients for this, but a simple greedy
worklist-driven combiner should be powerful enough to serve many important ones,
including the
[TF2XLA op expansion logic](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/compiler/tf2xla/kernels),
many of the pattern substitution passes of the
[TOCO compiler](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/toco)
for TF-Lite, many
[Grappler](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/grappler)
passes, and other general performance optimizations for applying identities.

The structure of this algorithm is straight-forward, here is pseudo code:

*   Walk a function in preorder, adding each operation to a worklist.
*   While the worklist is non-empty, pull something off the back (processing
    things generally in postorder)
    *   Perform matchAndRewrite on the operation. If failed, continue to the
        next operation.
    *   On success, add the newly created ops to the worklist and continue.

## Future directions

It is important to get implementation and usage experience with this, and many
patterns can be defined using this sort of framework. Over time, we can look to
make it easier to declare patterns in a declarative form (e.g. with the LLVM
tblgen tool or something newer/better). Once we have that, we can define an
internal abstraction for describing the patterns to match, allowing better high
level optimization of patterns (including fusion of the matching logic across
patterns, which the LLVM instruction selector does) and allow the patterns to be
defined without rebuilding the compiler itself.
