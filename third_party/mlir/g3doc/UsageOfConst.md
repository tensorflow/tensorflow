# Usage of 'Const' in MLIR, for core IR types

aka, where'd `const` go?

The MLIR data structures that represent the IR itself (Instruction, Block, etc)
form a graph-based data structure, and the compiler analyses and passes
frequently walk this graph (e.g. traversing from defs to users). The early
design of MLIR adopted the `const` model of LLVM, which is familiar and well
understood (even though the LLVM implementation is flawed in many ways).

The design team since decided to change to a different module, which eschews
`const` entirely for the core IR types: you should never see a `const` method on
`Operation`, should never see the type `const ValuePtr`, and you shouldn't feel
bad about this. That said, you *should* use `const` for non-IR types, like
`SmallVector`'s and many other things.

The document below explains this design point from the viewpoint of "why make a
change", to explain the rationale and the tradeoffs involved that led us to this
potentially controversial design point.

Bjarke Roune summarized the situation like this:

> In my opinion `const` correctness is highly valuable, catching many bugs and
> making it clear in a code base where the mutations happen. In my opinion
> `const` correctness still isn't worth it in particular for IR elements because
> of the special uses and properties of IRs, in particular that it is common to
> transfer a pointer/reference to an instruction from an analysis to an
> optimization which will change the instruction. The analysis should be const,
> the optimization needs to get a non-`const` pointer. So all analyses either
> end up being templates (and if they never get instantiated in a const context,
> then the point of `const` correctness has been defeated), you need to somehow
> launder the const in a safe way or there will be `const_cast`s. These options
> are all bad, probably so bad as to out-weigh the benefits of const.

# Reconsidering `const` in MLIR

This document argues this design is introducing significant sub-optimalities
into the MLIR codebase, argues that the cost/benefit tradeoff of this design is
a poor tradeoff, and proposes switching to a much simpler approach - eliminating
the use of const of these IR types entirely.

**Note:** **This document is only discussing things like `const Value` and
`const Operation*`. There is no proposed change for other types, e.g.
`SmallVector` references, the immutable types like `Attribute`, etc.**

## Background: The LLVM Const Model

The LLVM and MLIR data structures provide the IR data structures (like
`mlir::Operation`s and their users) as a structured cyclic graph data structure.
Clients of the IR typically walk up and down the graph, perform dynamic down
casting (of various sorts) to check for patterns, and use some high-abstraction
pattern matching and binding facilities to do their work.

The basic idea of LLVM's design is that these traversals of the IR should
preserve the const'ness of a pointer: if you have a const pointer to an
instruction and ask for its parent (or operand, users, etc), you should get a
const pointer to the block containing the instruction (or value defining the
operand, instruction using the instruction, etc). The instruction class looks
like this:

```
namespace llvm {
class Instruction : ...  {
  BasicBlock *Parent;
public:
  // A const instruction returns a const parent pointer.
  inline const BasicBlock *getParent() const { return Parent; }
  // A non-const instruction returns a non-const parent pointer.
  inline       BasicBlock *getParent()       { return Parent; }
…
};
}
```

The rationale for this design is that it would be const-incorrect to return a
non-const pointer from getParent, because you could then walk the block to find
the instruction again and get non-const references to the same instruction - all
without a `const_cast`.

This const model is simple and the C++ type system generally supports it through
code duplication of methods. That said, LLVM is actually inconsistent and buggy
about this. Even the core classes have bugs: `llvm::Instruction::getOperand()`
isn't currently const correct! There are other subsystems (e.g. the
`llvm/IR/PatternMatch.h` APIs) where you can perform a pattern match on a const
IR object and bind a non-const IR object.

LLVM is a mature technology with hundreds of people working on it. The fact that
it still isn't correctly following the const model it set out for strongly hints
that one of: 1) The design is too complicated to be practical, 2) the benefits
of the model aren't worth the cost of the complexity, or 3) both 1 and 2,
together in some combination.

## Advantages of Const-correctness in MLIR

Even though this doc argues for eliminating const from MLIR, it is important to
evaluate that as a tradeoff with the advantages the const model provides,
allowing us to do a cost/benefit tradeoff. These are the benefits we see:

The major advantage of allowing const on MLIR types is as a marker in APIs that
indicate that the function will not modify the specified values. For example,
the dominator APIs have a `dominates(const Block*, const Block*)` method, and
the consts provide a way of indicating that the call won't modify the blocks
passed in - similarly predicates like `Instruction::isTerminator() const` do not
modify the receiver object.

It is also an advantage that MLIR follows the generally prevailing pattern of
C++ code, which generally uses const. Consistency with the community norm is
important.

## Costs of Const-correctness in MLIR

As mentioned above, early work on MLIR adopted the same design as LLVM intended,
allowing const-correct traversals in the APIs. Here we discuss the various costs
of doing this by looking at some examples, listed in roughly increasing order of
severity.

### Pervasively duplicated accessors

Just as the getParent() example above shows, achieving this const model requires
that all of the graph traversal accessors be duplicated into const and non-const
versions. This causes API bloat and slows compile time, but these are minor
problems.

The more significant issue is that this duplication can be so significant that
the signal disappears in the noise, for example `mlir::Operation` ends up with
things like this, which is twice as much API surface area just to try to satisfy
const.

```c++
  operand_iterator operand_begin();
  operand_iterator operand_end();

  /// Returns an iterator on the underlying Value's (ValuePtr ).
  operand_range getOperands();

  // Support const operand iteration.
  using const_operand_iterator =
      OperandIterator<const Operation, const Value>;
  using const_operand_range = llvm::iterator_range<const_operand_iterator>;

  const_operand_iterator operand_begin() const;
  const_operand_iterator operand_end() const;

  /// Returns a const iterator on the underlying Value's (ValuePtr ).
  llvm::iterator_range<const_operand_iterator> getOperands() const;

  ArrayRef<OpOperand> getOpOperands() const {
    return getOperandStorage().getOperands();
  }
  MutableArrayRef<OpOperand> getOpOperands() {
    return getOperandStorage().getOperands();
  }

  OpOperand &getOpOperand(unsigned idx) { return getOpOperands()[idx]; }
  const OpOperand &getOpOperand(unsigned idx) const {
    return getOpOperands()[idx];
  }

```

### Templated accessors

A related issue is that having to provide both const and non-const versions of
accessors leads to us having to turn more code into templates than would
otherwise be desirable. Things like `ResultIterator` and `ResultTypeIterator`
are templates *_only_* because they are generic over const and non-const
versions of types. This leads to them being defined inline in headers (instead
of in .cpp files).

Thus, our const model is leading to more code in headers and more complexity in
the implementation.

### Const incorrect in practice

For some things, const is more trouble than it is worth, so they never get
updated.

This means that certain API in practice don't provide a const variant, leading
to pervasive use of `const_cast` to drop the const qualifier. For example the
logic in `Matchers.h` doesn't support const pointers at all (b/123355851), even
though matching and binding values themselves makes perfect sense for both const
and non-const values. Actually fixing this would cause massive code bloat and
complexity.

Other parts of the code are just outright incorrect. For example, the operation
cloning methods are defined on Operation like this:

```C++
Operation *clone(BlockAndValueMapping &mapper, MLIRContext *context) const;

Operation *clone(MLIRContext *context) const;
```

While it makes sense for a clone method to be `const` conceptually (the original
operation isn't modified) this is a violation of the model, since the returned
operation must be mutable, and provides access to the full graph of operands as
the original operation, violating the graph based const model we were shooting
for.

### The `OpPointer` and `ConstOpPointer` Classes

The "typed operation" classes for registered operations (e.g. like `DimOp` for
the "std.dim" operation in standard ops) contain a pointer to an operation and
provide typed APIs for processing it.

However, this is a problem for our current `const` design - `const DimOp` means
the pointer itself is immutable, not the pointee. The current solution for this
is the `OpPointer<>` and `ConstOpPointer<>` classes, which exist solely to
provide const correctness when referring to a typed operation. Instead of
referring to `DimOp` directly, we need to use `OpPointer<DimOp>` and
`ConstOpPointer<DimOp>` to preserve this constness.

While `auto` hides many instances of these `OpPointer` classes, their presence
leads to extremely ugly APIs. It also obscures the fact that the user does not
have a direct `DimOp` object, creating easy pitfalls with subtly incorrect
semantics:

```C++
// OpPointer encodes unnecessary and superfluous information into the API.
SmallVector<OpPointer<AffineForOp>, 8> stripmineSink(
  OpPointer<AffineForOp> forOp, uint64_t factor,
  ArrayRef<OpPointer<AffineForOp>> targets);
// Compared to the much cleaner and easier to read...
SmallVector<AffineForOp, 8> stripmineSink(AffineForOp forOp, uint64_t factor,
                                          ArrayRef<AffineForOp> targets);

// OpPointer is easy to misuse.
if (auto *dimOp = inst->dyn_cast<DimOp>()) {
  // This is actually undefined behavior because dyn_cast actually returns
  // OpPointer<DimOp>. OpPointer<DimOp> happily implicitly converts to DimOp *
  // creating undefined behavior that will execute correctly most of the time.
}
```

It would be much better to eliminate them entirely, and just pass around `DimOp`
directly. For example, instead of:

```C++
LogicalResult mlir::getIndexSet(MutableArrayRef<OpPointer<AffineForOp>> forOps,
                                FlatAffineConstraints *domain) {

```

It would be a lot nicer to just have:

```c++
LogicalResult mlir::getIndexSet(MutableArrayRef<AffineForOp> forOps,
                                FlatAffineConstraints *domain) {
```

Particularly since all of the `FooOp` classes are already semantically a smart
pointer to their underlying operation.

## Proposal: Remove `const` from IR objects

As we can see above, there is very little benefit to our const design and
significant cost, and given that the primary purpose of an IR is to represent
transformations of code, const is providing very little benefit.

As such, we propose eliminating support for const references in MLIR. This
implies the following changes to the codebase:

1.  All of the const-duplicated accessors would be eliminated, e.g.
    `Operation::getParent() const` would be removed. This is expected to remove
    approximately ~130 lines of code from just Operation.h alone.
1.  Const-only predicates would be changed to be non-const, e.g.
    `Operation::isTerminator() const` would have the const removed.
1.  Iterators and other types and functions that are templated to support
    `const` can have those template arguments removed.
1.  Types like `OpPointer` and `ConstOpPointer` that exist solely to propagate
    const can be entirely removed from the codebase.
1.  We can close bugs complaining about const incorrectness in the IR.
