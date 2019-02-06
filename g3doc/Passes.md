# MLIR Passes

This document describes the available MLIR passes and their contracts.

[TOC]

## Affine control lowering (`-lower-affine`) {#lower-affine-apply}

Convert instructions related to affine control into a graph of blocks using
operations from the standard dialect.

Loop statements are converted to a subgraph of blocks (initialization, condition
checking, subgraph of body blocks) with loop induction variable being passed as
the block argument of the condition checking block. Conditional statements are
converted to a subgraph of blocks (chain of condition checking with
short-circuit logic, subgraphs of 'then' and 'else' body blocks). `affine.apply`
operations are converted into sequences of primitive arithmetic operations that
have the same effect, using operands of the `index` type. Consequently, named
maps and sets may be removed from the module.

For example, `%r = affine.apply (d0, d1)[s0] -> (d0 + 2*d1 + s0)(%d0, %d1)[%s0]`
can be converted into:

```mlir
%d0 = <...>
%d1 = <...>
%s0 = <...>
%0 = constant 2 : index
%1 = muli %0, %d1
%2 = addi %d0, %1
%r = addi %2, %s0
```

### Input invariant

-   no `Tensor` types;

These restrictions may be lifted in the future.

### Output IR

Functions with `affine.for` and `if` instructions eliminated. These functions
may contain operations from the Standard dialect in addition to those already
present before the pass.

### Invariants

-   Functions without a body are not modified.
-   The semantics of the other functions is preserved.
-   Individual operations other than those mentioned above are not modified if
    they do not depend on the loop iterator value or on the result of
    `affine.apply`.

## Standard+Builtin to LLVM IR dialect conversion (`-convert-to-llvmir`) {#convert-to-llvmir}

Convert standard and builtin operations into the LLVM IR dialect operations.

### Input invariant

-   operations including: arithmetics on integers and floats, constants, direct
    calls, returns and branches;
-   no `tensor` types;
-   all `vector` are one-dimensional;
-   all blocks are reachable by following the successors of the first basic
    block;

If other operations are present and their results are required by the LLVM IR
dialect operations, the pass will fail.

### Output IR

Functions converted to LLVM IR. Function arguments types are converted
one-to-one. Function results are converted one-to-one and, in case more than 1
value is returned, packed into an LLVM IR struct type. Function calls and
returns are updated accordingly. Block argument types are updated to use LLVM IR
types.
