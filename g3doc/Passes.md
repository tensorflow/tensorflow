# MLIR Passes

This document describes the available MLIR passes and their contracts.

[TOC]

## ML to CFG conversion (`-convert-to-cfg`) {#convert-to-cfg}

Convert ML functions to equivalent CFG functions.

Individual operations are preserved. Loops are converted to a subgraph of basic
blocks (initialization, condition checking, subgraph of body blocks) with loop
induction variable being passed as the basic block argument of the condition
checking block.

### Input IR

MLIR with ML, CFG and External functions. The following restrictions apply to
the input ML functions:

-   0 or 1 return values;
-   no `Tensor` types;
-   no `if` conditions in the body.

These restrictions may be lifted in the future.

### Output IR

MLIR with CFG and External functions only. The CFG functions introduced by this
pass can contain any operations from BuiltIn and StandardOps dialects in
addition to the operations present in the source ML functions.

### Invariants

-   The CFG and External functions are not modified.
-   The CFG functions introduced by this pass have the same names as the
    replaced ML functions.
-   Individual operations other than control flow from the source ML functions
    are replicated in the produced CFG functions; their arguments may be updated
    to capture the corresponding SSA values after conversion (e.g., loop
    iterators become basic block arguments).

## `affine_apply` lowering (`-lower-affine-apply`) {#lower-affine-apply}

Convert `affine_apply` operations in CFG functions into arithmetic operations
they comrise. Arguments and results of all operations are of the `index` type.

For example, `%r = affine_apply (d0, d1)[s0] -> (d0 + 2*d1 + s0)(%d0, %d1)[%s0]`
can be converted into

```mlir
%d0 = <...>
%d1 = <...>
%s0 = <...>
%0 = constant 2 : index
%1 = muli %0, %d1
%2 = addi %d0, %1
%r = addi %2, %s0
```

### Input IR

MLIR with CFG and External functions.

ML functions are not allowed in the input since they *may* include syntactic
constructs equivalent to `affine_apply` that cannot be replaced, in particular
`for` loop bounds and `if` conditions. Lower ML functions to CFG functions to
expose all `affine_apply` operations before using this pass.

### Output IR

MLIR with CFG and External functions. CFG functions do not contain any
`affine_apply` operations. Consequently, named maps may be removed from the
module. CFG functions may use any operations from the StandardOps dialect in
addition to the already used dialects.

### Invariants

-   External functions are not modified.
-   The semantics of the CFG functions remains the same.
-   Operations other than `affine_apply` are not modified.
