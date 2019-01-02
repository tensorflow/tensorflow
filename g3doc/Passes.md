# MLIR Passes

This document describes the available MLIR passes and their contracts.

[TOC]

## Lower `if` and `for` (`-lower-if-and-for`) {#lower-if-and-for}

Lower the `if` and `for` instructions to the CFG equivalent.

Individual operations are preserved. Loops are converted to a subgraph of blocks
(initialization, condition checking, subgraph of body blocks) with loop
induction variable being passed as the block argument of the condition checking
block.

## `affine_apply` lowering (`-lower-affine-apply`) {#lower-affine-apply}

Convert `affine_apply` operations into arithmetic operations they comprise.
Arguments and results of all operations are of the `index` type.

For example, `%r = affine_apply (d0, d1)[s0] -> (d0 + 2*d1 + s0)(%d0, %d1)[%s0]`
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

`if` and `for` instructions should be eliminated before this pass.

### Output IR

Functions that do not contain any `affine_apply` operations. Consequently, named
maps may be removed from the module. CFG functions may use any operations from
the StandardOps dialect in addition to the already used dialects.

### Invariants

-   Operations other than `affine_apply` are not modified.
