# MLIR Passes

This document describes the available MLIR passes and their contracts.

[TOC]

## Affine control lowering (`-lower-affine`)

Convert operations related to affine control into a graph of blocks using
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

Functions with `affine.for` and `affine.if` operations eliminated. These
functions may contain operations from the Standard dialect in addition to those
already present before the pass.

### Invariants

-   Functions without a body are not modified.
-   The semantics of the other functions is preserved.
-   Individual operations other than those mentioned above are not modified if
    they do not depend on the loop iterator value or on the result of
    `affine.apply`.

## Conversion from Standard to LLVM IR dialect (`-convert-std-to-llvm`)

Convert standard operations into the LLVM IR dialect operations.

### Input invariant

-   operations including: arithmetic on integers and floats, constants, direct
    calls, returns and branches;
-   no `tensor` types;
-   all `vector` are one-dimensional;
-   all blocks are reachable by following the successors of the first basic
    block;

If other operations are present and their results are required by the LLVM IR
dialect operations, the pass will fail.  Any LLVM IR operations or types already
present in the IR will be kept as is.

### Output IR

Functions converted to LLVM IR. Function arguments types are converted
one-to-one. Function results are converted one-to-one and, in case more than 1
value is returned, packed into an LLVM IR struct type. Function calls and
returns are updated accordingly. Block argument types are updated to use LLVM IR
types.

## Data Copy DMA generation (`-affine-data-copy-generate`)

Replaces all loads and stores on memref's living in 'slowMemorySpace' by
introducing DMA operations (strided DMA if necessary) to transfer data to/from
`fastMemorySpace` and rewriting the original load's/store's to instead
load/store from the allocated fast memory buffers. Additional options specify
the identifier corresponding to the fast memory space and the amount of fast
memory space available. The pass traverses through the nesting structure,
recursing to inner levels if necessary to determine at what depth DMA transfers
need to be placed so that the allocated buffers fit within the memory capacity
provided. If this is not possible (for example, when the elemental type itself
is of size larger than the DMA capacity), an error with location information is
emitted. The DMA transfers are also hoisted up past all loops with respect to
which the transfers are invariant.

Input

```mlir
func @loop_nest_tiled() -> memref<256x1024xf32> {
  %0 = alloc() : memref<256x1024xf32>
  affine.for %i0 = 0 to 256 step 32 {
    affine.for %i1 = 0 to 1024 step 32 {
      affine.for %i2 = (d0) -> (d0)(%i0) to (d0) -> (d0 + 32)(%i0) {
        affine.for %i3 = (d0) -> (d0)(%i1) to (d0) -> (d0 + 32)(%i1) {
          %1 = affine.load %0[%i2, %i3] : memref<256x1024xf32>
        }
      }
    }
  }
  return %0 : memref<256x1024xf32>
}
```

Output (with flags: -affine-data-copy-generate -affine-data-copy-generate-fast-mem-space=2)

```mlir
module {
  func @loop_nest_tiled() -> memref<256x1024xf32> {
    %c262144 = constant 262144 : index
    %c0 = constant 0 : index
    %0 = alloc() : memref<256x1024xf32>
    %1 = alloc() : memref<256x1024xf32, 2>
    %2 = alloc() : memref<1xi32>
    affine.dma_start %0[%c0, %c0], %1[%c0, %c0], %2[%c0], %c262144 : memref<256x1024xf32>, memref<256x1024xf32, 2>, memref<1xi32>
    affine.dma_wait %2[%c0], %c262144 : memref<1xi32>
    affine.for %arg0 = 0 to 256 step 32 {
      affine.for %arg1 = 0 to 1024 step 32 {
        affine.for %arg2 = #map1(%arg0) to #map2(%arg0) {
          affine.for %arg3 = #map1(%arg1) to #map2(%arg1) {
            %3 = affine.load %1[%arg2, %arg3] : memref<256x1024xf32, 2>
          }
        }
      }
    }
    dealloc %2 : memref<1xi32>
    dealloc %1 : memref<256x1024xf32, 2>
    return %0 : memref<256x1024xf32>
  }
}
```

## Loop tiling (`-affine-loop-tile`)

Performs tiling or blocking of loop nests. It currently works on perfect loop
nests.

## Loop unroll (`-affine-loop-unroll`)

This pass implements loop unrolling. It is able to unroll loops with arbitrary
bounds, and generate a cleanup loop when necessary.

## Loop unroll and jam (`-affine-loop-unroll-jam`)

This pass implements unroll and jam for loops. It works on both perfect or
imperfect loop nests.

## Loop fusion (`-affine-loop-fusion`)

Performs fusion of loop nests using a slicing-based approach. The fused loop
nests, when possible, are rewritten to access significantly smaller local
buffers instead of the original memref's, and the latter are often
either completely optimized away or contracted. This transformation leads to
enhanced locality and lower memory footprint through the elimination or
contraction of temporaries / intermediate memref's. These benefits are sometimes
achieved at the expense of redundant computation through a cost model that
evaluates available choices such as the depth at which a source slice should be
materialized in the designation slice.

## Memref bound checking (`-memref-bound-check`)

Checks all load's and store's on memref's for out of bound accesses, and reports
any out of bound accesses (both overrun and underrun) with location information.

```mlir
test/Transforms/memref-bound-check.mlir:19:13: error: 'load' op memref out of upper bound access along dimension #2
      %x  = load %A[%idx0, %idx1] : memref<9 x 9 x i32>
            ^
test/Transforms/memref-bound-check.mlir:19:13: error: 'load' op memref out of lower bound access along dimension #2
      %x  = load %A[%idx0, %idx1] : memref<9 x 9 x i32>
            ^
```

## Memref dataflow optimization (`-memref-dataflow-opt`)

This pass performs store to load forwarding for memref's to eliminate memory
accesses and potentially the entire memref if all its accesses are forwarded.

Input

```mlir
func @store_load_affine_apply() -> memref<10x10xf32> {
  %cf7 = constant 7.0 : f32
  %m = alloc() : memref<10x10xf32>
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.store %cf7, %m[%i0, %i1] : memref<10x10xf32>
      %v0 = affine.load %m[%i0, %i1] : memref<10x10xf32>
      %v1 = addf %v0, %v0 : f32
    }
  }
  return %m : memref<10x10xf32>
}
```

Output

```mlir
module {
  func @store_load_affine_apply() -> memref<10x10xf32> {
    %cst = constant 7.000000e+00 : f32
    %0 = alloc() : memref<10x10xf32>
    affine.for %arg0 = 0 to 10 {
      affine.for %arg1 = 0 to 10 {
        affine.store %cst, %0[%arg0, %arg1] : memref<10x10xf32>
        %1 = addf %cst, %cst : f32
      }
    }
    return %0 : memref<10x10xf32>
  }
}

```

## Memref dependence analysis (`-memref-dependence-check`)

This pass performs dependence analysis to determine dependences between pairs of
memory operations (load's and store's) on memref's. Dependence analysis exploits
polyhedral information available (affine maps, expressions, and affine.apply
operations) to precisely represent dependences using affine constraints, while
also computing dependence vectors from them, where each component of the
dependence vector provides a lower and an upper bound on the dependence distance
along the corresponding dimension.

```mlir
test/Transforms/memref-dataflow-opt.mlir:232:7: note: dependence from 2 to 1 at depth 1 = ([1, 1], [-inf, +inf])
      store %cf9, %m[%idx] : memref<10xf32>
```

## Pipeline data transfer (`-affine-pipeline-data-transfer`)

This pass performs a transformation to overlap non-blocking DMA operations in a
loop with computations through double buffering. This is achieved by advancing
dma_start operations with respect to other operations.

Input

```mlir
func @pipelinedatatransfer() {
  %0 = alloc() : memref<256xf32>
  %1 = alloc() : memref<32xf32, 1>
  %2 = alloc() : memref<1xf32>
  %c0 = constant 0 : index
  %c128 = constant 128 : index
  affine.for %i0 = 0 to 8 {
    affine.dma_start %0[%i0], %1[%i0], %2[%c0], %c128 : memref<256xf32>, memref<32xf32, 1>, memref<1xf32>
    affine.dma_wait %2[%c0], %c128 : memref<1xf32>
    %3 = affine.load %1[%i0] : memref<32xf32, 1>
    %4 = "compute"(%3) : (f32) -> f32
    affine.store %4, %1[%i0] : memref<32xf32, 1>
  }
  return
}
```

Output

```mlir
module {
  func @pipelinedatatransfer() {
    %c8 = constant 8 : index
    %c0 = constant 0 : index
    %0 = alloc() : memref<256xf32>
    %c0_0 = constant 0 : index
    %c128 = constant 128 : index
    %1 = alloc() : memref<2x32xf32, 1>
    %2 = alloc() : memref<2x1xf32>
    affine.dma_start %0[%c0], %1[%c0 mod 2, %c0], %2[%c0 mod 2, symbol(%c0_0)], %c128 : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
    affine.for %arg0 = 1 to 8 {
      affine.dma_start %0[%arg0], %1[%arg0 mod 2, %arg0], %2[%arg0 mod 2, symbol(%c0_0)], %c128 : memref<256xf32>, memref<2x32xf32, 1>, memref<2x1xf32>
      %8 = affine.apply #map3(%arg0)
      %9 = affine.apply #map4(%8)
      %10 = affine.apply #map4(%8)
      affine.dma_wait %2[%8 mod 2, symbol(%c0_0)], %c128 : memref<2x1xf32>
      %11 = affine.load %1[%8 mod 2, %8] : memref<2x32xf32, 1>
      %12 = "compute"(%11) : (f32) -> f32
      affine.store %12, %1[%8 mod 2, %8] : memref<2x32xf32, 1>
    }
    %3 = affine.apply #map3(%c8)
    %4 = affine.apply #map4(%3)
    %5 = affine.apply #map4(%3)
    affine.dma_wait %2[%3 mod 2, symbol(%c0_0)], %c128 : memref<2x1xf32>
    %6 = affine.load %1[%3 mod 2, %3] : memref<2x32xf32, 1>
    %7 = "compute"(%6) : (f32) -> f32
    affine.store %7, %1[%3 mod 2, %3] : memref<2x32xf32, 1>
    dealloc %2 : memref<2x1xf32>
    dealloc %1 : memref<2x32xf32, 1>
    return
  }
}
```
