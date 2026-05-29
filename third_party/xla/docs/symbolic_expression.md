# Symbolic expressions and maps

A symbolic expression (`SymbolicExpr`) is a mathematical abstraction system that
enables symbolic tensor computations. A symbolic map (`SymbolicMap`) is a
collection of symbolic expressions, that mathematically represents tensor
mappings and transformations in the compilation pipeline. They act as the
"mathematical bridge" between a high-level HLO operation and the actual memory
addresses accessed by the GPU/CPU.

`SymbolicExpr` and `SymbolicMap` are XLA's custom implementations that supersede
the legacy `mlir::AffineExpr` and `mlir::AffineMap`.

## `SymbolicExpr`

A `SymbolicExpr` represents a node in an abstract syntax tree (AST). Unlike the
standard MLIR Affine Expression, it supports a wider range of operations
necessary for modern GPU tiling.

Example of a `SymbolicExpr`: `d0 + s0 * 8`

### Supported types

**Constants:** Fixed integer (`int64`) values.

**Variables:** Symbolic variables with *dimensions (d)* and *symbols (s)*. All
variables (dimensions and symbols) are treated as `VariableID`, and the
interpretation of a variable depends on its context within a Symbolic Map.

**Operations:** `add`, `mul` `mod`, `floorDiv`, `ceilDiv`, `min`, `max`.

*Supported operators: `+`, `-`, `*`, `/` (`floorDiv`), `%` (`mod`)*

### Example usage

```sh
v0 = CreateSymbolicVariable(0, context); // 0 is the var_id
v1 = CreateSymbolicVariable(1, context);

SymbolicExpr expr = (((v0 + 42) * v1.min(2).max(0)) / 2).ceilDiv(2);

int64_t result = expr.Evaluate({5, 1}); // Result: 12
```

### Key features

*   **Immutability**: Symbolic expressions are pointer-like handles to internal
    storage managed by
    [`mlir::MLIRContext`](https://mlir.llvm.org/doxygen/classmlir_1_1MLIRContext.html).
    They are automatically deduplicated to ensure uniqueness.

*   **Canonicalization (`Canonicalize()`)**: Symbolic expressions are
    algebraically simplified (with constant folding, identity elimination,
    associative property, distributive property, etc.) to ensure expressions are
    represented in a standard, minimal form. For example, `(d0 + 1) - 1` will be
    simplified to `d0`. This is vital for predictable equality checks
    (`operator==`) in expressions that are mathematically equivalent but
    structurally different.

## `SymbolicMap`

`SymbolicMap` represents a mathematical mapping of transformation between
coordinate systems, typically between input and output tensors.

Example of a `SymbolicMap`: `(d0, d1)[s0, s1] -> (d0 + s0, d1 * s1)`

### Example usage

```sh
SymbolicMap map = SymbolicMap::Get(
    context,
    2,                   // number of dimensions
    1,                   // number of symbols
    {d0 + s0, d1 * s1}); // SymbolicExprs

// Access components
int64_t num_dims = map.GetNumDims();
int64_t num_symbols = map.GetNumSymbols();
auto results = map.GetResults();
```

### Key operations

**Variable substitution (`ReplaceDimsAndSymbols()`)**: Map dimensions and/or
symbols can be substituted with other expressions, this enables re-mapping
coordinate spaces. For example:

```sh
// c2 and c3: SymbolicConstants
// sample_map: (d0, d1)[s0, s1] -> (d0 + s0, d1 * s1)

sample_map.ReplaceDimsAndSymbols(
      {d1, c2},  // New dimensions: Replace d0 with d1, d1 with c2
      {c3, d0},  // New symbols: Replace s0 with c3, s1 with d0
      2,         // New number of dimensions
      2)         // New number of symbols

// Result: (d1 + c3, c2 * d0)
```

**Composition (`Compose()`)**: Maps with compatible dimensions can be composed.
This enables chaining transformations, and forms the core mechanism for fusing
multiple HLO operations into a single indexing calculation. For example:

```sh
map1: (d0, d1)[s0] -> (d0 + s0, d1 * 2)
map2: (d0)[s0] -> (d0 - 10, d0 + s0)
map1.compose(map2): (new_d0)[new_s0_map1, new_s0_map2] -> ((new_d0 - 10) + new_s0_map1, (new_d0 + new_s0_map2) * 2)
```

**Optimization (`CompressDims()` / `CompressSymbols()`)**: Simplifies maps by
identifying and removing unused variables, keeping the generated indexing logic
as lean as possible. For example:

```sh
map1: (d0, d1, d2)[s0] -> (d0 + d2, s0 * 5) // d1 is unused
map1.CompressDims(): (new_d0, new_d1)[new_s0] -> (new_d0 + new_d1, new_s0 * 5)

map2: (d0)[s0, s1, s2] -> {d0 + s2, s0 * 5} // s1 is unused
map2.CompressSymbols(): (new_d0)[new_s0, new_s1] -> (new_d0 + new_s1, new_s0 * 5)
```

`SymbolicMap` forms the mathematical basis for `IndexingMap`, that describes how
tensor elements map to each other in [HLO semantics](./operation_semantics.md).

`IndexingMap` consists of symbolic maps with domain-specific constraints. It
enables shape and tiling analysis, and collapsing operation chains (like
consecutive reshaping, transpose, broadcast, etc.) into optimized indexing
calculation. Learn more with concrete examples in
[Indexing analysis](./indexing.md).
