# Affine Dialect

This dialect provides a powerful abstraction for affine operations and analyses.

[TOC]

## Restrictions on Dimension and Symbols {#restrictions-on-dimensions-and-symbols}

The affine dialect imposes certain restrictions on dimension and symbolic
identifiers to enable powerful analysis and transformation. A symbolic
identifier can be bound to an SSA value that is either an argument to the
function, a value defined at the top level of that function (outside of all
loops and if instructions), the result of a
[`constant` operation](LangRef.md#'constant'-operation), or the result of an
[`affine.apply` operation](#'affine.apply'-operation) that recursively takes as
arguments any symbolic identifiers. Dimensions may be bound not only to anything
that a symbol is bound to, but also to induction variables of enclosing
[`affine.for` operations](#'affine.for'-operation), and the result of an
[`affine.apply` operation](#'affine.apply'-operation) (which recursively may use
other dimensions and symbols).

## Operations {#operations}

#### 'affine.apply' operation {#'affine.apply'-operation}

Syntax:

``` {.ebnf}
operation ::= ssa-id `=` `affine.apply` affine-map dim-and-symbol-use-list
```

The `affine.apply` instruction applies an
[affine mapping](LangRef.md#affine-expressions) to a list of SSA values,
yielding a single SSA value. The number of dimension and symbol arguments to
affine.apply must be equal to the respective number of dimensional and symbolic
inputs to the affine mapping; the `affine.apply` instruction always returns one
value. The input operands and result must all have 'index' type.

Example:

```mlir {.mlir}
#map10 = (d0, d1) -> (floordiv(d0,8) + floordiv(d1,128))
...
%1 = affine.apply #map10 (%s, %t)

// Inline example.
%2 = affine.apply (i)[s0] -> (i+s0) (%42)[%n]
```

#### 'affine.for' operation {#'affine.for'-operation}

Syntax:

``` {.ebnf}
operation   ::= `affine.for` ssa-id `=` lower-bound `to` upper-bound
                      (`step` integer-literal)? `{` inst* `}`

lower-bound ::= `max`? affine-map dim-and-symbol-use-list | shorthand-bound
upper-bound ::= `min`? affine-map dim-and-symbol-use-list | shorthand-bound
shorthand-bound ::= ssa-id | `-`? integer-literal
```

The `affine.for` operation represents an affine loop nest, defining an SSA value
for its induction variable. This SSA value always has type
[`index`](LangRef.md#index-type), which is the size of the machine word.

The `affine.for` operation executes its body a number of times iterating from a
lower bound to an upper bound by a stride. The stride, represented by `step`, is
a positive constant integer which defaults to "1" if not present. The lower and
upper bounds specify a half-open range: the range includes the lower bound but
does not include the upper bound.

The lower and upper bounds of a `affine.for` operation are represented as an
application of an affine mapping to a list of SSA values passed to the map. The
[same restrictions](#restrictions-on-dimensions-and-symbols) hold for these SSA
values as for all bindings of SSA values to dimensions and symbols.

The affine mappings for the bounds may return multiple results, in which case
the `max`/`min` keywords are required (for the lower/upper bound respectively),
and the bound is the maximum/minimum of the returned values. There is no
semantic ambiguity, but MLIR syntax requires the use of these keywords to make
things more obvious to human readers.

Many upper and lower bounds are simple, so MLIR accepts two custom form
syntaxes: the form that accepts a single 'ssa-id' (e.g. `%N`) is shorthand for
applying that SSA value to a function that maps a single symbol to itself, e.g.,
`()[s]->(s)()[%N]`. The integer literal form (e.g. `-42`) is shorthand for a
nullary mapping function that returns the constant value (e.g. `()->(-42)()`).

Example showing reverse iteration of the inner loop:

```mlir {.mlir}
#map57 = (d0)[s0] -> (s0 - d0 - 1)

func @simple_example(%A: memref<?x?xf32>, %B: memref<?x?xf32>) {
  %N = dim %A, 0 : memref<?x?xf32>
  affine.for %i = 0 to %N step 1 {
    affine.for %j = 0 to %N {   // implicitly steps by 1
      %0 = affine.apply #map57(%j)[%N]
      %tmp = call @F1(%A, %i, %0) : (memref<?x?xf32>, index, index)->(f32)
      call @F2(%tmp, %B, %i, %0) : (f32, memref<?x?xf32>, index, index)->()
    }
  }
  return
}
```

#### 'affine.if' operation {#'affine.if'-operation}

Syntax:

``` {.ebnf}
operation    ::= `affine.if` if-inst-cond `{` inst* `}` (`else` `{` inst* `}`)?
if-inst-cond ::= integer-set dim-and-symbol-use-list
```

The `affine.if` operation restricts execution to a subset of the loop iteration
space defined by an integer set (a conjunction of affine constraints). A single
`affine.if` may end with an optional `else` clause.

The condition of the `affine.if` is represented by an
[integer set](LangRef.md#integer-sets) (a conjunction of affine constraints),
and the SSA values bound to the dimensions and symbols in the integer set. The
[same restrictions](#restrictions-on-dimensions-and-symbols) hold for these SSA
values as for all bindings of SSA values to dimensions and symbols.

Example:

```mlir {.mlir}
#set = (d0, d1)[s0]: (d0 - 10 >= 0, s0 - d0 - 9 >= 0,
                      d1 - 10 >= 0, s0 - d1 - 9 >= 0)
func @reduced_domain_example(%A, %X, %N) : (memref<10xi32>, i32, i32) {
  affine.for %i = 0 to %N {
     affine.for %j = 0 to %N {
       %0 = affine.apply #map42(%j)
       %tmp = call @S1(%X, %i, %0)
       affine.if #set(%i, %j)[%N] {
          %1 = affine.apply #map43(%i, %j)
          call @S2(%tmp, %A, %i, %1)
       }
    }
  }
  return
}
```
