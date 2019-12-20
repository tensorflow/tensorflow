# Affine Dialect

This dialect provides a powerful abstraction for affine operations and analyses.

[TOC]

## Polyhedral Structures

MLIR uses techniques from polyhedral compilation to make dependence analysis and
loop transformations efficient and reliable. This section introduces some of the
core concepts that are used throughout the document.

### Dimensions and Symbols

Dimensions and symbols are the two kinds of identifiers that can appear in the
polyhedral structures, and are always of [`index`](../LangRef.md#index-type)
type. Dimensions are declared in parentheses and symbols are declared in square
brackets.

Examples:

```mlir
// A 2d to 3d affine mapping.
// d0/d1 are dimensions, s0 is a symbol
#affine_map2to3 = (d0, d1)[s0] -> (d0, d1 + s0, d1 - s0)
```

Dimensional identifiers correspond to the dimensions of the underlying structure
being represented (a map, set, or more concretely a loop nest or a tensor); for
example, a three-dimensional loop nest has three dimensional identifiers. Symbol
identifiers represent an unknown quantity that can be treated as constant for a
region of interest.

Dimensions and symbols are bound to SSA values by various operations in MLIR and
use the same parenthesized vs square bracket list to distinguish the two.

Syntax:

```
// Uses of SSA values that are passed to dimensional identifiers.
dim-use-list ::= `(` ssa-use-list? `)`

// Uses of SSA values that are used to bind symbols.
symbol-use-list ::= `[` ssa-use-list? `]`

// Most things that bind SSA values bind dimensions and symbols.
dim-and-symbol-use-list ::= dim-use-list symbol-use-list?
```

SSA values bound to dimensions and symbols must always have 'index' type.

Example:

```mlir
#affine_map2to3 = (d0, d1)[s0] -> (d0, d1 + s0, d1 - s0)
// Binds %N to the s0 symbol in affine_map2to3.
%x = alloc()[%N] : memref<40x50xf32, #affine_map2to3>
```

### Restrictions on Dimensions and Symbols

The affine dialect imposes certain restrictions on dimension and symbolic
identifiers to enable powerful analysis and transformation. A symbolic
identifier can be bound to an SSA value that is either an argument to the
function, a value defined at the top level of that function (outside of all
loops and if operations), the result of a
[`constant` operation](Standard.md#constant-operation), or the result of an
[`affine.apply` operation](#affineapply-operation) that recursively takes as
arguments any symbolic identifiers, or the result of a [`dim`
operation](Standard.md#dim-operation) on either a memref that is a function
argument or a memref where the corresponding dimension is either static or a
dynamic one in turn bound to a symbolic identifier.  Dimensions may be bound not
only to anything that a symbol is bound to, but also to induction variables of
enclosing [`affine.for` operations](#affinefor-operation), and the result of an
[`affine.apply` operation](#affineapply-operation) (which recursively may use
other dimensions and symbols).

### Affine Expressions

Syntax:

```
affine-expr ::= `(` affine-expr `)`
              | affine-expr `+` affine-expr
              | affine-expr `-` affine-expr
              | `-`? integer-literal `*` affine-expr
              | affine-expr `ceildiv` integer-literal
              | affine-expr `floordiv` integer-literal
              | affine-expr `mod` integer-literal
              | `-`affine-expr
              | bare-id
              | `-`? integer-literal

multi-dim-affine-expr ::= `(` affine-expr (`,` affine-expr)* `)`
```

`ceildiv` is the ceiling function which maps the result of the division of its
first argument by its second argument to the smallest integer greater than or
equal to that result. `floordiv` is a function which maps the result of the
division of its first argument by its second argument to the largest integer
less than or equal to that result. `mod` is the modulo operation: since its
second argument is always positive, its results are always positive in our
usage. The `integer-literal` operand for ceildiv, floordiv, and mod is always
expected to be positive. `bare-id` is an identifier which must have type
[index](../LangRef.md#index-type). The precedence of operations in an affine
expression are ordered from highest to lowest in the order: (1)
parenthesization, (2) negation, (3) modulo, multiplication, floordiv, and
ceildiv, and (4) addition and subtraction. All of these operators associate from
left to right.

A _multidimensional affine expression_ is a comma separated list of
one-dimensional affine expressions, with the entire list enclosed in
parentheses.

**Context:** An affine function, informally, is a linear function plus a
constant. More formally, a function f defined on a vector $$\vec{v} \in
\mathbb{Z}^n$$ is a multidimensional affine function of $$\vec{v}$$ if
$$f(\vec{v})$$ can be expressed in the form $$M \vec{v} + \vec{c}$$ where $$M$$
is a constant matrix from $$\mathbb{Z}^{m \times n}$$ and $$\vec{c}$$ is a
constant vector from $$\mathbb{Z}$$. $$m$$ is the dimensionality of such an
affine function. MLIR further extends the definition of an affine function to
allow 'floordiv', 'ceildiv', and 'mod' with respect to positive integer
constants. Such extensions to affine functions have often been referred to as
quasi-affine functions by the polyhedral compiler community. MLIR uses the term
'affine map' to refer to these multidimensional quasi-affine functions. As
examples, $$(i+j+1, j)$$, $$(i \mod 2, j+i)$$, $$(j, i/4, i \mod 4)$$, $$(2i+1,
j)$$ are two-dimensional affine functions of $$(i, j)$$, but $$(i \cdot j,
i^2)$$, $$(i \mod j, i/j)$$ are not affine functions of $$(i, j)$$.

### Affine Maps

Syntax:

```
affine-map-inline
   ::= dim-and-symbol-id-lists `->` multi-dim-affine-expr
```

The identifiers in the dimensions and symbols lists must be unique. These are
the only identifiers that may appear in 'multi-dim-affine-expr'. Affine maps
with one or more symbols in its specification are known as "symbolic affine
maps", and those with no symbols as "non-symbolic affine maps".

**Context:** Affine maps are mathematical functions that transform a list of
dimension indices and symbols into a list of results, with affine expressions
combining the indices and symbols. Affine maps distinguish between
[indices and symbols](#dimensions-and-symbols) because indices are inputs to the
affine map when the map is called (through an operation such as
[affine.apply](#affineapply-operation)), whereas symbols are bound when
the map is established (e.g. when a memref is formed, establishing a
memory [layout map](../LangRef.md#layout-map)).

Affine maps are used for various core structures in MLIR. The restrictions we
impose on their form allows powerful analysis and transformation, while keeping
the representation closed with respect to several operations of interest.

#### Named affine mappings

Syntax:

```
affine-map-id ::= `#` suffix-id

// Definitions of affine maps are at the top of the file.
affine-map-def    ::= affine-map-id `=` affine-map-inline
module-header-def ::= affine-map-def

// Uses of affine maps may use the inline form or the named form.
affine-map ::= affine-map-id | affine-map-inline
```

Affine mappings may be defined inline at the point of use, or may be hoisted to
the top of the file and given a name with an affine map definition, and used by
name.

Examples:

```mlir
// Affine map out-of-line definition and usage example.
#affine_map42 = (d0, d1)[s0] -> (d0, d0 + d1 + s0 floordiv 2)

// Use an affine mapping definition in an alloc operation, binding the
// SSA value %N to the symbol s0.
%a = alloc()[%N] : memref<4x4xf32, #affine_map42>

// Same thing with an inline affine mapping definition.
%b = alloc()[%N] : memref<4x4xf32, (d0, d1)[s0] -> (d0, d0 + d1 + s0 floordiv 2)>
```

### Semi-affine maps

Semi-affine maps are extensions of affine maps to allow multiplication,
`floordiv`, `ceildiv`, and `mod` with respect to symbolic identifiers.
Semi-affine maps are thus a strict superset of affine maps.

Syntax of semi-affine expressions:

```
semi-affine-expr ::= `(` semi-affine-expr `)`
                   | semi-affine-expr `+` semi-affine-expr
                   | semi-affine-expr `-` semi-affine-expr
                   | symbol-or-const `*` semi-affine-expr
                   | semi-affine-expr `ceildiv` symbol-or-const
                   | semi-affine-expr `floordiv` symbol-or-const
                   | semi-affine-expr `mod` symbol-or-const
                   | bare-id
                   | `-`? integer-literal

symbol-or-const ::= `-`? integer-literal | symbol-id

multi-dim-semi-affine-expr ::= `(` semi-affine-expr (`,` semi-affine-expr)* `)`
```

The precedence and associativity of operations in the syntax above is the same
as that for [affine expressions](#affine-expressions).

Syntax of semi-affine maps:

```
semi-affine-map-inline
   ::= dim-and-symbol-id-lists `->` multi-dim-semi-affine-expr
```

Semi-affine maps may be defined inline at the point of use, or may be hoisted to
the top of the file and given a name with a semi-affine map definition, and used
by name.

```
semi-affine-map-id ::= `#` suffix-id

// Definitions of semi-affine maps are at the top of file.
semi-affine-map-def ::= semi-affine-map-id `=` semi-affine-map-inline
module-header-def ::= semi-affine-map-def

// Uses of semi-affine maps may use the inline form or the named form.
semi-affine-map ::= semi-affine-map-id | semi-affine-map-inline
```

### Integer Sets

An integer set is a conjunction of affine constraints on a list of identifiers.
The identifiers associated with the integer set are separated out into two
classes: the set's dimension identifiers, and the set's symbolic identifiers.
The set is viewed as being parametric on its symbolic identifiers. In the
syntax, the list of set's dimension identifiers are enclosed in parentheses
while its symbols are enclosed in square brackets.

Syntax of affine constraints:

```
affine-constraint ::= affine-expr `>=` `0`
                    | affine-expr `==` `0`
affine-constraint-conjunction ::= affine-constraint (`,` affine-constraint)*
```

Integer sets may be defined inline at the point of use, or may be hoisted to the
top of the file and given a name with an integer set definition, and used by
name.

```
integer-set-id ::= `#` suffix-id

integer-set-inline
   ::= dim-and-symbol-id-lists `:` '(' affine-constraint-conjunction? ')'

// Declarations of integer sets are at the top of the file.
integer-set-decl ::= integer-set-id `=` integer-set-inline

// Uses of integer sets may use the inline form or the named form.
integer-set ::= integer-set-id | integer-set-inline
```

The dimensionality of an integer set is the number of identifiers appearing in
dimension list of the set. The affine-constraint non-terminals appearing in the
syntax above are only allowed to contain identifiers from dims and symbols. A
set with no constraints is a set that is unbounded along all of the set's
dimensions.

Example:

```mlir
// A example two-dimensional integer set with two symbols.
#set42 = (d0, d1)[s0, s1]
   : (d0 >= 0, -d0 + s0 - 1 >= 0, d1 >= 0, -d1 + s1 - 1 >= 0)

// Inside a Region
affine.if #set42(%i, %j)[%M, %N] {
  ...
}
```

`d0` and `d1` correspond to dimensional identifiers of the set, while `s0` and
`s1` are symbol identifiers.

## Operations

#### 'affine.apply' operation

Syntax:

```
operation ::= ssa-id `=` `affine.apply` affine-map dim-and-symbol-use-list
```

The `affine.apply` operation applies an
[affine mapping](#affine-expressions) to a list of SSA values,
yielding a single SSA value. The number of dimension and symbol arguments to
affine.apply must be equal to the respective number of dimensional and symbolic
inputs to the affine mapping; the `affine.apply` operation always returns one
value. The input operands and result must all have 'index' type.

Example:

```mlir
#map10 = (d0, d1) -> (d0 floordiv 8 + d1 floordiv 128)
...
%1 = affine.apply #map10 (%s, %t)

// Inline example.
%2 = affine.apply (i)[s0] -> (i+s0) (%42)[%n]
```

#### 'affine.for' operation

Syntax:

```
operation   ::= `affine.for` ssa-id `=` lower-bound `to` upper-bound
                      (`step` integer-literal)? `{` op* `}`

lower-bound ::= `max`? affine-map dim-and-symbol-use-list | shorthand-bound
upper-bound ::= `min`? affine-map dim-and-symbol-use-list | shorthand-bound
shorthand-bound ::= ssa-id | `-`? integer-literal
```

The `affine.for` operation represents an affine loop nest. It has one region
containing its body. This region must contain one block that terminates with
[`affine.terminator`](#affineterminator-operation). *Note:* when `affine.for` is
printed in custom format, the terminator is omitted. The block has one argument
of [`index`](../LangRef.md#index-type) type that represents the induction
variable of the loop.

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

```mlir
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

#### 'affine.if' operation

Syntax:

```
operation    ::= `affine.if` if-op-cond `{` op* `}` (`else` `{` op* `}`)?
if-op-cond ::= integer-set dim-and-symbol-use-list
```

The `affine.if` operation restricts execution to a subset of the loop iteration
space defined by an integer set (a conjunction of affine constraints). A single
`affine.if` may end with an optional `else` clause.

The condition of the `affine.if` is represented by an
[integer set](#integer-sets) (a conjunction of affine constraints),
and the SSA values bound to the dimensions and symbols in the integer set. The
[same restrictions](#restrictions-on-dimensions-and-symbols) hold for these SSA
values as for all bindings of SSA values to dimensions and symbols.

The `affine.if` operation contains two regions for the "then" and "else"
clauses. The latter may be empty (i.e. contain no blocks), meaning the absence
of the else clause. When non-empty, both regions must contain exactly one block
terminating with [`affine.terminator`](#affineterminator-operation). *Note:*
when `affine.if` is printed in custom format, the terminator is omitted. These
blocks must not have any arguments.

Example:

```mlir
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

#### 'affine.load' operation

Syntax:

```
operation ::= ssa-id `=` `affine.load` ssa-use `[` multi-dim-affine-map-of-ssa-ids `]` `:` memref-type
```

The `affine.load` op reads an element from a memref, where the index for each
memref dimension is an affine expression of loop induction variables and
symbols. The output of 'affine.load' is a new value with the same type as the
elements of the memref. An affine expression of loop IVs and symbols must be
specified for each dimension of the memref. The keyword 'symbol' can be used to
indicate SSA identifiers which are symbolic.

Example:

```mlir

  Example 1:

    %1 = affine.load %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>

  Example 2: Uses 'symbol' keyword for symbols '%n' and '%m'.

    %1 = affine.load %0[%i0 + symbol(%n), %i1 + symbol(%m)]
      : memref<100x100xf32>

```

#### 'affine.store' operation

Syntax:

```
operation ::= ssa-id `=` `affine.store` ssa-use, ssa-use `[` multi-dim-affine-map-of-ssa-ids `]` `:` memref-type
```

The `affine.store` op writes an element to a memref, where the index for each
memref dimension is an affine expression of loop induction variables and
symbols. The 'affine.store' op stores a new value which is the same type as the
elements of the memref. An affine expression of loop IVs and symbols must be
specified for each dimension of the memref. The keyword 'symbol' can be used to
indicate SSA identifiers which are symbolic.

Example:

```mlir

    Example 1:

      affine.store %v0, %0[%i0 + 3, %i1 + 7] : memref<100x100xf32>

    Example 2: Uses 'symbol' keyword for symbols '%n' and '%m'.

      affine.store %v0, %0[%i0 + symbol(%n), %i1 + symbol(%m)]
        : memref<100x100xf32>

```

#### 'affine.dma_start' operation

Syntax:

```
operation ::= `affine.dma_Start` ssa-use `[` multi-dim-affine-map-of-ssa-ids `]`, `[` multi-dim-affine-map-of-ssa-ids `]`, `[` multi-dim-affine-map-of-ssa-ids `]`, ssa-use `:` memref-type
```

The `affine.dma_start` op starts a non-blocking DMA operation that transfers
data from a source memref to a destination memref. The source and destination
memref need not be of the same dimensionality, but need to have the same
elemental type. The operands include the source and destination memref's
each followed by its indices, size of the data transfer in terms of the
number of elements (of the elemental type of the memref), a tag memref with
its indices, and optionally at the end, a stride and a
number_of_elements_per_stride arguments. The tag location is used by an
AffineDmaWaitOp to check for completion. The indices of the source memref,
destination memref, and the tag memref have the same restrictions as any
affine.load/store. In particular, index for each memref dimension must be an
affine expression of loop induction variables and symbols.
The optional stride arguments should be of 'index' type, and specify a
stride for the slower memory space (memory space with a lower memory space
id), transferring chunks of number_of_elements_per_stride every stride until
%num_elements are transferred. Either both or no stride arguments should be
specified. The value of 'num_elements' must be a multiple of
'number_of_elements_per_stride'.


Example:

```mlir

For example, a DmaStartOp operation that transfers 256 elements of a memref
'%src' in memory space 0 at indices [%i + 3, %j] to memref '%dst' in memory
space 1 at indices [%k + 7, %l], would be specified as follows:

  %num_elements = constant 256
  %idx = constant 0 : index
  %tag = alloc() : memref<1xi32, 4>
  affine.dma_start %src[%i + 3, %j], %dst[%k + 7, %l], %tag[%idx],
    %num_elements :
      memref<40x128xf32, 0>, memref<2x1024xf32, 1>, memref<1xi32, 2>

  If %stride and %num_elt_per_stride are specified, the DMA is expected to
  transfer %num_elt_per_stride elements every %stride elements apart from
  memory space 0 until %num_elements are transferred.

  affine.dma_start %src[%i, %j], %dst[%k, %l], %tag[%idx], %num_elements,
    %stride, %num_elt_per_stride : ...

```

#### 'affine.dma_wait' operation

Syntax:

```
operation ::= `affine.dma_Start` ssa-use `[` multi-dim-affine-map-of-ssa-ids `]`, `[` multi-dim-affine-map-of-ssa-ids `]`, `[` multi-dim-affine-map-of-ssa-ids `]`, ssa-use `:` memref-type
```

The `affine.dma_start` op blocks until the completion of a DMA operation
associated with the tag element '%tag[%index]'. %tag is a memref, and %index
has to be an index with the same restrictions as any load/store index.
In particular, index for each memref dimension must be an affine expression of
loop induction variables and symbols. %num_elements is the number of elements
associated with the DMA operation. For example:

Example:

```mlir

  affine.dma_start %src[%i, %j], %dst[%k, %l], %tag[%index], %num_elements :
    memref<2048xf32, 0>, memref<256xf32, 1>, memref<1xi32, 2>
  ...
  ...
  affine.dma_wait %tag[%index], %num_elements : memref<1xi32, 2>

```

#### 'affine.min' operation

Syntax:

```
operation ::= ssa-id `=` `affine.min` affine-map dim-and-symbol-use-list
```

The `affine.min` operation applies an
[affine mapping](#affine-expressions) to a list of SSA values, and returns the
minimum value of all result expressions. The number of dimension and symbol
arguments to affine.min must be equal to the respective number of dimensional
and symbolic inputs to the affine mapping; the `affine.min` operation always
returns one value. The input operands and result must all have 'index' type.

Example:

```mlir

%0 = affine.min (d0)[s0] -> (1000, d0 + 512, s0) (%arg0)[%arg1]

```

#### `affine.terminator` operation

Syntax:

```
operation ::= `"affine.terminator"() : () -> ()`
```

Affine terminator is a special terminator operation for blocks inside affine
loops ([`affine.for`](#affinefor-operation)) and branches
([`affine.if`](#affineif-operation)). It unconditionally transmits the control
flow to the successor of the operation enclosing the region.

*Rationale*: bodies of affine operations are [blocks](../LangRef.md#blocks) that
must have terminators. Loops and branches represent structured control flow and
should not accept arbitrary branches as terminators.

This operation does _not_ have a custom syntax. However, affine control
operations omit the terminator in their custom syntax for brevity.
