# Standard Dialect

This dialect provides documentation for operations within the Standard dialect.

Note: This dialect is a collection of operations for several different concepts,
and should be split into multiple more-focused dialects accordingly.

[TOC]

TODO: shape, which returns a 1D tensor, and can take an unknown rank tensor as
input.

TODO: rank, which returns an index.

## Terminator operations

Terminator operations are required at the end of each block. They may contain a
list of successors, i.e. other blocks to which the control flow will proceed.

### 'br' terminator operation

Syntax:

```
operation ::= `br` successor
successor ::= bb-id branch-use-list?
branch-use-list ::= `(` ssa-use-list `:` type-list-no-parens `)`
```

The `br` terminator operation represents an unconditional jump to a target
block. The count and types of operands to the branch must align with the
arguments in the target block.

The MLIR branch operation is not allowed to target the entry block for a region.

### 'cond_br' terminator operation

Syntax:

```
operation ::= `cond_br` ssa-use `,` successor `,` successor
```

The `cond_br` terminator operation represents a conditional branch on a boolean
(1-bit integer) value. If the bit is set, then the first destination is jumped
to; if it is false, the second destination is chosen. The count and types of
operands must align with the arguments in the corresponding target blocks.

The MLIR conditional branch operation is not allowed to target the entry block
for a region. The two destinations of the conditional branch operation are
allowed to be the same.

The following example illustrates a function with a conditional branch operation
that targets the same block:

```mlir
func @select(i32, i32, i1) -> i32 {
^bb0(%a : i32, %b :i32, %flag : i1) :
    // Both targets are the same, operands differ
    cond_br %flag, ^bb1(%a : i32), ^bb1(%b : i32)

^bb1(%x : i32) :
    return %x : i32
}
```

### 'return' terminator operation

Syntax:

```
operation ::= `return` (ssa-use-list `:` type-list-no-parens)?
```

The `return` terminator operation represents the completion of a function, and
produces the result values. The count and types of the operands must match the
result types of the enclosing function. It is legal for multiple blocks in a
single function to return.

## Core Operations

### 'call' operation

Syntax:

```
operation ::=
    (ssa-id `=`)? `call` symbol-ref-id `(` ssa-use-list? `)` `:` function-type
```

The `call` operation represents a direct call to a function. The operands and
result types of the call must match the specified function type. The callee is
encoded as a function attribute named "callee".

Example:

```mlir
// Calling the function my_add.
%31 = call @my_add(%0, %1) : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
```

### 'call_indirect' operation

Syntax:

```
operation ::= `call_indirect` ssa-use `(` ssa-use-list? `)` `:` function-type
```

The `call_indirect` operation represents an indirect call to a value of function
type. Functions are first class types in MLIR, and may be passed as arguments
and merged together with block arguments. The operands and result types of the
call must match the specified function type.

Function values can be created with the
[`constant` operation](#constant-operation).

Example:

```mlir
%31 = call_indirect %15(%0, %1)
        : (tensor<16xf32>, tensor<16xf32>) -> tensor<16xf32>
```

### 'dim' operation

Syntax:

```
operation ::= ssa-id `=` `dim` ssa-id `,` integer-literal `:` type
```

The `dim` operation takes a memref or tensor operand and a dimension index, and
returns an [`index`](../LangRef.md#index-type) that is the size of that
dimension.

The `dim` operation is represented with a single integer attribute named
`index`, and the type specifies the type of the memref or tensor operand.

Examples:

```mlir
// Always returns 4, can be constant folded:
%x = dim %A, 0 : tensor<4 x ? x f32>

// Returns the dynamic dimension of %A.
%y = dim %A, 1 : tensor<4 x ? x f32>

// Equivalent generic form:
%x = "std.dim"(%A) {index = 0 : i64} : (tensor<4 x ? x f32>) -> index
%y = "std.dim"(%A) {index = 1 : i64} : (tensor<4 x ? x f32>) -> index
```

## Memory Operations

### 'alloc' operation

Syntax:

```
operation ::= ssa-id `=` `alloc` dim-and-symbol-use-list `:` memref-type
```

Allocates a new memref of specified type. Values required for dynamic dimension
sizes are passed as arguments in parentheses (in the same order in which they
appear in the shape signature of the memref) while the symbols required by the
layout map are passed in the square brackets in lexicographical order. If no
layout maps are specified in the memref, then an identity mapping is used.

The buffer referenced by a memref type is created by the `alloc` operation, and
destroyed by the `dealloc` operation.

Example:

```mlir
// Allocating memref for a fully static shape.
%A = alloc() : memref<1024x64xf32, #layout_map0, memspace0>

// %M, %N, %x, %y are SSA values of integer type.  M and N are bound to the
// two unknown dimensions of the type and x/y are bound to symbols in
// #layout_map1.
%B = alloc(%M, %N)[%x, %y] : memref<?x?xf32, #layout_map1, memspace1>
```

### 'alloc_static' operation

Syntax:

```
operation ::=
    ssa-id `=` `alloc_static` `(` integer-literal `)` :  memref-type
```

Allocates a new memref of specified type with a fixed base pointer location in
memory. 'alloc_static' does not support types that have dynamic shapes or that
require dynamic symbols in their layout function (use the
[`alloc` operation](#alloc-operation) in those cases).

Example:

```mlir
%A = alloc_static(0x1232a00) : memref<1024 x 64 x f32, #layout_map0, memspace0>
```

The `alloc_static` operation is used to represent code after buffer allocation
has been performed.

### 'dealloc' operation

Syntax:

```
operation ::= `dealloc` ssa-use `:` memref-type
```

Delineates the end of the lifetime of the memory corresponding to a memref
allocation. It is paired with an [`alloc`](#alloc-operation) or
[`alloc_static`](#alloc-static-operation) operation.

Example:

```mlir
dealloc %A : memref<128 x f32, #layout, memspace0>
```

### 'dma_start' operation

Syntax:

```
operation ::= `dma_start` ssa-use`[`ssa-use-list`]` `,`
               ssa-use`[`ssa-use-list`]` `,` ssa-use `,`
               ssa-use`[`ssa-use-list`]` (`,` ssa-use `,` ssa-use)?
              `:` memref-type `,` memref-type `,` memref-type
```

Starts a non-blocking DMA operation that transfers data from a source memref to
a destination memref. The operands include the source and destination memref's
each followed by its indices, size of the data transfer in terms of the number
of elements (of the elemental type of the memref), a tag memref with its
indices, and optionally two additional arguments corresponding to the stride (in
terms of number of elements) and the number of elements to transfer per stride.
The tag location is used by a dma_wait operation to check for completion. The
indices of the source memref, destination memref, and the tag memref have the
same restrictions as any load/store operation in a affine context (whenever DMA
operations appear in an affine context). See
[restrictions on dimensions and symbols](Affine.md#restrictions-on-dimensions-and-symbols)
in affine contexts. This allows powerful static analysis and transformations in
the presence of such DMAs including rescheduling, pipelining / overlap with
computation, and checking for matching start/end operations. The source and
destination memref need not be of the same dimensionality, but need to have the
same elemental type.

For example, a `dma_start` operation that transfers 32 vector elements from a
memref `%src` at location `[%i, %j]` to memref `%dst` at `[%k, %l]` would be
specified as shown below.

Example:

```mlir
%size = constant 32 : index
%tag = alloc() : memref<1 x i32, (d0) -> (d0), 4>
%idx = constant 0 : index
dma_start %src[%i, %j], %dst[%k, %l], %size, %tag[%idx] :
     memref<40 x 8 x vector<16xf32>, (d0, d1) -> (d0, d1), 0>,
     memref<2 x 4 x vector<16xf32>, (d0, d1) -> (d0, d1), 2>,
     memref<1 x i32>, (d0) -> (d0), 4>
```

### 'dma_wait' operation

Syntax:

```
operation ::= `dma_wait` ssa-use`[`ssa-use-list`]` `,` ssa-use `:` memref-type
```

Blocks until the completion of a DMA operation associated with the tag element
specified with a tag memref and its indices. The operands include the tag memref
followed by its indices and the number of elements associated with the DMA being
waited on. The indices of the tag memref have the same restrictions as
load/store indices.

Example:

```mlir
dma_wait %tag[%idx], %size : memref<1 x i32, (d0) -> (d0), 4>
```

### 'extract_element' operation

Syntax:

```
operation ::= ssa-id `=` `extract_element` ssa-use `[` ssa-use-list `]` `:` type
```

The `extract_element` op reads a tensor or vector and returns one element from
it specified by an index list. The output of the 'extract_element' is a new
value with the same type as the elements of the tensor or vector. The arity of
indices matches the rank of the accessed value (i.e., if a tensor is of rank 3,
then 3 indices are required for the extract. The indices should all be of
`index` type.

Examples:

```mlir
%3 = extract_element %v[%1, %2] : vector<4x4xi32>
%4 = extract_element %t[%1, %2] : tensor<4x4xi32>
%5 = extract_element %ut[%1, %2] : tensor<*xi32>
```

### 'load' operation

Syntax:

```
operation ::= ssa-id `=` `load` ssa-use `[` ssa-use-list `]` `:` memref-type
```

The `load` op reads an element from a memref specified by an index list. The
output of load is a new value with the same type as the elements of the memref.
The arity of indices is the rank of the memref (i.e., if the memref loaded from
is of rank 3, then 3 indices are required for the load following the memref
identifier).

In an `affine.if` or `affine.for` body, the indices of a load are restricted to
SSA values bound to surrounding loop induction variables,
[symbols](../LangRef.md#dimensions-and-symbols), results of a
[`constant` operation](#constant-operation), or the result of an `affine.apply`
operation that can in turn take as arguments all of the aforementioned SSA
values or the recursively result of such an `affine.apply` operation.

Example:

```mlir
%1 = affine.apply (d0, d1) -> (3*d0) (%i, %j)
%2 = affine.apply (d0, d1) -> (d1+1) (%i, %j)
%12 = load %A[%1, %2] : memref<8x?xi32, #layout, memspace0>

// Example of an indirect load (treated as non-affine)
%3 = affine.apply (d0) -> (2*d0 + 1)(%12)
%13 = load %A[%3, %2] : memref<4x?xi32, #layout, memspace0>
```

**Context:** The `load` and `store` operations are specifically crafted to fully
resolve a reference to an element of a memref, and (in affine `affine.if` and
`affine.for` operations) the compiler can follow use-def chains (e.g. through
[`affine.apply`](Affine.md#affineapply-operation) operations) to precisely
analyze references at compile-time using polyhedral techniques. This is possible
because of the
[restrictions on dimensions and symbols](Affine.md#restrictions-on-dimensions-and-symbols)
in these contexts.

### 'splat' operation

Syntax:

```
operation ::= ssa-id `=` `splat` ssa-use `:` ( vector-type | tensor-type )
```

Broadcast the operand to all elements of the result vector or tensor. The
operand has to be of either integer or float type. When the result is a tensor,
it has to be statically shaped.

Example:

```mlir
  %s = load %A[%i] : memref<128xf32>
  %v = splat %s : vector<4xf32>
  %t = splat %s : tensor<8x16xi32>
```

TODO: This operation is easy to extend to broadcast to dynamically shaped
tensors in the same way dynamically shaped memrefs are handled.
```mlir {.mlir}
// Broadcasts %s to a 2-d dynamically shaped tensor, with %m, %n binding
// to the sizes of the two dynamic dimensions.
%m = "foo"() : () -> (index)
%n = "bar"() : () -> (index)
%t = splat %s [%m, %n] : tensor<?x?xi32>
```

### 'store' operation

Syntax:

```
operation ::= `store` ssa-use `,` ssa-use `[` ssa-use-list `]` `:` memref-type
```

Store value to memref location given by indices. The value stored should have
the same type as the elemental type of the memref. The number of arguments
provided within brackets need to match the rank of the memref.

In an affine context, the indices of a store are restricted to SSA values bound
to surrounding loop induction variables,
[symbols](Affine.md#restrictions-on-dimensions-and-symbols), results of a
[`constant` operation](#constant-operation), or the result of an
[`affine.apply`](Affine.md#affineapply-operation) operation that can in turn
take as arguments all of the aforementioned SSA values or the recursively result
of such an `affine.apply` operation.

Example:

```mlir
store %100, %A[%1, 1023] : memref<4x?xf32, #layout, memspace0>
```

**Context:** The `load` and `store` operations are specifically crafted to fully
resolve a reference to an element of a memref, and (in polyhedral `affine.if`
and `affine.for` operations) the compiler can follow use-def chains (e.g.
through [`affine.apply`](Affine.md#affineapply-operation) operations) to
precisely analyze references at compile-time using polyhedral techniques. This
is possible because of the
[restrictions on dimensions and symbols](Affine.md#restrictions-on-dimensions-and-symbols)
in these contexts.

### 'tensor_load' operation

Syntax:

```
operation ::= ssa-id `=` `tensor_load` ssa-use-and-type
```

Create a tensor from a memref, making an independent copy of the element data.
The result value is a tensor whose shape and element type match the memref
operand.

Example:

```mlir
// Produces a value of tensor<4x?xf32> type.
%12 = tensor_load %10 : memref<4x?xf32, #layout, memspace0>
```

### 'tensor_store' operation

Syntax:

```
operation ::= `tensor_store` ssa-use `,` ssa-use `:` memref-type
```

Stores the contents of a tensor into a memref. The first operand is a value of
tensor type, the second operand is a value of memref type. The shapes and
element types of these must match, and are specified by the memref type.

Example:

```mlir
%9 = dim %8, 1 : tensor<4x?xf32>
%10 = alloc(%9) : memref<4x?xf32, #layout, memspace0>
tensor_store %8, %10 : memref<4x?xf32, #layout, memspace0>
```

## Unary Operations

### 'absf' operation

Syntax:

```
operation ::= ssa-id `=` `absf` ssa-use `:` type
```

Examples:

```mlir
// Scalar absolute value.
%a = absf %b : f64

// SIMD vector element-wise absolute value.
%f = absf %g : vector<4xf32>

// Tensor element-wise absolute value.
%x = absf %y : tensor<4x?xf8>
```

The `absf` operation computes the absolute value. It takes one operand and
returns one result of the same type. This type may be a float scalar type, a
vector whose element type is float, or a tensor of floats. It has no standard
attributes.

### 'ceilf' operation

Syntax:

```
operation ::= ssa-id `=` `ceilf` ssa-use `:` type
```

Examples:

```mlir
// Scalar ceiling value.
%a = ceilf %b : f64

// SIMD vector element-wise ceiling value.
%f = ceilf %g : vector<4xf32>

// Tensor element-wise ceiling value.
%x = ceilf %y : tensor<4x?xf8>
```

The `ceilf` operation computes the ceiling of a given value. It takes one
operand and returns one result of the same type. This type may be a float
scalar type, a vector whose element type is float, or a tensor of floats. It
has no standard attributes.

### 'cos' operation

Syntax:

```
operation ::= ssa-id `=` `cos` ssa-use `:` type
```

Examples:

```mlir
// Scalar cosine value.
%a = cos %b : f64

// SIMD vector element-wise cosine value.
%f = cos %g : vector<4xf32>

// Tensor element-wise cosine value.
%x = cos %y : tensor<4x?xf8>
```

The `cos` operation computes the cosine of a given value. It takes one operand
and returns one result of the same type. This type may be a float scalar type,
a vector whose element type is float, or a tensor of floats. It has no standard
attributes.

### 'exp' operation

Syntax:

```
operation ::= ssa-id `=` `exp` ssa-use `:` type
```

Examples:

```mlir
// Scalar natural exponential.
%a = exp %b : f64

// SIMD vector element-wise natural exponential.
%f = exp %g : vector<4xf32>

// Tensor element-wise natural exponential.
%x = exp %y : tensor<4x?xf8>
```

The `exp` operation takes one operand and returns one result of the same type.
This type may be a float scalar type, a vector whose element type is float, or a
tensor of floats. It has no standard attributes.

### 'negf' operation

Syntax:

```
operation ::= ssa-id `=` `negf` ssa-use `:` type
```

Examples:

```mlir
// Scalar negation value.
%a = negf %b : f64

// SIMD vector element-wise negation value.
%f = negf %g : vector<4xf32>

// Tensor element-wise negation value.
%x = negf %y : tensor<4x?xf8>
```

The `negf` operation computes the negation of a given value. It takes one
operand and returns one result of the same type. This type may be a float
scalar type, a vector whose element type is float, or a tensor of floats. It
has no standard attributes.

### 'tanh' operation

Syntax:

```
operation ::= ssa-id `=` `tanh` ssa-use `:` type
```

Examples:

```mlir
// Scalar hyperbolic tangent value.
%a = tanh %b : f64

// SIMD vector element-wise hyperbolic tangent value.
%f = tanh %g : vector<4xf32>

// Tensor element-wise hyperbolic tangent value.
%x = tanh %y : tensor<4x?xf8>
```

The `tanh` operation computes the hyperbolic tangent. It takes one operand and
returns one result of the same type. This type may be a float scalar type, a
vector whose element type is float, or a tensor of floats. It has no standard
attributes.

## Arithmetic Operations

Basic arithmetic in MLIR is specified by standard operations described in this
section.

### 'addi' operation

Syntax:

```
operation ::= ssa-id `=` `addi` ssa-use `,` ssa-use `:` type
```

Examples:

```mlir
// Scalar addition.
%a = addi %b, %c : i64

// SIMD vector element-wise addition, e.g. for Intel SSE.
%f = addi %g, %h : vector<4xi32>

// Tensor element-wise addition.
%x = addi %y, %z : tensor<4x?xi8>
```

The `addi` operation takes two operands and returns one result, each of these is
required to be the same type. This type may be an integer scalar type, a vector
whose element type is integer, or a tensor of integers. It has no standard
attributes.

### 'addf' operation

Syntax:

```
operation ::= ssa-id `=` `addf` ssa-use `,` ssa-use `:` type
```

Examples:

```mlir
// Scalar addition.
%a = addf %b, %c : f64

// SIMD vector addition, e.g. for Intel SSE.
%f = addf %g, %h : vector<4xf32>

// Tensor addition.
%x = addf %y, %z : tensor<4x?xbf16>
```

The `addf` operation takes two operands and returns one result, each of these is
required to be the same type. This type may be a floating point scalar type, a
vector whose element type is a floating point type, or a floating point tensor.

It has no standard attributes.

TODO: In the distant future, this will accept optional attributes for fast math,
contraction, rounding mode, and other controls.

### 'and' operation

Bitwise integer and.

Syntax:

```
operation ::= ssa-id `=` `and` ssa-use `,` ssa-use `:` type
```

Examples:

```mlir
// Scalar integer bitwise and.
%a = and %b, %c : i64

// SIMD vector element-wise bitwise integer and.
%f = and %g, %h : vector<4xi32>

// Tensor element-wise bitwise integer and.
%x = and %y, %z : tensor<4x?xi8>
```

The `and` operation takes two operands and returns one result, each of these is
required to be the same type. This type may be an integer scalar type, a vector
whose element type is integer, or a tensor of integers. It has no standard
attributes.

### 'cmpi' operation

Syntax:

```
operation ::= ssa-id `=` `cmpi` string-literal `,` ssa-id `,` ssa-id `:` type
```

Examples:

```mlir
// Custom form of scalar "signed less than" comparison.
%x = cmpi "slt", %lhs, %rhs : i32

// Generic form of the same operation.
%x = "std.cmpi"(%lhs, %rhs) {predicate = 2 : i64} : (i32, i32) -> i1

// Custom form of vector equality comparison.
%x = cmpi "eq", %lhs, %rhs : vector<4xi64>

// Generic form of the same operation.
%x = "std.cmpi"(%lhs, %rhs) {predicate = 0 : i64}
    : (vector<4xi64>, vector<4xi64>) -> vector<4xi1>
```

The `cmpi` operation is a generic comparison for integer-like types. Its two
arguments can be integers, vectors or tensors thereof as long as their types
match. The operation produces an i1 for the former case, a vector or a tensor of
i1 with the same shape as inputs in the other cases.

Its first argument is an attribute that defines which type of comparison is
performed. The following comparisons are supported:

-   equal (mnemonic: `"eq"`; integer value: `0`)
-   not equal (mnemonic: `"ne"`; integer value: `1`)
-   signed less than (mnemonic: `"slt"`; integer value: `2`)
-   signed less than or equal (mnemonic: `"sle"`; integer value: `3`)
-   signed greater than (mnemonic: `"sgt"`; integer value: `4`)
-   signed greater than or equal (mnemonic: `"sge"`; integer value: `5`)
-   unsigned less than (mnemonic: `"ult"`; integer value: `6`)
-   unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
-   unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
-   unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)

The result is `1` if the comparison is true and `0` otherwise. For vector or
tensor operands, the comparison is performed elementwise and the element of the
result indicates whether the comparison is true for the operand elements with
the same indices as those of the result.

Note: while the custom assembly form uses strings, the actual underlying
attribute has integer type (or rather enum class in C++ code) as seen from the
generic assembly form. String literals are used to improve readability of the IR
by humans.

This operation only applies to integer-like operands, but not floats. The main
reason being that comparison operations have diverging sets of attributes:
integers require sign specification while floats require various floating
point-related particularities, e.g., `-ffast-math` behavior, IEEE754 compliance,
etc
([rationale](../Rationale.md#splitting-floating-point-vs-integer-operations)).
The type of comparison is specified as attribute to avoid introducing ten
similar operations, taking into account that they are often implemented using
the same operation downstream
([rationale](../Rationale.md#specifying-comparison-kind-as-attribute)). The
separation between signed and unsigned order comparisons is necessary because of
integers being signless. The comparison operation must know how to interpret
values with the foremost bit being set: negatives in two's complement or large
positives
([rationale](../Rationale.md#specifying-sign-in-integer-comparison-operations)).

### 'constant' operation

Syntax:

```
operation ::= ssa-id `=` `constant` attribute-value `:` type
```

The `constant` operation produces an SSA value equal to some constant specified
by an attribute. This is the way that MLIR uses to form simple integer and
floating point constants, as well as more exotic things like references to
functions and (TODO!) tensor/vector constants.

The `constant` operation is represented with a single attribute named "value".
The type specifies the result type of the operation.

Examples:

```mlir
// Integer constant
%1 = constant 42 : i32

// Reference to function @myfn.
%3 = constant @myfn : (tensor<16xf32>, f32) -> tensor<16xf32>

// Equivalent generic forms
%1 = "std.constant"() {value = 42 : i32} : () -> i32
%3 = "std.constant"() {value = @myfn}
   : () -> ((tensor<16xf32>, f32) -> tensor<16xf32>)

```

MLIR does not allow direct references to functions in SSA operands because the
compiler is multithreaded, and disallowing SSA values to directly reference a
function simplifies this
([rationale](../Rationale.md#multithreading-the-compiler)).

### 'copysign' operation

Syntax:

```
operation ::= ssa-id `=` `copysign` ssa-use `:` type
```

Examples:

```mlir
// Scalar copysign value.
%a = copysign %b %c : f64

// SIMD vector element-wise copysign value.
%f = copysign %g %h : vector<4xf32>

// Tensor element-wise copysign value.
%x = copysign %y %z : tensor<4x?xf8>
```

The `copysign` returns a value with the magnitude of the first operand and the
sign of the second operand. It takes two operands and returns one result of the
same type. This type may be a float scalar type, a vector whose element type is
float, or a tensor of floats. It has no standard attributes.

### 'divis' operation

Signed integer division. Rounds towards zero. Treats the leading bit as sign,
i.e. `6 / -2 = -3`.

Note: the semantics of division by zero or signed division overflow (minimum
value divided by -1) is TBD; do NOT assume any specific behavior.

Syntax:

```
operation ::= ssa-id `=` `divis` ssa-use `,` ssa-use `:` type
```

Examples:

```mlir
// Scalar signed integer division.
%a = divis %b, %c : i64

// SIMD vector element-wise division.
%f = divis %g, %h : vector<4xi32>

// Tensor element-wise integer division.
%x = divis %y, %z : tensor<4x?xi8>
```

The `divis` operation takes two operands and returns one result, each of these
is required to be the same type. This type may be an integer scalar type, a
vector whose element type is integer, or a tensor of integers. It has no
standard attributes.

### 'diviu' operation

Unsigned integer division. Rounds towards zero. Treats the leading bit as the
most significant, i.e. for `i16` given two's complement representation, `6 /
-2 = 6 / (2^16 - 2) = 0`.

Note: the semantics of division by zero is TBD; do NOT assume any specific
behavior.

Syntax:

```
operation ::= ssa-id `=` `diviu` ssa-use `,` ssa-use `:` type
```

Examples:

```mlir
// Scalar unsigned integer division.
%a = diviu %b, %c : i64

// SIMD vector element-wise division.
%f = diviu %g, %h : vector<4xi32>

// Tensor element-wise integer division.
%x = diviu %y, %z : tensor<4x?xi8>
```

The `diviu` operation takes two operands and returns one result, each of these
is required to be the same type. This type may be an integer scalar type, a
vector whose element type is integer, or a tensor of integers. It has no
standard attributes.

### 'memref_cast' operation

Syntax:

```
operation ::= ssa-id `=` `memref_cast` ssa-use `:` type `to` type
```

Examples:

```mlir
// Discard static dimension information.
%3 = memref_cast %2 : memref<4x?xf32> to memref<?x?xf32>

// Convert to a type with more known dimensions.
%4 = memref_cast %3 : memref<?x?xf32> to memref<4x?xf32>

// Convert to a type with unknown rank.
%5 = memref_cast %3 : memref<?x?xf32> to memref<*xf32>

// Convert to a type with static rank.
%6 = memref_cast %5 : memref<*xf32> to memref<?x?xf32>
```

Convert a memref from one type to an equivalent type without changing any data
elements. The types are equivalent if 1. they both have the same static rank,
same element type, same mappings, same address space. The operation is invalid
if converting to a mismatching constant dimension, or 2. exactly one of the
operands have an unknown rank, and they both have the same element type and same
address space. The operation is invalid if both operands are of dynamic rank or
if converting to a mismatching static rank.

### 'mulf' operation

Syntax:

```
operation ::= ssa-id `=` `mulf` ssa-use `,` ssa-use `:` type
```

Examples:

```mlir
// Scalar multiplication.
%a = mulf %b, %c : f64

// SIMD pointwise vector multiplication, e.g. for Intel SSE.
%f = mulf %g, %h : vector<4xf32>

// Tensor pointwise multiplication.
%x = mulf %y, %z : tensor<4x?xbf16>
```

The `mulf` operation takes two operands and returns one result, each of these is
required to be the same type. This type may be a floating point scalar type, a
vector whose element type is a floating point type, or a floating point tensor.

It has no standard attributes.

TODO: In the distant future, this will accept optional attributes for fast math,
contraction, rounding mode, and other controls.

### 'or' operation

Bitwise integer or.

Syntax:

```
operation ::= ssa-id `=` `or` ssa-use `,` ssa-use `:` type
```

Examples:

```mlir
// Scalar integer bitwise or.
%a = or %b, %c : i64

// SIMD vector element-wise bitwise integer or.
%f = or %g, %h : vector<4xi32>

// Tensor element-wise bitwise integer or.
%x = or %y, %z : tensor<4x?xi8>
```

The `or` operation takes two operands and returns one result, each of these is
required to be the same type. This type may be an integer scalar type, a vector
whose element type is integer, or a tensor of integers. It has no standard
attributes.

### 'remis' operation

Signed integer division remainder. Treats the leading bit as sign, i.e. `6 %
-2 = 0`.

Note: the semantics of division by zero is TBD; do NOT assume any specific
behavior.

Syntax:

```
operation ::= ssa-id `=` `remis` ssa-use `,` ssa-use `:` type
```

Examples:

```mlir
// Scalar signed integer division remainder.
%a = remis %b, %c : i64

// SIMD vector element-wise division remainder.
%f = remis %g, %h : vector<4xi32>

// Tensor element-wise integer division remainder.
%x = remis %y, %z : tensor<4x?xi8>
```

The `remis` operation takes two operands and returns one result, each of these
is required to be the same type. This type may be an integer scalar type, a
vector whose element type is integer, or a tensor of integers. It has no
standard attributes.

### 'remiu' operation

Unsigned integer division remainder. Treats the leading bit as the most
significant, i.e. for `i16`, `6 % -2 = 6 % (2^16 - 2) = 6`.

Note: the semantics of division by zero is TBD; do NOT assume any specific
behavior.

Syntax:

```
operation ::= ssa-id `=` `remiu` ssa-use `,` ssa-use `:` type
```

Examples:

```mlir
// Scalar unsigned integer division remainder.
%a = remiu %b, %c : i64

// SIMD vector element-wise division remainder.
%f = remiu %g, %h : vector<4xi32>

// Tensor element-wise integer division remainder.
%x = remiu %y, %z : tensor<4x?xi8>
```

The `remiu` operation takes two operands and returns one result, each of these
is required to be the same type. This type may be an integer scalar type, a
vector whose element type is integer, or a tensor of integers. It has no
standard attributes.

### 'select' operation

Syntax:

```
operation ::= ssa-id `=` `select` ssa-use `,` ssa-use `,` ssa-use `:` type
```

Examples:

```mlir
// Custom form of scalar selection.
%x = select %cond, %true, %false : i32

// Generic form of the same operation.
%x = "std.select"(%cond, %true, %false) : (i1, i32, i32) -> i32

// Vector selection is element-wise
%vx = "std.select"(%vcond, %vtrue, %vfalse)
    : (vector<42xi1>, vector<42xf32>, vector<42xf32>) -> vector<42xf32>
```

The `select` operation chooses one value based on a binary condition supplied as
its first operand. If the value of the first operand is `1`, the second operand
is chosen, otherwise the third operand is chosen. The second and the third
operand must have the same type.

The operation applies to vectors and tensors elementwise given the _shape_ of
all operands is identical. The choice is made for each element individually
based on the value at the same position as the element in the condition operand.

The `select` operation combined with [`cmpi`](#cmpi-operation) can be used to
implement `min` and `max` with signed or unsigned comparison semantics.

### 'tensor_cast' operation

Syntax:

```
operation ::= ssa-id `=` `tensor_cast` ssa-use `:` type `to` type
```

Examples:

```mlir
// Convert from unknown rank to rank 2 with unknown dimension sizes.
%2 = "std.tensor_cast"(%1) : (tensor<*xf32>) -> tensor<?x?xf32>
%2 = tensor_cast %1 : tensor<*xf32> to tensor<?x?xf32>

// Convert to a type with more known dimensions.
%3 = "std.tensor_cast"(%2) : (tensor<?x?xf32>) -> tensor<4x?xf32>

// Discard static dimension and rank information.
%4 = "std.tensor_cast"(%3) : (tensor<4x?xf32>) -> tensor<?x?xf32>
%5 = "std.tensor_cast"(%4) : (tensor<?x?xf32>) -> tensor<*xf32>
```

Convert a tensor from one type to an equivalent type without changing any data
elements. The source and destination types must both be tensor types with the
same element type. If both are ranked, then the rank should be the same and
static dimensions should match. The operation is invalid if converting to a
mismatching constant dimension.

### 'xor' operation

Bitwise integer xor.

Syntax:

```
operation ::= ssa-id `=` `xor` ssa-use, ssa-use `:` type
```

Examples:

```mlir
// Scalar integer bitwise xor.
%a = xor %b, %c : i64

// SIMD vector element-wise bitwise integer xor.
%f = xor %g, %h : vector<4xi32>

// Tensor element-wise bitwise integer xor.
%x = xor %y, %z : tensor<4x?xi8>
```

The `xor` operation takes two operands and returns one result, each of these is
required to be the same type. This type may be an integer scalar type, a vector
whose element type is integer, or a tensor of integers. It has no standard
attributes.
