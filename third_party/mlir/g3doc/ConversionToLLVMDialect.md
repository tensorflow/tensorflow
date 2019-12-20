# Conversion to the LLVM Dialect

Conversion from the Standard to the [LLVM Dialect](Dialects/LLVM.md) can be
performed by the specialized dialect conversion pass by running

```sh
mlir-opt -convert-std-to-llvm <filename.mlir>
```

It performs type and operation conversions for a subset of operations from
standard dialect (operations on scalars and vectors, control flow operations) as
described in this document. We use the terminology defined by the
[LLVM IR Dialect description](Dialects/LLVM.md) throughout this document.

[TOC]

## Type Conversion

### Scalar Types

Scalar types are converted to their LLVM counterparts if they exist. The
following conversions are currently implemented.

-   `i*` converts to `!llvm.i*`
-   `f16` converts to `!llvm.half`
-   `f32` converts to `!llvm.float`
-   `f64` converts to `!llvm.double`

Note: `bf16` type is not supported by LLVM IR and cannot be converted.

### Index Type

Index type is converted to a wrapped LLVM IR integer with bitwidth equal to the
bitwidth of the pointer size as specified by the
[data layout](https://llvm.org/docs/LangRef.html#data-layout) of the LLVM module
[contained](Dialects/LLVM.md#context-and-module-association) in the LLVM Dialect
object. For example, on x86-64 CPUs it converts to `!llvm.i64`.

### Vector Types

LLVM IR only supports *one-dimensional* vectors, unlike MLIR where vectors can
be multi-dimensional. Vector types cannot be nested in either IR. In the
one-dimensional case, MLIR vectors are converted to LLVM IR vectors of the same
size with element type converted using these conversion rules. In the
n-dimensional case, MLIR vectors are converted to (n-1)-dimensional array types
of one-dimensional vectors.

For example, `vector<4 x f32>` converts to `!llvm<"<4 x float>">` and `vector<4
x 8 x 16 x f32>` converts to `!llvm<"[4 x [8 x <16 x float>]]">`.

### Memref Types

Memref types in MLIR have both static and dynamic information associated with
them. The dynamic information comprises the buffer pointer as well as sizes and
strides of any dynamically sized dimensions. Memref types are normalized and
converted to a descriptor that is only dependent on the rank of the memref. The
descriptor contains:

1.  the pointer to the data buffer, followed by
2.  the pointer to properly aligned data payload that the memref indexes,
    followed by
3.  a lowered `index`-type integer containing the distance between the beginning
    of the buffer and the first element to be accessed through the memref,
    followed by
4.  an array containing as many `index`-type integers as the rank of the memref:
    the array represents the size, in number of elements, of the memref along
    the given dimension. For constant MemRef dimensions, the corresponding size
    entry is a constant whose runtime value must match the static value,
    followed by
5.  a second array containing as many 64-bit integers as the rank of the MemRef:
    the second array represents the "stride" (in tensor abstraction sense), i.e.
    the number of consecutive elements of the underlying buffer.

For constant memref dimensions, the corresponding size entry is a constant whose
runtime value matches the static value. This normalization serves as an ABI for
the memref type to interoperate with externally linked functions. In the
particular case of rank `0` memrefs, the size and stride arrays are omitted,
resulting in a struct containing two pointers + offset.

Examples:

```mlir
memref<f32> -> !llvm<"{ float*, float*, i64 }">
memref<1 x f32> -> !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
memref<? x f32> -> !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
memref<10x42x42x43x123 x f32> -> !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64] }">
memref<10x?x42x?x123 x f32> -> !llvm<"{ float*, float*, i64, [5 x i64], [5 x i64]  }">

// Memref types can have vectors as element types
memref<1x? x vector<4xf32>> -> !llvm<"{ <4 x float>*, <4 x float>*, i64, [1 x i64], [1 x i64] }">
```

If the rank of the memref is unknown at compile time, the Memref is converted to
an unranked descriptor that contains:

1.  a 64-bit integer representing the dynamic rank of the memref, followed by
2.  a pointer to a ranked memref descriptor with the contents listed above.

Dynamic ranked memrefs should be used only to pass arguments to external library
calls that expect a unified memref type. The called functions can parse any
unranked memref descriptor by reading the rank and parsing the enclosed ranked
descriptor pointer.

Examples:

```mlir
// unranked descriptor
memref<*xf32> -> !llvm<"{i64, i8*}">
```

**In function signatures,** `memref` is passed as a _pointer_ to the structured
defined above to comply with the calling convention.

Example:

```mlir
// A function type with memref as argument
(memref<?xf32>) -> ()
// is transformed into the LLVM function with pointer-to-structure argument.
!llvm<"void({ float*, float*, i64, [1 x i64], [1 x i64]}*) ">
```

### Function Types

Function types get converted to LLVM function types. The arguments are converted
individually according to these rules. The result types need to accommodate the
fact that LLVM IR functions always have a return type, which may be a Void type.
The converted function always has a single result type. If the original function
type had no results, the converted function will have one result of the wrapped
`void` type. If the original function type had one result, the converted
function will have one result converted using these rules. Otherwise, the result
type will be a wrapped LLVM IR structure type where each element of the
structure corresponds to one of the results of the original function, converted
using these rules. In high-order functions, function-typed arguments and results
are converted to a wrapped LLVM IR function pointer type (since LLVM IR does not
allow passing functions to functions without indirection) with the pointee type
converted using these rules.

Examples:

```mlir
// zero-ary function type with no results.
() -> ()
// is converted to a zero-ary function with `void` result
!llvm<"void ()">

// unary function with one result
(i32) -> (i64)
// has its argument and result type converted, before creating the LLVM IR function type
!llvm<"i64 (i32)">

// binary function with one result
(i32, f32) -> (i64)
// has its arguments handled separately
!llvm<"i64 (i32, float)">

// binary function with two results
(i32, f32) -> (i64, f64)
// has its result aggregated into a structure type
!llvm<"{i64, double} (i32, f32)">

// function-typed arguments or results in higher-order functions
(() -> ()) -> (() -> ())
// are converted into pointers to functions
!llvm<"void ()* (void ()*)">
```

## Calling Convention

### Function Signature Conversion

LLVM IR functions are defined by a custom operation. The function itself has a
wrapped LLVM IR function type converted as described above. The function
definition operation uses MLIR syntax.

Examples:

```mlir
// zero-ary function type with no results.
func @foo() -> ()
// gets LLVM type void().
llvm.func @foo() -> ()

// function with one result
func @bar(i32) -> (i64)
// gets converted to LLVM type i64(i32).
func @bar(!llvm.i32) -> !llvm.i64

// function with two results
func @qux(i32, f32) -> (i64, f64)
// has its result aggregated into a structure type
func @qux(!llvm.i32, !llvm.float) -> !llvm<"{i64, double}">

// function-typed arguments or results in higher-order functions
func @quux(() -> ()) -> (() -> ())
// are converted into pointers to functions
func @quux(!llvm<"void ()*">) -> !llvm<"void ()*">
// the call flow is handled by the LLVM dialect `call` operation supporting both
// direct and indirect calls
```

### Result Packing

In case of multi-result functions, the returned values are inserted into a
structure-typed value before being returned and extracted from it at the call
site. This transformation is a part of the conversion and is transparent to the
defines and uses of the values being returned.

Example:

```mlir
func @foo(%arg0: i32, %arg1: i64) -> (i32, i64) {
  return %arg0, %arg1 : i32, i64
}
func @bar() {
  %0 = constant 42 : i32
  %1 = constant 17 : i64
  %2:2 = call @foo(%0, %1) : (i32, i64) -> (i32, i64)
  "use_i32"(%2#0) : (i32) -> ()
  "use_i64"(%2#1) : (i64) -> ()
}

// is transformed into

func @foo(%arg0: !llvm.i32, %arg1: !llvm.i64) -> !llvm<"{i32, i64}"> {
  // insert the vales into a structure
  %0 = llvm.mlir.undef :  !llvm<"{i32, i64}">
  %1 = llvm.insertvalue %arg0, %0[0] : !llvm<"{i32, i64}">
  %2 = llvm.insertvalue %arg1, %1[1] : !llvm<"{i32, i64}">

  // return the structure value
  llvm.return %2 : !llvm<"{i32, i64}">
}
func @bar() {
  %0 = llvm.mlir.constant(42 : i32) : !llvm.i32
  %1 = llvm.mlir.constant(17) : !llvm.i64

  // call and extract the values from the structure
  %2 = llvm.call @bar(%0, %1) : (%arg0: !llvm.i32, %arg1: !llvm.i32) -> !llvm<"{i32, i64}">
  %3 = llvm.extractvalue %2[0] : !llvm<"{i32, i64}">
  %4 = llvm.extractvalue %2[1] : !llvm<"{i32, i64}">

  // use as before
  "use_i32"(%3) : (!llvm.i32) -> ()
  "use_i64"(%4) : (!llvm.i64) -> ()
}
```

### Calling Convention for `memref`

For function _arguments_ of `memref` type, ranked or unranked, the type of the
argument is a _pointer_ to the memref descriptor type defined above. The caller
of such function is required to store the descriptor in memory and guarantee
that the storage remains live until the callee returns. The caller can than pass
the pointer to that memory as function argument. The callee loads from the
pointers it was passed as arguments in the entry block of the function, making
the descriptor passed in as argument available for use similarly to
ocally-defined descriptors.

This convention is implemented in the conversion of `std.func` and `std.call` to
the LLVM dialect. Conversions from other dialects should take it into account.
The motivation for this convention is to simplify the ABI for interfacing with
other LLVM modules, in particular those generated from C sources, while avoiding
platform-specific aspects until MLIR has a proper ABI modeling.

Example:

```mlir

func @foo(memref<?xf32>) -> () {
  %c0 = constant 0 : index
  load %arg0[%c0] : memref<?xf32>
  return
}

func @bar(%arg0: index) {
  %0 = alloc(%arg0) : memref<?xf32>
  call @foo(%0) : (memref<?xf32>)-> ()
  return
}

// Gets converted to the following IR.
// Accepts a pointer to the memref descriptor.
llvm.func @foo(!llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">) {
  // Loads the descriptor so that it can be used similarly to locally
  // created descriptors.
  %0 = llvm.load %arg0 : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">
}

llvm.func @bar(%arg0: !llvm.i64) {
  // ... Allocation ...
  // Definition of the descriptor.
  %7 = llvm.mlir.undef : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
  // ... Filling in the descriptor ...
  %14 = // The final value of the allocated descriptor.
  // Allocate the memory for the descriptor and store it.
  %15 = llvm.mlir.constant(1 : index) : !llvm.i64
  %16 = llvm.alloca %15 x !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }">
      : (!llvm.i64) -> !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">
  llvm.store %14, %16 : !llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">
  // Pass the pointer to the function.
  llvm.call @foo(%16) : (!llvm<"{ float*, float*, i64, [1 x i64], [1 x i64] }*">) -> ()
  llvm.return
}
```

*This convention may or may not apply if the conversion of MemRef types is
overridden by the user.*

## Repeated Successor Removal

Since the goal of the LLVM IR dialect is to reflect LLVM IR in MLIR, the dialect
and the conversion procedure must account for the differences between block
arguments and LLVM IR PHI nodes. In particular, LLVM IR disallows PHI nodes with
different values coming from the same source. Therefore, the LLVM IR dialect
disallows operations that have identical successors accepting arguments, which
would lead to invalid PHI nodes. The conversion process resolves the potential
PHI source ambiguity by injecting dummy blocks if the same block is used more
than once as a successor in an instruction. These dummy blocks branch
unconditionally to the original successors, pass them the original operands
(available in the dummy block because it is dominated by the original block) and
are used instead of them in the original terminator operation.

Example:

```mlir
  cond_br %0, ^bb1(%1 : i32), ^bb1(%2 : i32)
^bb1(%3 : i32)
  "use"(%3) : (i32) -> ()
```

leads to a new basic block being inserted,

```mlir
  cond_br %0, ^bb1(%1 : i32), ^dummy
^bb1(%3 : i32):
  "use"(%3) : (i32) -> ()
^dummy:
  br ^bb1(%4 : i32)
```

before the conversion to the LLVM IR dialect:

```mlir
  llvm.cond_br  %0, ^bb1(%1 : !llvm.i32), ^dummy
^bb1(%3 : !llvm<"i32">):
  "use"(%3) : (!llvm.i32) -> ()
^dummy:
  llvm.br ^bb1(%2 : !llvm.i32)
```

## Default Memref Model

### Memref Descriptor

Within a converted function, a `memref`-typed value is represented by a memref
_descriptor_, the type of which is the structure type obtained by converting
from the memref type. This descriptor holds all the necessary information to
produce an address of a specific element. In particular, it holds dynamic values
for static sizes, and they are expected to match at all times.

It is created by the allocation operation and is updated by the conversion
operations that may change static dimensions into dynamic and vice versa.

**Note**: LLVM IR conversion does not support `memref`s with layouts that are
not amenable to the strided form.

### Index Linearization

Accesses to a memref element are transformed into an access to an element of the
buffer pointed to by the descriptor. The position of the element in the buffer
is calculated by linearizing memref indices in row-major order (lexically first
index is the slowest varying, similar to C, but accounting for strides). The
computation of the linear address is emitted as arithmetic operation in the LLVM
IR dialect. Strides are extracted from the memref descriptor.

Accesses to zero-dimensional memref (that are interpreted as pointers to the
elemental type) are directly converted into `llvm.load` or `llvm.store` without
any pointer manipulations.

Examples:

An access to a zero-dimensional memref is converted into a plain load:

```mlir
// before
%0 = load %m[] : memref<f32>

// after
%0 = llvm.load %m : !llvm<"float*">
```

An access to a memref with indices:

```mlir
%0 = load %m[1,2,3,4] : memref<10x?x13x?xf32>
```

is transformed into the equivalent of the following code:

```mlir
// Compute the linearized index from strides. Each block below extracts one
// stride from the descriptor, multipllies it with the index and accumulates
// the total offset.
%stride1 = llvm.extractvalue[4, 0] : !llvm<"{float*, float*, i64, i64[4], i64[4]}">
%idx1 = llvm.mlir.constant(1 : index) !llvm.i64
%addr1 = muli %stride1, %idx1 : !llvm.i64

%stride2 = llvm.extractvalue[4, 1] : !llvm<"{float*, float*, i64, i64[4], i64[4]}">
%idx2 = llvm.mlir.constant(2 : index) !llvm.i64
%addr2 = muli %stride2, %idx2 : !llvm.i64
%addr3 = addi %addr1, %addr2 : !llvm.i64

%stride3 = llvm.extractvalue[4, 2] : !llvm<"{float*, float*, i64, i64[4], i64[4]}">
%idx3 = llvm.mlir.constant(3 : index) !llvm.i64
%addr4 = muli %stride3, %idx3 : !llvm.i64
%addr5 = addi %addr3, %addr4 : !llvm.i64

%stride4 = llvm.extractvalue[4, 3] : !llvm<"{float*, float*, i64, i64[4], i64[4]}">
%idx4 = llvm.mlir.constant(4 : index) !llvm.i64
%addr6 = muli %stride4, %idx4 : !llvm.i64
%addr7 = addi %addr5, %addr6 : !llvm.i64

// Add the linear offset to the address.
%offset = llvm.extractvalue[2] : !llvm<"{float*, float*, i64, i64[4], i64[4]}">
%addr8 = addi %addr7, %offset : !llvm.i64

// Obtain the aligned pointer.
%aligned = llvm.extractvalue[1] : !llvm<"{float*, float*, i64, i64[4], i64[4]}">

// Get the address of the data pointer.
%ptr = llvm.getelementptr %aligned[%addr8]
    : !llvm<"{float*, float*, i64, i64[4], i64[4]}"> -> !llvm<"float*">

// Perform the actual load.
%0 = llvm.load %ptr : !llvm<"float*">
```

For stores, the address computation code is identical and only the actual store
operation is different.

Note: the conversion does not perform any sort of common subexpression
elimination when emitting memref accesses.
