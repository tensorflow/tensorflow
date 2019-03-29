# LLVM IR Dialect

This dialect wraps the LLVM IR types and instructions into MLIR types and
operations. It provides several additional operations that are necessary to
cover for the differences in the IR structure (e.g., MLIR does not have `phi`
operations and LLVM IR does not have a `constant` operation).

In this document, we use "LLVM IR" to designate the
[intermediate representation of LLVM](https://llvm.org/docs/LangRef.html) and
"LLVM IR _dialect_" to refer to the MLIR dialect reflecting LLVM instructions
and types.

[TOC]

## Context and Module Association

The LLVM IR dialect object _contains_ an LLVM Context and an LLVM Module that it
uses to define, print, parse and manage LLVM IR types. These objects can be
obtained from the dialect object using `.getLLVMContext()` and
`getLLVMModule()`. All LLVM IR objects that interact with the LLVM IR dialect
must exist in the dialect's context.

## Types {#types}

The LLVM IR dialect defines a single MLIR type, `LLVM::LLVMType`, that can wrap
any existing LLVM IR type. Its syntax is as follows

``` {.ebnf}
type ::= `!llvm<"` llvm-canonical-type `">
llvm-canonical-type ::= <canonical textual representation defined by LLVM>
```

For example, one can use primitive types `!llvm<"i32">`, pointer types
`!llvm<"i8*">`, vector types `!llvm<"<4 x float>">` or structure types
`!llvm<"{i32, float}">`. The parsing and printing of the canonical form is
delegated to the LLVM assembly parser and printer.

LLVM IR dialect types contain an `llvm::Type*` object that can be obtained by
calling `.getUnderlyingType()` and used in LLVM API calls directly. These
objects are allocated within the LLVM context associated with the LLVM IR
dialect and may be linked to the properties of the associated LLVM module.

LLVM IR dialect type can be constructed from any `llvm::Type*` that is
associated with the LLVM context of the dialect. In this document, we use the
term "wrapped LLVM IR type" to refer to the LLVM IR dialect type containing a
specific LLVM IR type.

## Operations {#operations}

All operations in the LLVM IR dialect use the generic (verbose) form of MLIR
operations. The mnemonic of an operation is that used in LLVM IR prefixed with
"`llvm.`".

### LLVM IR operations

The following operations are currently supported. The semantics of these
operations corresponds to the semantics of the similarly-named LLVM IR
instructions.

#### Integer binary arithmetic operations

Take two arguments of wrapped LLVM IR integer type, produce one value of the
same type.

-   `add`
-   `sub`
-   `mul`
-   `udiv`
-   `sdiv`
-   `urem`
-   `srem`

Examples:

```mlir {.mlir}
// Integer addition.
%0 = "llvm.add"(%a, %b) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">

// Unsigned integer division.
%1 = "llvm.udiv"(%a, %b) : (!llvm<"i32">, !llvm<"i32">) -> !llvm<"i32">
```

#### Floating point binary arithmetic operations

Take two arguments of wrapped LLVM IR floating point type, produce one value of
the same type.

-   `fadd`
-   `fsub`
-   `fmul`
-   `fdiv`
-   `frem`

Examples:

```mlir {.mlir}
// Float addition.
%0 = "llvm.fadd"(%a, %b) : (!llvm<"float">, !llvm<"float">) -> !llvm<"float">

// Float division.
%1 = "llvm.fdiv"(%a, %b) : (!llvm<"float">, !llvm<"float">) -> !llvm<"float">
```

#### Memory-related operations

-   `<r> = alloca <size>`
-   `<r> = getelementptr <address>, <index> (, <index>)+`
-   `<r> = load <address>`
-   `store <value>, <address>`

In these operations, `<size>` must be a value of wrapped LLVM IR integer type,
`<address>` must be a value of wrapped LLVM IR pointer type, and `<value>` must
be a value of wrapped LLVM IR type that corresponds to the pointee type of
`<address>`.

The `index` operands are integer values whose semantics is identical to the
non-pointer arguments of LLVM IR's `getelementptr`.

Examples:

```mlir {.mlir}
// Allocate an array of 4 floats on stack
%c4 = "llvm.constant" {value: 4 : i64} : !llvm<"i64">
%0 = "llvm.alloca"(%c4) : (!llvm<"i64">) -> !llvm<"float*">

// Get the second element of the array (note 0-based indexing).
%c1 = "llvm.constant" {value: 1 : i64} : !llvm<"i64">
%1 = "llvm.getelementptr"(%0, %c1) : (!llvm<"float*">, !llvm<"i64">)
                                   -> !llvm<"float*">

// Store a constant into this element.
%cf = "llvm.constant" {value: 42.0 : f32} : !llvm<"float">
"llvm.store" %cf, %1 : (!llvm<"float">, !llvm<"float*">) -> ()

// Load the value from this element.
%3 = "llvm.load" %1 : (!llvm<"float*">) -> (!llvm<"float">)
```

#### Operations on values of aggregate type.

-   `<value> = extractvalue <struct> {position: [<index> (, <index>)+]}`
-   `<struct> = insertvalue <value>,<struct> {position: [<index> (, <index>)+]}`

In these operations, `<struct>` must be a value of wrapped LLVM IR structure
type and `<value>` must be a value that corresponds to one of the (nested)
structure element types.

The `position` attribute is a mandatory array attribute containing integer
attributes. It identifies the 0-based position of the element in the (nested)
structure type.

Examples:

```mlir {.mlir}
// Get the value third element of the second element of a structure.
%0 = "llvm.extractvalue"(%s) {position: [1, 2]} : (!llvm<"{i32, {i1, i8, i16}">) -> !llvm<"i16">

// Insert the value to the third element of the second element of a structure.
// Note that this returns a new structure-typed value.
%1 = "llvm.insertvalue"(%0, %s) {position: [1, 2]} :
  (!llvm<"i16">, !llvm<"{i32, {i1, i8, i16}">) -> !llvm<"{i32, {i1, i8, i16}">
```

#### Terminator operations.

Branch operations:

-   `br [<successor>(<operands>)]`
-   `cond_br <condition> [<true-successor>(<true-operands>),`
    `<false-successor>(<false-operands>)]`

In order to comply with MLIR design, branch operations in the LLVM IR dialect
pass arguments to basic blocks. Successors must be valid block MLIR identifiers
and operand lists for each of them must have the same types as the arguments of
the respective blocks. `<condition>` must be a wrapped LLVM IR `i1` type.

Since LLVM IR uses the name of the predecessor basic block to identify the
sources of a PHI node, it is invalid for two entries of the PHI node to indicate
different values coming from the same block. Therefore, `cond_br` in the LLVM IR
dialect disallows its successors to be the same block _if_ this block has
arguments.

Examples:

```mlir {.mlir}
// Branch without arguments.
^bb0:
  "llvm.br"() [^bb0] : () -> ()

// Branch and pass arguments.
^bb1(%arg: !llvm<"i32">):
  "llvm.br"() [^bb1(%arg : !llvm<"i32">)] : () -> ()

// Conditionally branch and pass arguments to one of the blocks.
"llvm.cond_br"(%cond) [^bb0, %bb1(%arg : !llvm<"i32">)] : (!llvm<"i1">) -> ()

// It's okay to use the same block without arguments, but probably useless.
"llvm.cond_br"(%cond) [^bb0, ^bb0] : (!llvm<"i1">) ->  ()

// ERROR: Passing different arguments to the same block in a conditional branch.
"llvm.cond_br"(%cond) [^bb1(%0 : !llvm<"i32">),
                       ^bb1(%1 : !llvm<"i32">)] : (!llvm<"i1">) -> ()

```

Call operations:

-   `<r> = call(<operands>)`
-   `call(<operands>)`

In LLVM IR, functions may return either 0 or 1 value. LLVM IR dialect implements
this behavior by providing a variadic `call` operation for 0- and 1-result
functions. Even though MLIR supports multi-result functions, LLVM IR dialect
disallows them.

The `call` operation supports both direct and indirect calls. Direct calls
require the `callee` attribute of function type to be present. Otherwise, the
call is considered indirect and expects the function as its first argument.

Examples:

```mlir {.mlir}
// Direct call without arguments and with one result.
%0 = "llvm.call"() {callee: @foo : () -> (!llvm<"float">)}
    : () -> (!llvm<"float">)

// Direct call with arguments and without a result.
"llvm.call"(%0) {callee: @bar : (!llvm<"float">) -> ()}
    : (!llvm<"float">) -> ()

// Indirect call with an argument and without a result.
"llvm.call"(%1, %0) : ((!llvm<"float">) -> (), !llvm<"float">) -> ()
```

#### Miscellaneous operations.

Integer comparisons: `icmp <lhs>, <rhs> {predicate: <int>}`. The following
predicate values are supported:

-   `0` - equality comparison;
-   `1` - inequality comparison;
-   `2` - signed less-than comparison
-   `3` - signed less-than-or-equal comparison
-   `4` - signed greater-than comparison
-   `5` - signed greater-than-or-equal comparison
-   `6` - unsigned less-than comparison
-   `7` - unsigned less-than-or-equal comparison
-   `8` - unsigned greater-than comparison
-   `9` - unsigned greater-than-or-equal comparison

Note: these constant values correspond to those used by MLIR's `cmpi` operation.

Bitwise reinterpretation: `bitcast <value>`.

Selection: `select <condition>, <lhs>, <rhs>`.

### Pseudo-operations {#pseudo-operations}

These operations do not have LLVM IR counterparts but are necessary to map LLVM
IR into MLIR.

#### `llvm.constant` {#constant-operation}

Unlike LLVM IR, MLIR does not have first-class constant values. Therefore, all
constants must be created as SSA values before being used in other operations.
`llvm.constant` creates such values for scalars and vectors. It has a mandatory
`value` attribute, which may be an integer, floating point attribute; splat,
dense or sparse attribute containing integers or floats. The type of the
attribute is one the corresponding MLIR standard types. The operation produces a
new SSA value of the specified LLVM IR dialect type.

Examples:

```mlir {.mlir}
// Integer constant
%0 = "llvm.constant"() {value: 42 : i32} -> !llvm<"i32">

// Floating point constant
%1 = "llvm.constant"() {value: 42.0 : f32} -> !llvm<"float">

// Splat vector constant
%2 = "llvm.constant"() {value: splat<vector<4xf32>, 1.0>}
      -> !llvm<"<4 x float>">
```

#### `llvm.undef` {#undef-operation}

Unlike LLVM IR, MLIR does not have first-class undefined values. Such values
must be created as SSA values using `llvm.undef`. This operation has no operands
or attributes. It creates an undefined value of the specified LLVM IR dialect
type wrapping an LLVM IR structure type.

Example:

```mlir {.mlir}
// Create a structure with a 32-bit integer followed by a float.
%0 = "llvm.undef"() -> !llvm<"{i32, float}">
```
