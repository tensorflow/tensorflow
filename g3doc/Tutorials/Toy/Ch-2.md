# Chapter 2: Emiting Basic MLIR

[TOC]

Now that we're familiar with our language and the AST, let see how MLIR can help
to compile Toy.

## Introduction: Multi-Level IR

Other compilers like LLVM (see the
[Kaleidoscope tutorial](https://llvm.org/docs/tutorial/LangImpl01.html)) offer
a fixed set of predefined types and, usually *low-level* / RISC-like,
instructions. It is up to the frontend for a given language to perform any
language specific type-checking, analysis, or transformation before emitting
LLVM IR. For example, clang will use its AST to perform static analysis but also
transformation like C++ template instantiation through AST cloning and rewrite.
Finally, languages with construction higher-level than C/C++ may require
non-trivial lowering from their AST to generate LLVM IR.

As a consequence, multiple frontends end up reimplementing significant pieces of
infrastructure to support the need for these analyses and transformation. MLIR
addresses this issue by being designed for extensibility. As such, there is
little to no pre-defined set of instructions (*operations* in MLIR
terminology) or types.

## MLIR Module, Functions, Blocks, and Operations

[Language reference](LangRef.md#operations)

In MLIR (like in LLVM), the top level structure for the IR is a Module
(equivalent to a translation unit in C/C++). A module contains a list of
functions, and each function has a list of blocks forming a CFG. Each block is a
list of operations that execute in sequence.

Operations in MLIR are similar to instructions in LLVM, however MLIR does not
have a closed set of operations. Instead, MLIR operations are fully extensible
and can have application-specific semantics.

Here is the MLIR assembly for the Toy 'transpose' operations:

```MLIR(.mlir)
%t_array = "toy.transpose"(%array) { inplace: true } : (!toy<"array<2, 3">) -> !toy<"array<3, 2">
```

Let's look at the anatomy of this MLIR operation:

-   it is identified by its name, which is expected to be a unique string (e.g.
    `toy.transpose`).
-   it takes as input zero or more operands (or arguments), which are SSA values
    defined by other operations or refering to function and block arguments
    (e.g. `%array`).
-   it produces zero or more results (we will limit ourselves to a single result
    in the context of Toy), which are SSA values (e.g. `%t_array`).
-   it has zero or more attributes, which are special operands that are always
    constant (e.g. `inplace: true`).
-   Lastly the type of the operation appears at the end in a functional form,
    spelling the types of the arguments in parentheses and the type of the
    return values afterward.

Finally, in MLIR every operation has a mandatory source location associated with
it. Contrary to LLVM where debug info locations are metadata and can be dropped,
in MLIR the location is a core requirement which translates in APIs manipulating
operations requiring it. Dropping a location becomes an explicit choice and
cannot happen by mistake.


## Opaque Builder API

Operations and types can be created with only their string names using the
raw builder API. This allows MLIR to parse, represent, and round-trip any valid
IR. For example, the following can round-trip through *mlir-opt*:

```MLIR(.mlir)
func @some_func(%arg0: !random_dialect<"custom_type">) -> !another_dialect<"other_type"> {
  %result = "custom.operation"(%arg0) : (!random_dialect<"custom_type">) -> !another_dialect<"other_type">
  return %result : !another_dialect<"other_type">
}
```

Here MLIR will enforce some structural constraints (SSA, block termination,
return operand type coherent with function return type, etc.) but otherwise the
types and the operation are completely opaque.

We will take advantage of this facility to emit MLIR for Toy by traversing the
AST. Our types will be prefixed with "!toy" and our operation name with "toy.".
MLIR refers to this prefix as a *dialect*, we will introduce this with more
details in the [next chapter](Ch-3.md).

Programmatically creating an opaque operation like the one above involves using
the `mlir::OperationState` structure which group all the basic elements needs to
build an operation with an `mlir::Builder`:

-   The name of the operation.
-   A location for debugging purpose. It is mandatory, but can be explicitly set
    to "unknown".
-   The list of operand values.
-   The types for returned values.
-   The list of attributes.
-   A list of successors (for branches mostly).

To build the `custom.operation` from the listing above, assuming you have a
`Value *` handle to `%arg0`, is as simple as:

```c++
// The return type for the operation: `!another_dialect<"other_type">`
auto another_dialect_prefix = mlir::Identifier::get("another_dialect", &context);
auto returnType = mlir::UnknownType::get(another_dialect_prefix,
                                         "custom_type", &context);
// Creation of the state defining the operation:
mlir::OperationState state(&context, location, "custom.operation");
state.types.push_back(returnType);
state.operands.push_back(arg0);
// Using a builder to create the operation and insert it where the builder
// insertion point is currently set.
auto customOperation = builder->createOperation(state);
// An operation is not an SSA value (unlike LLVM), because it can return
// multiple SSA value, the resulting value can be obtained:
Value *result = customOperation->getResult(0);
```

This approach is used in `Ch2/mlir/MLIRGen.cpp` to implement a naive MLIR
generation through a simple depth-first search traversal of the Toy AST. Here is
how we create a `toy.transpose` operation:

```
mlir::Operation *createTransposeOp(FuncBuilder *builder,
                                   mlir::Value *input_array) {
  // We bundle our custom type in a `toy` dialect.
  auto toyDialect = mlir::Identifier::get("toy", builder->getContext());
  // Create a custom type, in the MLIR assembly it is:  !toy<"array<2, 2>">
  auto type = mlir::UnknownType::get(toyDialect, "array<2, 2>", builder->getContext());

  // Fill the `OperationState` with the required fields
  mlir::OperationState result(builder->getContext(), location, "toy.transpose");
  result.types.push_back(type);  // return type
  result.operands.push_back(input_value); // argument
  Operation *newTransposeOp = builder->createOperation(result);
  return newTransposeOp;
}
```

## Complete Toy Example

FIXME: It would be nice to have an idea for the **need** of a custom **type** in
Toy? Right now `toy<array>` could be replaced directly by unranked `tensor<*>`
and `toy<array<YxZ>>` could be replaced by a `memref<YxZ>`.

At this point we can already generate our "Toy IR" without having registered
anything with MLIR. A simplified version of the previous example:

```Toy {.toy}
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return a * transpose(b);
}

def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(b, a);
  print(d);
}
```

Results in the following IR:

```MLIR(.mlir)
func @multiply_transpose(%arg0: !toy<"array">, %arg1: !toy<"array">)
  attributes  {toy.generic: true} loc("test/codegen.toy":2:1) {
  %0 = "toy.transpose"(%arg1) : (!toy<"array">) -> !toy<"array"> loc("test/codegen.toy":3:14)
  %1 = "toy.mul"(%arg0, %0) : (!toy<"array">, !toy<"array">) -> !toy<"array"> loc("test/codegen.toy":3:14)
  "toy.return"(%1) : (!toy<"array">) -> () loc("test/codegen.toy":3:3)
}

func @main() loc("test/codegen.toy":6:1) {
  %0 = "toy.constant"() {value: dense<tensor<2x3xf64>, [[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]>} : () -> !toy<"array<2, 3>"> loc("test/codegen.toy":7:17)
  %1 = "toy.reshape"(%0) : (!toy<"array<2, 3>">) -> !toy<"array<2, 3>"> loc("test/codegen.toy":7:3)
  %2 = "toy.constant"() {value: dense<tensor<6xf64>, [1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]>} : () -> !toy<"array<6>"> loc("test/codegen.toy":8:17)
  %3 = "toy.reshape"(%2) : (!toy<"array<6>">) -> !toy<"array<2, 3>"> loc("test/codegen.toy":8:3)
  %4 = "toy.generic_call"(%1, %3, %1, %3) {callee: "multiply_transpose"} : (!toy<"array<2, 3>">, !toy<"array<2, 3>">, !toy<"array<2, 3>">, !toy<"array<2, 3>">) -> !toy<"array"> loc("test/codegen.toy":9:11)
  %5 = "toy.generic_call"(%3, %1, %3, %1) {callee: "multiply_transpose"} : (!toy<"array<2, 3>">, !toy<"array<2, 3>">, !toy<"array<2, 3>">, !toy<"array<2, 3>">) -> !toy<"array"> loc("test/codegen.toy":10:11)
  "toy.print"(%5) : (!toy<"array">) -> () loc("test/codegen.toy":11:3)
  "toy.return"() : () -> () loc("test/codegen.toy":6:1)
}
```

You can build `toyc` and try yourself: `toyc test/codegen.toy -emit=mlir
-mlir-print-debuginfo`. We can also check our RoundTrip: `toyc test/codegen.toy
-emit=mlir -mlir-print-debuginfo > codegen.mlir` followed by `toyc codegen.mlir
-emit=mlir`.

Notice how these MLIR operations are prefixed with `toy.` ; by convention we use
this similarly to a "namespace" in order to avoid conflicting with other
operations with the same name. Similarly the syntax for types wraps an arbitrary
string representing our custom types within our "namespace" `!toy<...>`. Of
course at this point MLIR does not know anything about Toy, and so there is no
semantic associated with the operations and types, everything is opaque and
string-based. The only thing enforced by MLIR here is that the IR is in SSA
form: values are defined once, and uses appears after their definition.

This can be observed by crafting what should be an invalid IR for Toy and see it
round-trip without tripping the verifier:

```MLIR(.mlir)
// RUN: toyc %s -emit=mlir
func @main() {
  %0 = "toy.print"() : () -> !toy<"array<2, 3>">
}
```

There are multiple problems here: first the `toy.print` is not a terminator,
then it should take an operand, and not return any value.

In the [next chapter](Ch-2.md) we will register our dialect and operations with
MLIR, plug in the verifier, and add nicer APIs to manipulate our operations.
