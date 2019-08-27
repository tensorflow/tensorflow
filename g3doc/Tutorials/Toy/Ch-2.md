# Chapter 2: Emitting Basic MLIR

[TOC]

Now that we're familiar with our language and the AST, let see how MLIR can help
to compile Toy.

## Introduction: Multi-Level IR

Other compilers like LLVM (see the
[Kaleidoscope tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html))
offer a fixed set of predefined types and, usually *low-level* / RISC-like,
instructions. It is up to the frontend for a given language to perform any
language specific type-checking, analysis, or transformation before emitting
LLVM IR. For example, clang will use its AST to perform static analysis but also
transformations like C++ template instantiation through AST cloning and rewrite.
Finally, languages with construction at a higher-level than C/C++ may require
non-trivial lowering from their AST to generate LLVM IR.

As a consequence, multiple frontends end up reimplementing significant pieces of
infrastructure to support the need for these analyses and transformation. MLIR
addresses this issue by being designed for extensibility. As such, there are
little to no pre-defined instructions (*operations* in MLIR terminology) or
types.

## MLIR Dialects and Operations

[Language reference](../../LangRef.md#dialects)

In MLIR, the core unit of abstraction and computation is an `Operation`, similar
in many ways to LLVM instructions. Operations can be used to represent all of
the core IR structures in LLVM: instructions, globals(like functions), modules,
etc; however MLIR does not have a closed set of operations. Instead, the MLIR
operation set is fully extensible and operations can have application-specific
semantics.

MLIR supports this extensibility with the concept of
[Dialects](../../LangRef.md#dialects). Among other things, Dialects provide a
grouping mechanism for operations under a unique `namespace`. Dialects will be a
discussed a bit more in the [next chapter](Ch-3.md).

Here is the MLIR assembly for the Toy 'transpose' operations:

```MLIR(.mlir)
%t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
```

Let's look at the anatomy of this MLIR operation:

-   it is identified by its name, which is expected to be a unique string (e.g.
    `toy.transpose`).
    *   the operation name is split in two parts: the dialect namespace prefix,
        and the specific op name. This can be read as the `transpose` operation
        in the `toy` dialect.
-   it takes as input zero or more operands (or arguments), which are SSA values
    defined by other operations or referring to block arguments (e.g.
    `%tensor`).
-   it produces zero or more results (we will limit ourselves to single result
    operations in the context of Toy), which are SSA values (e.g. `%t_tensor`).
-   it has zero or more attributes, which are special operands that are always
    constant (e.g. `inplace = true`).
-   lastly, the type of the operation appears at the end in a functional form,
    spelling the types of the arguments in parentheses and the type of the
    return values afterward.

Finally, in MLIR every operation has a mandatory source location associated with
it. Contrary to LLVM where debug info locations are metadata and can be dropped,
in MLIR the location is a core requirement which translates in APIs manipulating
operations requiring it. Dropping a location becomes an explicit choice and
cannot happen by mistake.

## Opaque API

MLIR is designed to be a completely extensible system, as such the
infrastructure has the capability to opaquely represent operations (as well as
attributes, types, etc.) that have not been registered. This allows MLIR to
parse, represent, and round-trip any valid IR. For example, the following can
round-trip through *mlir-opt*:

```MLIR(.mlir)
func @some_func(%arg0: !random_dialect<"custom_type">) -> !another_dialect<"other_type"> {
  %result = "custom.operation"(%arg0) { attr = #random_dialect<"custom_attribute"> } : (!random_dialect<"custom_type">) -> !another_dialect<"other_type">
  return %result : !another_dialect<"other_type">
}
```

Here MLIR will enforce some structural constraints (SSA, block termination,
etc.) but otherwise the types and the `custom.operation` are completely opaque.

We will take advantage of this facility for the initial emission of MLIR for Toy
by traversing the AST. Our operation names will be prefixed `toy.` in
preparation for a `toy` dialect, which we will introduce with more details in
the [next chapter](Ch-3.md).

Programmatically creating an opaque operation, like the one above, involves
using the `mlir::OperationState` structure which group all the basic elements
needed to build an operation with an `mlir::OpBuilder`:

-   The name of the operation.
-   A location for debugging purposes. It is mandatory, but can be explicitly
    set to `unknown`.
-   A list of operand values.
-   A list of types for result values.
-   A list of attributes.
-   A list of successors blocks (for branches mostly).
-   A list of regions (for structural operations like functions).

To build the `custom.operation` from the listing above, assuming you have a
`Value *` handle to `%arg0`, is as simple as:

```c++
// Creation of the state defining the operation:
mlir::OperationState state(location, "custom.operation");
state.addOperands(arg0);

// The return type for the operation: `!another_dialect<"other_type">`
auto anotherDialectPrefix = mlir::Identifier::get("another_dialect", &context);
auto returnType = mlir::OpaqueType::get(another_dialect_prefix,
                                        "custom_type", &context);
state.addTypes(returnType);


// Using a builder to create the operation and insert it where the builder
// insertion point is currently set.
Operation *customOperation = builder.createOperation(state);

// An operation is not an SSA value (unlike LLVM), because it can return
// multiple SSA values, the resulting value can be obtained:
Value *result = customOperation->getResult(0);
```

This approach is used in `Ch2/mlir/MLIRGen.cpp` to implement a naive MLIR
generation through a simple depth-first search traversal of the Toy AST. Here is
how we create a `toy.transpose` operation:

```c++
mlir::Operation *createTransposeOp(OpBuilder &builder,
                                   mlir::Value *input_tensor) {
  // Fill the `OperationState` with the required fields.
  mlir::OperationState result(location, "toy.transpose");
  result.addOperands(input_tensor);

  // We use the MLIR tensor type for 'toy' types.
  auto type = builder.getTensorType({2, 2}, builder.getF64Type());
  result.addTypes(type);

  // Create the transpose operation.
  Operation *newTransposeOp = builder->createOperation(result);
  return newTransposeOp;
}
```

## Complete Toy Example

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
module {
  func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>)
  attributes  {toy.generic} {
    %0 = "toy.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64> loc("test/codegen.toy":3:14)
    %1 = "toy.mul"(%arg0, %0) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64> loc("test/codegen.toy":3:14)
    "toy.return"(%1) : (tensor<*xf64>) -> () loc("test/codegen.toy":3:3)
  } loc("test/codegen.toy":2:1)
  func @main() {
    %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64> loc("test/codegen.toy":7:17)
    %1 = "toy.reshape"(%0) : (tensor<2x3xf64>) -> tensor<2x3xf64> loc("test/codegen.toy":7:3)
    %2 = "toy.constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : () -> tensor<6xf64> loc("test/codegen.toy":8:17)
    %3 = "toy.reshape"(%2) : (tensor<6xf64>) -> tensor<2x3xf64> loc("test/codegen.toy":8:3)
    %4 = "toy.generic_call"(%1, %3) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/codegen.toy":9:11)
    %5 = "toy.generic_call"(%3, %1) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/codegen.toy":10:11)
    "toy.print"(%5) : (tensor<*xf64>) -> () loc("test/codegen.toy":11:3)
    "toy.return"() : () -> () loc("test/codegen.toy":6:1)
  } loc("test/codegen.toy":6:1)
} loc("test/codegen.toy"0:0)
```

You can build `toyc-ch2` and try yourself: `toyc-ch2 test/codegen.toy -emit=mlir
-mlir-print-debuginfo`. We can also check our RoundTrip: `toyc-ch2
test/codegen.toy -emit=mlir -mlir-print-debuginfo 2> codegen.mlir` followed by
`toyc-ch2 codegen.mlir -emit=mlir`.

At this point MLIR does not know anything about Toy, so there are no semantics
associated with the operations, everything is opaque and string-based. The only
thing enforced by MLIR here is that the IR is in SSA form: values are defined
once, and uses appear after their definition.

This can be observed by crafting what should be an invalid IR for Toy and see it
round-trip without tripping the verifier:

```MLIR(.mlir)
// RUN: toyc %s -emit=mlir

func @main() {
  %0 = "toy.print"() : () -> tensor<2x3xf64>
}
```

There are multiple problems here: the `toy.print` operation is not a terminator,
it should take an operand, and it shouldn't return any values.

In the [next chapter](Ch-3.md) we will register our dialect and operations with
MLIR, plug into the verifier, and add nicer APIs to manipulate our operations.

