# Chapter 2: Emitting Basic MLIR

[TOC]

Now that we're familiar with our language and the AST, let's see how MLIR can
help to compile Toy.

## Introduction: Multi-Level Intermediate Representation

Other compilers, like LLVM (see the
[Kaleidoscope tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html)),
offer a fixed set of predefined types and (usually *low-level* / RISC-like)
instructions. It is up to the frontend for a given language to perform any
language-specific type-checking, analysis, or transformation before emitting
LLVM IR. For example, Clang will use its AST to perform not only static analysis
but also transformations, such as C++ template instantiation through AST cloning
and rewrite. Finally, languages with construction at a higher-level than C/C++
may require non-trivial lowering from their AST to generate LLVM IR.

As a consequence, multiple frontends end up reimplementing significant pieces of
infrastructure to support the need for these analyses and transformation. MLIR
addresses this issue by being designed for extensibility. As such, there are few
pre-defined instructions (*operations* in MLIR terminology) or types.

## Interfacing with MLIR

[Language reference](../../LangRef.md)

MLIR is designed to be a completely extensible infrastructure; there is no
closed set of attributes (think: constant metadata), operations, or types. MLIR
supports this extensibility with the concept of
[Dialects](../../LangRef.md#dialects). Dialects provide a grouping mechanism for
abstraction under a unique `namespace`.

In MLIR, [`Operations`](../../LangRef.md#operations) are the core unit of
abstraction and computation, similar in many ways to LLVM instructions.
Operations can have application-specific semantics and can be used to represent
all of the core IR structures in LLVM: instructions, globals (like functions),
modules, etc.

Here is the MLIR assembly for the Toy `transpose` operations:

```mlir
%t_tensor = "toy.transpose"(%tensor) {inplace = true} : (tensor<2x3xf64>) -> tensor<3x2xf64> loc("example/file/path":12:1)
```

Let's break down the anatomy of this MLIR operation:

-   `%t_tensor`

    *   The name given to the result defined by this operation (which includes
        [a prefixed sigil to avoid collisions](../../LangRef.md#identifiers-and-keywords)).
        An operation may define zero or more results (in the context of Toy, we
        will limit ourselves to single-result operations), which are SSA values.
        The name is used during parsing but is not persistent (e.g., it is not
        tracked in the in-memory representation of the SSA value).

-   `"toy.transpose"`

    *   The name of the operation. It is expected to be a unique string, with
        the namespace of the dialect prefixed before the "`.`". This can be read
        as the `transpose` operation in the `toy` dialect.

-   `(%tensor)`

    *   A list of zero or more input operands (or arguments), which are SSA
        values defined by other operations or referring to block arguments.

-   `{ inplace = true }`

    *   A dictionary of zero or more attributes, which are special operands that
        are always constant. Here we define a boolean attribute named 'inplace'
        that has a constant value of true.

-   `(tensor<2x3xf64) -> tensor<3x2xf64>`

    *   This refers to the type of the operation in a functional form, spelling
        the types of the arguments in parentheses and the type of the return
        values afterward.

-   `loc("example/file/path":12:1)`

    *   This is the location in the source code from which this operation
        originated.

Shown here is the general form of an operation. As described above, the set of
operations in MLIR is extensible. This means that the infrastructure must be
able to opaquely reason about the structure of an operation. This is done by
boiling down the composition of an operation into discrete pieces:

-   A name for the operation.
-   A list of SSA operand values.
-   A list of [attributes](../../LangRef.md#attributes).
-   A list of [types](../../LangRef.md#type-system) for result values.
-   A [source location](../../Diagnostics.md#source-locations) for debugging
    purposes.
-   A list of successors [blocks](../../LangRef.md#blocks) (for branches,
    mostly).
-   A list of [regions](../../LangRef.md#regions) (for structural operations
    like functions).

In MLIR, every operation has a mandatory source location associated with it.
Contrary to LLVM, where debug info locations are metadata and can be dropped, in
MLIR, the location is a core requirement, and APIs depend on and manipulate it.
Dropping a location is thus an explicit choice which cannot happen by mistake.

To provide an illustration: If a transformation replaces an operation by
another, that new operation must still have a location attached. This makes it
possible to track where that operation came from.

It's worth noting that the mlir-opt tool - a tool for testing
compiler passes - does not include locations in the output by default. The
`-mlir-print-debuginfo` flag specifies to include locations. (Run `mlir-opt
--help` for more options.)

### Opaque API

MLIR is designed to be a completely extensible system, and as such, the
infrastructure has the capability to opaquely represent all of its core
components: attributes, operations, types, etc. This allows MLIR to parse,
represent, and [round-trip](../../Glossary.md#round-trip) any valid IR. For
example, we could place our Toy operation from above into an `.mlir` file and
round-trip through *mlir-opt* without registering anything:

```mlir
func @toy_func(%tensor: tensor<2x3xf64>) -> tensor<3x2xf64> {
  %t_tensor = "toy.transpose"(%tensor) { inplace = true } : (tensor<2x3xf64>) -> tensor<3x2xf64>
  return %t_tensor : tensor<3x2xf64>
}
```

In the cases of unregistered attributes, operations, and types, MLIR will
enforce some structural constraints (SSA, block termination, etc.), but
otherwise they are completely opaque. This can be useful for bootstrapping
purposes, but it is generally advised against. Opaque operations must be treated
conservatively by transformations and analyses, and they are much harder to
construct and manipulate.

This handling can be observed by crafting what should be an invalid IR for Toy
and seeing it round-trip without tripping the verifier:

```mlir
// RUN: toyc %s -emit=mlir

func @main() {
  %0 = "toy.print"() : () -> tensor<2x3xf64>
}
```

There are multiple problems here: the `toy.print` operation is not a terminator;
it should take an operand; and it shouldn't return any values. In the next
section, we will register our dialect and operations with MLIR, plug into the
verifier, and add nicer APIs to manipulate our operations.

## Defining a Toy Dialect

To effectively interface with MLIR, we will define a new Toy dialect. This
dialect will properly model the semantics of the Toy language, as well as
provide an easy avenue for high-level analysis and transformation.

```c++
/// This is the definition of the Toy dialect. A dialect inherits from
/// mlir::Dialect and registers custom attributes, operations, and types (in its
/// constructor). It can also override some general behavior exposed via virtual
/// methods, which will be demonstrated in later chapters of the tutorial.
class ToyDialect : public mlir::Dialect {
 public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// Provide a utility accessor to the dialect namespace. This is used by
  /// several utilities.
  static llvm::StringRef getDialectNamespace() { return "toy"; }
};
```

The dialect can now be registered in the global registry:

```c++
  mlir::registerDialect<ToyDialect>();
```

Any new `MLIRContext` created from now on will contain an instance of the Toy
dialect and invoke specific hooks for things like parsing attributes and types.

## Defining Toy Operations

Now that we have a `Toy` dialect, we can start registering operations. This will
allow for providing semantic information that the rest of the system can hook
into. Let's walk through the creation of the `toy.constant` operation:

```mlir
 %4 = "toy.constant"() {value = dense<1.0> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
```

This operation takes zero operands, a
[dense elements](../../LangRef.md#dense-elements-attribute) attribute named
`value`, and returns a single result of
[TensorType](../../LangRef.md#tensor-type). An operation inherits from the
[CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)
`mlir::Op` class which also takes some optional [*traits*](../../Traits.md) to
customize its behavior. These traits may provide additional accessors,
verification, etc.

```c++
class ConstantOp : public mlir::Op<ConstantOp,
                     /// The ConstantOp takes zero inputs.
                     mlir::OpTrait::ZeroOperands,
                     /// The ConstantOp returns a single result.
                     mlir::OpTrait::OneResult,
                     /// The ConstantOp is pure and has no visible side-effects.
                     mlir::OpTrait::HasNoSideEffect> {

 public:
  /// Inherit the constructors from the base Op class.
  using Op::Op;

  /// Provide the unique name for this operation. MLIR will use this to register
  /// the operation and uniquely identify it throughout the system.
  static llvm::StringRef getOperationName() { return "toy.constant"; }

  /// Return the value of the constant by fetching it from the attribute.
  mlir::DenseElementsAttr getValue();

  /// Operations can provide additional verification beyond the traits they
  /// define. Here we will ensure that the specific invariants of the constant
  /// operation are upheld, for example the result type must be of TensorType.
  LogicalResult verify();

  /// Provide an interface to build this operation from a set of input values.
  /// This interface is used by the builder to allow for easily generating
  /// instances of this operation:
  ///   mlir::OpBuilder::create<ConstantOp>(...)
  /// This method populates the given `state` that MLIR uses to create
  /// operations. This state is a collection of all of the discrete elements
  /// that an operation may contain.
  /// Build a constant with the given return type and `value` attribute.
  static void build(mlir::Builder *builder, mlir::OperationState &state,
                    mlir::Type result, mlir::DenseElementsAttr value);
  /// Build a constant and reuse the type from the given 'value'.
  static void build(mlir::Builder *builder, mlir::OperationState &state,
                    mlir::DenseElementsAttr value);
  /// Build a constant by broadcasting the given 'value'.
  static void build(mlir::Builder *builder, mlir::OperationState &state,
                    double value);
};
```

and we register this operation in the `ToyDialect` constructor:

```c++
ToyDialect::ToyDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx) {
  addOperations<ConstantOp>();
}
```

### Op vs Operation: Using MLIR Operations

Now that we have defined an operation, we will want to access and transform it.
In MLIR, there are two main classes related to operations: `Operation` and `Op`.
Operation is the actual opaque instance of the operation, and represents the
general API into an operation instance. An `Op` is the base class of a derived
operation, like `ConstantOp`, and acts as smart pointer wrapper around a
`Operation*`. This means that when we define our Toy operations, we are actually
providing a clean interface for building and interfacing with the `Operation`
class; this is why our `ConstantOp` defines no class fields. Therefore, we
always pass these classes around by value, instead of by reference or pointer
(*passing by value* is a common idiom and applies similarly to attributes,
types, etc). We can always get an instance of our toy operation by using LLVM's
casting infrastructure:

```c++
void processConstantOp(mlir::Operation *op) {
  ConstantOp op = llvm::dyn_cast<ConstantOp>(op);

  // This operation is not an instance of `ConstantOp`.
  if (!op)
    return;

  // Get the internal operation instance back.
  mlir::Operation *internalOp = op.getOperation();
  assert(internalOp == op && "these operation instances are the same");
}
```

### Using the Operation Definition Specification (ODS) Framework

In addition to specializing the `mlir::Op` C++ template, MLIR also supports
defining operations in a declarative manner. This is achieved via the
[Operation Definition Specification](../../OpDefinitions.md) framework. Facts
regarding an operation are specified concisely into a TableGen record, which
will be expanded into an equivalent `mlir::Op` C++ template specialization at
compile time. Using the ODS framework is the desired way for defining operations
in MLIR given the simplicity, conciseness, and general stability in the face of
C++ API changes.

Lets see how to define the ODS equivalent of our ConstantOp:

The first thing to do is to define a link to the Toy dialect that we defined in
C++. This is used to link all of the operations that we will define to our
dialect:

```tablegen
// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Toy_Dialect : Dialect {
  // The namespace of our dialect, this corresponds 1-1 with the string we
  // provided in `ToyDialect::getDialectNamespace`.
  let name = "toy";

  // The C++ namespace that the dialect class definition resides in.
  let cppNamespace = "toy";
}
```

Now that we have defined a link to the Toy dialect, we can start defining
operations. Operations in ODS are defined by inheriting from the `Op` class. To
simplify our operation definitions, we will define a base class for operations
in the Toy dialect.

```tablegen
// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Toy_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```

With all of the preliminary pieces defined, we can begin to define the constant
operation.

We define a toy operation by inheriting from our base 'Toy_Op' class above. Here
we provide the mnemonic and a list of traits for the operation. The
[mnemonic](../../OpDefinitions.md#operation-name) here matches the one given in
`ConstantOp::getOperationName` without the dialect prefix; `toy.`. The constant
operation here is also marked as 'NoSideEffect'. This is an ODS trait, and
matches one-to-one with the trait we providing when defining `ConstantOp`:
`mlir::OpTrait::HasNoSideEffect`. Missing here from our C++ definition are the
`ZeroOperands` and `OneResult` traits; these will be automatically inferred
based upon the `arguments` and `results` fields we define later.

```tablegen
def ConstantOp : Toy_Op<"constant", [NoSideEffect]> {
}
```

At this point you probably might want to know what the C++ code generated by
TableGen looks like. Simply run the `mlir-tblgen` command with the
`gen-op-decls` or the `gen-op-defs` action like so:

```
${build_root}/bin/mlir-tblgen -gen-op-defs ${mlir_src_root}/examples/toy/Ch2/include/toy/Ops.td -I ${mlir_src_root}/include/
```

Depending on the selected action, this will print either the `ConstantOp` class
declaration or its implementation. Comparing this output to the hand-crafted
implementation is incredibly useful when getting started with TableGen.

#### Defining Arguments and Results

With the shell of the operation defined, we can now provide the
[inputs](../../OpDefinitions.md#operation-arguments) and
[outputs](../../OpDefinitions.md#operation-results) to our operation. The
inputs, or arguments, to an operation may be attributes or types for SSA operand
values. The results correspond to a set of types for the values produced by the
operation:

```tablegen
def ConstantOp : Toy_Op<"constant", [NoSideEffect]> {
  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);
}
```

By providing a name to the arguments or results, e.g. `$value`, ODS will
automatically generate a matching accessor: `DenseElementsAttr
ConstantOp::value()`.

#### Adding Documentation

The next step after defining the operation is to document it. Operations may
provide
[`summary` and `description`](../../OpDefinitions.md#operation-documentation)
fields to describe the semantics of the operation. This information is useful
for users of the dialect and can even be used to auto-generate Markdown
documents.

```tablegen
def ConstantOp : Toy_Op<"constant", [NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documenatation of the operations within our dialect.
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The generic call operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);
}
```

#### Verifying Operation Semantics

At this point we've already covered a majority of the original C++ operation
definition. The next piece to define is the verifier. Luckily, much like the
named accessor, the ODS framework will automatically generate a lot of the
necessary verification logic based upon the constraints we have given. This
means that we don't need to verify the structure of the return type, or even the
input attribute `value`. In many cases, additional verification is not even
necessary for ODS operations. To add additional verification logic, an operation
can override the [`verifier`](../../OpDefinitions.md#custom-verifier-code)
field. The `verifier` field allows for defining a C++ code blob that will be run
as part of `ConstantOp::verify`. This blob can assume that all of the other
invariants of the operation have already been verified:

```tablegen
def ConstantOp : Toy_Op<"constant", [NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documenatation of the operations within our dialect.
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The generic call operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);

  // Add additional verification logic to the constant operation. Here we invoke
  // a static `verify` method in a C++ source file. This codeblock is executed
  // inside of ConstantOp::verify, so we can use `this` to refer to the current
  // operation instance.
  let verifier = [{ return ::verify(*this); }];
}
```

#### Attaching `build` Methods

The final missing component here from our original C++ example are the `build`
methods. ODS can generate some simple build methods automatically, and in this
case it will generate our first build method for us. For the rest, we define the
[`builders`](../../OpDefinitions.md#custom-builder-methods) field. This field
takes a list of `OpBuilder` objects that take a string corresponding to a list
of C++ parameters, as well as an optional code block that can be used to specify
the implementation inline.

```tablegen
def ConstantOp : Toy_Op<"constant", [NoSideEffect]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documenatation of the operations within our dialect.
  let summary = "constant operation";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

      %0 = "toy.constant"()
         { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
        : () -> tensor<2x3xf64>
  }];

  // The constant operation takes an attribute as the only input.
  // `F64ElementsAttr` corresponds to a 64-bit floating-point ElementsAttr.
  let arguments = (ins F64ElementsAttr:$value);

  // The generic call operation returns a single value of TensorType.
  // F64Tensor corresponds to a 64-bit floating-point TensorType.
  let results = (outs F64Tensor);

  // Add additional verification logic to the constant operation. Here we invoke
  // a static `verify` method in a c++ source file. This codeblock is executed
  // inside of ConstantOp::verify, so we can use `this` to refer to the current
  // operation instance.
  let verifier = [{ return ::verify(*this); }];

  // Add custom build methods for the constant operation. These methods populate
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<"Builder *builder, OperationState &result, "
              "DenseElementsAttr value", [{
      // Call into an autogenerated `build` method.
      build(builder, result, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value. This builder
    // creates a declaration for `ConstantOp::build` with the given parameters.
    OpBuilder<"Builder *builder, OperationState &result, double value">
  ];
}
```

Above we introduce several of the concepts for defining operations in the ODS
framework, but there are many more that we haven't had a chance to: regions,
variadic operands, etc. Check out the
[full specification](../../OpDefinitions.md) for more details.

## Complete Toy Example

At this point we can generate our "Toy IR". A simplified version of the previous
example:

```.toy
# User defined generic function that operates on unknown shaped arguments.
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
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

```mlir
module {
  func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = "toy.transpose"(%arg0) : (tensor<*xf64>) -> tensor<*xf64> loc("test/codegen.toy":5:10)
    %1 = "toy.transpose"(%arg1) : (tensor<*xf64>) -> tensor<*xf64> loc("test/codegen.toy":5:25)
    %2 = "toy.mul"(%0, %1) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64> loc("test/codegen.toy":5:25)
    "toy.return"(%2) : (tensor<*xf64>) -> () loc("test/codegen.toy":5:3)
  } loc("test/codegen.toy":4:1)
  func @main() {
    %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64> loc("test/codegen.toy":9:17)
    %1 = "toy.reshape"(%0) : (tensor<2x3xf64>) -> tensor<2x3xf64> loc("test/codegen.toy":9:3)
    %2 = "toy.constant"() {value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00]> : tensor<6xf64>} : () -> tensor<6xf64> loc("test/codegen.toy":10:17)
    %3 = "toy.reshape"(%2) : (tensor<6xf64>) -> tensor<2x3xf64> loc("test/codegen.toy":10:3)
    %4 = "toy.generic_call"(%1, %3) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/codegen.toy":11:11)
    %5 = "toy.generic_call"(%3, %1) {callee = @multiply_transpose} : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64> loc("test/codegen.toy":12:11)
    "toy.print"(%5) : (tensor<*xf64>) -> () loc("test/codegen.toy":13:3)
    "toy.return"() : () -> () loc("test/codegen.toy":8:1)
  } loc("test/codegen.toy":8:1)
} loc("test/codegen.toy":0:0)
```

You can build `toyc-ch2` and try yourself: `toyc-ch2
test/Examples/Toy/Ch2/codegen.toy -emit=mlir -mlir-print-debuginfo`. We can also
check our RoundTrip: `toyc-ch2 test/Examples/Toy/Ch2/codegen.toy -emit=mlir
-mlir-print-debuginfo 2> codegen.mlir` followed by `toyc-ch2 codegen.mlir
-emit=mlir`. You should also use `mlir-tblgen` on the final definition file and
study the generated C++ code.

At this point, MLIR knows about our Toy dialect and operations. In the
[next chapter](Ch-3.md), we will leverage our new dialect to implement some
high-level language-specific analyses and transformations for the Toy language.
