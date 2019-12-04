# Introduction to MLIR Operation Traits

[TOC]

MLIR allows for a truly open operation ecosystem, as any dialect may define
operations that suit a specific level of abstraction. `Traits` are a mechanism
in which to abstract implementation details and properties that are common
across many different operations. `Traits` may be used to specify special
properties and constraints of the operation, including whether the operation has
side effects or whether its output has the same type as the input. Some examples
of traits are `Commutative`, `SingleResult`, `Terminator`, etc. See the more
[comprehensive list](#traits) below for more examples of what is possible.

## Defining a Trait

Traits may be defined in C++ by inheriting from the
`OpTrait::TraitBase<ConcreteType, TraitType>` class. This base class takes as
template parameters:

*   ConcreteType
    -   The concrete operation type that this trait was attached to.
*   TraitType
    -   The type of the trait class that is being defined, for use with the
        [`Curiously Recurring Template Pattern`](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern).

A derived trait class is expected to take a single template that corresponds to
the `ConcreteType`. An example trait definition is shown below:

```c++
template <typename ConcreteType>
class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
};
```

Derived traits may also provide a `verifyTrait` hook, that is called when
verifying the concrete operation. The trait verifiers will currently always be
invoked before the main `Op::verify`.

```c++
template <typename ConcreteType>
class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
public:
  /// Override the 'verifyTrait' hook to add additional verification on the
  /// concrete operation.
  static LogicalResult verifyTrait(Operation *op) {
    // ...
  }
};
```

Note: It is generally good practice to define the implementation of the
`verifyTrait` hook out-of-line as a free function when possible to avoid
instantiating the implementation for every concrete operation type.

### Parametric Traits

The above demonstrates the definition of a simple self-contained trait. It is
also often useful to provide some static parameters to the trait to control its
behavior. Given that the definition of the trait class is rigid, i.e. we must
have a single template argument for the concrete operation, the templates for
the parameters will need to be split out. An example is shown below:

```c++
template <int Parameter>
class MyParametricTrait {
public:
  template <typename ConcreteType>
  class Impl : public OpTrait::TraitBase<ConcreteType, Impl> {
    // Inside of 'Impl' we have full access to the template parameters
    // specified above.
  };
};
```

## Attaching a Trait

Traits may be used when defining a derived operation type, by simply adding the
name of the trait class to the `Op` class after the concrete operation type:

```c++
/// Here we define 'MyOp' along with the 'MyTrait' and `MyParameteric trait
/// classes we defined previously.
class MyOp : public Op<MyOp, MyTrait, MyParametricTrait<10>::Impl> {};
```

To use a trait in the [ODS](OpDefinitions.md) framework, we need to provide a
definition of the trait class. This can be done using the `NativeOpTrait` and
`ParamNativeOpTrait` classes. `ParamNativeOpTrait` provides a mechanism in which
to specify arguments to a parametric trait class with an internal `Impl`.

```td
// The argument is the c++ trait class name.
def MyTrait : NativeOpTrait<"MyTrait">;

// The first argument is the parent c++ class name. The second argument is a
// string containing the parameter list.
class MyParametricTrait<int prop>
  : NativeOpTrait<"MyParametricTrait", !cast<string>(!head(parameters))>;
```

These can then be used in the `traits` list of an op definition:

```td
def OpWithInferTypeInterfaceOp : Op<...[MyTrait, MyParametricTrait<10>]> { ... }
```

See the documentation on [operation definitions](OpDefinitions.md) for more
details.

## Using a Trait

Traits may be used to provide additional methods, static fields, or other
information directly on the concrete operation. `Traits` internally become
`Base` classes of the concrete operation, so all of these are directly
accessible. To expose this information opaquely to transformations and analyses,
[`interfaces`](Interfaces.md) may be used.

To query if a specific operation contains a specific trait, the `hasTrait<>`
method may be used. This takes as a template parameter the trait class, which is
the same as the one passed when attaching the trait to an operation.

```c++
Operation *op = ..;
if (op->hasTrait<MyTrait>() || op->hasTrait<MyParametricTrait<10>::Impl>())
  ...;
```

## Trait List

MLIR provides a suite of traits that provide various functionalities that are
common across many different operations. Below is a list of some key traits that
may be used directly by any dialect. The format of the header for each trait
section goes as follows:

*   `Header`
    -   (`C++ class` -- `ODS class`(if applicable))

### Broadcastable

*   `OpTrait::BroadcastableTwoOperandsOneResult` -- `Broadcastable`

This trait provides the API for operations that are known to have
[broadcast-compatible](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
operand and result types. Specifically, starting from the most varying
dimension, each dimension pair of the two operands' types should either be the
same or one of them is one. Also, the result type should have the corresponding
dimension equal to the larger one, if known. Shapes are checked partially if
ranks or dimensions are not known. For example, an op with `tensor<?x2xf32>` and
`tensor<2xf32>` as operand types and `tensor<3x2xf32>` as the result type is
broadcast-compatible.

Ths trait assumes the op has two operands and one result, and it asserts if the
pre-condition is not satisfied.

### Commutative

*   `OpTrait::IsCommutative` -- `Commutative`

This trait adds the property that the operation is commutative, i.e. `X op Y ==
Y op X`

### Function-Like

*   `OpTrait::FunctionLike`

This trait provides APIs for operations that behave like functions. In
particular:

-   Ops must be symbols, i.e. also have the `Symbol` trait;
-   Ops have a single region with multiple blocks that corresponds to the body
    of the function;
-   the absence of a region corresponds to an external function;
-   arguments of the first block of the region are treated as function
    arguments;
-   they can have argument and result attributes that are stored in dictionary
    attributes on the operation itself.

This trait does *NOT* provide type support for the functions, meaning that
concrete Ops must handle the type of the declared or defined function.
`getTypeAttrName()` is a convenience function that returns the name of the
attribute that can be used to store the function type, but the trait makes no
assumption based on it.

### HasParent

*   `OpTrait::HasParent<typename ParentOpType>` -- `HasParent<string op>`

This trait provides APIs and verifiers for operations that can only be nested
within regions that are attached to operations of `ParentOpType`.

### IsolatedFromAbove

*   `OpTrait::IsIsolatedFromAbove` -- `IsolatedFromAbove`

This trait signals that the regions of an operations are known to be isolated
from above. This trait asserts that the regions of an operation will not
capture, or reference, SSA values defined above the region scope. This means
that the following is invalid if `foo.region_op` is defined as
`IsolatedFromAbove`:

```mlir
%result = constant 10 : i32
foo.region_op {
  foo.yield %result : i32
}
```

This trait is an important structural property of the IR, and enables operations
to have [passes](WritingAPass.md) scheduled under them.

### NoSideEffect

*   `OpTrait::HasNoSideEffect` -- `NoSideEffect`

This trait signifies that the operation is pure and has no visible side effects.

### Single Block with Implicit Terminator

*   `OpTrait::SingleBlockImplicitTerminator<typename TerminatorOpType>` :
    `SingleBlockImplicitTerminator<string op>`

This trait provides APIs and verifiers for operations with regions that have a
single block that must terminate with `TerminatorOpType`.

### Symbol

*   `OpTrait::Symbol` -- `Symbol`

This trait is used for operations that define a `Symbol`.

TODO(riverriddle) Link to the proper document detailing the design of symbols.

### SymbolTable

*   `OpTrait::SymbolTable` -- `SymbolTable`

This trait is used for operations that define a `SymbolTable`.

TODO(riverriddle) Link to the proper document detailing the design of symbols.

### Terminator

*   `OpTrait::IsTerminator` -- `Terminator`

This trait provides verification and functionality for operations that are known
to be [terminators](LangRef.md#terminator-operations).
