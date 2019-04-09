# Linalg Dialect

This chapter describes the design and implementation of a simple linear algebra
dialect in MLIR. The objective of the `linalg` dialect is to demonstrate that
the MLIR infrastructure is a great fit for implementing high-level operations
and lower them gradually to LLVM by reusing existing components and lowering
paths. In particular, `linalg` is built upon the type system of the
[`affine`](../../Dialects/Affine.md) dialect, which allows partial lowering to
be implemented with relative ease.

The `linalg` dialect is introduced gradually following this outline:

1.  Type system and type-building operations.
2.  Compute operations.
3.  Lowerings between the `linalg` operations into `linalg` + `affine`
    operations.
4.  Tiling transformations.
5.  A simple tiling and fusion transformation.

The Toy language tutorial already introduced core MLIR concepts and best
practices, the `linalg` dialect operates mostly at the level of the C++ API and
in particular makes use of [declarative builders](DeclarativeBuilders.md), for
terser IR emitting expressions. Without loss of generality, anything in this
section can also be implemented with `mlir::Builder` and enough
`getInsertionPoint` and `setInsertionPoint` manipulations.

The implementation follows a few conventions to decouple, at each step, the
newly introduced concepts and code from ones introduced previously without
duplicating the whole code base in each directory. The code for concepts
introduced at a particular step `k` live in the `Linalgk/include/linalgk` and
`Linalgk/lib` directories and is linked into the `Linalgk` library.

Lastly, note that simplifying assumptions are made to cut down on boilerplate
and help focus on the core concepts. In particular, parsing the linalg dialect
is currently not supported as it is used as an intermediary dialect. This does
not impact the ability to lower all the way to LLVM with proper verified IR at
each step of the lowering, or to execute the compiled binary.

# Linalg Part 1: Type system

We first describe the `linalg` type system.

## RangeType and RangeOp

A
[RangeType](https://github.com/tensorflow/mlir/blob/master/examples/Linalg/Linalg1/include/linalg1/RangeType.h)
is a simple triple of `index` values. It represents a minimal range abstraction
`(min, max, step)`. `RangeType` is a fully defined type and is constructed
without any additional type argument. Its implementation illustrates the minimal
amount of information required to implement a new custom MLIR type.

```
class RangeType : public mlir::Type::TypeBase<RangeType, mlir::Type> {
public:
  // Used to implement llvm-style cast.
  using Base::Base;
  /// Construction hook.
  static RangeType get(mlir::MLIRContext *context) {
    /// Custom, uniqu'ed construction in the mlir::MLIRContext.
    return Base::get(context, LinalgTypes::Range);
  }
  /// Used to implement llvm-style cast.
  static bool kindof(unsigned kind) { return kind == LinalgTypes::Range; }
};
```

Unlike more complex types, RangeType does not require a hashing key for
unique'ing in the `MLIRContext`. Note that all MLIR types derive from
`mlir::Type::TypeBase` and expose `using Base::Base` to enable generic hooks to
work properly (in this instance for llvm-style casts. RangeType does not even
require an implementation file as the above represents the whole code for the
type.

The `linalg` dialect type `RangeType` pretty-prints simply as `!linalg.range`.

A `linalg::RangeOp`, defined
[here](https://github.com/tensorflow/mlir/blob/master/examples/Linalg/Linalg1/include/linalg1/RangeOp.h),
is the operation that produces ssa-values of `RangeType`. It pretty-prints as

```
  %0 = linalg.range %min, %max, %range : !linalg.range
```

The implementation of the `RangeOp::build` method and `RangeOp::verify`
[methods](https://github.com/tensorflow/mlir/blob/master/examples/Linalg/Linalg1/lib/RangeOp.cpp)
are straightforward.

A RangeType is used throughout to step over iteration domains (i.e. loop
iterations via loop bounds and steps) as well as over the view data abstraction.
A `LoopNestRangeBuilder` helper class is
[introduced](https://github.com/tensorflow/mlir/blob/master/examples/Linalg/Linalg1/include/linalg1/Common.h)
to allow emission of loop nests from an `llvm::ArrayRef<mlir::Value*>` where
each `mlir::Value` is a `linalg.range`.

### Simplifying assumption

The `linalg.range` type is generally unrestricted beyond havind elements of
`index` type. however it is used to build loop nests using the `affine.for`
[operation](../../Dialects/Affine.md) whose restrictions it inherits, at the
point where `affine.for` operations are materialized. This is a tradeoff to
reuse existing MLIR operations that are already known to lower to LLVM. As a
consequence, the `step` in a `linalg.range` must be a static constant and cannot
be symbolic.

## ViewType and ViewOp

A
[ViewType](https://github.com/tensorflow/mlir/blob/master/examples/Linalg/Linalg1/include/linalg1/ViewType.h)
represents a multi-dimensional range abstraction to iterate over an underlying
storage type. It is backed by a data type, in our case objects of
[MemRefType](https://github.com/tensorflow/mlir/blob/master/include/mlir/IR/StandardTypes.h).
A ViewType is a parameterized type which has a base element type and a rank. It
is thus slightly more complex than RangeType and requires unique'ing in the
enclosing MLIRContext.

This is materialized by the existence of a storage type and a `hashKey` in the
implementation
[file](https://github.com/tensorflow/mlir/blob/master/examples/Linalg/Linalg1/lib/ViewType.cpp).

```
struct ViewTypeStorage : public mlir::TypeStorage {
  /// Underlying Key type to transport the payload needed to construct a custom
  /// type in a generic way.
  struct Key {
    Key(Type elementType, unsigned rank)
        : elementType(elementType), rank(rank) {}
    Type elementType;
    unsigned rank;
  };
  ...
};
```

The `ViewTypeStorage` is not visible outside of the `ViewType` implementation
and is referred to from `ViewType` as such: `class ViewType : public
mlir::Type::TypeBase<ViewType, mlir::Type, ViewTypeStorage> { ... }`

A two dimensional ViewType over a f32 storage pretty-prints as `view<?x?xf32>`.

A `linalg::ViewOp`, defined
[here](https://github.com/tensorflow/mlir/blob/master/examples/Linalg/Linalg1/lib/ViewOp.cpp),
is the operation that produces ssa-values of `ViewType` from an ssa-value of
type `MemRefType`. A ViewOp has operands called "indexings" which can be either
of `index` or `!linalg.range` type. The rationale is that `index` reduces the
rank of a ViewType by 1 while a `!linalg.range` keeps the rank unchanged. This
behavior is a convention that we have found useful during the implementation in
order to fold chains of slice operations (introduced in the following paragraph)
and capture enough information in the ViewOp so it can be lowered to LLVM.

The entry point to the builder is the method: `static void
ViewOp::build(mlir::Builder *b, mlir::OperationState *result, mlir::Value
*memRef, llvm::ArrayRef<mlir::Value *> indexings = {});`

A `ViewOp` pretty-prints as: `%1 = linalg.view %0[%m, %n, %k] :
!linalg.view<?x?xf32>`

This signifies that `%0` is a three dimensional `MemRef` of `f32` elemental type
and that the `%1` view uses an `index` into one of the dimensions and two
`!linalg.range` for the two other dimensions.

The implementation of the `ViewOp::build` and `ViewOp::verify`
[methods](https://github.com/tensorflow/mlir/blob/master/examples/Linalg/Linalg1/lib/ViewOp.cpp)
are simple.

### Simplifying assumption

We choose to reuse the existing MLIR
`MemRef`[type](https://github.com/tensorflow/mlir/blob/master/include/mlir/IR/StandardTypes.h)
as the underlying data structure. This avoids the need to redefine a new
abstraction and simplifies lowering all the way to LLVM.

## SliceOp

A slice is a subview that is fully contained within its parent view and is
constructed using a `SliceOp`. A SliceOp takes an ssa-value of type
`linalg.view` and an "indexing" to produce a new `linalg.view` of rank:

1.  Equal to the rank of the original view, if the indexing is a
    `!linalg.range`.
2.  Equal to the rank of the original view minus one, if the indexing is an
    `index`.

A slice op has an integer attribute which specifies the dimension of the parent
view it slices and pretty-prints as:

```
%2 = linalg.slice %1[*, *, %0, *] : !linalg.view<?x?x?xf32>
```

In this particular case, %2 slices dimension `2` of the four dimensional view
%1. The returned `!linalg.view<?x?x?xf32>` indicates that the indexing is
rank-reducing and that %0 is an `index`.

The implementation of the `SliceOp::build` and `SliceOp::verify`
[methods](https://github.com/tensorflow/mlir/blob/master/examples/Linalg/Linalg1/lib/SliceOp.cpp)
are simple.

### Simplifying assumption

In this tutorial we do not enforce the strict subview property or perform bounds
check analysis and instead assume that the code is correct by construction.

## Notable remarks

The declaration for the classes implementing the operations we described have
common traits that enable certain API shortcuts and other behaviors. For
instance, the `mlir::OpTrait::OneResult` makes the `getResult()` method
available to the class.

```

class RangeOp : public mlir::Op<RangeOp, mlir::OpTrait::NOperands<3>::Impl,
                                mlir::OpTrait::OneResult,
                                mlir::OpTrait::HasNoSideEffect> { ... };

class ViewOp : public mlir::Op<ViewOp, mlir::OpTrait::VariadicOperands,
                               mlir::OpTrait::OneResult,
                               mlir::OpTrait::HasNoSideEffect> { ... } ;

class SliceOp : public mlir::Op<SliceOp, mlir::OpTrait::NOperands<2>::Impl,
                                mlir::OpTrait::OneResult,
                                mlir::OpTrait::HasNoSideEffect> { ... };
```

One particular trait of interest is `mlir::OpTrait::HasNoSideEffect` which
enables constant folding and dead code elimination in the `canonicalizerPass`.

## Dialect Registration

Similarly to Toy, the dialect must be registered so that the pretty-printer and
verifier can be enabled. Without registration, only the custom op form can be
printed. Beware of ops printed in custom op form, when a short-hand form exists,
because there is a high chance the IR verification is not enabled.

To register the Linalg dialect, call
`mlir::registerDialect<linalg::LinalgDialect>();`.

### Note on code organization

Registration occurs by constructing a new `LinalgDialect` which registers the
proper types and ops at construction time, with sanity checks guarding against
multiple registrations of the same symbols. At that point, the constructor needs
to be statically aware of all the types and ops. Since our code structure
chooses to isolate independent portions of the tutorial, and certain ops are
introduced in later parts, we explicitly separate `DialectConstruction.cpp` in
its separate library. Linking with the proper library enables the types that
have been declared so far.

## Putting it all together

The
[example](https://github.com/tensorflow/mlir/blob/master/examples/Linalg/Linalg1/Example.cpp)
demonstrates how to construct some simple IR snippets that pass through the
verifier checks. We introduce a custom op called `some_consumer` to ensure that
dead-code elimination does not optimize these simple examples out of existence.
