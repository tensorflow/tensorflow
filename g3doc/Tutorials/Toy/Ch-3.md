# Chapter 3: Defining and Registering a Dialect in MLIR

In the previous chapter, we saw how to emit a custom IR for Toy in MLIR using
opaque operations. In this chapter we will register our Dialect with MLIR to
start making the Toy IR more robust and friendly to use.

Dialects in MLIR allow for registering operations and types with an MLIRContext.
They also must reserve a "namespace" to avoid collision with other registered
dialects. These registered operations are no longer opaque to MLIR: for example
we can teach the MLIR verifier to enforce some invariants on the IR.

```c++
/// This is the definition of the Toy dialect. A dialect inherits from
/// mlir::Dialect and registers custom operations and types (in its constructor).
/// It can also overridde general behavior of dialects exposed as virtual
/// methods, for example regarding verification and parsing/printing.
class ToyDialect : public mlir::Dialect {
 public:
  explicit ToyDialect(mlir::MLIRContext *ctx);

  /// Parse a type registered to this dialect. Overridding this method is
  /// required for dialects that have custom types.
  /// Technically this is only needed to be able to round-trip to textual IR.
  mlir::Type parseType(llvm::StringRef tyData,
                       mlir::Location loc) const override;

  /// Print a type registered to this dialect. Overridding this method is
  /// only required for dialects that have custom types.
  /// Technically this is only needed to be able to round-trip to textual IR.
  void printType(mlir::Type type, llvm::raw_ostream &os) const override;
};
```

The dialect can now be registered in the global registry:

```c++
  mlir::registerDialect<ToyDialect>();
```

Any new `MLIRContext` created from now on will recognize the `toy` prefix when
parsing new types and invoke our `parseType` method. We will see later how to
enable custom operations, but first let's define a custom type to handle Toy
arrays.

# Custom Type Handling

As you may have noticed in the previous chapter, dialect specific types in MLIR
are serialized as strings. In the case of Toy, an example would be
`!toy<"array<2, 3>">`. MLIR will find the ToyDialect from the `!toy` prefix but
it is up to the dialect itself to translate the content of the string into a
proper type.

First we need to define the class representing our type. In MLIR, types are
references to immutable and uniqued objects owned by the MLIRContext. As such,
our `ToyArrayType` will only be a wrapper around a pointer to an uniqued
instance of `ToyArrayTypeStorage` in the Context and provide the public facade
API to interact with the type.

```c++
class ToyArrayType : public mlir::Type::TypeBase<ToyArrayType, mlir::Type,
                                                 detail::ToyArrayTypeStorage> {
 public:
  /// Returns the dimensions for this Toy array, or an empty range for a generic array.
  llvm::ArrayRef<int64_t> getShape();

  /// Predicate to test if this array is generic (shape haven't been inferred yet).
  bool isGeneric() { return getShape().empty(); }

  /// Return the rank of this array (0 if it is generic)
  int getRank() { return getShape().size(); }

  /// Get the unique instance of this Type from the context.
  /// A ToyArrayType is only defined by the shape of the array.
  static ToyArrayType get(mlir::MLIRContext *context,
                          llvm::ArrayRef<int64_t> shape = {});

  /// Support method to enable LLVM-style RTTI type casting.
  static bool kindof(unsigned kind) { return kind == ToyTypeKind::TOY_ARRAY; }
};
```

Implementing `getShape()` for example is just about retrieving the pointer to
the uniqued instance and forwarding:

```c++
llvm::ArrayRef<int64_t> ToyArrayType::getShape() {
  return getImpl()->getShape();
}
```

The calls to `getImpl()` give access to the `ToyArrayTypeStorage` that holds the
information for this type. For details about how the storage of the type works,
we'll refer you to `Ch3/mlir/ToyDialect.cpp`.

Finally, the Toy dialect can register the type with MLIR, and implement some
custom parsing for our types:

```c++
ToyDialect::ToyDialect(mlir::MLIRContext *ctx)
    // note the `toy` prefix that we reserve here.
    : mlir::Dialect("toy", ctx) {
  // Register our custom type with MLIR.
  addTypes<ToyArrayType>();
}

/// Parse a type registered to this dialect, we expect only Toy arrays.
mlir::Type ToyDialect::parseType(StringRef tyData,
                                 mlir::Location loc) const {
  // Sanity check: we only support array or array<...>
  if (!tyData.startswith("array")) {
    getContext()->emitError(loc, "Invalid Toy type '" + tyData +
                                     "', array expected");
    return nullptr;
  }
  // Drop the "array" prefix from the type name, we expect either an empty
  // string or just the shape.
  tyData = tyData.drop_front(StringRef("array").size());
  // This is the generic array case without shape, early return it.
  if (tyData.empty())
    return ToyArrayType::get(getContext());

  // Use a regex to parse the shape (for efficient we should store this regex in
  // the dialect itself).
  SmallVector<StringRef, 4> matches;
  auto shapeRegex = llvm::Regex("^<([0-9]+)(, ([0-9]+))*>$");
  if (!shapeRegex.match(tyData, &matches)) {
    getContext()->emitError(loc, "Invalid toy array shape '" + tyData + "'");
    return nullptr;
  }
  SmallVector<int64_t, 4> shape;
  // Iterate through the captures, skip the first one which is the full string.
  for (auto dimStr :
       llvm::make_range(std::next(matches.begin()), matches.end())) {
    if (dimStr.startswith(","))
      continue; // POSIX misses non-capturing groups.
    if (dimStr.empty())
      continue; // '*' makes it an optional group capture
    // Convert the capture to an integer
    unsigned long long dim;
    if (getAsUnsignedInteger(dimStr, /* Radix = */ 10, dim)) {
      getContext()->emitError(loc, Twine("Couldn't parse dimension as integer, matched: ") + dimStr);
      return mlir::Type();
    }
    shape.push_back(dim);
  }
  // Finally we collected all the dimensions in the shape,
  // create the array type.
  return ToyArrayType::get(getContext(), shape);
}
```

And we also update our IR generation from the Toy AST to use our new type
instead of an opaque one:

```c++
template <typename T> mlir::Type getType(T shape) {
  SmallVector<int64_t, 8> shape64(shape.begin(), shape.end());
  return ToyArrayType::get(&context, shape64);
}
```

From now on, MLIR knows how to parse types that are wrapped in `!toy<...>` and
these won't be opaque anymore. The first consequence is that bogus IR with
respect to our type won't be loaded anymore:

```bash(.sh)
$ echo 'func @foo() -> !toy<"bla">' | toyc -emit=mlir -x mlir -
loc("<stdin>":1:21): error: Invalid Toy type 'bla', array expected
$ echo 'func @foo() -> !toy<"array<>">' | toyc -emit=mlir -x mlir -
loc("<stdin>":1:21): error: Invalid toy array shape '<>'
$ echo 'func @foo() -> !toy<"array<1, >">' | toyc -emit=mlir -x mlir -
loc("<stdin>":1:21): error: Invalid toy array shape '<1, >'
$ echo 'func @foo() -> !toy<"array<1, 2, 3>">' | toyc -emit=mlir -x mlir -
func @foo() -> !toy<"array<1, 3>">
```

## Defining a C++ Class for an Operation

After defining our custom type, we will register all the operations for the Toy
language. Let's walk through the creation of the `toy.generic_call` operation:

```MLIR(.mlir)
 %4 = "toy.generic_call"(%1, %3) {callee: "my_func"}
         : (!toy<"array<2, 3>">, !toy<"array<2, 3>">) -> !toy<"array">
```

This operation takes a variable number of operands, all of which are expected to
be Toy arrays, and return a single result. An operation inherit from `mlir::Op`
and add some optional *traits* to customize its behavior.

```c++
class GenericCallOp
    : public mlir::Op<GenericCallOp, mlir::OpTrait::VariadicOperands,
                      mlir::OpTrait::OneResult> {

 public:
  /// MLIR will use this to register the operation with the parser/printer.
  static llvm::StringRef getOperationName() { return "toy.generic_call"; }

  /// Operations can add custom verification beyond the traits they define.
  /// We will ensure that all the operands are Toy arrays.
  bool verify();

  /// Interface to the builder to allow:
  ///   mlir::FuncBuilder::create<GenericCallOp>(...)
  /// This method populate the `state` that MLIR use to create operations.
  /// The `toy.generic_call` operation accepts a callee name and a list of
  /// arguments for the call.
  static void build(mlir::FuncBuilder *builder, mlir::OperationState *state,
                    llvm::StringRef callee,
                    llvm::ArrayRef<mlir::Value *> arguments);

  /// Return the name of the callee by fetching it from the attribute.
  llvm::StringRef getCalleeName();

 private:
  friend class mlir::Operation;
  using Op::Op;
};
```

and we register this operation in the `ToyDialect` constructor:

```c++
ToyDialect::ToyDialect(mlir::MLIRContext *ctx) : mlir::Dialect("toy", ctx) {
  addOperations<GenericCallOp>();
  addTypes<ToyArrayType>();
}
```

After creating classes for each of our operations, our dialect is ready and we
have now better invariants enforced in our IR, and nicer API to implement
analyses and transformations in the [next chapter](Ch-4.md).

## Using TableGen

FIXME: complete

## Revisiting the Builder API

We can now update `MLIRGen.cpp`, previously our use of the builder was very
generic and creating a call operation looked like:

```
    // Calls to user-defined function are mapped to a custom call that takes
    // the callee name as an attribute.
    mlir::OperationState result(&context, location, "toy.generic_call");
    result.types.push_back(getType(VarType{}));
    result.operands = std::move(operands);
    for (auto &expr : call.getArgs()) {
      auto *arg = mlirGen(*expr);
      if (!arg)
        return nullptr;
      result.operands.push_back(arg);
    }
    auto calleeAttr = builder->getStringAttr(call.getCallee());
    result.attributes.push_back(builder->getNamedAttr("callee", calleeAttr));
    return builder->createOperation(result)->getResult(0);
```

We replace it with this new version:

```c++
    for (auto &expr : call.getArgs()) {
      auto *arg = mlirGen(*expr);
      if (!arg)
        return nullptr;
      operands.push_back(arg);
    }
    return builder->create<GenericCallOp>(location, call.getCallee(), operands)->getResult();
```

This interface offers better type safety, with some invariant enforced at the
API level. For instance the `GenericCallOp` exposes now a `getResult()` method
that does not take any argument, while before MLIR assumed the general cases and
left open the possibility to have multiple returned values. The API was
`getResult(int resultNum)`.

# Putting It All Together

After writing a class for each of our operation and implementing custom
verifier, we try again the same example of invalid IR from the previous chapter:

```bash(.sh)
$ cat test/invalid.mlir
func @main() {
  %0 = "toy.print"()  : () -> !toy<"array<2, 3>">
}
$ toyc test/invalid.mlir -emit=mlir
loc("test/invalid.mlir":2:8): error: 'toy.print' op requires a single operand
```

This time the IR is correctly rejected by the verifier!

In the [next chapter](Ch-4.md) we will leverage our new dialect to implement
some high-level language-specific analyses and transformations for the Toy
language.
