# Chapter 7: Adding a Composite Type to Toy

[TOC]

In the [previous chapter](Ch-6.md), we demonstrated an end-to-end compilation
flow from our Toy front-end to LLVM IR. In this chapter, we will extend the Toy
language to support a new composite `struct` type.

## Defining a `struct` in Toy

The first thing we need to define is the interface of this type in our `toy`
source language. The general syntax of a `struct` type in Toy is as follows:

```toy
# A struct is defined by using the `struct` keyword followed by a name.
struct MyStruct {
  # Inside of the struct is a list of variable declarations without initializers
  # or shapes, which may also be other previously defined structs.
  var a;
  var b;
}
```

Structs may now be used in functions as variables or parameters by using the
name of the struct instead of `var`. The members of the struct are accessed via
a `.` access operator. Values of `struct` type may be initialized with a
composite initializer, or a comma-separated list of other initializers
surrounded by `{}`. An example is shown below:

```toy
struct Struct {
  var a;
  var b;
}

# User defined generic function may operate on struct types as well.
def multiply_transpose(Struct value) {
  # We can access the elements of a struct via the '.' operator.
  return transpose(value.a) * transpose(value.b);
}

def main() {
  # We initialize struct values using a composite initializer.
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};

  # We pass these arguments to functions like we do with variables.
  var c = multiply_transpose(value);
  print(c);
}
```

## Defining a `struct` in MLIR

In MLIR, we will also need a representation for our struct types. MLIR does not
provide a type that does exactly what we need, so we will need to define our
own. We will simply define our `struct` as an unnamed container of a set of
element types. The name of the `struct` and its elements are only useful for the
AST of our `toy` compiler, so we don't need to encode it in the MLIR
representation.

### Defining the Type Class

#### Reserving a Range of Type Kinds

Types in MLIR rely on having a unique `kind` value to ensure that casting checks
remain extremely efficient
([rationale](../../Rationale.md#reserving-dialect-type-kinds)). For `toy`, this
means we need to explicitly reserve a static range of type `kind` values in the
symbol registry file
[DialectSymbolRegistry](https://github.com/tensorflow/mlir/blob/master/include/mlir/IR/DialectSymbolRegistry.def).

```c++
DEFINE_SYM_KIND_RANGE(LINALG) // Linear Algebra Dialect
DEFINE_SYM_KIND_RANGE(TOY)    // Toy language (tutorial) Dialect

// The following ranges are reserved for experimenting with MLIR dialects in a
// private context without having to register them here.
DEFINE_SYM_KIND_RANGE(PRIVATE_EXPERIMENTAL_0)
```

These definitions will provide a range in the Type::Kind enum to use when
defining the derived types.

```c++
/// Create a local enumeration with all of the types that are defined by Toy.
namespace ToyTypes {
enum Types {
  Struct = mlir::Type::FIRST_TOY_TYPE,
};
} // end namespace ToyTypes
```

#### Defining the Type Class

As mentioned in [chapter 2](Ch-2.md), [`Type`](../../LangRef.md#type-system)
objects in MLIR are value-typed and rely on having an internal storage object
that holds the actual data for the type. The `Type` class in itself acts as a
simple wrapper around an internal `TypeStorage` object that is uniqued within an
instance of an `MLIRContext`. When constructing a `Type`, we are internally just
constructing and uniquing an instance of a storage class.

When defining a new `Type` that requires additional information beyond just the
`kind` (e.g. the `struct` type, which requires additional information to hold
the element types), we will need to provide a derived storage class. The
`primitive` types that don't have any additional data (e.g. the
[`index` type](../../LangRef.md#index-type)) don't require a storage class.

##### Defining the Storage Class

Type storage objects contain all of the data necessary to construct and unique a
type instance. Derived storage classes must inherit from the base
`mlir::TypeStorage` and provide a set of aliases and hooks that will be used by
the `MLIRContext` for uniquing. Below is the definition of the storage instance
for our `struct` type, with each of the necessary requirements detailed inline:

```c++
/// This class represents the internal storage of the Toy `StructType`.
struct StructTypeStorage : public mlir::TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage. For our struct type, we will unique each instance structurally on
  /// the elements that it contains.
  using KeyTy = llvm::ArrayRef<mlir::Type>;

  /// A constructor for the type storage instance.
  StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
      : elementTypes(elementTypes) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == elementTypes; }

  /// Define a hash function for the key type. This is used when uniquing
  /// instances of the storage.
  /// Note: This method isn't necessary as both llvm::ArrayRef and mlir::Type
  /// have hash functions available, so we could just omit this entirely.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_value(key);
  }

  /// Define a construction function for the key type from a set of parameters.
  /// These parameters will be provided when constructing the storage instance
  /// itself, see the `StructType::get` method further below.
  /// Note: This method isn't necessary because KeyTy can be directly
  /// constructed with the given parameters.
  static KeyTy getKey(llvm::ArrayRef<mlir::Type> elementTypes) {
    return KeyTy(elementTypes);
  }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`. The given allocator must be used for *all* necessary dynamic
  /// allocations used to create the type storage and its internal.
  static StructTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                      const KeyTy &key) {
    // Copy the elements from the provided `KeyTy` into the allocator.
    llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);

    // Allocate the storage instance and construct it.
    return new (allocator.allocate<StructTypeStorage>())
        StructTypeStorage(elementTypes);
  }

  /// The following field contains the element types of the struct.
  llvm::ArrayRef<mlir::Type> elementTypes;
};
```

##### Defining the Type Class

With the storage class defined, we can add the definition for the user-visible
`StructType` class. This is the class that we will actually interface with.

```c++
/// This class defines the Toy struct type. It represents a collection of
/// element types. All derived types in MLIR must inherit from the CRTP class
/// 'Type::TypeBase'. It takes as template parameters the concrete type
/// (StructType), the base class to use (Type), and the storage class
/// (StructTypeStorage).
class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
                                               StructTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static bool kindof(unsigned kind) { return kind == ToyTypes::Struct; }

  /// Create an instance of a `StructType` with the given element types. There
  /// *must* be at least one element type.
  static StructType get(llvm::ArrayRef<mlir::Type> elementTypes) {
    assert(!elementTypes.empty() && "expected at least 1 element type");

    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. The first two parameters are the context to unique in and
    // the kind of the type. The parameters after the type kind are forwarded to
    // the storage instance.
    mlir::MLIRContext *ctx = elementTypes.front().getContext();
    return Base::get(ctx, ToyTypes::Struct, elementTypes);
  }

  /// Returns the element types of this struct type.
  llvm::ArrayRef<mlir::Type> getElementTypes() {
    // 'getImpl' returns a pointer to the internal storage instance.
    return getImpl()->elementTypes;
  }

  /// Returns the number of element type held by this struct.
  size_t getNumElementTypes() { return getElementTypes().size(); }
};
```

We register this type in the `ToyDialect` constructor in a similar way to how we
did with operations:

```c++
ToyDialect::ToyDialect(mlir::MLIRContext *ctx)
    : mlir::Dialect(getDialectNamespace(), ctx) {
  addTypes<StructType>();
}
```

With this we can now use our `StructType` when generating MLIR from Toy. See
examples/toy/Ch7/mlir/MLIRGen.cpp for more details.

### Parsing and Printing

At this point we can use our `StructType` during MLIR generation and
transformation, but we can't output or parse `.mlir`. For this we need to add
support for parsing and printing instances of the `StructType`. This can be done
by overriding the `parseType` and `printType` methods on the `ToyDialect`.

```c++
class ToyDialect : public mlir::Dialect {
public:
  /// Parse an instance of a type registered to the toy dialect.
  mlir::Type parseType(mlir::DialectAsmParser &parser) const override;

  /// Print an instance of a type registered to the toy dialect.
  void printType(mlir::Type type,
                 mlir::DialectAsmPrinter &printer) const override;
};
```

These methods take an instance of a high-level parser or printer that allows for
easily implementing the necessary functionality. Before going into the
implementation, let's think about the syntax that we want for the `struct` type
in the printed IR. As described in the
[MLIR language reference](../../LangRef.md#dialect-types), dialect types are
generally represented as: `! dialect-namespace < type-data >`, with a pretty
form available under certain circumstances. The responsibility of our `Toy`
parser and printer is to provide the `type-data` bits. We will define our
`StructType` as having the following form:

```
  struct-type ::= `struct` `<` type (`,` type)* `>`
```

#### Parsing

An implementation of the parser is shown below:

```c++
/// Parse an instance of a type registered to the toy dialect.
mlir::Type ToyDialect::parseType(mlir::DialectAsmParser &parser) const {
  // Parse a struct type in the following form:
  //   struct-type ::= `struct` `<` type (`,` type)* `>`

  // NOTE: All MLIR parser function return a ParseResult. This is a
  // specialization of LogicalResult that auto-converts to a `true` boolean
  // value on failure to allow for chaining, but may be used with explicit
  // `mlir::failed/mlir::succeeded` as desired.

  // Parse: `struct` `<`
  if (parser.parseKeyword("struct") || parser.parseLess())
    return Type();

  // Parse the element types of the struct.
  SmallVector<mlir::Type, 1> elementTypes;
  do {
    // Parse the current element type.
    llvm::SMLoc typeLoc = parser.getCurrentLocation();
    mlir::Type elementType;
    if (parser.parseType(elementType))
      return nullptr;

    // Check that the type is either a TensorType or another StructType.
    if (!elementType.isa<mlir::TensorType>() &&
        !elementType.isa<StructType>()) {
      parser.emitError(typeLoc, "element type for a struct must either "
                                "be a TensorType or a StructType, got: ")
          << elementType;
      return Type();
    }
    elementTypes.push_back(elementType);

    // Parse the optional: `,`
  } while (succeeded(parser.parseOptionalComma()));

  // Parse: `>`
  if (parser.parseGreater())
    return Type();
  return StructType::get(elementTypes);
}
```

#### Printing

An implementation of the printer is shown below:

```c++
/// Print an instance of a type registered to the toy dialect.
void ToyDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  // Currently the only toy type is a struct type.
  StructType structType = type.cast<StructType>();

  // Print the struct type according to the parser format.
  printer << "struct<";
  mlir::interleaveComma(structType.getElementTypes(), printer);
  printer << '>';
}
```

Before moving on, let's look at a quick of example showcasing the functionality
we have now:

```toy
struct Struct {
  var a;
  var b;
}

def multiply_transpose(Struct value) {
}
```

Which generates the following:

```mlir
module {
  func @multiply_transpose(%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>) {
    "toy.return"() : () -> ()
  }
}
```

### Operating on `StructType`

Now that the `struct` type has been defined, and we can round-trip it through
the IR. The next step is to add support for using it within our operations.

#### Updating Existing Operations

A few of our existing operations will need to be updated to handle `StructType`.
The first step is to make the ODS framework aware of our Type so that we can use
it in the operation definitions. A simple example is shown below:

```td
// Provide a definition for the Toy StructType for use in ODS. This allows for
// using StructType in a similar way to Tensor or MemRef.
def Toy_StructType :
    Type<CPred<"$_self.isa<StructType>()">, "Toy struct type">;

// Provide a definition of the types that are used within the Toy dialect.
def Toy_Type : AnyTypeOf<[F64Tensor, Toy_StructType]>;
```

We can then update our operations, e.g. `ReturnOp`, to also accept the
`Toy_StructType`:

```td
def ReturnOp : Toy_Op<"return", [Terminator, HasParent<"FuncOp">]> {
  ...
  let arguments = (ins Variadic<Toy_Type>:$input);
  ...
}
```

#### Adding New `Toy` Operations

In addition to the existing operations, we will be adding a few new operations
that will provide more specific handling of `structs`.

##### `toy.struct_constant`

This new operation materializes a constant value for a struct. In our current
modeling, we just use an [array attribute](../../LangRef.md#array-attribute)
that contains a set of constant values for each of the `struct` elements.

```mlir
  %0 = "toy.struct_constant"() {
    value = [dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>]
  } : () -> !toy.struct<tensor<*xf64>>
```

##### `toy.struct_access`

This new operation materializes the Nth element of a `struct` value.

```mlir
  %0 = "toy.struct_constant"() {
    value = [dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>]
  } : () -> !toy.struct<tensor<*xf64>>
  %1 = "toy.struct_access"(%0) {index = 0 : i64} : (!toy.struct<tensor<*xf64>>) -> tensor<*xf64>
```

With these operations, we can revisit our original example:

```toy
struct Struct {
  var a;
  var b;
}

# User defined generic function may operate on struct types as well.
def multiply_transpose(Struct value) {
  # We can access the elements of a struct via the '.' operator.
  return transpose(value.a) * transpose(value.b);
}

def main() {
  # We initialize struct values using a composite initializer.
  Struct value = {[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]};

  # We pass these arguments to functions like we do with variables.
  var c = multiply_transpose(value);
  print(c);
}
```

and finally get a full MLIR module:

```mlir
module {
  func @multiply_transpose(%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64> {
    %0 = "toy.struct_access"(%arg0) {index = 0 : i64} : (!toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64>
    %1 = "toy.transpose"(%0) : (tensor<*xf64>) -> tensor<*xf64>
    %2 = "toy.struct_access"(%arg0) {index = 1 : i64} : (!toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64>
    %3 = "toy.transpose"(%2) : (tensor<*xf64>) -> tensor<*xf64>
    %4 = "toy.mul"(%1, %3) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    "toy.return"(%4) : (tensor<*xf64>) -> ()
  }
  func @main() {
    %0 = "toy.struct_constant"() {value = [dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>, dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>]} : () -> !toy.struct<tensor<*xf64>, tensor<*xf64>>
    %1 = "toy.generic_call"(%0) {callee = @multiply_transpose} : (!toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64>
    "toy.print"(%1) : (tensor<*xf64>) -> ()
    "toy.return"() : () -> ()
  }
}
```

#### Optimizing Operations on `StructType`

Now that we have a few operations operating on `StructType`, we also have many
new constant folding opportunities.

After inlining, the MLIR module in the previous section looks something like:

```mlir
module {
  func @main() {
    %0 = "toy.struct_constant"() {value = [dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>, dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>]} : () -> !toy.struct<tensor<*xf64>, tensor<*xf64>>
    %1 = "toy.struct_access"(%0) {index = 0 : i64} : (!toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64>
    %2 = "toy.transpose"(%1) : (tensor<*xf64>) -> tensor<*xf64>
    %3 = "toy.struct_access"(%0) {index = 1 : i64} : (!toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64>
    %4 = "toy.transpose"(%3) : (tensor<*xf64>) -> tensor<*xf64>
    %5 = "toy.mul"(%2, %4) : (tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>
    "toy.print"(%5) : (tensor<*xf64>) -> ()
    "toy.return"() : () -> ()
  }
}
```

We have several `toy.struct_access` operations that access into a
`toy.struct_constant`. As detailed in [chapter 3](Ch-3.md), we can add folders
for these `toy` operations by setting the `hasFolder` bit on the operation
definition and providing a definition of the `*Op::fold` method.

```c++
/// Fold constants.
OpFoldResult ConstantOp::fold(ArrayRef<Attribute> operands) { return value(); }

/// Fold struct constants.
OpFoldResult StructConstantOp::fold(ArrayRef<Attribute> operands) {
  return value();
}

/// Fold simple struct access operations that access into a constant.
OpFoldResult StructAccessOp::fold(ArrayRef<Attribute> operands) {
  auto structAttr = operands.front().dyn_cast_or_null<mlir::ArrayAttr>();
  if (!structAttr)
    return nullptr;

  size_t elementIndex = index().getZExtValue();
  return structAttr.getValue()[elementIndex];
}
```

To ensure that MLIR generates the proper constant operations when folding our
`Toy` operations, i.e. `ConstantOp` for `TensorType` and `StructConstant` for
`StructType`, we will need to provide an override for the dialect hook
`materializeConstant`. This allows for generic MLIR operations to create
constants for the `Toy` dialect when necessary.

```c++
mlir::Operation *ToyDialect::materializeConstant(mlir::OpBuilder &builder,
                                                 mlir::Attribute value,
                                                 mlir::Type type,
                                                 mlir::Location loc) {
  if (type.isa<StructType>())
    return builder.create<StructConstantOp>(loc, type,
                                            value.cast<mlir::ArrayAttr>());
  return builder.create<ConstantOp>(loc, type,
                                    value.cast<mlir::DenseElementsAttr>());
}
```

With this, we can now generate code that can be generated to LLVM without any
changes to our pipeline.

```mlir
module {
  func @main() {
    %0 = "toy.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %1 = "toy.transpose"(%0) : (tensor<2x3xf64>) -> tensor<3x2xf64>
    %2 = "toy.mul"(%1, %1) : (tensor<3x2xf64>, tensor<3x2xf64>) -> tensor<3x2xf64>
    "toy.print"(%2) : (tensor<3x2xf64>) -> ()
    "toy.return"() : () -> ()
  }
}
```

You can build `toyc-ch7` and try yourself: `toyc-ch7
test/Examples/Toy/Ch7/struct-codegen.toy -emit=mlir`. More details on defining
custom types can be found in
[DefiningAttributesAndTypes](../../DefiningAttributesAndTypes.md).
