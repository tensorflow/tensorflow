# Quickstart tutorial to defining custom dialect attributes and types

This document is a quickstart to defining dialect specific extensions to the
[attribute](LangRef.md#attributes) and [type system](LangRef.md#type-system).
The main part of the tutorial focuses on defining types, but the instructions
are nearly identical for defining attributes.

See [MLIR specification](LangRef.md) for more information about MLIR, the
structure of the IR, operations, etc.

## Types

Types in MLIR (like attributes, locations, and many other things) are
value-typed. This means that instances of `Type` should be passed around
by-value, as opposed to by-pointer or by-reference. The `Type` class in itself
acts as a wrapper around an internal storage object that is uniqued within an
instance of an `MLIRContext`.

### Reserving a range of type kinds

Types in MLIR rely on having a unique `kind` value to ensure that casting checks
remain extremely
efficient([rationale](Rationale.md#reserving-dialect-type-kinds). For a dialect
author, this means that a range of type `kind` values must be explicitly, and
statically, reserved. A dialect can reserve a range of values by adding a new
entry to the
[DialectSymbolRegistry](https://github.com/tensorflow/mlir/blob/master/include/mlir/IR/DialectSymbolRegistry.def).
To support out-of-tree and experimental dialects, the registry predefines a set
of privates ranges, `PRIVATE_EXPERIMENTAL_[0-9]`, that are free for immediate
use.

```c++
DEFINE_SYM_KIND_RANGE(LINALG) // Linear Algebra Dialect
DEFINE_SYM_KIND_RANGE(TOY)    // Toy language (tutorial) Dialect

// The following ranges are reserved for experimenting with MLIR dialects in a
// private context without having to register them here.
DEFINE_SYM_KIND_RANGE(PRIVATE_EXPERIMENTAL_0)
```

For the sake of this tutorial, we will use the predefined
`PRIVATE_EXPERIMENTAL_0` range. These definitions will provide a range in the
Type::Kind enum to use when defining the derived types.

```c++
namespace MyTypes {
enum Kinds {
  // These kinds will be used in the examples below.
  Simple = Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_0_TYPE,
  Complex
};
}
```

### Defining the type class

As described above, `Type` objects in MLIR are value-typed and rely on having an
implicitly internal storage object that holds the actual data for the type. When
defining a new `Type` it isn't always necessary to define a new storage class.
So before defining the derived `Type`, it's important to know which of the two
classes of `Type` we are defining. Some types are `primitives` meaning they do
not have any parameters and are singletons uniqued by kind, like the
[`index` type](LangRef.md#index-type). Parametric types on the other hand, have
additional information that differentiates different instances of the same
`Type` kind. For example the [`integer` type](LangRef.md#integer-type) has a
bitwidth, making `i8` and `i16` be different instances of
[`integer` type](LangRef.md#integer-type).

#### Simple non-parametric types

For simple parameterless types, we can jump straight into defining the derived
type class. Given that these types are uniqued solely on `kind`, we don't need
to provide our own storage class.

```c++
/// This class defines a simple parameterless type. All derived types must
/// inherit from the CRTP class 'Type::TypeBase'. It takes as template
/// parameters the concrete type (SimpleType), and the base class to use (Type).
/// 'Type::TypeBase' also provides several utility methods to simplify type
/// construction.
class SimpleType : public Type::TypeBase<SimpleType, Type> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static bool kindof(unsigned kind) { return kind == MyTypes::Simple; }

  /// This method is used to get an instance of the 'SimpleType'. Given that
  /// this is a parameterless type, it just needs to take the context for
  /// uniquing purposes.
  static SimpleType get(MLIRContext *context) {
    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type.
    return Base::get(context, MyTypes::Simple);
  }
};
```

#### Parametric types

Parametric types are those that have additional construction or uniquing
constraints outside of the type `kind`. As such, these types require defining a
type storage class.

##### Defining a type storage

Type storage objects contain all of the data necessary to construct and unique a
parametric type instance. The storage classes must obey the following:

*   Inherit from the base type storage class `TypeStorage`.
*   Define a type alias, `KeyTy`, that maps to a type that uniquely identifies
    an instance of the parent type.
*   Provide a construction method that is used to allocate a new instance of the
    storage class.
    -   `Storage *construct(TypeStorageAllocator &, const KeyTy &key)`
*   Provide a comparison method between the storage and `KeyTy`.
    -   `bool operator==(const KeyTy &) const`
*   Provide a method to generate the `KeyTy` from a list of arguments passed to
    the uniquer. (Note: This is only necessary if the `KeyTy` cannot be default
    constructed from these arguments).
    -   `static KeyTy getKey(Args...&& args)`
*   Provide a method to hash an instance of the `KeyTy`. (Note: This is not
    necessary if an `llvm::DenseMapInfo<KeyTy>` specialization exists)
    -   `static llvm::hash_code hashKey(const KeyTy &)`

Let's look at an example:

```c++
/// Here we define a storage class for a ComplexType, that holds a non-zero
/// integer and an integer type.
struct ComplexTypeStorage : public TypeStorage {
  ComplexTypeStorage(unsigned nonZeroParam, Type integerType)
      : nonZeroParam(nonZeroParam), integerType(integerType) {}

  /// The hash key for this storage is a pair of the integer and type params.
  using KeyTy = std::pair<unsigned, Type>;

  /// Define the comparison function for the key type.
  bool operator==(const KeyTy &key) const {
    return key == KeyTy(nonZeroParam, integerType);
  }

  /// Define a hash function for the key type.
  /// Note: This isn't necessary because std::pair, unsigned, and Type all have
  /// hash functions already available.
  static llvm::hash_code hashKey(const KeyTy &key) {
    return llvm::hash_combine(key.first, key.second);
  }

  /// Define a construction function for the key type.
  /// Note: This isn't necessary because KeyTy can be directly constructed with
  /// the given parameters.
  static KeyTy getKey(unsigned nonZeroParam, Type integerType) {
    return KeyTy(nonZeroParam, integerType);
  }

  /// Define a construction method for creating a new instance of this storage.
  static ComplexTypeStorage *construct(TypeStorageAllocator &allocator,
                                       const KeyTy &key) {
    return new (allocator.allocate<ComplexTypeStorage>())
        ComplexTypeStorage(key.first, key.second);
  }

  unsigned nonZeroParam;
  Type integerType;
};
```

##### Type class definition

Now that the storage class has been created, the derived type class can be
defined. This structure is similar to the
[simple type](#simple-non-parametric-types), except for a bit more of the
functionality of `Type::TypeBase` is put to use.

```c++
/// This class defines a parametric type. All derived types must inherit from
/// the CRTP class 'Type::TypeBase'. It takes as template parameters the
/// concrete type (ComplexType), the base class to use (Type), and the storage
/// class (ComplexTypeStorage). 'Type::TypeBase' also provides several utility
/// methods to simplify type construction and verification.
class ComplexType : public Type::TypeBase<ComplexType, Type,
                                          ComplexTypeStorage> {
public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// This static method is used to support type inquiry through isa, cast,
  /// and dyn_cast.
  static bool kindof(unsigned kind) { return kind == MyTypes::Complex; }

  /// This method is used to get an instance of the 'ComplexType'. This method
  /// asserts that all of the construction invariants were satisfied. To
  /// gracefully handle failed construction, getChecked should be used instead.
  static ComplexType get(MLIRContext *context, unsigned param, Type type) {
    // Call into a helper 'get' method in 'TypeBase' to get a uniqued instance
    // of this type. All parameters to the storage class are passed after the
    // type kind.
    return Base::get(context, MyTypes::Complex, param, type);
  }

  /// This method is used to get an instance of the 'ComplexType', defined at
  /// the given location. If any of the construction invariants are invalid,
  /// errors are emitted with the provided location and a null type is returned.
  /// Note: This method is completely optional.
  static ComplexType getChecked(MLIRContext *context, unsigned param, Type type,
                                Location location) {
    // Call into a helper 'getChecked' method in 'TypeBase' to get a uniqued
    // instance of this type. All parameters to the storage class are passed
    // after the type kind.
    return Base::getChecked(location, context, MyTypes::Complex, param, type);
  }

  /// This method is used to verify the construction invariants passed into the
  /// 'get' and 'getChecked' methods. Note: This method is completely optional.
  static LogicalResult verifyConstructionInvariants(
      llvm::Optional<Location> loc, MLIRContext *context, unsigned param,
      Type type) {
    // Our type only allows non-zero parameters.
    if (param == 0) {
      if (loc)
        context->emitError(loc) << "non-zero parameter passed to 'ComplexType'";
      return failure();
    }
    // Our type also expects an integer type.
    if (!type.isa<IntegerType>()) {
      if (loc)
        context->emitError(loc) << "non integer-type passed to 'ComplexType'";
      return failure();
    }
    return success();
  }

  /// Return the parameter value.
  unsigned getParameter() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->nonZeroParam;
  }

  /// Return the integer parameter type.
  IntegerType getParameterType() {
    // 'getImpl' returns a pointer to our internal storage instance.
    return getImpl()->integerType;
  }
};
```

### Registering types with a Dialect

Once the dialect types have been defined, they must then be registered with a
`Dialect`. This is done via similar mechanism to
[operations](LangRef.md#operations), `addTypes`.

```c++
struct MyDialect : public Dialect {
  MyDialect(MLIRContext *context) : Dialect(/*name=*/"mydialect", context) {
    /// Add these types to the dialect.
    addTypes<SimpleType, ComplexType>();
  }
};
```

### Parsing and Printing

As a final step after registration, a dialect must override the `printType` and
`parseType` hooks. These enable native support for roundtripping the type in the
textual IR.

## Attributes

As stated in the introduction, the process for defining dialect attributes is
nearly identical to that of defining dialect types. That key difference is that
the things named `*Type` are generally now named `*Attr`.

*   `Type::TypeBase` -> `Attribute::AttrBase`
*   `TypeStorageAllocator` -> `AttributeStorageAllocator`
*   `addTypes` -> `addAttributes`

Aside from that, all of the interfaces for uniquing and storage construction are
all the same.
