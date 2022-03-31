/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TF_TYPE_ATTRIBUTES
#define TF_TYPE_ATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/SubElementInterfaces.td"
include "tensorflow/core/ir/types/dialect.td"

// Base class for TFType dialect attributes.
class TFType_Attr<string name, list<Trait> traits = []>
    : AttrDef<TFTypeDialect, name, traits>;

// LINT.IfChange
def TFType_FullTypeId : I32EnumAttr<"FullTypeId", "", [
  // The default represents an uninitialized values.
  I32EnumAttrCase<"TFT_UNSET", 0, "unset">,

  // Type symbols. Used to construct more complex type expressions like
  // algebraic data types.

  // Type variables may serve as placeholder for any other type ID in type
  // templates.
  //
  // Examples:
  //   TFT_DATASET[TFT_VAR["T"]] is a Dataset returning a type indicated by "T".
  //   TFT_TENSOR[TFT_VAR["T"]] is a Tensor of n element type indicated by "T".
  //   TFT_TENSOR[TFT_VAR["T"]], TFT_TENSOR[TFT_VAR["T"]] are two tensors of
  //     identical element types.
  //   TFT_TENSOR[TFT_VAR["P"]], TFT_TENSOR[TFT_VAR["Q"]] are two tensors of
  //     independent element types.
  //
  I32EnumAttrCase<"TFT_VAR", 1, "var">,

  // Wildcard type. Describes a parameter of unknown type. In TensorFlow, that
  // can mean either a "Top" type (accepts any type), or a dynamically typed
  // object whose type is unknown in context.
  // Important: "unknown" does not necessarily mean undeterminable!
  I32EnumAttrCase<"TFT_ANY", 2, "any">,

  // The algebraic product type. This is an algebraic type that may be used just
  // for logical grouping. Not to confused with TFT_TUPLE which describes a
  // concrete object of several elements.
  //
  // Example:
  //   TFT_DATASET[TFT_PRODUCT[TFT_TENSOR[TFT_INT32], TFT_TENSOR[TFT_FLOAT64]]]
  //     is a Dataset producing two tensors, an integer one and a float one.
  //
  I32EnumAttrCase<"TFT_PRODUCT", 3, "product">,

  // Represents a named field, with the name stored in the attribute.
  //
  // Parametrization:
  //   TFT_NAMED[<type>]{<name>}
  //   * <type> is the type of the field
  //   * <name> is the field name, as string (thpugh can theoretically be an int
  //     as well)
  //
  // Example:
  //   TFT_RECORD[
  //     TFT_NAMED[TFT_TENSOR[TFT_INT32]]{'foo'},
  //     TFT_NAMED[TFT_TENSOR[TFT_FLOAT32]]{'bar'},
  //   ]
  //     is a structure with two fields, an int tensor "foo" and a float tensor
  //     "bar".
  I32EnumAttrCase<"TFT_NAMED", 4, "named">,

  // Template definition. Expands the variables by repeating a template as
  // arguments of container.
  //
  // Parametrization:
  //   TFT_FOR_EACH[<container_type>, <template>, <expansions>]
  //   * <container_type> is the type of the container that the template will be
  //     expanded into
  //   * <template> is any type definition that potentially contains type
  //     variables
  //   * <expansions> is a TFT_VAR and may include more types in the future
  //
  // Example:
  //   TFT_FOR_EACH[
  //         TFT_PRODUCT,
  //         TFT_TENSOR[TFT_VAR["t"]],
  //         TFT_VAR["t"]
  //     ]
  //     will substitute a T = TFT_INT32 to TFT_PRODUCT[TFT_TENSOR[TFT_INT32]]
  //     and a T = (TFT_INT32, TFT_INT64) to
  //     TFT_PRODUCT[TFT_TENSOR[TFT_INT32], TFT_TENSOR[TFT_INT64]].
  I32EnumAttrCase<"TFT_FOR_EACH", 20, "for_each">,

  // Callable types describe functions and ops.
  //
  // Parametrization:
  //   TFT_CALLABLE[<arg type>, <return type>]
  //   * <arg type> is the type of the arguments; TFT_PRODUCT represents
  //   multiple
  //     arguments.
  //   * <return type> is the return type; TFT_PRODUCT represents multiple
  //     return values (that means that callables returning multiple things
  //     don't necessarily return a single tuple).
  //
  // Example:
  //   TFT_CALLABLE[
  //     TFT_ANY,
  //     TFT_PRODUCT[TFT_TENSOR[TFT_INT32], TFT_TENSOR[TFT_FLOAT64]],
  //   ]
  //     is a callable with unspecified (for now) input arguments, and
  //     two return values of type tensor.
  //
  I32EnumAttrCase<"TFT_CALLABLE", 100, "callable">,

  // Concrete type IDs, representing "proper" data types that can describe
  // runtime TensorFlow objects.

  // The usual Tensor. This is a parametric type.
  //
  // Parametrization:
  //   TFT_TENSOR[<element type>, <shape type>]
  //   * <element type> is currently limited to one of the element types
  //     defined below.
  //   * <shape type> is not yet defined, and may only be TFT_UNKNOWN for now.
  //
  // A TFT_SHAPE type will be defined in the future.
  //
  // Example:
  //   TFT_TENSOR[TFT_INT32, TFT_UNKNOWN]
  //     is a Tensor of int32 element type and unknown shape.
  //
  // TODO(mdan): Define TFT_SHAPE and add more examples.
  I32EnumAttrCase<"TFT_TENSOR", 1000, "tensor">,

  // Array (or tensorflow::TensorList in the variant type registry).
  // Note: this is not to be confused with the deprecated `TensorArray*` ops
  // which are not supported by FullType.
  // This type represents a random-access list whose elements can be
  // described by a single type. Although immutable, Array is expected to
  // support efficient mutation semantics (i.e. element update) in the
  // user-facing API.
  // The element type may be generic or even TFT_ANY for a heterogenous list.
  //
  // Parametrization:
  //   TFT_ARRAY[<element type>]
  //   * <element type> may be any concrete type.
  //
  // Examples:
  //   TFT_ARRAY[TFT_TENSOR[TFT_INT32]] is a TensorArray holding int32 Tensors
  //     of any shape.
  //   TFT_ARRAY[TFT_TENSOR[TFT_UNKNOWN]] is a TensorArray holding Tensors of
  //     mixed element types.
  //   TFT_ARRAY[TFT_UNKNOWN] is a TensorArray holding any element type.
  //   TFT_ARRAY[] is equivalent to TFT_ARRAY[TFT_UNKNOWN].
  //   TFT_ARRAY[TFT_ARRAY[]] is an array or arrays (of unknown types).
  I32EnumAttrCase<"TFT_ARRAY", 1001, "array">,

  // Optional (or tensorflow::OptionalVariant in the variant type registry).
  // This type represents a value that may either hold an element of a single
  // specified type, or nothing at all.
  //
  // Parametrization:
  //   TFT_OPTIONAL[<element type>]
  //   * <element type> may be any concrete type.
  //
  // Examples:
  //   TFT_OPTIONAL[TFT_TENSOR[TFT_INT32]] is an Optional holding an int32
  //     Tensor of any shape.
  I32EnumAttrCase<"TFT_OPTIONAL", 1002, "optional">,

  // Literal types describe compile-time constant values.
  // Literal types may also participate in dependent types.
  //
  // Parametrization:
  //   TFT_LITERAL[<value type>]{<value>}
  //   * <value type> may be any concrete type compatible that can hold <value>
  //   * <value> is the type's attribute, and holds the actual literal value
  //
  // Examples:
  //   TFT_LITERAL[TFT_INT32]{1} is the compile-time constant 1.
  I32EnumAttrCase<"TFT_LITERAL", 1003, "literal">,

  // Type attributes. These always appear in the parametrization of a type,
  // never alone. For example, there is no such thing as a "bool" TensorFlow
  // object (for now).

  // The bool element type.
  // TODO(mdan): Quantized types, legacy representations (e.g. ref)
  I32EnumAttrCase<"TFT_BOOL", 200, "bool">,
  // Integer element types.
  I32EnumAttrCase<"TFT_UINT8", 201, "uint8">,
  I32EnumAttrCase<"TFT_UINT16", 202, "uint16">,
  I32EnumAttrCase<"TFT_UINT32", 203, "uint32">,
  I32EnumAttrCase<"TFT_UINT64", 204, "uint64">,
  I32EnumAttrCase<"TFT_INT8", 205, "int8">,
  I32EnumAttrCase<"TFT_INT16", 206, "int16">,
  I32EnumAttrCase<"TFT_INT32", 207, "int32">,
  I32EnumAttrCase<"TFT_INT64", 208, "int64">,
  // Floating-point element types.
  I32EnumAttrCase<"TFT_HALF", 209, "half">,
  I32EnumAttrCase<"TFT_FLOAT", 210, "float">,
  I32EnumAttrCase<"TFT_DOUBLE", 211, "double">,
  I32EnumAttrCase<"TFT_BFLOAT16", 215, "bfloat16">,
  // TODO(mdan): Represent as TFT_COMPLEX[TFT_DOUBLE] instead?
  I32EnumAttrCase<"TFT_COMPLEX64", 212, "complex64">,
  I32EnumAttrCase<"TFT_COMPLEX128", 213, "complex128">,
  // The string element type.
  I32EnumAttrCase<"TFT_STRING", 214, "string">,

  // Other types that we don't know yet whether they will become part of the
  // core type system or be consisdered third-party (and consequently moved to
  // user-defined type mechanisms). Presently, they are effectively in the core
  // type system, because key compilation passes like Placer account for their
  // existence.

  // Datasets created by tf.data ops and APIs. Datasets have generator/iterable
  // semantics, that is, one can construct an iterator from them. Like
  // Array, they are considered to return elements that can be described
  // by a single type. Unlike Array, they do not support random access or
  // mutation, and can potentially produce an infinite number of elements.
  // A datasets can produce logical structures (e.g. multiple elements). This
  // is expressed using TFT_PRODUCT.
  //
  //
  // Parametrization: TFT_ARRAY[<element type>].
  //   * <element type> may be a concrete type or a type symbol. It represents
  //     the data type of the elements produced by the dataset.
  //
  // Examples:
  //   TFT_DATSET[TFT_TENSOR[TFT_INT32]] is a Dataset producing single int32
  //     Tensors of unknown shape.
  //   TFT_DATSET[TFT_PRODUCT[TFT_TENSOR[TFT_INT32], TFT_TENSOR[TFT_FLOAT32]] is
  //     a Dataset producing pairs of Tensors, one integer and one float.
  // Note: The high ID number is to prepare for the eventuality that Datasets
  // will be supported by user types in the future.
  I32EnumAttrCase<"TFT_DATASET", 10102, "dataset">,

  // A ragged tensor created by tf.ragged ops and APIs.
  //
  // Parametrization: TFT_RAGGED[<element_type>].
  I32EnumAttrCase<"TFT_RAGGED", 10103, "ragged">,

  // A mutex lock tensor, produced by tf.raw_ops.MutexLock.
  // Unlike strict execution models, where ownership of a lock is denoted by
  // "running after the lock has been acquired", in non-strict mode, lock
  // ownership is in the true sense: "the op argument representing the lock is
  // available".
  // Mutex locks are the dynamic counterpart of control dependencies.
  // TODO(mdan): Properly document this thing.
  //
  // Parametrization: TFT_MUTEX_LOCK[].
  I32EnumAttrCase<"TFT_MUTEX_LOCK", 10202, "mutex_lock">,

  // The equivalent of a Tensor with DT_VARIANT dtype, kept here to simplify
  // translation. This type should not normally appear after type inference.
  // Note that LEGACY_VARIANT != ANY: TENSOR[INT32] is a subtype of ANY, but is
  // not a subtype of LEGACY_VARIANT.
  I32EnumAttrCase<"TFT_LEGACY_VARIANT", 10203, "legacy_variant">
]> {
  let cppNamespace = "::mlir::tf_type";
  string cppType = "int32_t";
  let genSpecializedAttr = 0;
}

def TFType_FullTypeArgsAttr : ArrayRefParameter<"::mlir::tf_type::FullTypeAttr", "args">;

def TFType_FullTypeAttrAttr : Attr<Or<[StrAttr.predicate, SI64Attr.predicate]>,
  "FullType literal attr"> {
  let storageType = "Attribute";
  let returnType = "Attribute";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$0";
  string cppType = "Attribute";
  let isOptional = 1;
}

def TFType_FullTypeAttr : AttrDef<TFTypeDialect, "FullType"> {
  let summary = "FullType";
  let parameters = (ins
      TFType_FullTypeId:$type_id,
      TFType_FullTypeArgsAttr:$args,
      TFType_FullTypeAttrAttr:$attr
  );
  let mnemonic = "full_type";
  let hasCustomAssemblyFormat = 1;
  // Format is effectively: type_id ('<' $args^ '>')? ($attr?)
}
// LINT.ThenChange(../../framework/full_type.proto)

//===----------------------------------------------------------------------===//
// FuncAttr
//===----------------------------------------------------------------------===//

def TFType_FuncAttr : TFType_Attr<"Func", [
    DeclareAttrInterfaceMethods<SubElementAttrInterface,
        ["replaceImmediateSubAttribute"]>
  ]> {
  let mnemonic = "func";
  let summary = "Models the `AttrValue.value.func` proto attribute value as a "
    "pair of SymbolRef and DictionaryAttr";
  let description = [{
    This attributes matches the protobuf `AttrValue.value.func` with a
    `SymbolRefAttr`, for the `NameAttrList.name` `string` and a `DictionaryAttr`
    for the `NameAttrList.attr` `map<string, AttrValue>`. It is currently
    printed and parsed for the following format:

      #tf_type.func<@symbol, {attr = "value"}>

    where the first element is the `SymbolRefAttr` and the second element is the
    `DictionaryAttr`.

    So that the symbol reference and any symbol references nested in the
    `DictionaryAttr` are visible to symbol tables, this attribute implements the
    `SubElementAttrInterface`.
  }];

  let parameters = (ins
    "SymbolRefAttr":$name,
    "DictionaryAttr":$attrs
  );
  let builders = [
    AttrBuilder<(ins "StringRef":$name, "DictionaryAttr":$attr), [{
      return $_get($_ctxt, SymbolRefAttr::get($_ctxt, name), attr);
    }]>
  ];
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Placeholder
//===----------------------------------------------------------------------===//

def TFType_PlaceholderAttr : TFType_Attr<"Placeholder"> {
  let mnemonic = "placeholder";
  let summary = "Placeholder attributes are string referring to a function "
    "attribute to be substituted on instantiation";
  let description = [{
    This is matching the `placeholder` Attribute type in protobuf storage. This
    is just a string, but we need a dedicated type for roundtrip purpose.
  }];
  let parameters = (ins
    StringRefParameter<"value">:$value
  );
  let hasCustomAssemblyFormat = 1;
}

def TFGraph_TypeOrPlaceholder
    : Attr<Or<[TypeAttr.predicate, TFType_PlaceholderAttr.predicate]>,
          "a type or placeholder attribute"> {
  let returnType = "::mlir::Attribute";
  let convertFromStorage = "$_self";
}

//===----------------------------------------------------------------------===//
// ShapeAttr
//===----------------------------------------------------------------------===//

def TFType_ShapeAttrDef : TFType_Attr<"Shape"> {
  let mnemonic = "shape";
  let summary = "A shape either unranked or is modelled an array of int64";
  let description = [{
    This attributes matches the `ShapedType` MLIR Type content into an attribute
    value. It contains a flag to indicate if it unranked, and if ranked it
    exposes an array of integer modeling the individual dimensions. A value of
    `ShapedType::kDynamicDim` indicates a dynamic dimension.
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "bool":$unranked
  );
  let builders = [
    // Returns a shape attribute for the provided `dimension` array. If the
    // `dimensions` aren't provided, then the shape attribute is unranked.
    // For ranked shapes, the value of the each individual dimension size must
    // be >= 0 or `ShapedType::kDynamicDim`. The value of
    // `ShapedType::kDynamicDim` means the dimension is dynamic. Otherwise, the
    // dimension is static.
    AttrBuilder<(ins "llvm::Optional<ArrayRef<int64_t>>":$dimensions)>,
    // Returns a Shape attribute from a TensorFlow ShapedType type.
    AttrBuilder<(ins "ShapedType":$shaped_type)>
  ];
  let extraClassDeclaration = [{
    // Returns true if this shape is ranked and has only known dimensions size.
    bool hasStaticShape() const;

    // Returns true if this shape attribute has a statically known rank.
    bool hasRank() const;

    // Returns the rank. Aborts if unranked.
    int64_t getRank() const;

    // Returns the shape array if ranked, or None if unranked.
    llvm::Optional<ArrayRef<int64_t>> getValue() const;
  }];
  let hasCustomAssemblyFormat = 1;
}

// An array of TF shapes.
def TFGraph_ShapesAttr
    : TypedArrayAttrBase<TFType_ShapeAttrDef, "An array of shapes.">;

//===----------------------------------------------------------------------===//
// VersionAttr
//===----------------------------------------------------------------------===//

def TFType_VersionAttr : TFType_Attr<"Version"> {
  let mnemonic = "version";
  let summary = "An Attribute describing the version for a TensorFlow Graph";
  let parameters = (ins
    "int32_t":$producer,
    "int32_t":$minConsumer,
    ArrayRefParameter<"int32_t">:$badConsumers
  );
  let hasCustomAssemblyFormat = 1;
}

#endif
