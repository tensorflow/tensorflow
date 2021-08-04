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

include "tensorflow/core/ir/types/dialect.td"

// Base class for TFType dialect attributes.
class TFType_Attr<string name> : AttrDef<TFTypeDialect, name>;

//===----------------------------------------------------------------------===//
// FuncAttr
//===----------------------------------------------------------------------===//

def TFType_FuncAttr : TFType_Attr<"Func"> {
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
}


//===----------------------------------------------------------------------===//
// ShapeAttr
//===----------------------------------------------------------------------===//

def TFType_ShapeAttrDef : TFType_Attr<"Shape"> {
  let mnemonic = "shape";
  let returnType = "ShapeAttr";
  let convertFromStorage = "$_self.cast<ShapeAttr>()";
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
}


//===----------------------------------------------------------------------===//
// VersionAttr
//===----------------------------------------------------------------------===//

def TFType_VersionAttr : TFType_Attr<"Version"> {
  let mnemonic = "version";
  let returnType = "VersionAttr";
  let convertFromStorage = "$_self.cast<VersionAttr>()";
  let summary = "An Attribute describing the version for a TensorFlow Graph";
  let parameters = (ins
    "int32_t":$producer,
    "int32_t":$minConsumer,
    ArrayRefParameter<"int32_t">:$badConsumers
  );
}

#endif
