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

#ifndef TF_ATTRIBUTES
#define TF_ATTRIBUTES

include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"


// Base class for TF dialect attributes.
class TF_Attr<string name> : AttrDef<TF_Dialect, name>;

//===----------------------------------------------------------------------===//
// FuncAttr
//===----------------------------------------------------------------------===//

def TF_FuncAttr : TF_Attr<"Func"> {
  let mnemonic = "func";
  let summary = "Models the `AttrValue.value.func` proto attribute value as a "
    "pair of SymbolRef and DictionaryAttr.";
  let description = [{
    This attributes matches the protobuf `AttrValue.value.func` with a
    `SymbolRefAttr`, for the `NameAttrList.name` `string` and a `DictionaryAttr`
    for the `NameAttrList.attr` `map<string, AttrValue>`. It is currently
    printed and parsed for the following format:

      #tf.func<@symbol, {attr = "value"}>

    where the first element is the `SymbolRefAttr` and the second element is the
    `DictionaryAttr`.  }];

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

def TF_PlaceholderAttr : TF_Attr<"Placeholder"> {
  let mnemonic = "placeholder";
  let summary = "Placeholder attributes are string referring a function "
    "attribute to be substituted on instantiation.";
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

def TF_ShapeAttrDef : TF_Attr<"Shape"> {
  let mnemonic = "shape";
  let returnType = "ShapeAttr";
  let convertFromStorage = "$_self.cast<ShapeAttr>()";
  let summary = "A shape either unranked or is modelled an array of int64.";
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
    // Returns true if this shape is ranked and no dimension has unknown size
    // (ShapedType::kDynamicDim).
    bool hasStaticShape() const;

    bool hasRank() const;

    // If this is ranked, return the rank. Otherwise, abort.
    int64_t getRank() const;

    // Return the shape array if ranked.
    llvm::Optional<ArrayRef<int64_t>> getValue() const;
  }];
}

#endif
