/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef COMPOSITE_UTILS_TD
#define COMPOSITE_UTILS_TD

include "mlir/IR/PatternBase.td"

// See the function doc in the header file.
def GetNhwcReturnTypeFromNchw: NativeCodeCall<
  "GetNhwcReturnTypeFromNchw((*$0.begin()).getDefiningOp())">;


// When given a DenseIntElementsAttr containing I64 elements, this extracts
// one I32IntegerAttr from the given index.
class GetI32At<int index>: NativeCodeCall<
  "$_builder.getI32IntegerAttr(static_cast<int32_t>(*($0.getValues<int64_t>().begin() + " # index #")))">;

def GetI32DenseAttr: NativeCodeCall<
  "DenseI64AttrToI32Attr($0, $_builder)">;

// Receives a composite DictionaryAttr and returns the value of the Attribute
// with the key `attr_name` as the type provided by `attr_type`.
class GetCompositeAttributeAs<string attr_name, string attr_type>:
  NativeCodeCall<"$0.get(\"" # attr_name # "\").dyn_cast<" # attr_type # ">()">;

// Receives a composite DictionaryAttr and returns the value of the Attribute
// with the key `attr_name` as a DenseIntElementsAttr.
class GetAsVectorAttr<string attr_name>:
  GetCompositeAttributeAs<attr_name, "DenseIntElementsAttr">;

class IsBoolAttrEqual<string true_or_false> : Constraint<CPred<
  "$0.getValue() == "#true_or_false#"">>;

// Receives a composite DictionaryAttr as an argument and checks if one of the
// its attributes (with the name `attr_name`) is of type `attribute` and has
// the value `val`.
class IsCompositeAttribute<string attr_name, Attr attribute, string val>:
  Constraint<CPred<
    "$0.get(\"" # attr_name # "\") == " # !subst("$0", val, attribute.constBuilderCall)>>;

// Receives a composite DictionaryAttr as an argument and checks if has a
// BoolAttr with the name `attr_name` and value `val`.
class IsBoolCompositeAttribute <string attr_name, string val>:
  IsCompositeAttribute<attr_name, BoolAttr, val>;

// Receives a composite DictionaryAttr as an argument and checks if has a
// StrAttr with the name `attr_name` and value `val`.
class IsStrCompositeAttribute <string attr_name, string val>:
  IsCompositeAttribute<attr_name, StrAttr, "\"" # val # "\"">;

#endif
