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

#endif
