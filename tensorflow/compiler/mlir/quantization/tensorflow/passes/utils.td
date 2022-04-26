/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
include "mlir/IR/OpBase.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"

def NonOpaqueElementsAttr : ElementsAttrBase<
  CPred<"!$_self.isa<OpaqueElementsAttr>()">,
  "non-opaque constant tensor">;

// Checks if the data format is "NHWC".
def IsDataFormatNHWC : ConstantAttr<TF_ConvnetDataFormatAttr, "\"NHWC\"">;

// Checks if the element value has a float type.
def IsFloatElementsAttr : ElementsAttrBase<
  CPred<"$_self.isa<ElementsAttr>() && "
        "getElementTypeOrSelf($_self.cast<ElementsAttr>().getType()).isa<FloatType>()">,
        "float constant tensor">;

// Checks if the boolean value is false.
def IsFalseBoolAttr : AttrConstraint<
  CPred<"!$_self.cast<BoolAttr>().getValue()">>;

// Checks if the value has only one user.
def HasOneUse : Constraint<CPred<"$0.hasOneUse()">>;
