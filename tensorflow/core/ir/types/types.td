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

#ifndef TF_TYPES_TYPES
#define TF_TYPES_TYPES

include "mlir/IR/AttrTypeBase.td"
include "tensorflow/core/ir/types/dialect.td"

// All of the types will extend this class.
class TFType_Type<string name> : TypeDef<TFTypeDialect, name> { }

def ControlType : TFType_Type<"Control"> {
  let mnemonic = "control";
  let description = [{
    This type models a control dependency between two TensorFlow nodes.
  }];
}

def OpaqueTensorType : TFType_Type<"OpaqueTensor"> {
  let mnemonic = "tensor";
  let description = [{
    This type models an opaque tensor: the shape and dtypes are unknown. This is
    particularly useful in generic functions.
  }];
}

def TFGraph_Tensor
    : Type<Or<[AnyTensor.predicate, OpaqueTensorType.predicate]>,
           "a tensor or opaque tensor type">;

// A value that is either a tensor or a control type.
def TFGraph_TensorOrControlType
    : Type<Or<[TFGraph_Tensor.predicate, ControlType.predicate]>,
           "any tensor or control type">;

#endif // TYPES
