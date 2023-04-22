/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This is the operation definition file for TensorFlow operations with
// implementation available only in TFRT.

#ifndef TFRT_OPS
#define TFRT_OPS

include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_base.td"
include "mlir/IR/OpBase.td"

def TF__TfrtSetResourceOp : TF_Op<"_TfrtSetResource", []> {
  let summary = "Set values as TFRT runtime's static resource.";

  let description = [{
    Setting a tensor value in TFRT runtime's static resource manager, using
    index as its identifier. It can be retrieved by _TfrtGetResource using the
    same index.
  }];

  let arguments = (ins
    TF_Tensor:$arg,
    I64Attr:$index
  );
}

def TF__TfrtGetResourceOp : TF_Op<"_TfrtGetResource", [NoSideEffect, TF_NoConstantFold]> {
  let summary = "Get values from TFRT runtime's static resource.";

  let description = [{
    Getting tensor values from TFRT runtime's static resource manager, using
    the indices that are previously set in _TfrtSetResource op.
  }];

  let arguments = (ins
    I64ArrayAttr:$indices
  );

  let results = (outs
    Variadic<TF_Tensor>:$results
  );
}

#endif // TFRT_OPS
