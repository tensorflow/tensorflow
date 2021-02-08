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

def TF__JitFusedMatMulOp : TF_Op<"_JitFusedMatMul", [NoSideEffect, SameOperandsAndResultElementType]> {
  let summary = [{
    MatMul operation with an output fusion compiled at runtime via MLIR codegen.
  }];

  let description = [{
The inputs to the MatMul are specified by `a` and `b`. The series of operations
that follows is specified by the `fusion` attribute, which is a list of output
kernel names specified as strings (e.g. "BiasAdd"). They are performed in order,
where the (first) input to each op is the output of the preceding op. The first
input and the output of each fused_op must be of type T.

Supported list of fusions is defined by ContractionOutputKernelBuilder
implementations.

*WARN*: This is a TFRT only operations, and it does not exist in TF. This
operation is only added by the ContractionFusion pass.
  }];

  let arguments = (ins
    TensorOf<[F32]>:$a,
    TensorOf<[F32]>:$b,
    Variadic<TensorOf<[F32]>>:$additional_args,

    DefaultValuedAttr<BoolAttr, "false">:$transpose_a,
    DefaultValuedAttr<BoolAttr, "false">:$transpose_b,
    DefaultValuedAttr<StrArrayAttr, "{}">:$fusion
  );

  let results = (outs
    TensorOf<[F32]>:$product
  );

  TF_DerivedOperandTypeAttr T = TF_DerivedOperandTypeAttr<0>;
}

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

def TF__TfrtGetResourceOp : TF_Op<"_TfrtGetResource", []> {
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
