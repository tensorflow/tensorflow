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

#ifndef GML_ST_EXTENSION_OPS
#define GML_ST_EXTENSION_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops_base.td"
include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface.td"

def GMLST_DynamicBroadcastInDimOp : GMLST_Op<"dynamic_broadcast_in_dim",
    [NoSideEffect,
     TypesMatchWith<"result and init types match", "init", "result", "$_self">,
     DeclareOpInterfaceMethods<FusionInterface>]> {
  let summary = [{Destination-style twin for `mhlo.dynamic_broadcast_in_dim`}];
  let arguments = (ins
    AnyTensor:$operand,
    AnyTensor:$init,
    I64ElementsAttr:$broadcast_dimensions,
    OptionalAttr<I64ElementsAttr>:$known_expanding_dimensions,
    OptionalAttr<I64ElementsAttr>:$known_nonexpanding_dimensions
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat = [{
    `ins` `(` $operand `:` type($operand) `)`
    `outs` `(` $init `:` type($init) `)`
    attr-dict
  }];
  let hasVerifier = 0;
}

#endif // GML_ST_EXTENSION_OPS