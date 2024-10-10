/* Copyright 2024 The OpenXLA Authors.

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


#ifndef XLA_SERVICE_GPU_FUSIONS_TRITON_XLA_TRITON_OPS_TD_
#define XLA_SERVICE_GPU_FUSIONS_TRITON_XLA_TRITON_OPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "triton/Dialect/Triton/IR/TritonTypeInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

def XlaTritonDialect : Dialect {
  let name = "triton_xla";

  let description = [{
    This dialect contains ops included in the xla extension point for Triton.
  }];

  let cppNamespace = "::mlir::triton::xla";
}

class TTXLA_Op<string mnemonic, list<Trait> traits = []> :
      Op<XlaTritonDialect, mnemonic, traits> {
}

def TTXLA_SparseDotOp : TTXLA_Op<"sparse_dot", [
        Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>,
        TypesMatchWith<"result's type matches accumulator's type", "d", "c", "$_self">]> {
    let summary = "sparse dot";

    let arguments = (ins
      TT_TensorOrMemDesc:$a,
      TT_TensorOrMemDesc:$b,
      TT_FpIntTensor:$c,
      TT_IntTensor: $aMeta);
    let results = (outs TT_FpIntTensor:$d);
    let assemblyFormat = [{
      $a`,` $b`,` $c`,` $aMeta attr-dict
        `:` type($a) `meta` type($aMeta) `*` type($b) `->` type($d)
    }];
    let hasVerifier = 1;
}

#endif // XLA_SERVICE_GPU_FUSIONS_TRITON_XLA_TRITON_OPS_TD_
