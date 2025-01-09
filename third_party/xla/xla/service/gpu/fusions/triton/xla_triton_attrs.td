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

#ifndef XLA_SERVICE_GPU_FUSIONS_TRITON_XLA_TRITON_ATTRS_TD_
#define XLA_SERVICE_GPU_FUSIONS_TRITON_XLA_TRITON_ATTRS_TD_

include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "xla/service/gpu/fusions/triton/xla_triton_dialect.td"

def TTXLA_SparseDotMetaEncodingAttr : DistributedEncoding<"SparseDotMetaEncoding",
"sparse_dot_meta_encoding", [], XlaTritonDialect> {
  let mnemonic = "sparse_dot_meta";

  let parameters = (ins "Attribute":$parent);
  let assemblyFormat = "`<``{` struct(params) `}``>`";
  let extraClassDeclaration = extraDistributedDeclaration # [{
    SmallVector<unsigned> getContigPerThread() {
      return getSizePerThread();
    };
  }];
}


#endif // XLA_SERVICE_GPU_FUSIONS_TRITON_XLA_TRITON_ATTRS_TD_
