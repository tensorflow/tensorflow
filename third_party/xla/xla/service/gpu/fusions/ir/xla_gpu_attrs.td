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

#ifndef XLA_SERVICE_GPU_FUSIONS_MLIR_ATTRS
#define XLA_SERVICE_GPU_FUSIONS_MLIR_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "xla/service/gpu/fusions/ir/xla_gpu_dialect.td"

class XLAGPU_Attr<string name, list<Trait> traits = []> :
      AttrDef<XlaGpuDialect, name, traits> {
}

def XLAGPU_AffineMapParameter :
    AttrOrTypeParameter<"::mlir::AffineMap", ""> {
}

def XLAGPU_DimVarsParameter : ArrayRefParameter<"::xla::gpu::DimVar",
                                                "DimVarArray"> {
}

def XLAGPU_RangeVarsParameter : ArrayRefParameter<"::xla::gpu::RangeVar",
                                                  "RangeVarArray"> {
}

def XLAGPU_ConstraintsParameter :
    ArrayRefParameter<"::std::pair<::mlir::AffineExpr, ::xla::gpu::Interval>",
                      "ContraintsArray"> {
}

def XLAGPU_IndexingMapAttr : XLAGPU_Attr<"IndexingMap"> {
  let summary = "An Attribute representing an indexing map.";
  let mnemonic = "indexing_map";
  let description = [{This attribute stores an indexing map. See
    https://openxla.org/xla/indexing for more details.
  }];
  let parameters = (ins XLAGPU_AffineMapParameter:$map,
                        XLAGPU_DimVarsParameter:$dim_vars,
                        XLAGPU_RangeVarsParameter:$range_vars,
                        XLAGPU_ConstraintsParameter:$constraints);
  let hasCustomAssemblyFormat = 1;
  let builders =  [
    AttrBuilder<(ins "const ::xla::gpu::IndexingMap&":$indexing_map)>,
  ];
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    // Returns the indexing map constructed from IndexingMapAttr.
    xla::gpu::IndexingMap getIndexingMap();
  }];
}

#endif // XLA_SERVICE_GPU_FUSIONS_MLIR_ATTRS
