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

#ifndef XLA_CODEGEN_EMITTERS_IR_XLA_ATTRS
#define XLA_CODEGEN_EMITTERS_IR_XLA_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "xla/codegen/emitters/ir/xla_dialect.td"

class XLA_Attr<string name, list<Trait> traits = []> :
      AttrDef<XlaDialect, name, traits> {
}

def XLA_AffineMapParameter :
    AttrOrTypeParameter<"::mlir::AffineMap", ""> {
}

def XLA_IndexingMapVariableParameter
  : ArrayRefParameter<"::xla::IndexingMap::Variable",
                      "IndexingMapVariableArray"> {
}

def XLA_ConstraintsParameter :
    ArrayRefParameter<"::std::pair<::mlir::AffineExpr, ::xla::Interval>",
                      "ContraintsArray"> {
}

def XLA_IndexingMapAttr : XLA_Attr<"IndexingMap"> {
  let summary = "An Attribute representing an indexing map.";
  let mnemonic = "indexing_map";
  let description = [{This attribute stores an indexing map. See
    https://openxla.org/xla/indexing for more details.
  }];
  let parameters = (ins XLA_AffineMapParameter:$map,
                        XLA_IndexingMapVariableParameter:$dim_vars,
                        XLA_IndexingMapVariableParameter:$range_vars,
                        XLA_ConstraintsParameter:$constraints);
  let hasCustomAssemblyFormat = 1;
  let builders =  [
    AttrBuilder<(ins "const ::xla::IndexingMap&":$indexing_map)>,
  ];
  let genVerifyDecl = 1;
  let extraClassDeclaration = [{
    // Returns the indexing map constructed from IndexingMapAttr.
    xla::IndexingMap getIndexingMap() const;

    // Returns the number of indexing map results.
    int64_t getNumResults() const;
  }];
}

// Note: settle for BackendKind to avoid clashing with the existing
// xla::Backend and xla::BackendType types.
def XLA_BackendKind : I32EnumAttr<"BackendKind", "XLA Backend kind (or type)", [
    I32EnumAttrCase<"kCpu", 0, "cpu">,
    I32EnumAttrCase<"kGpu", 1, "gpu">,
    I32EnumAttrCase<"kTpu", 2, "tpu">
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::xla";
}
def XLA_BackendKindAttr :
    EnumAttr<XlaDialect, XLA_BackendKind, "backend_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def XLA_ExtraBackendOptionsAttr
  : ArrayOfAttr<XlaDialect,
                "ExtraBackendOptions",
                "extra_backend_options",
                "mlir::StringAttr"> {
}

def Workgroup_Dimension : I32EnumAttr<"WorkgroupDimension",
    "a dimension, either 'x', 'y', or 'z'",
    [
      I32EnumAttrCase<"x", 0>,
      I32EnumAttrCase<"y", 1>,
      I32EnumAttrCase<"z", 2>
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::xla";
}
def Workgroup_DimensionAttr : EnumAttr<XlaDialect, Workgroup_Dimension, "dim">;

#endif // XLA_CODEGEN_EMITTERS_IR_XLA_ATTRS
