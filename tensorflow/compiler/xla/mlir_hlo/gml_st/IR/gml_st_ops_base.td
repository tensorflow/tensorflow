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

#ifndef GML_ST_OPS_BASE
#define GML_ST_OPS_BASE

include "mlir/Dialect/Utils/StructuredOpsUtils.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def GmlSt_Dialect : Dialect {
  let name = "gml_st";
  let cppNamespace = "::mlir::gml_st";
  let description = [{
    The GmlSt (Google ML Structured) dialect is intended to hold operations,
    types and transformations to assist structured code generation.
  }];

  let dependentDialects = ["tensor::TensorDialect"];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

class GMLST_Op<string mnemonic, list<Trait> traits> :
    Op<GmlSt_Dialect, mnemonic, traits> {
  let hasVerifier = 1;
}

def IteratorTypeEnum : EnumAttr<GmlSt_Dialect, IteratorType, "iterator_type"> {
  let assemblyFormat = "`<` $value `>`";
}

def IteratorTypeArrayAttr : TypedArrayAttrBase<IteratorTypeEnum,
    "Iterator type should be an enum.">;

#endif // GML_ST_OPS_BASE
