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

#ifndef DTENSOR_DIALECT
#define DTENSOR_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


// ODS Definition for the dialect, see https://mlir.llvm.org/docs/OpDefinitions/
// for more information.


//===----------------------------------------------------------------------===//
// DTensor dialect definitions
//===----------------------------------------------------------------------===//

def DTensorDialect : Dialect {
  let name = "dtensor";

  let summary = "Dialect for dtensor";
  let description = [{
    This dialect contains key operations for DTensors. Currently it holds two
    attributes; one that represents layouts and one for representing a mesh.
  }];

  let cppNamespace = "::mlir::dtensor";

  let extraClassDeclaration = [{
// Parses an instance of an attribute registered to the dialect.
Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

// Prints an instance of an attribute registered to the dialect.
void printAttribute(Attribute attr, DialectAsmPrinter &os) const override;

private:
// Register the attributes of this dialect.
void registerAttributes();

public:
  }];
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

//===----------------------------------------------------------------------===//
// DTENSOR dialect types definitions
//===----------------------------------------------------------------------===//


#endif // DTENSOR_DIALECT
