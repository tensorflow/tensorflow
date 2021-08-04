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

#ifdef TFG_DIALECT
#else
#define TFG_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


// ODS Definition for the dialect, see https://mlir.llvm.org/docs/OpDefinitions/
// for more information.


//===----------------------------------------------------------------------===//
// TFGraph dialect definitions
//===----------------------------------------------------------------------===//

def TFGraphDialect : Dialect {
  let name = "tfg";

  let summary = "This dialect models TensorFlow Graphs as encoded in GraphDef.";
  let description = [{
    This dialect is modeling TensorFlow GraphDefs and intended to provide a high
    level of fidelity.
  }];

  let extraClassDeclaration = [{
    Identifier getNameAttrIdentifier() { return *name_key_; }
    static constexpr StringRef getNameAttrKey() { return "_mlir_name"; }

    Identifier getDeviceAttrIdentifier() { return *device_key_; }
    static constexpr StringRef getDeviceAttrKey() { return "_mlir_device"; }

    Identifier getAssignedDeviceAttrIdentifier() {
      return *assigned_device_key_;
    }
    static constexpr StringRef getAssignedDeviceAttrKey() {
      return "_mlir_assigned_device";
    }

    // Cached accessor for the control type.
    ControlType getControlType() { return control_ty_; }

    // Returns the hook to parse an operation belonging to this dialect, even
    // if unregistered.
    Optional<ParseOpHook> getParseOperationHook(StringRef opName) const
      override;

    // Print an operation that belongs to this dialect if unregistered.
    LogicalResult printOperation(Operation *op,
                                 OpAsmPrinter &printer) const override;
  private:
    // Cached identifier for efficiency purpose.
    llvm::Optional<Identifier> name_key_;
    llvm::Optional<Identifier> device_key_;
    llvm::Optional<Identifier> assigned_device_key_;
    ControlType control_ty_;
  }];

  let cppNamespace = "::mlir::tfg";
}

//===----------------------------------------------------------------------===//
// TFGraph dialect types definitions
//===----------------------------------------------------------------------===//

include "tensorflow/core/ir/types/types.td"

// Base class for the operations in this dialect.
class TFGraph_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<TFGraphDialect, mnemonic, traits>;

#endif // TFG_DIALECT
