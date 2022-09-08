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

#ifndef TFG_DIALECT
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

    The attribute mappings from GraphDef are listed down below,

    Graph/Function Attributes:
    FunctionDef.attr will prepand with "tf" prefix
    FunctionDef.signature.name <-> "sym_name"
    FunctionDef.signature.description <-> "description"
    FunctionDef.signature.is_stateful <-> "is_stateful"
    FunctionDef.signature.gradient <-> "gradient"
    FunctionDef.resource_arg_unique_id <-> "resource_arg_unique_ids_keys"
    FunctionDef.resource_arg_unique_id <-> "resource_arg_unique_ids_values"

    Input Attributes:
    FunctionDef.signature.input_arg.name <-> "tfg.name"
    FunctionDef.signature.input_arg.description <-> "tfg.description"
    FunctionDef.signature.input_arg.handle_data <-> "tfg.handle_data"
    FunctionDef.signature.input_arg.is_ref <-> "tfg.is_ref"
    FunctionDef.arg_attr will prepand with "tf" prefix

    Output Attributes:
    FunctionDef.signature.output_arg.name <-> "tfg.name"
    FunctionDef.signature.output_arg.description <-> "tfg.description"
    FunctionDef.signature.output_arg.handle_data <-> "tfg.handle_data"
    FunctionDef.signature.output_arg.type <-> "tfg.dtype"
    FunctionDef.signature.control_output <-> "tfg.control_ret_name_"

    Node Attributes:
    NodeDef.device <-> "_mlir_device"
    NodeDef.name <-> "_mlir_name"
    NodeDef.attr <-> "_output_shape"
    NodeDef.experimental_type <-> "_mlir_fulltype"
  }];

  let extraClassDeclaration = [{
    StringAttr getNameAttrIdentifier() const { return name_key_; }
    static constexpr StringLiteral getNameAttrKey() { return {"_mlir_name"}; }

    StringAttr getDeviceAttrIdentifier() const { return device_key_; }
    static constexpr StringLiteral getDeviceAttrKey() {
      return {"_mlir_device"};
    }

    StringAttr getAssignedDeviceAttrIdentifier() const {
      return assigned_device_key_;
    }
    static constexpr StringLiteral getAssignedDeviceAttrKey() {
      return {"_mlir_assigned_device"};
    }

    StringAttr getFullTypeAttrIdentifier() const { return fulltype_key_; }
    static constexpr StringLiteral getFullTypeAttrKey() {
      return {"_mlir_fulltype"};
    }

    StringAttr getTfgNameAttrIdentifier() const { return tfg_name_key_; }
    static constexpr StringRef getTfgNameAttrKey() { return "tfg.name"; }

    StringAttr getTfgDescriptionAttrIdentifier() const {
      return tfg_description_key_;
    }
    static constexpr StringRef getTfgDescriptionAttrKey() {
      return {"tfg.description"};
    }

    StringAttr getTfgIsRefAttrIdentifier() const { return tfg_is_ref_key_; }
    static constexpr StringRef getTfgIsRefAttrKey() { return {"tfg.is_ref"}; }

    StringAttr getTfgHandleDataAttrIdentifier() const {
      return tfg_handle_data_key_;
    }
    static constexpr StringRef getTfgHandleDataAttrKey() {
      return {"tfg.handle_data"};
    }

    StringAttr getTfgFullTypeAttrIdentifier() const {
      return tfg_full_type_key_;
    }
    static constexpr StringRef getTfgFullTypeAttrKey() {
      return {"tfg.experimental_full_type"};
    }

    StringAttr getLiftedGraphFuncNameAttrIdentifier() const {
      return lifted_graph_func_name_;
    }
    static constexpr StringRef getLiftedGraphFuncNameKey() {
      return {"_mlir_lifted_graph"};
    }

    // Cached accessor for the control type.
    ControlType getControlType() const { return control_ty_; }

    // Print an operation that belongs to this dialect if unregistered.
    void printCustomTfOp(Operation *op, OpAsmPrinter &printer) const;

    // Returns the hook to parse an operation belonging to this dialect, even
    // if unregistered.
    Optional<ParseOpHook> getParseOperationHook(StringRef opName) const
      override;

    // Returns the took to print an operation belonging to this dialect, even
    // if unregistered.
    llvm::unique_function<void(Operation *, OpAsmPrinter &)>
    getOperationPrinter(Operation *op) const override;

    // Functions for checking operation categories.
    #define GET_OP_CATEGORIES
    #include "tensorflow/core/ir/tf_op_names.inc"

  private:
    // Fallback implementation of OpAsmOpInterface.
    TFGraphOpAsmInterface *fallbackOpAsmInterface_ = nullptr;

    // Cached TensorFlow operation names.
    #define GET_OP_NAME_DECLS
    #include "tensorflow/core/ir/tf_op_names.inc"

    // Cached identifier for efficiency purpose.
    StringAttr assigned_device_key_;
    StringAttr device_key_;
    StringAttr fulltype_key_;
    StringAttr lifted_graph_func_name_;
    StringAttr name_key_;
    StringAttr tfg_description_key_;
    StringAttr tfg_full_type_key_;
    StringAttr tfg_handle_data_key_;
    StringAttr tfg_is_ref_key_;
    StringAttr tfg_name_key_;

    // Cached control type.
    ControlType control_ty_;
  }];

  let cppNamespace = "::mlir::tfg";

  let useDefaultAttributePrinterParser = 1;
  let hasNonDefaultDestructor = 1;
  let hasOperationInterfaceFallback = 1;
  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}

#endif // TFG_DIALECT
