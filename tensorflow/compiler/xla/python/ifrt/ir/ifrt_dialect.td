/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_IFRT_DIALECT_TD_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_IFRT_DIALECT_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/DialectBase.td"

def Ifrt_Dialect : Dialect {
  let name = "ifrt";
  let summary = "IFRT dialect";
  let cppNamespace = "::xla::ifrt";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let hasOperationAttrVerify = 1;
  let hasRegionArgAttrVerify = 1;
  let usePropertiesForAttributes = 0;
}

def Ifrt_ShardingParameter :
    AttrOrTypeParameter<"::xla::ifrt::ShardingParam", ""> {
  let parser = "::xla::ifrt::ShardingParam::Parse($_parser)";
}

def Ifrt_DevicesAttr : AttrDef<Ifrt_Dialect, "IfrtDevices"> {
  let mnemonic = "devices";
  let summary = "Represents a list of device ids.";

  let parameters = (ins ArrayRefParameter<"int">:$ids);
  let assemblyFormat = "`[` $ids `]`";

  let returnType = "::llvm::ArrayRef<int>";
  let convertFromStorage = "$_self";
  let constBuilderCall = "$_builder.getAttr<::xla::ifrt::IfrtDevicesAttr>($0)";

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    operator ::llvm::ArrayRef<int>() const;
  }];
}

def Ifrt_ShardingAttr : AttrDef<Ifrt_Dialect, "IfrtSharding"> {
  let mnemonic = "sharding";
  let summary = "ShardingParam as an attribute.";

  let parameters = (ins Ifrt_ShardingParameter:$sharding);
  let assemblyFormat = "`<` $sharding `>`";
}

def Ifrt_ArrayType : TypeDef<Ifrt_Dialect, "IfrtArray"> {
  let mnemonic = "array";
  let summary = "An Ifrt array sharded on a set of devices.";

  let parameters = (ins
    Builtin_RankedTensor:$shape,
    Ifrt_ShardingParameter:$sharding,
    Ifrt_DevicesAttr:$devices_attr);

  let builders = [
    TypeBuilder<(ins
      "::mlir::RankedTensorType":$shape,
      "::xla::ifrt::ShardingParam":$sharding,
      "::llvm::ArrayRef<int>":$devices), [{
        return Base::get($_ctxt, shape, sharding,
                         ::xla::ifrt::IfrtDevicesAttr::get($_ctxt, devices));
      }]>
  ];

  let assemblyFormat = "`<` $shape`,` $sharding`,` $devices_attr`>`";

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    // Get device ids from `devices_attr`.
    ::llvm::ArrayRef<int> getDevices() const;
  }];
}

def Ifrt_ControlType : TypeDef<Ifrt_Dialect, "IfrtControl"> {
  let mnemonic = "control";
  let summary = [{
    Represents execution dependency.

    When an op takes a control as argument, it will not be scheduled for
    execution before the op emitting the control has finished execution.
  }];
}

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_IFRT_DIALECT_TD_
