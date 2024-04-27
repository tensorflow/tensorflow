/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_IFRT_DIALECT_TD_
#define XLA_PYTHON_IFRT_IR_IFRT_DIALECT_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/DialectBase.td"
include "xla/python/ifrt/ir/ifrt_interfaces.td"

//===---------------------------------------------------------------------------
// Dialect
//===---------------------------------------------------------------------------

def Ifrt_Dialect : Dialect {
  let name = "ifrt";
  let summary = "IFRT dialect";
  let cppNamespace = "::xla::ifrt";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let hasOperationAttrVerify = 1;
  let hasRegionArgAttrVerify = 1;
}

//===---------------------------------------------------------------------------
// Attributes
//===---------------------------------------------------------------------------

def Ifrt_DevicesAttr : AttrDef<Ifrt_Dialect, "IfrtDevices"> {
  let mnemonic = "devices";
  let summary = "Represents a list of logical device ids.";
  let description = [{
    Logical device ids are 0-based indices within a list of concrete devices
    an IFRT module is compiled for. These ids are logical in order to
    ensure that an IFRT IR module is hermetically specified, and thus can
    be compiled for different concrete device ids without modifications.
  }];

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

def Ifrt_ShardingParameter :
    AttrOrTypeParameter<"::xla::ifrt::ShardingParam", ""> {
  let parser = "::xla::ifrt::ShardingParam::Parse($_parser)";
}

def Ifrt_ShardingParamAttr : AttrDef<Ifrt_Dialect, "IfrtShardingParam", [
    DeclareAttrInterfaceMethods<Ifrt_ShardingAttrInterface>
]> {
  let mnemonic = "sharding_param";
  let summary = "ShardingParam as an attribute.";

  let parameters = (ins Ifrt_ShardingParameter:$sharding);
  let assemblyFormat = "`<` $sharding `>`";

  let genVerifyDecl = 1;
}

def Ifrt_UnspecifiedShardingAttr : AttrDef<Ifrt_Dialect,
    "IfrtUnspecifiedSharding", [
      DeclareAttrInterfaceMethods<Ifrt_ShardingAttrInterface>
]> {
  let mnemonic = "sharding_unspecified";
  let summary = "Attribute to be used when sharding is unspecified.";

  let parameters = (ins);
  let assemblyFormat = "";

  let genVerifyDecl = 1;
}

//===---------------------------------------------------------------------------
// Types
//===---------------------------------------------------------------------------

def Ifrt_ArrayType : TypeDef<Ifrt_Dialect, "IfrtArray"> {
  let mnemonic = "array";
  let summary = "An Ifrt array sharded on a set of devices.";

  let parameters = (ins
    Builtin_RankedTensor:$shape,
    "::xla::ifrt::IfrtShardingAttrInterface":$sharding_attr,
    Ifrt_DevicesAttr:$devices_attr);

  let builders = [
    TypeBuilder<(ins
      "::mlir::RankedTensorType":$shape,
      "::xla::ifrt::IfrtShardingAttrInterface":$sharding_attr,
      "::llvm::ArrayRef<int>":$devices), [{
        return Base::get(
          $_ctxt, shape, sharding_attr,
          ::xla::ifrt::IfrtDevicesAttr::get($_ctxt, devices));
      }]>,
    TypeBuilder<(ins
      "::mlir::RankedTensorType":$shape,
      "::llvm::ArrayRef<int>":$devices), [{
        return Base::get(
          $_ctxt, shape, ::xla::ifrt::IfrtUnspecifiedShardingAttr::get($_ctxt),
          ::xla::ifrt::IfrtDevicesAttr::get($_ctxt, devices));
      }]>,
    TypeBuilder<(ins
      "::mlir::RankedTensorType":$shape,
      "::xla::ifrt::ShardingParam":$sharding,
      "::llvm::ArrayRef<int>":$devices), [{
        return Base::get(
          $_ctxt, shape,
          ::xla::ifrt::IfrtShardingParamAttr::get($_ctxt, sharding),
          ::xla::ifrt::IfrtDevicesAttr::get($_ctxt, devices));
      }]>
  ];

  let assemblyFormat = "`<` $shape`,` $sharding_attr`,` $devices_attr`>`";

  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    // Get logical device ids from `devices_attr`.
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

#endif  // XLA_PYTHON_IFRT_IR_IFRT_DIALECT_TD_
