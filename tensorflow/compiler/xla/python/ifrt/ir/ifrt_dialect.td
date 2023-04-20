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
}

def Ifrt_ShardingParameter :
    AttrOrTypeParameter<"::xla::ifrt::ShardingParam", ""> {
  let parser = "::xla::ifrt::ShardingParam::Parse($_parser)";
}

def Ifrt_ArrayType : TypeDef<Ifrt_Dialect, "IfrtArray"> {
  let mnemonic = "array";
  let summary = "An Ifrt array sharded on a set of devices.";

  let parameters = (ins
    Builtin_RankedTensor:$shape,
    Ifrt_ShardingParameter:$sharding,
    ArrayRefParameter<"int64_t">:$devices);

  let assemblyFormat = "`<` $shape`,` $sharding`,` `[`$devices`]` `>`";

  let genVerifyDecl = 1;
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
