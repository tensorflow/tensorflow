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


#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_IFRT_OPS_TD_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_IFRT_OPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "tensorflow/compiler/xla/python/ifrt/ir/ifrt_dialect.td"

class Ifrt_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ifrt_Dialect, mnemonic, traits>;

def Ifrt_ReshardOp : Ifrt_Op<"Reshard"> {
  let summary = "Reshards a host tensor or device array";
  let description = [{
    Copies the host tensor or device array to a new sharding.

    The input and output must have the same global shape.

    The input and output are allowed to have the same sharding and devices. In
    this case, this ReshardOp will have no transfer across devices.
  }];

  let arguments = (ins
    Ifrt_ArrayType:$input,
    Variadic<Ifrt_ControlType>:$controlInputs);
  let results = (outs
    Ifrt_ArrayType:$output);

  let hasVerifier = 1;
}

def Ifrt_AssembleOp : Ifrt_Op<"Assemble", [AttrSizedOperandSegments]> {
  let summary = "Assembles single device arrays to a sharded array";
  let description = [{
    Builds a larger array out of individual per-device arrays.

    On each device, the local shard of the result array is the input array.
    Thus, in an efficient implementation, no buffer transfer is needed.

    The inputs' `device` parameter must have size 1. They should be distinct and
    combine to the output's `device` list.
  }];

  let arguments = (ins
    Variadic<Ifrt_ArrayType>:$inputs,
    Variadic<Ifrt_ControlType>:$controlInputs);
  let results = (outs
    Ifrt_ArrayType:$output);

  let hasVerifier = 1;
}

def Ifrt_DisassembleOp : Ifrt_Op<"Disassemble"> {
  let summary = "Disassembles a sharded array to single device arrays";
  let description = [{
    Breaks an array up into per-device arrays.

    On each device, the result array is its local shard in the input array.
    Thus, in an efficient implementation, no buffer transfer is needed.

    The outputs' `devices` parameter must have size 1. They should be distinct
    and combine to the input's `devices` list.
  }];

  let arguments = (ins
    Ifrt_ArrayType:$input,
    Variadic<Ifrt_ControlType>:$controlInputs);
  let results = (
    outs Variadic<Ifrt_ArrayType>:$outputs);

  let hasVerifier = 1;
}

def Ifrt_CallOp : Ifrt_Op<"Call",
    [AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<CallOpInterface>,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Call some function on a set of devices";
  let description = [{
    The callee is a global FuncOp to be sharded onto the given set of devices.
    It's up to the IFRT implementation to define which dialects are legal in the
    callee.

    The callee's inputs/outputs should be regular tensors, the shape of which
    are the same as inputs/outputs' global shape, correspondingly. Every
    input/output of the callee must have a `ifrt.sharding` attribute matching
    that on the corresponding input/output of this CallOp.

    The callee doesn't specify the device placement. It's specified at the
    `devices` attribute of this CallOp. Every input/output must be placed on
    a subset of these devices.
  }];

  let arguments = (ins
    Variadic<Ifrt_ArrayType>:$inputs,
    Variadic<Ifrt_ControlType>:$controlInputs,

    SymbolRefAttr:$callee,
    DenseI64ArrayAttr:$devices);
  let results = (outs
    Variadic<Ifrt_ArrayType>:$outputs,
    Ifrt_ControlType:$controlOutput);

  let assemblyFormat = [{
    $callee `(` $inputs `)` oilist(`after` $controlInputs) attr-dict `:` functional-type($inputs, $outputs)
  }];
  let hasVerifier = 1;
}

def Ifrt_AfterOp : Ifrt_Op<"After"> {
  let summary = "Get a control handle for array materialization.";
  let description = [{
    When depending on the `controlOutput`, the op will be not be scheduled until
    all `inputs` are materialized. For example, this could mean the data
    transfer is completed from a `Reshard` op.

    Moreover, this op provides fine-grained control dependency on `CallOp`'s
    result. Consider
      %0, %1, %ctrl_0 = "ifrt.Call" @callee() ...
    The following 3 control handles may resolve at difference time instances:
      "ifrt.After"(%0)
      "ifrt.After"(%1)
      %ctrl_0
  }];

  let arguments = (ins Variadic<Ifrt_ArrayType>:$inputs);
  let results = (outs Ifrt_ControlType:$controlOutput);
}

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_IFRT_OPS_TD_
