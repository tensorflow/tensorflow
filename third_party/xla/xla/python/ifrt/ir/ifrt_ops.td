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


#ifndef XLA_PYTHON_IFRT_IR_IFRT_OPS_TD_
#define XLA_PYTHON_IFRT_IR_IFRT_OPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "xla/python/ifrt/ir/ifrt_dialect.td"
include "xla/python/ifrt/ir/ifrt_interfaces.td"

class Ifrt_Op<string mnemonic, list<Trait> traits = []> :
    Op<Ifrt_Dialect, mnemonic, traits>;

def Ifrt_ReshardOp
    : Ifrt_Op<"Reshard", [AttrSizedOperandSegments, NestedInIfrtFunc]> {
  let summary = "Reshards a sequence of host tensors or device arrays";
  let description = [{
    Copies the host tensors or device arrays to a new sharding.

    The corresponding output of each input must have the same global shape
    as the input.

    An input and its corresponding output are allowed to have the same sharding
    and devices. In this case, this ReshardOp will have no transfer across
    devices.
  }];

  let arguments = (ins
    Variadic<Ifrt_ArrayType>:$inputs,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$donated,
    Variadic<Ifrt_ControlType>:$control_inputs);
  let results = (outs
    Variadic<Ifrt_ArrayType>:$outputs,
    Ifrt_ControlType:$control_output);

  let assemblyFormat = [{
    `(` $inputs `)` oilist(`after` $control_inputs) attr-dict
    `:` functional-type($inputs, $outputs)
  }];

  let hasVerifier = 1;
}

def Ifrt_CopyArraysOp
    : Ifrt_Op<"CopyArrays", [AttrSizedOperandSegments, NestedInIfrtFunc]> {
  let summary = "Copies arrays to a new set of devices";
  let description = [{
    Copies the input arrays to the output arrays.

    This op requires that all input arrays have use the same devices and memory
    king, and all the output arrays use the same devices and memory king. Note
    that the devices and memory kind used by the input and output arrays might
    not be the same. Moreover, the corresponding output array of an input array
    must have the same sharding as the op can not reshard.
  }];

  let arguments = (ins
    Variadic<Ifrt_ArrayType>:$inputs,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$donated,
    Variadic<Ifrt_ControlType>:$control_inputs);
  let results = (outs
    Variadic<Ifrt_ArrayType>:$outputs,
    Ifrt_ControlType:$control_output);

  let assemblyFormat = [{
    `(` $inputs `)` oilist(`after` $control_inputs) attr-dict
    `:` functional-type($inputs, $outputs)
  }];

  let hasVerifier = 1;
}

def Ifrt_AssembleOp
    : Ifrt_Op<"Assemble", [AttrSizedOperandSegments, NestedInIfrtFunc]> {
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
    Variadic<Ifrt_ControlType>:$control_inputs);
  let results = (outs
    Ifrt_ArrayType:$output);

  let hasVerifier = 1;
}

def Ifrt_DisassembleOp : Ifrt_Op<"Disassemble", [NestedInIfrtFunc]> {
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
    Variadic<Ifrt_ControlType>:$control_inputs);
  let results = (
    outs Variadic<Ifrt_ArrayType>:$outputs);

  let hasVerifier = 1;
}

def Ifrt_RemapArraysOp
    : Ifrt_Op<"RemapArrays", [NestedInIfrtFunc]> {
  let summary = "Remaps shards from input `Array`s to create new `Array`s.";
  let description = [{
    A metadata-only op that remaps shards across input `Array`s to create
    new `Array`s based on a plan remapping array shards. The op can shuffle
    or extract input array shards without causing a data copy or transfer. It
    must meet the following requirements:
     - All input/output `Array`s must have the same `DType` and per-shard shape.
     - An input shard can be used at most once.
     - Every output shard must have exactly one input shard mapped.
  }];

  let arguments = (ins
    Variadic<Ifrt_ArrayType>:$inputs,
    Ifrt_ArrayMappingAttrArrayAttr:$mappings,
    DefaultValuedOptionalAttr<BoolAttr, "false">:$donated);
  let results = (outs Variadic<Ifrt_ArrayType>:$outputs);

  let assemblyFormat = [{
    `(` $inputs `)` `mappings` `=` $mappings attr-dict
    `:` functional-type($inputs, $outputs)
  }];

  let hasVerifier = 1;
}

def Ifrt_CallOp : Ifrt_Op<"Call",
    [AttrSizedOperandSegments,
     NestedInIfrtFunc,
     IfrtCallLike<"::mlir::func::FuncOp">,
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

    `io_aliases` represents pairs of inputs and outputs, where the input buffer
    may be aliased and used as the output buffer. The aliased pair must have the
    same byte size. It's up to IFRT implementations whether to respect this
    hint or not. Alternatively, if the index of an input is In
    `donated_input_indices` then the input buffer might be donated to the
    callee if an output with the same byte size is found.
  }];

  let arguments = (ins
    Variadic<Ifrt_ArrayType>:$inputs,
    Variadic<Ifrt_ControlType>:$control_inputs,
    SymbolRefAttr:$callee,
    Ifrt_DevicesAttr:$devices,
    DefaultValuedAttr<Ifrt_IoAliasesAttr, "{}">:$io_aliases,
    DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$donated_input_indices);
  let results = (outs
    Variadic<Ifrt_ArrayType>:$outputs,
    Ifrt_ControlType:$control_output);

  let assemblyFormat = [{
    $callee `(` $inputs `)` oilist(`after` $control_inputs)
    `on` `devices` $devices attr-dict `:` functional-type($inputs, $outputs)
  }];
  let hasVerifier = 1;
}

def Ifrt_CallLoadedExecutableOp : Ifrt_Op<"CallLoadedExecutable",
    [AttrSizedOperandSegments,
     NestedInIfrtFunc,
     IfrtCallLike<"LoadedExecutableOp">,
     DeclareOpInterfaceMethods<CallOpInterface>,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Call some loaded executable";
  let description = [{
    The callee is a LoadedExecutableOp that declares the signature of the
    LoadedExecutable. The actual LoadedExecutable instance is bound through
    CompileOptions.

    The callee is placed on devices because it's loaded. Every input/output must
    be placed on a subset of these devices.

    `io_aliases` represents pairs of inputs and outputs, where the input buffer
    may be aliased and used as the output buffer. The aliased pair must have the
    same byte size. It's up to IFRT implementations whether to respect this
    hint or not. Alternatively, if the index of an input is In
    `donated_input_indices` then the input buffer might be donated to the
    callee if an output with the same byte size is found.
  }];

  let arguments = (ins
    Variadic<Ifrt_ArrayType>:$inputs,
    Variadic<Ifrt_ControlType>:$control_inputs,
    SymbolRefAttr:$callee,
    DefaultValuedAttr<Ifrt_IoAliasesAttr, "{}">:$io_aliases,
    DefaultValuedAttr<DenseI32ArrayAttr, "{}">:$donated_input_indices);
  let results = (outs
    Variadic<Ifrt_ArrayType>:$outputs,
    Ifrt_ControlType:$control_output);

  let assemblyFormat = [{
    $callee `(` $inputs `)` oilist(`after` $control_inputs) attr-dict
    `:` functional-type($inputs, $outputs)
  }];
  let hasVerifier = 1;
}

def Ifrt_LoadedExecutableOp : Ifrt_Op<"LoadedExecutable",
    [DeclareOpInterfaceMethods<Symbol>]> {
  let summary = "Declare a loaded executable";
  let description = [{
    The inputs/outputs of `function_type` should be IFRT arrays. They should be
    placed on a subset of the `devices`.
  }];

  let arguments = (ins
    SymbolNameAttr: $sym_name,
    TypeAttrOf<FunctionType>: $function_type,
    Ifrt_DevicesAttr:$devices
  );

  let assemblyFormat = [{
    $sym_name `on` `devices` $devices attr-dict `:` $function_type
  }];
  let hasVerifier = 1;
}

def Ifrt_AfterOp : Ifrt_Op<"After", [NestedInIfrtFunc]> {
  let summary = "Get a control handle for array materialization.";
  let description = [{
    When depending on the `control_output`, the op will be not be scheduled
    until all `inputs` are materialized. For example, this could mean the data
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
  let results = (outs Ifrt_ControlType:$control_output);
}

#endif  // XLA_PYTHON_IFRT_IR_IFRT_OPS_TD_
