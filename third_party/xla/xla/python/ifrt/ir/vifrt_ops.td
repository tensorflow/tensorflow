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

#ifndef XLA_PYTHON_IFRT_IR_VIFRT_OPS_TD_
#define XLA_PYTHON_IFRT_IR_VIFRT_OPS_TD_

include "mlir/IR/OpBase.td"
include "xla/python/ifrt/ir/vifrt_dialect.td"
include "xla/python/ifrt/ir/vifrt_interfaces.td"

// VIFRT is a bare mininum versioned copy of IFRT. In the pursuit of minimality,
// it adopts the following conventions (similar to VHLO):
// 1) Use Vifrt_AnyType or Variading<Vifrt_AnyType> for all operands and
//    results.
// 2) Use traits only when strictly necessary (e.g., `AttrSizedOperandSegments`
//    when multiple variable operands are present).
// 3) Use Vifrt_AnyAttr for all attributes.
// 4) Don't use verifiers.

class Vifrt_Op<string mnemonic, string min_version, string max_version, list<Trait> traits = []> :
    Op<Vifrt_Dialect, mnemonic,
      [DeclareOpInterfaceMethods<Vifrt_VersionedOpInterface>] # traits> {
  let extraClassDefinition = [{
    ::xla::ifrt::Version $cppClass::getMinVersion() {
      return ::xla::ifrt::Version(}] # !subst(".", ", ", min_version) # [{);
    }
    ::xla::ifrt::Version $cppClass::getMaxVersion() {
      }] # !if(
        !eq(max_version, "current"),
        [{ return ::xla::ifrt::Version::getCurrentVersion(); }],
        [{ return ::xla::ifrt::Version(}] # !subst(".", ", ", max_version) # [{); }]
      ) # [{
    }
  }];
}

def Vifrt_ReshardOpV1 : Vifrt_Op<"ReshardV1", "0.1.0", "current",
    [AttrSizedOperandSegments]> {
  let arguments = (ins
    Variadic<Vifrt_AnyType>:$inputs,
    Vifrt_AnyAttr:$donated,
    Variadic<Vifrt_AnyType>:$control_inputs);
  let results = (outs
    Variadic<Vifrt_AnyType>:$outputs,
    Vifrt_AnyType:$control_output);
}

def Vifrt_CopyArraysOpV1 : Vifrt_Op<"CopyArraysV1", "0.1.0", "current",
    [AttrSizedOperandSegments]> {
  let arguments = (ins
    Variadic<Vifrt_AnyType>:$inputs,
    Vifrt_AnyAttr:$donated,
    Variadic<Vifrt_AnyType>:$control_inputs);
  let results = (outs
    Variadic<Vifrt_AnyType>:$outputs,
    Vifrt_AnyType:$control_output);
}

def Vifrt_AssembleOpV1
    : Vifrt_Op<"AssembleV1", "0.1.0", "current", [AttrSizedOperandSegments]> {
  let arguments = (ins
    Variadic<Vifrt_AnyType>:$inputs,
    Variadic<Vifrt_AnyType>:$control_inputs);
  let results = (outs Vifrt_AnyType:$output);
}

def Vifrt_DisassembleOpV1 : Vifrt_Op<"DisassembleV1", "0.1.0", "current", []> {
  let arguments = (ins
    Vifrt_AnyType:$input,
    Variadic<Vifrt_AnyType>:$control_inputs);
  let results = (outs Variadic<Vifrt_AnyType>:$outputs);
}

def Vifrt_RemapArraysOpV1 : Vifrt_Op<"RemapArraysV1", "0.1.0", "current", []> {
  let arguments = (ins
    Variadic<Vifrt_AnyType>:$inputs,
    Vifrt_AnyAttr:$mappings,
    Vifrt_AnyAttr:$donated);
  let results = (outs Variadic<Vifrt_AnyType>:$outputs);
}

def Vifrt_CallOpV1 : Vifrt_Op<"CallV1", "0.1.0", "current",
    [AttrSizedOperandSegments]> {
  let arguments = (ins
    Variadic<Vifrt_AnyType>:$inputs,
    Variadic<Vifrt_AnyType>:$control_inputs,
    Vifrt_AnyAttr:$callee,
    Vifrt_AnyAttr:$devices,
    Vifrt_AnyAttr:$io_aliases,
    Vifrt_AnyAttr:$donated_input_indices);
  let results = (outs
    Variadic<Vifrt_AnyType>:$outputs,
    Vifrt_AnyType:$control_output);
}

def Vifrt_CallLoadedExecutableOpV1 : Vifrt_Op<"CallLoadedExecutableV1", "0.1.0",
    "current", [AttrSizedOperandSegments]> {
  let arguments = (ins
    Variadic<Vifrt_AnyType>:$inputs,
    Variadic<Vifrt_AnyType>:$control_inputs,
    Vifrt_AnyAttr:$callee,
    Vifrt_AnyAttr:$io_aliases,
    Vifrt_AnyAttr:$donated_input_indices);
  let results = (outs
    Variadic<Vifrt_AnyType>:$outputs,
    Vifrt_AnyType:$control_output);
}

def Vifrt_LoadedExecutableOpV1 : Vifrt_Op<"LoadedExecutableV1", "0.1.0",
    "current", []> {
  let arguments = (ins
    Vifrt_AnyAttr:$sym_name,
    Vifrt_AnyAttr:$function_type,
    Vifrt_AnyAttr:$devices
  );
}

def Vifrt_AfterOpV1 : Vifrt_Op<"AfterV1", "0.1.0", "current", []> {
  let arguments = (ins Variadic<Vifrt_AnyType>:$inputs);
  let results = (outs Vifrt_AnyType:$control_output);
}

#endif  // XLA_PYTHON_IFRT_IR_VIFRT_OPS_TD_