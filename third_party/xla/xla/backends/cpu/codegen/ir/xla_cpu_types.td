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

#ifndef XLA_BACKENDS_CPU_CODEGEN_IR_XLA_CPU_TYPES
#define XLA_BACKENDS_CPU_CODEGEN_IR_XLA_CPU_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypes.td"
include "xla/backends/cpu/codegen/ir/xla_cpu_dialect.td"

class XLACPU_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<XlaCpuDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// !xla_cpu.call_frame
//===----------------------------------------------------------------------===//

def XLACPU_CallFrame : XLACPU_Type<"CallFrame", "call_frame"> {
  let summary = "XLA:CPU host kernel call frame";
}

#endif // XLA_BACKENDS_CPU_CODEGEN_IR_XLA_CPU_TYPES
