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

#ifndef XLA_BACKENDS_CPU_CODEGEN_EMITTERS_IR_XLA_CPU_DIALECT
#define XLA_BACKENDS_CPU_CODEGEN_EMITTERS_IR_XLA_CPU_DIALECT

include "mlir/IR/DialectBase.td"

def XlaCpuDialect : Dialect {
  let name = "xla_cpu";

  let description = [{
    This dialect contains ops required for lowering HLO to LLVM for XLA:CPU
    backend and runtime.
  }];

  let dependentDialects = [
    "xla::XlaDialect",
  ];

  let cppNamespace = "::xla::cpu";
  let useDefaultTypePrinterParser = 1;
}

#endif // XLA_BACKENDS_CPU_CODEGEN_EMITTERS_IR_XLA_CPU_DIALECT
