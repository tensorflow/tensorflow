/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_MLIR_XLA_CPU_DIALECT_TD_
#define XLA_MLIR_XLA_CPU_DIALECT_TD_

include "mlir/IR/OpBase.td"

def XlaCpuDialect : Dialect {
  let name = "xla_cpu";

  let summary = "Enums and operations for the xla_cpu dialect";
  let description = [{
    This dialect contains operations that bridge the gap between HLO and the
    CPU runtime.
  }];
  let cppNamespace = "::mlir::xla_cpu";
}

#endif  // XLA_MLIR_XLA_CPU_DIALECT_TD_
