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

#ifndef XLA_MLIR_XLA_CPU_ENUMS_TD_
#define XLA_MLIR_XLA_CPU_ENUMS_TD_

include "xla/mlir/xla_cpu/ir/xla_cpu_dialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/PatternBase.td"

def ALL_REDUCE_SUM     : I32EnumAttrCase<"ALL_REDUCE_SUM", 0>;
def ALL_REDUCE_PRODUCT : I32EnumAttrCase<"ALL_REDUCE_PRODUCT", 1>;
def ALL_REDUCE_MIN     : I32EnumAttrCase<"ALL_REDUCE_MIN", 2>;
def ALL_REDUCE_MAX     : I32EnumAttrCase<"ALL_REDUCE_MAX", 3>;

def XlaCpuReductionKind : I32EnumAttr<"ReductionKind",
    "Type of reduction to apply.",
    [ALL_REDUCE_SUM, ALL_REDUCE_PRODUCT, ALL_REDUCE_MIN, ALL_REDUCE_MAX]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::xla_cpu";
}

def XlaCpuReductionKindEnum : EnumAttr<XlaCpuDialect, XlaCpuReductionKind, "reduction_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // XLA_MLIR_XLA_CPU_ENUMS_TD_
