/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_PASS_DETAIL_H
#define MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_PASS_DETAIL_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace func {
class FuncOp;
}  // namespace func
namespace scf {
class SCFDialect;
}  // namespace scf
}  // namespace mlir

#define GEN_PASS_CLASSES
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h.inc"

#endif  // MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_PASS_DETAIL_H
