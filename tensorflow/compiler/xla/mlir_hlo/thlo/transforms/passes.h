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

#ifndef MLIR_HLO_DIALECT_THLO_TRANSFORMS_PASSES_H
#define MLIR_HLO_DIALECT_THLO_TRANSFORMS_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {

template <typename T>
class OperationPass;

namespace func {
class FuncOp;
}  // namespace func

namespace thlo {

#define GEN_PASS_DECL_THLOLEGALIZESORTPASS
#include "thlo/transforms/thlo_passes.h.inc"

/// Lowers sort to Arith, MemRef, and SCF
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeSortPass();

#define GEN_PASS_REGISTRATION
#include "thlo/transforms/thlo_passes.h.inc"

}  // namespace thlo
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_THLO_TRANSFORMS_PASSES_H
