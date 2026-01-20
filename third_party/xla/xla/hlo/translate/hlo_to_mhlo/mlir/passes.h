/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSLATE_HLO_TO_MHLO_MLIR_PASSES_H_
#define XLA_HLO_TRANSLATE_HLO_TO_MHLO_MLIR_PASSES_H_

// NOLINTBEGIN: Used in passes.h.inc
#include <memory>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/DialectConversion.h"
// NOLINTEND

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DECL
#include "xla/hlo/translate/hlo_to_mhlo/mlir/passes.h.inc"

}  // namespace stablehlo_ext
}  // namespace mlir

#endif  // XLA_HLO_TRANSLATE_HLO_TO_MHLO_MLIR_PASSES_H_
