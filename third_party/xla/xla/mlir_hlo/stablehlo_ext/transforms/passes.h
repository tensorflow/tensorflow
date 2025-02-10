/* Copyright 2023 The StableHLO Authors.

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

#ifndef STABLEHLO_EXT_TRANSFORMS_PASSES_H
#define STABLEHLO_EXT_TRANSFORMS_PASSES_H

#include <memory>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DECL
#include "stablehlo_ext/transforms/passes.h.inc"

void createChloLegalizeToStablehloPipeline(OpPassManager &pm);
std::unique_ptr<OperationPass<func::FuncOp>> createStablehloFlattenTuplePass();

#define GEN_PASS_REGISTRATION
#include "stablehlo_ext/transforms/passes.h.inc"

}  // namespace stablehlo_ext
}  // namespace mlir

#endif  // STABLEHLO_EXT_TRANSFORMS_PASSES_H
