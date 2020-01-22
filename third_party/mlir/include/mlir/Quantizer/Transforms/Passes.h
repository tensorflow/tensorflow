//===- Passes.h - Quantizer passes  -----------------------------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file defines entry points to create passes to perform various kinds
// of quantization related transforms.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_QUANTIZER_TRANSFORMS_PASSES_H
#define MLIR_QUANTIZER_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace quantizer {

class SolverContext;
class TargetConfiguration;

/// Creates a pass that infers quantized types based on metadata discovered
/// in the computation.
std::unique_ptr<OpPassBase<ModuleOp>>
createInferQuantizedTypesPass(SolverContext &solverContext,
                              const TargetConfiguration &config);

/// Creates a pass which removes any instrumentation and hint ops which have
/// no effect on final runtime.
std::unique_ptr<OpPassBase<FuncOp>> createRemoveInstrumentationPass();

/// Adds default (dummy) statistics to ops that can benefit from runtime stats.
/// Meant for testing.
std::unique_ptr<OpPassBase<FuncOp>> createAddDefaultStatsPass();

} // namespace quantizer
} // namespace mlir

#endif // MLIR_QUANTIZER_TRANSFORMS_PASSES_H
