//===- Passes.h - Quantizer passes  -----------------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
