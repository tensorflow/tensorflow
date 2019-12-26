//===- FxpMathConfig.h - Reference fixed point config -----------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a TargetConfiguration for reference fixed-point math
// quantization scheme based on the FxpMathOps (plus a small category of
// extension ops that can be added from other dialects).
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_QUANTIZER_CONFIGURATIONS_FXPMATHCONFIG_H
#define MLIR_QUANTIZER_CONFIGURATIONS_FXPMATHCONFIG_H

#include "mlir/Quantizer/Support/Configuration.h"
#include "mlir/Quantizer/Support/Metadata.h"

namespace mlir {
namespace quantizer {

/// Target configuration for a reference affine/fixed-point quantization
/// scheme defined in terms of the FxpMathOps dialect. This can be extended
/// with select ops from other dialects by way of the following public
/// methods:
///   - addValueIdentityOp
class FxpMathTargetConfig : public TargetConfiguration {
public:
  /// Creates an FxpMathTargetConfig instance which can be further customized.
  static std::unique_ptr<FxpMathTargetConfig> create(SolverContext &context);

protected:
  FxpMathTargetConfig(SolverContext &context) : TargetConfiguration(context) {}
};

} // namespace quantizer
} // namespace mlir

#endif // MLIR_QUANTIZER_CONFIGURATIONS_FXPMATHCONFIG_H
