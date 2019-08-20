//===- FxpMathConfig.h - Reference fixed point config -----------*- C++ -*-===//
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
