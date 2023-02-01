/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_XLA_LEGALIZE_TARGETS_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_XLA_LEGALIZE_TARGETS_H_

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir {
namespace mhlo {

// Returns a ConversionTarget that includes default legalized MLIR dialects
// for conversion to XLA.
// If legalize_chlo is true, the resulting conversion target cannot have CHLO.
mlir::ConversionTarget GetDefaultLegalConversionTargets(
    MLIRContext& mlir_context, bool legalize_chlo);

}  // namespace mhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_XLA_LEGALIZE_TARGETS_H_
