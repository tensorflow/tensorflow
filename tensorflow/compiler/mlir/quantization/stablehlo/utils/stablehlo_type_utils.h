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

#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_UTILS_STABLEHLO_TYPE_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_UTILS_STABLEHLO_TYPE_UTILS_H_

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo

namespace mlir::quant::stablehlo {

// Checks if an op is from StableHLO dialect.
inline bool IsStablehloOp(Operation* op) {
  return op->getDialect()->getNamespace() ==
         mlir::stablehlo::StablehloDialect::getDialectNamespace();
}

}  // namespace mlir::quant::stablehlo

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_UTILS_STABLEHLO_TYPE_UTILS_H_
