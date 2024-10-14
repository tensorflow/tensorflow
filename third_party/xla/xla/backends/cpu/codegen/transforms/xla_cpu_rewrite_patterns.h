/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_CODEGEN_TRANSFORMS_XLA_CPU_REWRITE_PATTERNS_H_
#define XLA_BACKENDS_CPU_CODEGEN_TRANSFORMS_XLA_CPU_REWRITE_PATTERNS_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace xla::cpu {

// Populates type conversion and legality constraints for lowering XLA:CPU
// types to LLVM types.
void PopulateXlaCpuTypeConversionAndLegality(mlir::TypeConverter& converter,
                                             mlir::ConversionTarget& target);

// Populates rewrite patterns for converting XLA:CPU ops to LLVM ops.
void PopulateXlaCpuConversionPatterns(mlir::RewritePatternSet& patterns);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_TRANSFORMS_XLA_CPU_REWRITE_PATTERNS_H_
