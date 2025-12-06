/* Copyright 2025 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================*/

#ifndef XLA_CODEGEN_EMITTERS_TRANSFORMS_GPU_CONVERSION_PATTERNS_H_
#define XLA_CODEGEN_EMITTERS_TRANSFORMS_GPU_CONVERSION_PATTERNS_H_

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "xla/codegen/device_spec.h"

namespace xla {
namespace emitters {

// Populates GPU dialect conversion patterns for lowering to LLVM.
mlir::LogicalResult PopulateGpuDialectConversionPatterns(
    const DeviceSpec& device_spec, mlir::LLVMTypeConverter& type_converter,
    mlir::RewritePatternSet& patterns, mlir::LLVMConversionTarget& target,
    mlir::MLIRContext* context);

}  // namespace emitters
}  // namespace xla

#endif  // XLA_CODEGEN_EMITTERS_TRANSFORMS_GPU_CONVERSION_PATTERNS_H_
