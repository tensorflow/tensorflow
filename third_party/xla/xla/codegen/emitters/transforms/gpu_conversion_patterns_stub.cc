/* Copyright 2024 The OpenXLA Authors.
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

#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/device_spec.h"
#include "xla/codegen/emitters/transforms/gpu_conversion_patterns.h"
#include "xla/tsl/platform/logging.h"

namespace xla {
namespace emitters {

mlir::LogicalResult PopulateGpuDialectConversionPatterns(
    const DeviceSpec& device_spec, mlir::LLVMTypeConverter& type_converter,
    mlir::RewritePatternSet& patterns, mlir::LLVMConversionTarget& target,
    mlir::MLIRContext* context) {
  LOG(FATAL) << "GPU lowering patterns are only available in GPU builds";
  return mlir::failure();
}

}  // namespace emitters
}  // namespace xla
