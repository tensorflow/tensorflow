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

#include "xla/codegen/emitters/transforms/gpu_conversion_patterns.h"

#include <string>

#include "llvm/Support/LogicalResult.h"
#include "mlir/Conversion/GPUToLLVMSPV/GPUToLLVMSPVPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/device_spec.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace emitters {

mlir::LogicalResult PopulateGpuDialectConversionPatterns(
    const DeviceSpec& device_spec, mlir::LLVMTypeConverter& type_converter,
    mlir::RewritePatternSet& patterns, mlir::LLVMConversionTarget& target,
    mlir::MLIRContext* context) {
  if (device_spec.IsAmdGpu()) {
    std::string chipset =
        device_spec.gpu().rocm_compute_capability().gfx_version();
    llvm::FailureOr<mlir::amdgpu::Chipset> maybeChipset =
        mlir::amdgpu::Chipset::parse(chipset);
    if (failed(maybeChipset)) {
      mlir::emitError(mlir::UnknownLoc::get(context),
                      "Invalid chipset name: " + chipset);
      return mlir::failure();
    }
    mlir::populateGpuToROCDLConversionPatterns(type_converter, patterns,
                                               mlir::gpu::amd::Runtime::Unknown,
                                               *maybeChipset);
    mlir::configureGpuToROCDLConversionLegality(target);
  } else if (device_spec.IsIntelGpu()) {
    populateGpuToLLVMSPVConversionPatterns(type_converter, patterns);
    populateGpuMemorySpaceAttributeConversions(type_converter);
  } else {
    mlir::populateGpuToNVVMConversionPatterns(type_converter, patterns);
    mlir::configureGpuToNVVMConversionLegality(target);
  }
  return mlir::success();
}

}  // namespace emitters
}  // namespace xla
