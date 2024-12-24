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
#ifndef XLA_BACKENDS_GPU_CODEGEN_TRANSFORMS_PASSES_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRANSFORMS_PASSES_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DECL
#include "xla/backends/gpu/codegen/transforms/passes.h.inc"

// Returns the range of a given value, if it can be statically determined.
std::optional<Interval> GetRange(mlir::Value value);

// Returns the range for the induction variable, if it can be statically
// determined.
std::optional<Interval> GetIVRange(mlir::Value iv);

std::unique_ptr<mlir::Pass> CreateConvertFloatNvidiaPass();
std::optional<std::unique_ptr<mlir::Pass>> MaybeCreateConvertFloatNvidiaPass(
    const se::DeviceDescription& device_description);
std::unique_ptr<mlir::Pass> CreateConvertPureCallOpsPass();
std::unique_ptr<mlir::Pass> CreateEraseDeadFunctionsPass();
std::unique_ptr<mlir::Pass> CreateExpandFloatOpsPass();
std::unique_ptr<mlir::Pass> CreateFlattenTensorsPass();
std::unique_ptr<mlir::Pass> CreateLowerTensorsPass(
    const std::string& gpu_device_info = "");
std::unique_ptr<mlir::Pass> CreateLowerTensorsPass(
    const se::DeviceDescription& device_description);
std::unique_ptr<mlir::Pass> CreateLowerToLLVMPass(
    const std::string& gpu_device_info = "");
std::unique_ptr<mlir::Pass> CreateLowerToLLVMPass(
    const se::DeviceDescription& device_description);
std::unique_ptr<mlir::Pass> CreateLowerXlaGpuToScfPass(int64_t warp_size = 32);
std::unique_ptr<mlir::Pass> CreateLowerXlaGpuLoopsToScfPass();
std::unique_ptr<mlir::Pass> CreateMergePointersToSameSlicePass();
std::unique_ptr<mlir::Pass> CreateOptimizeLoopsPass();
std::unique_ptr<mlir::Pass> CreateFuseLoopsPass();
std::unique_ptr<mlir::Pass> CreatePeelLoopsPass();
std::unique_ptr<mlir::Pass> CreatePropagateSliceIndicesPass();
std::unique_ptr<mlir::Pass> CreateSimplifyAffinePass();
std::unique_ptr<mlir::Pass> CreateSimplifyArithPass();
std::unique_ptr<mlir::Pass> CreateUnswitchLoopsPass();
std::unique_ptr<mlir::Pass> CreateVectorizeLoadsAndStoresPass();

#define GEN_PASS_REGISTRATION
#include "xla/backends/gpu/codegen/transforms/passes.h.inc"

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRANSFORMS_PASSES_H_
