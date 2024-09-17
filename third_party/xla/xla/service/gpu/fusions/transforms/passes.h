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
#ifndef XLA_SERVICE_GPU_FUSIONS_TRANSFORMS_PASSES_H_
#define XLA_SERVICE_GPU_FUSIONS_TRANSFORMS_PASSES_H_

#include <memory>
#include <optional>
#include <string>

#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

#define GEN_PASS_DECL
#include "xla/service/gpu/fusions/transforms/passes.h.inc"

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
    bool is_amd_gpu = false, const std::string& gpu_arch = "6.0");
std::unique_ptr<mlir::Pass> CreateLowerToLLVMPass(bool use_rocdl);
std::unique_ptr<mlir::Pass> CreateLowerXlaGpuToScfPass();
std::unique_ptr<mlir::Pass> CreateLowerXlaGpuLoopsToScfPass();
std::unique_ptr<mlir::Pass> CreateMergePointersToSameSlicePass();
std::unique_ptr<mlir::Pass> CreateOptimizeLoopsPass();
std::unique_ptr<mlir::Pass> CreatePeelLoopsPass();
std::unique_ptr<mlir::Pass> CreatePropagateSliceIndicesPass();
std::unique_ptr<mlir::Pass> CreateSimplifyAffinePass();
std::unique_ptr<mlir::Pass> CreateSimplifyArithPass();
std::unique_ptr<mlir::Pass> CreateUnswitchLoopsPass();
std::unique_ptr<mlir::Pass> CreateVectorizeLoadsAndStoresPass();

#define GEN_PASS_REGISTRATION
#include "xla/service/gpu/fusions/transforms/passes.h.inc"

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_TRANSFORMS_PASSES_H_
