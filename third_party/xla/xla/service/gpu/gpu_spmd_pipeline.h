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

#ifndef XLA_SERVICE_GPU_GPU_SPMD_PIPELINE_H_
#define XLA_SERVICE_GPU_GPU_SPMD_PIPELINE_H_

#include <optional>

#include "absl/functional/function_ref.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/algebraic_simplifier.h"
#include "xla/service/hlo_pass_pipeline.h"

namespace xla {
namespace gpu {

// Adds SPMD passes to the pipeline.
void AddSPMDPasses(
    const HloModule* hlo_module,
    const AlgebraicSimplifierOptions& layout_insensitive_algsimp_opts,
    const se::GpuComputeCapability& compute_capability,
    HloPassPipeline& spmd_pipeline,
    std::optional<const absl::FunctionRef<void(HloPassPipeline&)>>
        auto_sharding_func = std::nullopt);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GPU_SPMD_PIPELINE_H_
