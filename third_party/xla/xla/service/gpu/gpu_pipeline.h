/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_GPU_PIPELINE_H_
#define XLA_SERVICE_GPU_GPU_PIPELINE_H_

#include "absl/status/statusor.h"
#include "xla/hlo/pass/hlo_pass_context.h"
#include "xla/hlo/pass/hlo_pipeline.pb.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"
#include "xla/service/hlo_module_config.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// Temporarily exposed while `gpu_compiler.cc` and backend subclasses still
// build parts of the pipeline in C++. Move into `gpu_pipeline.cc` once the
// entire pipeline is migrated.
enum class AlgebraicSimplifierMode {
  kLayoutInsensitive,
  kPostFusionSimplification,
  kLayoutNormalization,
  kPostLayoutAssignment,
  kAfterSimplifyFPConversions,
  kGpuConvoluationCanonicalization,
};

AlgebraicSimplifierOptions GetAlgebraicSimplifierOptions(
    AlgebraicSimplifierMode mode, const DebugOptions& debug_options,
    bool is_cuda_platform);

// Returns the default HLO pass pipeline of the GPU compiler. So far it covers
// only the pre-SPMD-partitioner passes.
absl::StatusOr<HloPipelineProto> GetDefaultCompilationPipeline(
    const HloModuleConfig& config, const HloPassContext& context);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_GPU_PIPELINE_H_
