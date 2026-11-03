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

#include "xla/hlo/pass/hlo_pipeline.pb.h"
#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"

namespace xla::gpu {

// Returns the default HLO pass pipeline of the GPU compiler. So far it covers
// only the pre-SPMD-partitioner passes.
HloPipelineProto GetDefaultCompilationPipeline(
    const AlgebraicSimplifierOptions& layout_insensitive_algsimp_opts);

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_GPU_PIPELINE_H_
