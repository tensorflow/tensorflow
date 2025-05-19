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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_TRITON_FUSION_NUMERICS_VERIFIER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_TRITON_FUSION_NUMERICS_VERIFIER_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/gpu/autotuning/autotuner_compile_util.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// For each Triton fusion in the Hlo module this pass checks that the output
// of the fusion generated via Triton matches the output of the fusion if
// generated with the regular emitters.
class TritonFusionNumericsVerifier : public HloModulePass {
 public:
  explicit TritonFusionNumericsVerifier(const AutotuneConfig& config)
      : config_(config) {}

  static absl::string_view Name() { return "triton-numerics-verifier"; }
  absl::string_view name() const override { return Name(); }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  using FusionCacheKey = std::string;

  int CacheHitsForTestingOnly() const { return cache_hits_; }

 private:
  AutotuneConfig config_;

  // In some models there are many identical fusions. These are cached to avoid
  // expensive recomputations.
  absl::flat_hash_map<FusionCacheKey, absl::Status> fusion_result_cache_;
  int cache_hits_ = 0;  // used for testing only.
};

namespace triton_fusion_numerics_pass_internal {
// These are exposed only for testing. Do not use.
absl::StatusOr<ScopedShapedBuffer> CompileAndRunFusion(
    AutotunerCompileUtil& util, const HloFusionInstruction& fusion,
    const AutotuneConfig& config, const DebugOptions& debug_opts,
    bool disable_triton);
absl::Status CompareBuffers(const ScopedShapedBuffer& current,
                            const ScopedShapedBuffer& expected,
                            const Shape& shape, const DebugOptions& debug_opts,
                            se::Stream* stream);
absl::Status ForAllTritonFusions(
    const HloModule& module,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    absl::AnyInvocable<absl::Status(const HloFusionInstruction&)> fn);
}  // namespace triton_fusion_numerics_pass_internal

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TRANSFORMS_TRITON_FUSION_NUMERICS_VERIFIER_H_
