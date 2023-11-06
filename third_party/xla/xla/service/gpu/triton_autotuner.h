/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XLA_SERVICE_GPU_TRITON_AUTOTUNER_H_
#define XLA_SERVICE_GPU_TRITON_AUTOTUNER_H_

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "tsl/platform/threadpool.h"

namespace xla {
namespace gpu {

// Find best tiling configuration for each triton fusion outlined.
class TritonAutotuner : public HloModulePass {
 public:
  explicit TritonAutotuner(const AutotuneConfig& config,
                           tsl::thread::ThreadPool* thread_pool)
      : config_(config), thread_pool_(thread_pool) {}

  absl::string_view name() const override { return "triton-autotuner"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  AutotuneConfig config_;
  tsl::thread::ThreadPool* thread_pool_;
};

// TODO(b/266210099): have a way to generate/load these dynamically.
// Returns a list of possible tilings for a GEMM performed in Triton.
std::vector<TritonGemmConfig> GetPossibleMatmulAutotuneConfigs(
    const HloDotInstruction& dot, se::CudaComputeCapability compute_capability,
    const DebugOptions& debug_options, bool exhaustive_tiling_search = false);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRITON_AUTOTUNER_H_
