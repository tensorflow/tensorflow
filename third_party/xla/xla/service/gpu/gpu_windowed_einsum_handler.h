/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_GPU_WINDOWED_EINSUM_HANDLER_H_
#define XLA_SERVICE_GPU_GPU_WINDOWED_EINSUM_HANDLER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla::gpu {

// This pass is targeting the windowed einsum optimization
// in the SPMD pipeline. It rewrites all-gather+gemm or
// gemm+reduce-scatter into sharded loops to achieve overlap
// between sharded gemms and communication. This pass will
// optimize it on GPU by annotating independent gemms with
// stream ids in the backend config. By running them in different
// streams, we can practically achieve overlap between gemms too.
class GpuWindowedEinsumHandler : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "gpu-windowed-einsum-handler";
  }

  struct WindowedEinsumAgLoops {
    WindowedEinsumAgLoops(HloInstruction* loop) : loop(loop) {}
    HloInstruction* loop;
    bool consumed = false;
  };

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  constexpr static const char* kWindowedEinsumRsLoopName =
      "windowed_dot_general_body_rs";
  constexpr static const char* kWindowedEinsumAgLoopName =
      "windowed_dot_general_body_ag";

 private:
  std::vector<WindowedEinsumAgLoops> all_ag_loops_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_GPU_WINDOWED_EINSUM_HANDLER_H_
