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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_SYCL_GEMM_WORKSPACE_PASS_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_SYCL_GEMM_WORKSPACE_PASS_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Resizes the cuBLASLt matmul workspace to the size oneDNN's matmul primitive
// requires for its scratchpad (oneDNN's term for workspace).
// Added only by IntelGpuCompiler, after the generic GPU post-layout-assignment
// pipeline.
class SyclGemmWorkspacePass : public HloModulePass {
 public:
  explicit SyclGemmWorkspacePass(se::GpuComputeCapability gpu_version)
      : gpu_version_(gpu_version) {}

  absl::string_view name() const override { return "sycl-gemm-workspace"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::GpuComputeCapability gpu_version_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_SYCL_GEMM_WORKSPACE_PASS_H_
