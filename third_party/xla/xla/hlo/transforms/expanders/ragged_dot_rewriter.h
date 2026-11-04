/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_EXPANDERS_RAGGED_DOT_REWRITER_H_
#define XLA_HLO_TRANSFORMS_EXPANDERS_RAGGED_DOT_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"
#include "xla/stream_executor/semantic_version.h"

namespace xla {

inline const stream_executor::dnn::VersionInfo
    kMinCudnnVersionForRaggedDotFusion(9, 22);

// The ragged-dot wgrad path for kRaggedContracting lowers to cuDNN
// moe_grouped_matmul_bwd, which requires cuDNN 9.24 or newer, cuBLASLt 13.5 or
// newer from CUDA Toolkit 13.3, and compute capability SM90 through SM119.
inline const stream_executor::dnn::VersionInfo
    kMinCudnnVersionForRaggedDotWgradFusion(9, 24);
inline constexpr stream_executor::SemanticVersion
    kMinCudaToolkitVersionForRaggedDotWgradFusion{13, 3, 0};

bool SupportsCudnnRaggedDotWgrad(
    const stream_executor::GpuComputeCapability& gpu_compute_capability,
    stream_executor::dnn::VersionInfo cudnn_version,
    stream_executor::SemanticVersion toolkit_version);

// RaggedDotRewriter converts ragged dots to general dots through expansion.
class RaggedDotRewriter : public HloModulePass {
 public:
  explicit RaggedDotRewriter(
      se::GpuComputeCapability gpu_compute_capability,
      stream_executor::dnn::VersionInfo cudnn_version =
          stream_executor::dnn::VersionInfo(),
      stream_executor::SemanticVersion toolkit_version = {0, 0, 0})
      : gpu_compute_capability_(gpu_compute_capability),
        cudnn_version_(cudnn_version),
        toolkit_version_(toolkit_version) {}

  absl::string_view name() const override { return "ragged_dot_rewriter"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  stream_executor::GpuComputeCapability gpu_compute_capability_;
  stream_executor::dnn::VersionInfo cudnn_version_;
  stream_executor::SemanticVersion toolkit_version_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_EXPANDERS_RAGGED_DOT_REWRITER_H_
