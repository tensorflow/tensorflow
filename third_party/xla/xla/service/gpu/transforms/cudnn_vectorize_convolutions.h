/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_CUDNN_VECTORIZE_CONVOLUTIONS_H_
#define XLA_SERVICE_GPU_TRANSFORMS_CUDNN_VECTORIZE_CONVOLUTIONS_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/dnn.h"

namespace xla {
namespace gpu {

// Changes the shape of cudnn convolutions to allow faster "vectorized"
// algorithms.
//
// On sm61+ will convert int8_t convolutions from
//
//   - [N, C, H, W] to [N, C/4, H, W, 4],
//
// assuming C is divisible by 4.
//
// On sm75+ will convert int8_t convolutions from
//
//   - [N, C, H, W]      to [N, C/32, H, W, 32],
//   - [N, C/4, H, W, 4] to [N, C/32, H, W, 32], and
//   - [N, C, H, W]      to [N,  C/4, H, W,  4] (same as sm61+),
//
// assuming C is divisible by 4 or 32.
//
// This pass will not pad the channel dim to a multiple of 4 or 32, so you
// should run CudnnPadForConvolutions before this.
class CudnnVectorizeConvolutions : public HloModulePass {
 public:
  explicit CudnnVectorizeConvolutions(
      se::CudaComputeCapability compute_capability,
      se::dnn::VersionInfo cudnn_version)
      : compute_capability_(compute_capability),
        cudnn_version_(cudnn_version) {}

  absl::string_view name() const override {
    return "cudnn_vectorize_convolutions";
  }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const se::CudaComputeCapability compute_capability_;
  const se::dnn::VersionInfo cudnn_version_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_CUDNN_VECTORIZE_CONVOLUTIONS_H_
