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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_TRITON_COLLECTIVE_FUSION_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_TRITON_COLLECTIVE_FUSION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Fuses collective communication operations (like AllReduceStart) with adjacent
// GEMM operations into a custom fusion of type kTritonCollectiveFusionKind.
class TritonCollectiveFusion : public HloModulePass {
 public:
  explicit TritonCollectiveFusion(
      const se::DeviceDescription& device_description)
      : device_description_(device_description) {}

  absl::string_view name() const override { return "triton-collective-fusion"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const se::DeviceDescription& device_description_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_TRITON_COLLECTIVE_FUSION_H_
