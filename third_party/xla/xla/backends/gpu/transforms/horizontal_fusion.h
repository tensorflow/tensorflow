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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_HORIZONTAL_FUSION_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_HORIZONTAL_FUSION_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Groups consecutive independent small-grid fusion instructions in a scheduled
// HLO module into an horizontal fusions whose fused computation contains nested
// fusions and a root tuple of their outputs.
class HorizontalFusion : public HloModulePass {
 public:
  struct Options {
    // Maximum number of blocks in the resulting horizontal fusion, as a factor
    // of device_info.core_count() (total_blocks <= max_blocks_per_sm_factor *
    // device_info.core_count()).
    int64_t max_blocks_per_sm_factor = 1;
    // Maximum number of nested kernels in a single horizontal fusion.
    int64_t max_nested_kernels = 8;
  };

  explicit HorizontalFusion(const se::DeviceDescription& device_info)
      : HorizontalFusion(device_info, Options{}) {}

  HorizontalFusion(const se::DeviceDescription& device_info,
                   const Options& options)
      : device_info_(device_info), options_(options) {}

  absl::string_view name() const override { return "horizontal-fusion"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::DeviceDescription device_info_;
  Options options_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_HORIZONTAL_FUSION_H_
