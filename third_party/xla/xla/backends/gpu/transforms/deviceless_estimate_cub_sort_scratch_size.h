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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_DEVICELESS_ESTIMATE_CUB_SORT_SCRATCH_SIZE_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_DEVICELESS_ESTIMATE_CUB_SORT_SCRATCH_SIZE_H_

#include <cstdint>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/shape.h"
#include "xla/stream_executor/semantic_version.h"

namespace xla::gpu {

// Updates the scratch size of CUB sort custom calls to match the actual scratch
// size. Unlike `EstimateCubSortScratchSize` this pass doesn't require the
// device to be present. It looks up the scratch size in a bundled lookup table.
class DevicelessEstimateCubSortScratchSize : public HloModulePass {
 public:
  explicit DevicelessEstimateCubSortScratchSize(
      std::string platform_name, std::string device_name,
      stream_executor::SemanticVersion cub_version)
      : platform_name_(std::move(platform_name)),
        device_name_(std::move(device_name)),
        cub_version_(cub_version) {}

  absl::string_view name() const override {
    return "deviceless-estimate-cub-sort-scratch-size";
  }

 protected:
  absl::Status RunOnSortInstruction(HloCustomCallInstruction* custom_call);
  absl::StatusOr<bool> RunOnComputation(HloComputation* computation);

  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::StatusOr<int64_t> CalculateDevicelessScratchSize(
      HloCustomCallInstruction* custom_call, const Shape& key_shape,
      bool is_pairs, int64_t num_items, int64_t batch_size);

  std::string platform_name_;
  std::string device_name_;
  stream_executor::SemanticVersion cub_version_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_DEVICELESS_ESTIMATE_CUB_SORT_SCRATCH_SIZE_H_
