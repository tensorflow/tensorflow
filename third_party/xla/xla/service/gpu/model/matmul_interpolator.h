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

#ifndef XLA_SERVICE_GPU_MODEL_MATMUL_INTERPOLATOR_H_
#define XLA_SERVICE_GPU_MODEL_MATMUL_INTERPOLATOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/interpolator.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

class MatmulInterpolator {
 public:
  static absl::StatusOr<std::unique_ptr<MatmulInterpolator>> Create(
      const HloInstructionProfileList& profiles,
      const se::DeviceDescription& device_info);

  // Returns the estimated runtime for a supported `collective`.
  std::optional<absl::Duration> EstimatedRuntime(const HloInstruction& instr);

 private:
  // Uses `EuclideanNNInterpolator` to figure get the closest neighbour from
  // profiles.
  explicit MatmulInterpolator(
      std::unique_ptr<EuclideanNNInterpolator<int64_t, 4>> interpolator)
      : interpolator_(std::move(interpolator)) {}

  std::unique_ptr<EuclideanNNInterpolator<int64_t, 4>> interpolator_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_MODEL_MATMUL_INTERPOLATOR_H_
