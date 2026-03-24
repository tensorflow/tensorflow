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

#ifndef XLA_SERVICE_GPU_MODEL_GPU_COLLECTIVE_PERFORMANCE_MODEL_H_
#define XLA_SERVICE_GPU_MODEL_GPU_COLLECTIVE_PERFORMANCE_MODEL_H_

#include <array>
#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

class GpuPerformanceWithCollectiveModel : public GpuPerformanceModelBase {
 public:
  static absl::Duration ComputeCollectiveTime(
      const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
      const se::DeviceDescription& gpu_device_info);

 private:
  static absl::Duration ComputeAllreduceTime(
      const HloInstruction& instr, const GpuHloCostAnalysis* cost_analysis,
      const se::DeviceDescription& gpu_device_info);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_GPU_COLLECTIVE_PERFORMANCE_MODEL_H_
